// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/cpu_tuning.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>

#if defined(_WIN32)
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h> // Get/SetThreadPriority
#endif

#if defined(_WIN32) && defined(XLLAMA_USE_LLAMA)
    #include "ggml.h"              // GGML_OP_COUNT
    #include "xllama/ggml_d3d12.h" // d3d12_flag_enabled
    #include "xllama/path_utils.h"
    #include "xllama/utf8_utils.h"

// Defined by patches/0002-msvc-cpu-relax.patch in ggml-cpu.c; every Windows
// build that links ggml applies the UWP patches first. An ORT-only build has
// no ggml, hence the XLLAMA_USE_LLAMA guard.
extern "C" int xllama_ggml_cpu_relax_pause;
// Defined by patches/0003-cpu-op-timing.patch in ggml-cpu.c.
extern "C" int xllama_ggml_cpu_op_timing;
extern "C" int64_t xllama_ggml_cpu_op_us[GGML_OP_COUNT];
extern "C" int64_t xllama_ggml_cpu_op_n[GGML_OP_COUNT];
extern "C" int64_t xllama_ggml_cpu_barrier_us;
extern "C" int64_t xllama_ggml_cpu_graphs;
#endif

namespace xllama {

int parse_cpu_knob_int(std::string_view text, int lo, int hi, int fallback) {
    std::size_t b = 0, e = text.size();
    while (b < e && std::isspace(static_cast<unsigned char>(text[b])))
        ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(text[e - 1])))
        --e;
    if (b == e || e - b > 9)
        return fallback;
    long v = 0;
    bool neg = false;
    std::size_t i = b;
    if (text[i] == '-' || text[i] == '+') {
        neg = text[i] == '-';
        if (++i == e)
            return fallback;
    }
    for (; i < e; ++i) {
        if (!std::isdigit(static_cast<unsigned char>(text[i])))
            return fallback;
        v = v * 10 + (text[i] - '0');
    }
    if (neg)
        v = -v;
    return v < lo || v > hi ? fallback : static_cast<int>(v);
}

// Knobs only matter where ggml is linked (d3d12_flag_enabled lives in
// ggml_d3d12.cpp, compiled to nothing in an ORT-only build).
#if defined(_WIN32) && defined(XLLAMA_USE_LLAMA)
namespace {
// Knob file text; empty when absent.
std::string read_knob_text(const char* name) {
    FILE* f = _wfopen(utf8_to_wstring(resolve_local_path(name)).c_str(), L"rb");
    if (!f)
        return {};
    char buf[32] = {};
    const std::size_t n = std::fread(buf, 1, sizeof(buf), f);
    std::fclose(f);
    return std::string(buf, n);
}
} // namespace

CpuDecodeTuning read_cpu_decode_tuning() {
    CpuDecodeTuning t;
    t.decode_threads = parse_cpu_knob_int(read_knob_text("decode_threads.txt"), 1, 64, 0);
    t.decode_poll = parse_cpu_knob_int(read_knob_text("decode_poll.txt"), 0, 100, -1);
    t.relax_pause = d3d12_flag_enabled(read_knob_text("cpu-relax-pause.txt"));
    t.timing = d3d12_flag_enabled(read_knob_text("cpu-timing.txt"));
    t.op_timing = d3d12_flag_enabled(read_knob_text("cpu-op-timing.txt"));
    t.decode_prio = parse_cpu_knob_int(read_knob_text("decode_prio.txt"), 0, 2, -1);
    return t;
}
#else
CpuDecodeTuning read_cpu_decode_tuning() {
    return {};
}
#endif

#if defined(_WIN32) && defined(XLLAMA_USE_LLAMA)
void apply_cpu_relax_pause(bool on) {
    xllama_ggml_cpu_relax_pause = on ? 1 : 0;
}
void apply_cpu_op_timing(bool on) {
    xllama_ggml_cpu_op_timing = on ? 1 : 0;
}

bool take_cpu_op_stats(CpuOpStats* out) {
    if (out) {
        out->op_us.assign(xllama_ggml_cpu_op_us, xllama_ggml_cpu_op_us + GGML_OP_COUNT);
        out->op_n.assign(xllama_ggml_cpu_op_n, xllama_ggml_cpu_op_n + GGML_OP_COUNT);
        out->barrier_us = xllama_ggml_cpu_barrier_us;
        out->graphs = xllama_ggml_cpu_graphs;
    }
    std::fill(xllama_ggml_cpu_op_us, xllama_ggml_cpu_op_us + GGML_OP_COUNT, 0);
    std::fill(xllama_ggml_cpu_op_n, xllama_ggml_cpu_op_n + GGML_OP_COUNT, 0);
    xllama_ggml_cpu_barrier_us = 0;
    xllama_ggml_cpu_graphs = 0;
    return true;
}
#else
void apply_cpu_relax_pause(bool) {}
void apply_cpu_op_timing(bool) {}
bool take_cpu_op_stats(CpuOpStats*) {
    return false;
}
#endif

#if defined(_WIN32)
int current_thread_priority() {
    const int p = GetThreadPriority(GetCurrentThread());
    return p == THREAD_PRIORITY_ERROR_RETURN ? kNoThreadPriority : p;
}

void set_current_thread_priority(int prio) {
    if (prio != kNoThreadPriority)
        SetThreadPriority(GetCurrentThread(), prio);
}
#else
int current_thread_priority() {
    return kNoThreadPriority;
}

void set_current_thread_priority(int) {}
#endif

std::string format_cpu_tuning(const CpuDecodeTuning& t, int batch_threads) {
    // An unset decode_poll / decode_prio prints "default": any explicit value,
    // even ggml's own (50, 0), also splits off the prefill pool.
    const std::string poll = t.decode_poll >= 0 ? std::to_string(t.decode_poll) : "default";
    const std::string prio = t.decode_prio >= 0 ? std::to_string(t.decode_prio) : "default";
    char buf[200];
    std::snprintf(buf, sizeof(buf),
                  "[xllama] cpu-tuning decode_threads=%d batch_threads=%d decode_poll=%s "
                  "relax_pause=%d timing=%d op_timing=%d decode_prio=%s\n",
                  t.decode_threads > 0 ? t.decode_threads : batch_threads, batch_threads,
                  poll.c_str(), t.relax_pause ? 1 : 0, t.timing ? 1 : 0, t.op_timing ? 1 : 0,
                  prio.c_str());
    return buf;
}

std::string format_cpu_op_profile(const char* phase, const std::vector<CpuOpRow>& rows,
                                  double barrier_ms, std::int64_t graphs, int tokens) {
    if (!phase)
        phase = "-";
    std::vector<const CpuOpRow*> order;
    double ops_ms = 0.0;
    for (const CpuOpRow& r : rows) {
        if (r.count <= 0)
            continue;
        order.push_back(&r);
        ops_ms += r.ms;
    }
    std::stable_sort(order.begin(), order.end(),
                     [](const CpuOpRow* a, const CpuOpRow* b) { return a->ms > b->ms; });
    std::string out;
    char buf[200];
    for (const CpuOpRow* r : order) {
        std::snprintf(buf, sizeof(buf), "[xllama] cpu-op phase=%s op=%s ms=%.3f count=%lld\n",
                      phase, r->op.c_str(), r->ms, static_cast<long long>(r->count));
        out += buf;
    }
    char per[32];
    if (tokens > 0)
        std::snprintf(per, sizeof(per), "%.2f",
                      static_cast<double>(graphs) / static_cast<double>(tokens));
    else
        std::snprintf(per, sizeof(per), "missing");
    std::snprintf(buf, sizeof(buf),
                  "[xllama] cpu-sync phase=%s graphs=%lld graphs_per_token=%s barrier_ms=%.3f "
                  "ops_ms=%.3f tokens=%d\n",
                  phase, static_cast<long long>(graphs), per, barrier_ms, ops_ms, tokens);
    out += buf;
    return out;
}

std::string format_cpu_phase(const char* phase, const CpuPhaseTimes& t) {
    char buf[200];
    std::snprintf(buf, sizeof(buf),
                  "[xllama] cpu-phase phase=%s llama_decode_ms=%.3f sample_ms=%.3f emit_ms=%.3f "
                  "tokens=%d\n",
                  phase, t.llama_decode_ms, t.sample_ms, t.emit_ms, t.tokens);
    return buf;
}

} // namespace xllama
