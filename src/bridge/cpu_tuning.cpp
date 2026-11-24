// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/cpu_tuning.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>

#if defined(_WIN32) && defined(XLLAMA_USE_LLAMA)
    #include "xllama/ggml_d3d12.h" // d3d12_flag_enabled
    #include "xllama/path_utils.h"
    #include "xllama/utf8_utils.h"

// Defined by patches/0002-msvc-cpu-relax.patch in ggml-cpu.c; every Windows
// build that links ggml applies the UWP patches first. An ORT-only build has
// no ggml, hence the XLLAMA_USE_LLAMA guard.
extern "C" int xllama_ggml_cpu_relax_pause;
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
#else
void apply_cpu_relax_pause(bool) {}
#endif

std::string format_cpu_tuning(const CpuDecodeTuning& t, int batch_threads) {
    char buf[160];
    std::snprintf(buf, sizeof(buf),
                  "[xllama] cpu-tuning decode_threads=%d batch_threads=%d decode_poll=%d "
                  "relax_pause=%d timing=%d\n",
                  t.decode_threads > 0 ? t.decode_threads : batch_threads, batch_threads,
                  t.decode_poll >= 0 ? t.decode_poll : 50, t.relax_pause ? 1 : 0, t.timing ? 1 : 0);
    return buf;
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
