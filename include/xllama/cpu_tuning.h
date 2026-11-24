// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// CPU-side knobs and timing for a GGUF decode whose matmuls run on d3d12.
// After the d3d12 cuts a large share of a decode token is CPU time outside
// the backend (norms, KV writes, attention, residual adds, many small CPU
// splits per token, sampling). These knobs exist to measure and A/B that part; all
// default off, so an absent knob leaves the shipping behaviour unchanged.
// Fork experiment: only run_inference (bench.flag, CLI) reads them; LlamaSession
// (chat, LAN API) does not until a knob earns its place by an A/B.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace xllama {

struct CpuDecodeTuning {
    // decode_threads.txt: threads of the decode (1-token ubatch) pool, 1..64.
    // 0 = the decode pool has the prefill thread count.
    int decode_threads = 0;
    // decode_poll.txt: ggml threadpool polling level of the decode pool,
    // 0..100. -1 = ggml default (50). Any value also splits off a separate
    // prefill pool (2x the OS threads), so A/B it against decode_poll.txt=50,
    // not against an absent knob.
    int decode_poll = -1;
    // cpu-relax-pause.txt = "1": _mm_pause in ggml-cpu spin loops under MSVC
    // (patches/0002-msvc-cpu-relax.patch).
    bool relax_pause = false;
    // cpu-timing.txt = "1": log one cpu-phase line per decode (not with
    // prompt lookup, whose speculative samples the timers do not cover).
    bool timing = false;
    // cpu-op-timing.txt = "1": per-op CPU time of the decode from thread 0 of
    // the ggml-cpu pool (patches/0003-cpu-op-timing.patch), logged as cpu-op
    // and cpu-sync lines. Also logs the cpu-phase line, so one run gives both
    // sides of the cpu-op + barrier ~ llama_decode_ms - d3d12 wall_ms check.
    bool op_timing = false;
    // decode_prio.txt: ggml_sched_priority of the decode pool, 0..2 =
    // NORMAL / MEDIUM (ABOVE_NORMAL) / HIGH (HIGHEST); REALTIME is out of
    // range. -1 = ggml default (NORMAL). Like decode_poll, any value splits
    // off a separate prefill pool, so A/B it against decode_prio.txt=0.
    // ggml_threadpool_new also applies the priority to the CALLING thread
    // (the inference thread, which is thread 0 of both pools), so prefill of
    // that run runs raised too; GgufCpuThreadpools::reset puts the thread's
    // previous priority back when the run ends.
    int decode_prio = -1;

    bool any() const {
        return decode_threads > 0 || decode_poll >= 0 || relax_pause || timing || op_timing ||
               decode_prio >= 0;
    }
};

// Integer in [lo, hi] from a knob file's text (surrounding whitespace
// allowed); |fallback| when empty, not a number, trailing junk or out of range.
int parse_cpu_knob_int(std::string_view text, int lo, int hi, int fallback);

// Reads the knobs from LocalState on Windows builds with ggml. Elsewhere
// it returns the defaults (on Linux resolve_local_path is the bare name).
CpuDecodeTuning read_cpu_decode_tuning();

// Sets the patched ggml-cpu spin-pause flag (Windows builds with ggml; a
// no-op elsewhere). A plain global read by pool workers: call it before the
// threadpools are created.
void apply_cpu_relax_pause(bool on);

// Sets the patched ggml-cpu op-timing flag (patches/0003-cpu-op-timing.patch;
// Windows builds with ggml, a no-op elsewhere). Read by thread 0 per graph.
void apply_cpu_op_timing(bool on);

// Per-op CPU time accumulated by the patched ggml-cpu since the last take.
// Indexed by ggml_op; the caller names them (ggml_op_name), so this header
// stays free of ggml.
struct CpuOpStats {
    std::vector<std::int64_t> op_us;
    std::vector<std::int64_t> op_n;
    std::int64_t barrier_us = 0; // thread 0 waiting in ggml_barrier
    std::int64_t graphs = 0;     // ggml-cpu graph computes (CPU splits)
};

// Copies the counters into |out| (when non-null) and zeroes them. Call only
// while the pool is idle (between llama_decode calls). False and no-op where
// the patch is not linked (Linux, ORT-only builds).
bool take_cpu_op_stats(CpuOpStats* out);

struct CpuOpRow {
    std::string op;
    double ms = 0.0;
    std::int64_t count = 0;
};

// One "[xllama] cpu-op phase=… op=… ms=… count=…" line per row with count > 0,
// by ms descending (ties keep input order), then
// "[xllama] cpu-sync phase=… graphs=… graphs_per_token=… barrier_ms=… ops_ms=… tokens=…".
// graphs_per_token is "missing" when tokens <= 0.
std::string format_cpu_op_profile(const char* phase, const std::vector<CpuOpRow>& rows,
                                  double barrier_ms, std::int64_t graphs, int tokens);

// Calling thread's OS priority (GetThreadPriority) and setter, so a decode_prio
// run can undo what ggml_threadpool_new did to it. kNoThreadPriority off
// Windows or on error; setting it is a no-op.
inline constexpr int kNoThreadPriority = -1000;
int current_thread_priority();
void set_current_thread_priority(int prio);

// "[xllama] cpu-tuning decode_threads=2 batch_threads=6 decode_poll=50 relax_pause=1 timing=0
// op_timing=0 decode_prio=default\n" (one line). decode_threads prints the effective value;
// decode_poll and decode_prio print "default" when unset.
std::string format_cpu_tuning(const CpuDecodeTuning& t, int batch_threads);

struct CpuPhaseTimes {
    // Wall time of llama_decode of sampled tokens: CPU splits AND d3d12
    // submissions/waits. CPU outside the backend = this - d3d12-phase wall_ms.
    double llama_decode_ms = 0.0;
    double sample_ms = 0.0; // sampler chain
    double emit_ms = 0.0;   // detokenize + stream + stop-sequence check
    int tokens = 0;         // tokens through llama_decode
};

// "[xllama] cpu-phase phase=decode llama_decode_ms=… sample_ms=… emit_ms=… tokens=N\n"
std::string format_cpu_phase(const char* phase, const CpuPhaseTimes& t);

} // namespace xllama
