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

#include <string>
#include <string_view>

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

    bool any() const {
        return decode_threads > 0 || decode_poll >= 0 || relax_pause || timing;
    }
};

// Integer in [lo, hi] from a knob file's text (surrounding whitespace
// allowed); |fallback| when empty, not a number, trailing junk or out of range.
int parse_cpu_knob_int(std::string_view text, int lo, int hi, int fallback);

// Reads the four knobs from LocalState on Windows builds with ggml. Elsewhere
// it returns the defaults (on Linux resolve_local_path is the bare name).
CpuDecodeTuning read_cpu_decode_tuning();

// Sets the patched ggml-cpu spin-pause flag (Windows builds with ggml; a
// no-op elsewhere). A plain global read by pool workers: call it before the
// threadpools are created.
void apply_cpu_relax_pause(bool on);

// "[xllama] cpu-tuning decode_threads=2 batch_threads=6 decode_poll=50 relax_pause=1 timing=0\n"
// decode_threads and decode_poll print the effective values.
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
