// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/cpu_tuning.h"

#include <doctest/doctest.h>

using namespace xllama;

TEST_CASE("parse_cpu_knob_int accepts an in-range integer with whitespace") {
    CHECK(parse_cpu_knob_int("2", 1, 64, 0) == 2);
    CHECK(parse_cpu_knob_int(" 64\r\n", 1, 64, 0) == 64);
    CHECK(parse_cpu_knob_int("0", 0, 100, -1) == 0);
    CHECK(parse_cpu_knob_int("+7", 1, 64, 0) == 7);
}

TEST_CASE("parse_cpu_knob_int falls back on empty, junk or out of range") {
    CHECK(parse_cpu_knob_int("", 1, 64, 0) == 0);
    CHECK(parse_cpu_knob_int("  \n", 1, 64, 0) == 0);
    CHECK(parse_cpu_knob_int("0", 1, 64, 0) == 0);
    CHECK(parse_cpu_knob_int("65", 1, 64, 0) == 0);
    CHECK(parse_cpu_knob_int("-1", 0, 100, -1) == -1);
    CHECK(parse_cpu_knob_int("3x", 1, 64, 0) == 0);
    CHECK(parse_cpu_knob_int("-", 0, 100, -1) == -1);
    CHECK(parse_cpu_knob_int("99999999999", 0, 100, -1) == -1);
}

TEST_CASE("cpu tuning defaults are off and read as defaults off Windows") {
    const CpuDecodeTuning t;
    CHECK_FALSE(t.any());
#if !defined(_WIN32)
    CHECK_FALSE(read_cpu_decode_tuning().any());
#endif
}

TEST_CASE("format_cpu_tuning prints effective values") {
    CpuDecodeTuning t;
    CHECK(format_cpu_tuning(t, 6) == "[xllama] cpu-tuning decode_threads=6 batch_threads=6 "
                                     "decode_poll=default relax_pause=0 timing=0 op_timing=0 "
                                     "decode_prio=default\n");
    t.decode_threads = 2;
    t.decode_poll = 0;
    t.relax_pause = true;
    t.timing = true;
    t.op_timing = true;
    t.decode_prio = 2;
    CHECK(t.any());
    CHECK(format_cpu_tuning(t, 6) == "[xllama] cpu-tuning decode_threads=2 batch_threads=6 "
                                     "decode_poll=0 relax_pause=1 timing=1 op_timing=1 "
                                     "decode_prio=2\n");
}

TEST_CASE("op_timing and decode_prio alone count as tuning") {
    CpuDecodeTuning t;
    t.op_timing = true;
    CHECK(t.any());
    CpuDecodeTuning p;
    p.decode_prio = 0; // NORMAL, but it still splits the pools
    CHECK(p.any());
}

TEST_CASE("format_cpu_tuning tells an explicit ggml default from an unset knob") {
    CpuDecodeTuning t;
    t.decode_poll = 50;
    t.decode_prio = 0;
    CHECK(format_cpu_tuning(t, 6) == "[xllama] cpu-tuning decode_threads=6 batch_threads=6 "
                                     "decode_poll=50 relax_pause=0 timing=0 op_timing=0 "
                                     "decode_prio=0\n");
}

TEST_CASE("thread priority helpers are inert off Windows") {
#if !defined(_WIN32)
    CHECK(current_thread_priority() == kNoThreadPriority);
    set_current_thread_priority(2); // no-op, must not crash
#endif
}

TEST_CASE("decode_prio knob range is 0..2, REALTIME excluded") {
    CHECK(parse_cpu_knob_int("0", 0, 2, -1) == 0);
    CHECK(parse_cpu_knob_int("2\n", 0, 2, -1) == 2);
    CHECK(parse_cpu_knob_int("3", 0, 2, -1) == -1);
    CHECK(parse_cpu_knob_int("-1", 0, 2, -1) == -1);
}

TEST_CASE("take_cpu_op_stats is a no-op without the ggml-cpu patch") {
#if !defined(_WIN32)
    CpuOpStats s;
    s.graphs = 7;
    CHECK_FALSE(take_cpu_op_stats(&s));
    CHECK_FALSE(take_cpu_op_stats(nullptr));
    CHECK(s.graphs == 7);
    CHECK(s.op_us.empty());
    apply_cpu_op_timing(true); // must not crash or link ggml
#endif
}

TEST_CASE("format_cpu_op_profile sorts by ms, drops zero counts, adds a sync line") {
    const std::vector<CpuOpRow> rows = {{"ADD", 1.5, 10},
                                        {"NONE", 0.0, 0},
                                        {"FLASH_ATTN_EXT", 9.25, 28},
                                        {"MUL", 1.5, 4},
                                        {"RMS_NORM", 0.0, 3}};
    CHECK(format_cpu_op_profile("decode", rows, 2.5, 512, 128) ==
          "[xllama] cpu-op phase=decode op=FLASH_ATTN_EXT ms=9.250 count=28\n"
          "[xllama] cpu-op phase=decode op=ADD ms=1.500 count=10\n"
          "[xllama] cpu-op phase=decode op=MUL ms=1.500 count=4\n"
          "[xllama] cpu-op phase=decode op=RMS_NORM ms=0.000 count=3\n"
          "[xllama] cpu-sync phase=decode graphs=512 graphs_per_token=4.00 barrier_ms=2.500 "
          "ops_ms=12.250 tokens=128\n");
}

TEST_CASE("format_cpu_op_profile reports missing graphs_per_token without tokens") {
    CHECK(format_cpu_op_profile("decode", {}, 0.0, 3, 0) ==
          "[xllama] cpu-sync phase=decode graphs=3 graphs_per_token=missing barrier_ms=0.000 "
          "ops_ms=0.000 tokens=0\n");
}

TEST_CASE("format_cpu_op_profile prints a dash for a null phase") {
    CHECK(format_cpu_op_profile(nullptr, {}, 0.0, 0, 1) ==
          "[xllama] cpu-sync phase=- graphs=0 graphs_per_token=0.00 barrier_ms=0.000 "
          "ops_ms=0.000 tokens=1\n");
}

TEST_CASE("format_cpu_phase prints one machine-readable line") {
    CpuPhaseTimes t;
    t.llama_decode_ms = 1234.5678;
    t.sample_ms = 12.25;
    t.emit_ms = 0.5;
    t.tokens = 128;
    CHECK(format_cpu_phase("decode", t) ==
          "[xllama] cpu-phase phase=decode llama_decode_ms=1234.568 "
          "sample_ms=12.250 emit_ms=0.500 tokens=128\n");
}
