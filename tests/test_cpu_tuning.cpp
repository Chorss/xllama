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
                                     "decode_poll=50 relax_pause=0 timing=0\n");
    t.decode_threads = 2;
    t.decode_poll = 0;
    t.relax_pause = true;
    t.timing = true;
    CHECK(t.any());
    CHECK(format_cpu_tuning(t, 6) == "[xllama] cpu-tuning decode_threads=2 batch_threads=6 "
                                     "decode_poll=0 relax_pause=1 timing=1\n");
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
