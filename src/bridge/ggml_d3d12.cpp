// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

// Built only with the llama.cpp backend (it links ggml); the ORT-only UWP
// variant compiles this file to nothing.
#ifdef XLLAMA_USE_LLAMA

    #include "xllama/ggml_d3d12.h"

    #include "ggml-cpu.h"

    #include <algorithm>
    #include <cmath>
    #include <cstdio>
    #include <cstring>
    #include <ctime>
    #include <limits>
    #include <optional>
    #include <string>
    #include <string_view>
    #include <vector>

namespace xllama {

// --- Pure rules (host-tested) ---

bool d3d12_weight_type_supported(ggml_type t) {
    return t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q6_K;
}

D3d12Dispatch d3d12_mm_dispatch(std::int64_t n, std::int64_t ncols) {
    D3d12Dispatch d;
    if (n <= 0 || ncols < 0)
        return d;
    if (ncols == 0) { // no output rows in this ubatch: a no-op, still ours
        d.ok = true;
        return d;
    }
    const std::int64_t gx = (n + kD3d12MmvRows - 1) / kD3d12MmvRows;
    d.ok = gx <= kD3d12MaxGroups && ncols <= kD3d12MaxGroups;
    if (d.ok) {
        d.groups_x = static_cast<std::uint32_t>(gx);
        d.groups_y = static_cast<std::uint32_t>(ncols);
    }
    return d;
}

int d3d12_mm_threads(std::int64_t k) {
    return k / kD3d12Chunk >= kD3d12LongKChunks ? kD3d12MmvThreadsLong : kD3d12MmvThreadsShort;
}

bool d3d12_mm_supported(const D3d12MatmulDesc& d) {
    return d3d12_weight_type_supported(d.src0_type) && d.src0_in_weight_buffer &&
           d.src0_contiguous && d.src1_contiguous && d.src1_type == GGML_TYPE_F32 &&
           d.dst_type == GGML_TYPE_F32 && d.ne00 > 0 && d.ne00 % kD3d12Chunk == 0 &&
           d.ne00 == d.ne10 && d.ne02 == 1 && d.ne03 == 1 && d.ne12 == 1 && d.ne13 == 1 &&
           d3d12_mm_dispatch(d.ne01, d.ne11).ok;
}

bool d3d12_get_rows_type_supported(ggml_type t) {
    return t == GGML_TYPE_Q6_K;
}

bool d3d12_get_rows_supported(const D3d12GetRowsDesc& d) {
    return d3d12_get_rows_type_supported(d.src0_type) && d.src0_in_weight_buffer &&
           d.src0_contiguous && d.src1_contiguous && d.src1_type == GGML_TYPE_I32 &&
           d.dst_type == GGML_TYPE_F32 && d.ne00 > 0 && d.ne00 % kD3d12Chunk == 0 && d.ne02 == 1 &&
           d.ne03 == 1 && d.ne10 >= 0 && d.ne10 <= kD3d12MaxGroups && d.ne11 == 1 && d.ne12 == 1 &&
           d.ne00 / kD3d12Chunk <= kD3d12MaxGroups;
}

bool d3d12_add_supported(const D3d12AddDesc& d) {
    const std::int64_t groups_x = (d.ne00 + kD3d12AddThreads - 1) / kD3d12AddThreads;
    return d.src0_type == GGML_TYPE_F32 && d.src1_type == GGML_TYPE_F32 &&
           d.dst_type == GGML_TYPE_F32 && d.src1_in_weight_buffer && d.src0_rows_contiguous &&
           d.src1_contiguous && d.ne00 > 0 && d.ne10 == d.ne00 && d.ne11 == 1 && d.ne12 == 1 &&
           d.ne13 == 1 && d.ne01 >= 0 && d.ne02 == 1 && d.ne03 == 1 &&
           groups_x <= kD3d12MaxGroups && d.ne01 <= kD3d12MaxGroups;
}

bool d3d12_rope_supported(const D3d12RopeDesc& d) {
    // Same span as the dispatch in submit_ops.
    const std::int64_t span = d3d12_rope_span(d.n_dims, d.ne0);
    const std::int64_t groups = (span + kD3d12RopeThreads - 1) / kD3d12RopeThreads;
    return d.src0_type == GGML_TYPE_F32 && d.dst_type == GGML_TYPE_F32 && d.same_shape &&
           d.src0_contiguous && d.dst_contiguous && !d.src0_in_weight_buffer &&
           !d.src0_in_non_host_buffer && !d.pos_in_non_host_buffer && d.pos_type == GGML_TYPE_I32 &&
           !d.has_freq_factors &&
           (d.mode == kD3d12RopeModeNormal || d.mode == kD3d12RopeModeNeox) &&
           d.ext_factor == 0.0f && d.n_offs == 0 && d.n_dims > 0 && d.n_dims % 2 == 0 &&
           d.n_dims <= d.ne0 && d.ne0 > 0 && d.ne1 > 0 && d.ne2 > 0 && d.ne3 == 1 &&
           d.ne_pos == d.ne2 && d.ne1 <= kD3d12MaxGroups && d.ne2 <= kD3d12MaxGroups &&
           groups <= kD3d12MaxGroups;
}

void d3d12_rope_cache(const D3d12RopeParams& p, std::int32_t pos, float* table) {
    // Same float operations, in the same order, as ggml-cpu ops.cpp
    // (ggml_rope_cache_init with freq_factors == null and ext_factor == 0).
    const float theta_scale = powf(p.freq_base, -2.0f / static_cast<float>(p.n_dims));
    float theta = static_cast<float>(static_cast<std::int64_t>(pos));
    for (int i0 = 0; i0 < p.n_dims; i0 += 2) {
        const float theta_interp = p.freq_scale * (theta / 1.0f);
        table[i0 + 0] = cosf(theta_interp) * p.attn_factor;
        table[i0 + 1] = sinf(theta_interp) * p.attn_factor;
        theta *= theta_scale;
    }
}

void d3d12_rope_emulate(const D3d12RopeParams& p, int mode, std::int64_t ne0, std::int64_t rows,
                        std::int64_t tokens, const std::int32_t* pos, const float* x, float* y) {
    std::vector<float> table(static_cast<std::size_t>(p.n_dims));
    const int half = p.n_dims / 2;
    for (std::int64_t t = 0; t < tokens; ++t) {
        d3d12_rope_cache(p, pos[t], table.data());
        for (std::int64_t r = 0; r < rows; ++r) {
            const float* xs = x + (t * rows + r) * ne0;
            float* ys = y + (t * rows + r) * ne0;
            for (int pr = 0; pr < half; ++pr) {
                const float c = table[2 * pr], s = table[2 * pr + 1];
                const int i0 = mode == kD3d12RopeModeNormal ? 2 * pr : pr;
                const int i1 = mode == kD3d12RopeModeNormal ? 2 * pr + 1 : pr + half;
                const float x0 = xs[i0], x1 = xs[i1];
                ys[i0] = x0 * c - x1 * s;
                ys[i1] = x0 * s + x1 * c;
            }
            for (std::int64_t i = p.n_dims; i < ne0; ++i)
                ys[i] = xs[i];
        }
    }
}

bool d3d12_place_tied_embedding(ggml_type embd_type, bool has_output_weight) {
    return !has_output_weight && d3d12_get_rows_type_supported(embd_type);
}

// --- Host emulation of shaders/ggml_d3d12_mmv_*.hlsl ---

namespace {

// Little-endian loads with the shaders' 2-byte-aligned dword trick.
std::uint32_t ld32(const std::uint8_t* w, std::uint32_t a) {
    auto dw = [&](std::uint32_t at) {
        std::uint32_t v;
        std::memcpy(&v, w + at, 4);
        return v;
    };
    const std::uint32_t b = a & ~3u;
    const std::uint32_t lo = dw(b);
    if ((a & 3u) == 0u)
        return lo;
    return (lo >> 16) | (dw(b + 4u) << 16);
}

std::uint32_t ld16(const std::uint8_t* w, std::uint32_t a) {
    std::uint32_t v;
    std::memcpy(&v, w + (a & ~3u), 4);
    return (a & 2u) != 0u ? (v >> 16) : (v & 0xffffu);
}

float h2f(std::uint32_t h) {
    return ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(h));
}

std::int32_t nib(std::uint32_t word, std::uint32_t i, std::uint32_t shift) {
    return static_cast<std::int32_t>((word >> (8u * i + shift)) & 0xFu);
}

std::int32_t sbyte(std::uint32_t word, std::uint32_t i) {
    return static_cast<std::int8_t>((word >> (8u * i)) & 0xffu);
}

void q4k_scale_min(std::uint32_t j, const std::uint8_t* sc, std::int32_t* d, std::int32_t* m) {
    std::uint32_t dd, mm;
    if (j < 4u) {
        dd = sc[j] & 63u;
        mm = sc[j + 4u] & 63u;
    } else {
        dd = (sc[j + 4u] & 0xFu) | ((sc[j - 4u] >> 6) << 4);
        mm = (sc[j + 4u] >> 4) | ((sc[j] >> 6) << 4);
    }
    *d = static_cast<std::int32_t>(dd);
    *m = static_cast<std::int32_t>(mm);
}

// One thread's contribution for one row and one 256-element chunk. `xq` is
// one activation column in the vec_dot type's ggml blocks (block_q8_0 for
// Q4_0, block_q8_K for Q4_K / Q6_K). Integer sums are exact; the float terms
// follow ggml's vec_dot (#312).
float thread_chunk(ggml_type t, const std::uint8_t* w, std::uint32_t row_off, std::uint32_t blk,
                   std::uint32_t itid, const std::uint8_t* xq) {
    if (t == GGML_TYPE_Q4_0) {
        const std::uint32_t b = itid >> 1, h = itid & 1u;
        const std::uint8_t* xb = xq + (blk * 8u + b) * 34u; // block_q8_0
        const float dx = h2f(ld16(xb, 0));
        const auto* qx = reinterpret_cast<const std::int8_t*>(xb + 2);
        const std::uint32_t bb = row_off + (blk * 8u + b) * 18u;
        const float d = h2f(ld16(w, bb));
        const std::uint32_t q0 = ld32(w, bb + 2u + 8u * h), q1 = ld32(w, bb + 6u + 8u * h);
        std::int32_t sumi = 0;
        for (std::uint32_t i = 0; i < 4; ++i) {
            const std::uint32_t lo = 8u * h + i, hi = 16u + 8u * h + i;
            sumi += (nib(q0, i, 0) - 8) * qx[lo] + (nib(q1, i, 0) - 8) * qx[lo + 4] +
                    (nib(q0, i, 4) - 8) * qx[hi] + (nib(q1, i, 4) - 8) * qx[hi + 4];
        }
        return (d * dx) * static_cast<float>(sumi);
    }
    float dy;
    std::memcpy(&dy, xq + blk * 292u, 4); // block_q8_K: float d; int8 qs[256]; ...
    const auto* qx = reinterpret_cast<const std::int8_t*>(xq + blk * 292u + 4u);
    if (t == GGML_TYPE_Q4_K) {
        const std::uint32_t il = itid >> 2, ir = itid & 3u;
        const std::uint32_t e_lo = il * 64u + ir * 8u, e_hi = e_lo + 32u;
        const std::uint32_t bb = row_off + blk * 144u;
        const float d = h2f(ld16(w, bb)), dmin = h2f(ld16(w, bb + 2u));
        std::int32_t sc1, m1, sc2, m2;
        q4k_scale_min(2u * il, w + bb + 4u, &sc1, &m1);
        q4k_scale_min(2u * il + 1u, w + bb + 4u, &sc2, &m2);
        const std::uint32_t qa = ld32(w, bb + 16u + il * 32u + ir * 8u);
        const std::uint32_t qb = ld32(w, bb + 20u + il * 32u + ir * 8u);
        std::int32_t lo = 0, hi = 0, sl = 0, sh = 0;
        for (std::uint32_t i = 0; i < 4; ++i) {
            lo += nib(qa, i, 0) * qx[e_lo + i] + nib(qb, i, 0) * qx[e_lo + 4 + i];
            hi += nib(qa, i, 4) * qx[e_hi + i] + nib(qb, i, 4) * qx[e_hi + 4 + i];
            sl += qx[e_lo + i] + qx[e_lo + 4 + i];
            sh += qx[e_hi + i] + qx[e_hi + 4 + i];
        }
        return (d * dy) * static_cast<float>(sc1 * lo + sc2 * hi) -
               (dmin * dy) * static_cast<float>(m1 * sl + m2 * sh);
    }
    // Q6_K
    const std::uint32_t v = itid >> 3, l0 = 4u * (itid & 7u), is = l0 >> 4;
    const std::uint32_t e = 128u * v + l0;
    const std::uint32_t bb = row_off + blk * 210u;
    const std::uint32_t qla = ld32(w, bb + 64u * v + l0);
    const std::uint32_t qlb = ld32(w, bb + 64u * v + 32u + l0);
    const std::uint32_t qh = ld32(w, bb + 128u + 32u * v + l0);
    const std::uint32_t s0 = ld32(w, bb + 192u + 8u * v), s1 = ld32(w, bb + 196u + 8u * v);
    const float d = h2f(ld16(w, bb + 208u));
    auto q6 = [](std::uint32_t ql, std::uint32_t qhw, std::uint32_t i, std::uint32_t qs,
                 std::uint32_t hs) {
        const std::uint32_t lo4 = (ql >> (8u * i + qs)) & 0xFu;
        const std::uint32_t hi2 = (qhw >> (8u * i + hs)) & 3u;
        return static_cast<std::int32_t>(lo4 | (hi2 << 4)) - 32;
    };
    std::int32_t t1 = 0, t2 = 0, t3 = 0, t4 = 0;
    for (std::uint32_t i = 0; i < 4; ++i) {
        t1 += q6(qla, qh, i, 0, 0) * qx[e + i];
        t2 += q6(qlb, qh, i, 0, 2) * qx[e + 32 + i];
        t3 += q6(qla, qh, i, 4, 4) * qx[e + 64 + i];
        t4 += q6(qlb, qh, i, 4, 6) * qx[e + 96 + i];
    }
    const std::int32_t tsum =
        sbyte(s0, is) * t1 + sbyte(s0, is + 2) * t2 + sbyte(s1, is) * t3 + sbyte(s1, is + 2) * t4;
    return (d * dy) * static_cast<float>(tsum);
}

} // namespace

ggml_type d3d12_activation_type(ggml_type weight) {
    return weight == GGML_TYPE_Q4_0 ? GGML_TYPE_Q8_0 : GGML_TYPE_Q8_K;
}

void d3d12_quantize_activations(ggml_type weight, const float* x, std::size_t x_stride, int k,
                                int ncols, std::uint8_t* out) {
    const ggml_type at = d3d12_activation_type(weight);
    const auto from_float = ggml_get_type_traits_cpu(at)->from_float;
    const std::size_t row = ggml_row_size(at, k);
    for (int c = 0; c < ncols; ++c)
        from_float(x + static_cast<std::size_t>(c) * x_stride, out + c * row, k);
}

std::uint32_t d3d12_f32_div_rn(std::uint32_t a, std::uint32_t b) {
    const std::uint32_t sign = (a ^ b) & 0x80000000u;
    const std::uint32_t ea = (a >> 23) & 0xffu, eb = (b >> 23) & 0xffu;
    std::uint32_t ma = a & 0x7fffffu, mb = b & 0x7fffffu;
    const bool a_nan = ea == 0xffu && ma != 0, b_nan = eb == 0xffu && mb != 0;
    const bool a_inf = ea == 0xffu && ma == 0, b_inf = eb == 0xffu && mb == 0;
    const bool a_zero = ea == 0 && ma == 0, b_zero = eb == 0 && mb == 0;
    if (a_nan || b_nan || (a_inf && b_inf) || (a_zero && b_zero))
        return 0x7fc00000u;
    if (a_inf || b_zero)
        return sign | 0x7f800000u;
    if (b_inf || a_zero)
        return sign;
    // Normalize both significands to [2^23, 2^24).
    int xa = static_cast<int>(ea), xb = static_cast<int>(eb);
    if (ea == 0) {
        xa = 1;
        while ((ma & 0x800000u) == 0) {
            ma <<= 1;
            --xa;
        }
    } else {
        ma |= 0x800000u;
    }
    if (eb == 0) {
        xb = 1;
        while ((mb & 0x800000u) == 0) {
            mb <<= 1;
            --xb;
        }
    } else {
        mb |= 0x800000u;
    }
    int e = xa - xb + 127;
    if (ma < mb) {
        ma <<= 1;
        --e;
    }
    // 26 quotient bits in [2^25, 2^26): 24 significand bits, a round bit and
    // one more; the remainder is the sticky bit.
    std::uint32_t q = 0, r = ma;
    for (int i = 0; i < 26; ++i) {
        q <<= 1;
        if (r >= mb) {
            r -= mb;
            q |= 1u;
        }
        r <<= 1;
    }
    std::uint32_t sticky = r != 0 ? 1u : 0u;
    if (e >= 255)
        return sign | 0x7f800000u;
    const bool subnormal = e <= 0;
    if (subnormal) {
        const std::uint32_t sh = static_cast<std::uint32_t>(1 - e);
        if (sh > 26) {
            sticky |= q != 0 ? 1u : 0u;
            q = 0;
        } else {
            sticky |= (q & ((1u << sh) - 1u)) != 0 ? 1u : 0u;
            q >>= sh;
        }
    }
    std::uint32_t mant = q >> 2;
    const std::uint32_t round = (q >> 1) & 1u;
    sticky |= q & 1u;
    if (round != 0 && (sticky != 0 || (mant & 1u) != 0))
        ++mant;
    // A normal significand carries its implicit bit, so (e - 1) << 23 adds the
    // exponent; a rounding carry or an overflow to 255 lands on the right bits.
    return sign | (subnormal ? mant : (static_cast<std::uint32_t>(e - 1) << 23) + mant);
}

std::uint16_t d3d12_f32_to_f16_rn(std::uint32_t a) {
    const std::uint32_t sign = (a >> 16) & 0x8000u;
    const std::uint32_t e = (a >> 23) & 0xffu;
    std::uint32_t m = a & 0x7fffffu;
    if (e == 0xffu)
        return static_cast<std::uint16_t>(sign | (m != 0 ? 0x7e00u : 0x7c00u));
    const int e16 = static_cast<int>(e) - 127 + 15;
    std::uint32_t mant, rem, half, bits;
    if (e16 >= 1) {
        mant = m >> 13;
        rem = m & 0x1fffu;
        half = 0x1000u;
        bits = (static_cast<std::uint32_t>(e16) << 10) + mant;
    } else {
        if (e == 0) // fp32 zero or subnormal: far below half the fp16 minimum
            return static_cast<std::uint16_t>(sign);
        m |= 0x800000u;
        const std::uint32_t s = static_cast<std::uint32_t>(14 - e16);
        if (s >= 32)
            return static_cast<std::uint16_t>(sign);
        mant = m >> s;
        rem = m & ((1u << s) - 1u);
        half = 1u << (s - 1);
        bits = mant;
    }
    if (rem > half || (rem == half && (mant & 1u) != 0))
        ++bits;
    if (bits >= 0x7c00u)
        bits = 0x7c00u;
    return static_cast<std::uint16_t>(sign | bits);
}

namespace {

std::uint32_t f32_bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

float bits_f32(std::uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// _mm256_cvtps_epi32: NaN and out-of-range give INT_MIN.
std::int32_t cvt_i32(float r) {
    return std::fabs(r) < 2147483648.0f ? static_cast<std::int32_t>(r)
                                        : std::numeric_limits<std::int32_t>::min();
}

} // namespace

namespace {

// 96-bit unsigned integers in three 32-bit words (w[0] lowest): the shader has
// no 64-bit integers to rely on, so the host twin uses none either.
struct U96 {
    std::uint32_t w[3] = {0, 0, 0};
};

U96 u96(std::uint32_t lo) {
    U96 r;
    r.w[0] = lo;
    return r;
}

bool is_zero96(const U96& a) {
    return (a.w[0] | a.w[1] | a.w[2]) == 0;
}

U96 add96(const U96& a, const U96& b) {
    U96 r;
    std::uint32_t carry = 0;
    for (int i = 0; i < 3; ++i) {
        const std::uint32_t s = a.w[i] + b.w[i];
        const std::uint32_t c1 = s < a.w[i] ? 1u : 0u;
        const std::uint32_t s2 = s + carry;
        const std::uint32_t c2 = s2 < s ? 1u : 0u;
        r.w[i] = s2;
        carry = c1 | c2;
    }
    return r;
}

U96 sub96(const U96& a, const U96& b) { // a >= b
    U96 r;
    std::uint32_t borrow = 0;
    for (int i = 0; i < 3; ++i) {
        const std::uint32_t d = a.w[i] - b.w[i];
        const std::uint32_t b1 = a.w[i] < b.w[i] ? 1u : 0u;
        const std::uint32_t d2 = d - borrow;
        const std::uint32_t b2 = d < borrow ? 1u : 0u;
        r.w[i] = d2;
        borrow = b1 | b2;
    }
    return r;
}

bool ge96(const U96& a, const U96& b) {
    for (int i = 2; i >= 0; --i)
        if (a.w[i] != b.w[i])
            return a.w[i] > b.w[i];
    return true;
}

U96 shl96(const U96& a, int s) { // 0 <= s < 96
    U96 r;
    const int ws = s / 32, bs = s % 32;
    for (int i = 2; i >= 0; --i) {
        const int j = i - ws;
        std::uint32_t v = 0;
        if (j >= 0) {
            v = a.w[j] << bs;
            if (bs != 0 && j >= 1)
                v |= a.w[j - 1] >> (32 - bs);
        }
        r.w[i] = v;
    }
    return r;
}

// a >> s; *sticky |= any bit shifted out. s >= 0, any size.
U96 shr96(const U96& a, int s, std::uint32_t* sticky) {
    if (s >= 96) {
        *sticky |= is_zero96(a) ? 0u : 1u;
        return U96{};
    }
    const int ws = s / 32, bs = s % 32;
    std::uint32_t lost = 0;
    for (int i = 0; i < ws; ++i)
        lost |= a.w[i];
    if (bs != 0)
        lost |= a.w[ws] & ((1u << bs) - 1u);
    U96 r;
    for (int i = 0; i < 3; ++i) {
        const int j = i + ws;
        std::uint32_t v = 0;
        if (j <= 2) {
            v = a.w[j] >> bs;
            if (bs != 0 && j + 1 <= 2)
                v |= a.w[j + 1] << (32 - bs);
        }
        r.w[i] = v;
    }
    *sticky |= lost != 0 ? 1u : 0u;
    return r;
}

int top96(const U96& a) { // highest set bit, -1 for zero
    for (int i = 2; i >= 0; --i)
        if (a.w[i] != 0) {
            int b = 31;
            while ((a.w[i] >> b) == 0)
                --b;
            return i * 32 + b;
        }
    return -1;
}

} // namespace

std::uint32_t d3d12_f32_fma_rn(std::uint32_t a, std::uint32_t b, std::uint32_t c) {
    const std::uint32_t sp = (a ^ b) & 0x80000000u, sc = c & 0x80000000u;
    const std::uint32_t ea = (a >> 23) & 0xffu, eb = (b >> 23) & 0xffu, ec = (c >> 23) & 0xffu;
    std::uint32_t ma = a & 0x7fffffu, mb = b & 0x7fffffu, mc = c & 0x7fffffu;
    const bool a_nan = ea == 0xffu && ma != 0, b_nan = eb == 0xffu && mb != 0,
               c_nan = ec == 0xffu && mc != 0;
    const bool a_inf = ea == 0xffu && ma == 0, b_inf = eb == 0xffu && mb == 0,
               c_inf = ec == 0xffu && mc == 0;
    const bool a_zero = ea == 0 && ma == 0, b_zero = eb == 0 && mb == 0,
               c_zero = ec == 0 && mc == 0;
    if (a_nan || b_nan || c_nan)
        return 0x7fc00000u;
    if (a_inf || b_inf) {
        if (a_zero || b_zero || (c_inf && sc != sp))
            return 0x7fc00000u;
        return sp | 0x7f800000u;
    }
    if (c_inf)
        return c;
    if (a_zero || b_zero) {
        if (c_zero)
            return sp & sc; // -0 only when both are -0 (round to nearest)
        return c;
    }
    // value = m * 2^x; subnormals keep their smaller m, top96 finds the top bit.
    const int xa = (ea == 0 ? 1 : static_cast<int>(ea)) - 150;
    const int xb = (eb == 0 ? 1 : static_cast<int>(eb)) - 150;
    const int xc = (ec == 0 ? 1 : static_cast<int>(ec)) - 150;
    if (ea != 0)
        ma |= 0x800000u;
    if (eb != 0)
        mb |= 0x800000u;
    if (ec != 0)
        mc |= 0x800000u;
    // Exact 48-bit product from 12-bit halves (each partial product < 2^24).
    const std::uint32_t al = ma & 0xfffu, ah = ma >> 12, bl = mb & 0xfffu, bh = mb >> 12;
    U96 prod =
        add96(add96(shl96(u96(ah * bh), 24), shl96(u96(ah * bl + al * bh), 12)), u96(al * bl));
    const int xp = xa + xb;
    U96 cm = u96(mc);
    // One window for both: LSB exponent L, top bit at most 93 (room for a carry).
    const int tp = xp + top96(prod);
    const int tc = c_zero ? tp : xc + top96(cm);
    const int hi = std::max(tp, tc);
    int lsb = c_zero ? xp : std::min(xp, xc);
    if (hi - lsb > 93)
        lsb = hi - 93;
    std::uint32_t st_p = 0, st_c = 0;
    prod = xp >= lsb ? shl96(prod, xp - lsb) : shr96(prod, lsb - xp, &st_p);
    cm = c_zero ? U96{} : (xc >= lsb ? shl96(cm, xc - lsb) : shr96(cm, lsb - xc, &st_c));
    // Bits below the window are jammed into the LSB: they then only decide the
    // rounding, which sits at least 40 bits higher whenever anything was lost.
    prod.w[0] |= st_p;
    cm.w[0] |= st_c;
    U96 sum;
    std::uint32_t sign;
    if (sp == sc) {
        sum = add96(prod, cm);
        sign = sp;
    } else if (ge96(prod, cm)) {
        sum = sub96(prod, cm);
        sign = sp;
    } else {
        sum = sub96(cm, prod);
        sign = sc;
    }
    if (is_zero96(sum))
        return 0; // exact cancellation: +0 when rounding to nearest
    const int t = top96(sum);
    const int e = lsb + t; // exponent of the top bit
    if (e > 127)
        return sign | 0x7f800000u;
    const int out_lsb = e <= -126 ? -149 : e - 23;
    const int sh = out_lsb - lsb;
    std::uint32_t mant, round = 0, sticky = 0;
    if (sh <= 0) {
        mant = shl96(sum, -sh).w[0];
    } else {
        const U96 r = shr96(sum, sh - 1, &sticky);
        round = r.w[0] & 1u;
        mant = r.w[0] >> 1;
    }
    if (round != 0 && (sticky != 0 || (mant & 1u) != 0))
        ++mant;
    // At e == -126 the implicit bit (2^23) is exponent field 1, so both forms
    // agree there; a rounding carry moves into the exponent on its own.
    const std::uint32_t bits =
        e <= -126 ? mant : (static_cast<std::uint32_t>(e + 126) << 23) + mant;
    return sign | std::min(bits, 0x7f800000u);
}

namespace {

std::uint32_t fbits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

float fval(std::uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

float fmadd(float a, float b, float c) {
    return fval(d3d12_f32_fma_rn(fbits(a), fbits(b), fbits(c)));
}

// ggml_v_expf, AVX2 + FMA, one lane. Every lane takes the same branch result
// whatever its neighbours do: the slow path blends per lane, and its !c lanes
// compute fma(k, j, k) == fma(j, k, k).
float v_expf_lane(float x) {
    const float r = 0x1.8p23f;
    const float z = fmadd(x, 0x1.715476p+0f, r);
    const float n = z - r;
    const float b = fmadd(-n, 0x1.7f7d1cp-20f, fmadd(-n, 0x1.62e4p-1f, x));
    const std::uint32_t e = fbits(z) << 23;
    const float k = fval(e + fbits(1.0f));
    const bool c = std::fabs(n) > 126.0f;
    const float u = b * b;
    const float j = fmadd(fmadd(fmadd(0x1.0e4020p-7f, b, 0x1.573e2ep-5f), u,
                                fmadd(0x1.555e66p-3f, b, 0x1.fffdb6p-2f)),
                          u, 0x1.ffffecp-1f * b);
    if (!c)
        return fmadd(j, k, k);
    const std::uint32_t g = n <= 0.0f ? 0x82000000u : 0u;
    const float s1 = fval(g + 0x7f000000u);
    const float s2 = fval(e - g);
    if (std::fabs(n) > 192.0f)
        return s1 * s1;
    return fmadd(s2, j, s2) * s1;
}

} // namespace

float d3d12_swiglu_emulate(float x, float g) {
    const float neg_x = 0.0f - x;
    const float den = 1.0f + v_expf_lane(neg_x);
    const float silu = fval(d3d12_f32_div_rn(fbits(x), fbits(den)));
    return silu * g;
}

bool d3d12_swiglu_supported(const D3d12SwigluDesc& d) {
    return d.glu_op_swiglu && d.two_sources && !d.swapped && d.src0_type == GGML_TYPE_F32 &&
           d.src1_type == GGML_TYPE_F32 && d.dst_type == GGML_TYPE_F32 && d.rows_contiguous &&
           d.same_shapes && d.ne0 > 0 && d.ne0 % 8 == 0 && d.ne1 >= 0 && d.ne2 == 1 && d.ne3 == 1 &&
           (d.ne0 + kD3d12SwigluThreads - 1) / kD3d12SwigluThreads <= kD3d12MaxGroups &&
           d.ne1 <= kD3d12MaxGroups;
}

void d3d12_quantize_q8_0_emulate(const float* x, int k, std::uint8_t* out) {
    for (int b = 0; b < k / 32; ++b, x += 32, out += 34) {
        std::uint32_t amax = 0; // max |x| on bits: same order as floats, no FTZ
        for (int j = 0; j < 32; ++j)
            amax = std::max(amax, f32_bits(x[j]) & 0x7fffffffu);
        const std::uint32_t d = d3d12_f32_div_rn(amax, f32_bits(127.0f));
        const std::uint16_t dh = d3d12_f32_to_f16_rn(d);
        std::memcpy(out, &dh, 2);
        const float id = amax != 0 ? bits_f32(d3d12_f32_div_rn(f32_bits(127.0f), amax)) : 0.0f;
        for (int j = 0; j < 32; ++j) {
            const std::int32_t q = cvt_i32(std::nearbyint(x[j] * id)); // HLSL round(): ties even
            out[2 + j] = static_cast<std::uint8_t>(std::clamp(q, -128, 127)); // packs saturation
        }
    }
}

void d3d12_quantize_q8_K_emulate(const float* x, int k, std::uint8_t* out) {
    for (int b = 0; b < k / 256; ++b, x += 256, out += 292) {
        // First index of the largest |x| (quantize_row_q8_K_ref's strict >).
        float amax = 0.0f, max = 0.0f;
        for (int j = 0; j < 256; ++j) {
            const float ax = std::fabs(x[j]);
            if (ax > amax) {
                amax = ax;
                max = x[j];
            }
        }
        std::int8_t qs[256] = {};
        float d = 0.0f;
        if (amax != 0.0f) {
            const float iscale = bits_f32(d3d12_f32_div_rn(f32_bits(-127.0f), f32_bits(max)));
            for (int j = 0; j < 256; ++j) {
                const float fv = iscale * x[j];
                // nearest_int: the low mantissa bits of fv + 1.5 * 2^23. A
                // non-finite fv (|max| tiny enough for iscale to overflow)
                // stores 0 on x86 for both inf and the default NaN.
                std::int32_t v = 0;
                if (std::isfinite(fv))
                    v = static_cast<std::int32_t>(f32_bits(fv + 12582912.0f) & 0x007fffffu) -
                        0x00400000;
                v = std::min(127, v);
                qs[j] = static_cast<std::int8_t>(static_cast<std::uint8_t>(v & 0xff));
            }
            d = bits_f32(d3d12_f32_div_rn(f32_bits(1.0f), f32_bits(iscale)));
        }
        std::memcpy(out, &d, 4);
        std::memcpy(out + 4, qs, 256);
        for (int g = 0; g < 16; ++g) {
            std::int32_t s = 0;
            for (int i = 0; i < 16; ++i)
                s += qs[g * 16 + i];
            const std::int16_t s16 = static_cast<std::int16_t>(s);
            std::memcpy(out + 260 + g * 2, &s16, 2);
        }
    }
}

void d3d12_mmv_emulate(ggml_type t, const std::uint8_t* w, std::size_t w_row_bytes, const float* x,
                       std::size_t x_stride, float* y, std::size_t y_stride, int n, int k,
                       int ncols) {
    if (!w || !x || !y || n <= 0 || k <= 0 || k % kD3d12Chunk != 0 || ncols <= 0 ||
        !d3d12_weight_type_supported(t))
        return;
    const std::size_t xq_row = ggml_row_size(d3d12_activation_type(t), k);
    std::vector<std::uint8_t> xq(xq_row * static_cast<std::size_t>(ncols) + 4);
    d3d12_quantize_activations(t, x, x_stride, k, ncols, xq.data());
    const std::uint32_t nchunk = static_cast<std::uint32_t>(k / kD3d12Chunk);
    const std::uint32_t threads = static_cast<std::uint32_t>(d3d12_mm_threads(k));
    const std::uint32_t in_flight = threads / 16u;
    float acc[kD3d12MmvThreadsLong];
    for (int c = 0; c < ncols; ++c) {
        const std::uint8_t* xc = xq.data() + static_cast<std::size_t>(c) * xq_row;
        for (int row = 0; row < n; ++row) {
            const std::uint32_t row_off = static_cast<std::uint32_t>(row * w_row_bytes);
            for (std::uint32_t tid = 0; tid < threads; ++tid) {
                float a = 0.f;
                for (std::uint32_t blk = tid >> 4; blk < nchunk; blk += in_flight)
                    a += thread_chunk(t, w, row_off, blk, tid & 15u, xc);
                acc[tid] = a;
            }
            for (int stride = static_cast<int>(threads) / 2; stride > 0; stride >>= 1)
                for (int i = 0; i < stride; ++i)
                    acc[i] += acc[i + stride];
            y[static_cast<std::size_t>(c) * y_stride + row] = acc[0];
        }
    }
}

void d3d12_get_rows_emulate(ggml_type t, const std::uint8_t* w, std::size_t w_row_bytes,
                            const std::int32_t* ids, int n_ids, int k, float* y,
                            std::size_t y_stride) {
    if (!w || !ids || !y || n_ids <= 0 || k <= 0 || k % kD3d12Chunk != 0 ||
        !d3d12_get_rows_type_supported(t))
        return;
    auto ld8 = [&](std::uint32_t a) { return (ld32(w, a & ~3u) >> ((a & 3u) * 8u)) & 0xffu; };
    // One thread group per (256-element chunk, id); thread tid = 32v + l writes
    // elements 128v + l + {0, 32, 64, 96}, as dequantize_row_q6_K's inner loop.
    for (int i = 0; i < n_ids; ++i) {
        for (std::uint32_t blk = 0; blk < static_cast<std::uint32_t>(k / kD3d12Chunk); ++blk) {
            const std::uint32_t bb = static_cast<std::uint32_t>(ids[i] * w_row_bytes) + blk * 210u;
            const float d = h2f(ld16(w, bb + 208u));
            for (std::uint32_t tid = 0; tid < 64u; ++tid) {
                const std::uint32_t v = tid >> 5, l = tid & 31u, is = l >> 4;
                const std::uint32_t qla = ld8(bb + 64u * v + l), qlb = ld8(bb + 64u * v + l + 32u);
                const std::uint32_t qh = ld8(bb + 128u + 32u * v + l);
                auto sc = [&](std::uint32_t j) {
                    return static_cast<float>(
                        static_cast<std::int8_t>(ld8(bb + 192u + 8u * v + is + j)));
                };
                const int q1 = static_cast<int>((qla & 0xFu) | (((qh >> 0) & 3u) << 4)) - 32;
                const int q2 = static_cast<int>((qlb & 0xFu) | (((qh >> 2) & 3u) << 4)) - 32;
                const int q3 = static_cast<int>((qla >> 4) | (((qh >> 4) & 3u) << 4)) - 32;
                const int q4 = static_cast<int>((qlb >> 4) | (((qh >> 6) & 3u) << 4)) - 32;
                float* out = y + static_cast<std::size_t>(i) * y_stride + blk * 256u + 128u * v + l;
                out[0] = d * sc(0) * static_cast<float>(q1);
                out[32] = d * sc(2) * static_cast<float>(q2);
                out[64] = d * sc(4) * static_cast<float>(q3);
                out[96] = d * sc(6) * static_cast<float>(q4);
            }
        }
    }
}

// --- Selftest CSV ---

const char* d3d12_selftest_csv_header() {
    return "type,n,k,ncols,rel_err,gpu_ms,packed_gbs,ok,d3d12_ran,host,date,error\n";
}

bool d3d12_flag_enabled(std::string_view config) {
    const auto first = config.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos || config[first] != '1')
        return false;
    return config.find_first_not_of(" \t\r\n", first + 1) == std::string_view::npos;
}

const char* d3d12_cols_class(std::int64_t ncols) {
    if (ncols <= 1)
        return "1";
    if (ncols < 64)
        return "2-63";
    if (ncols < 512)
        return "64-511";
    return "512+";
}

void D3d12DispatchProfile::add(const D3d12DispatchKind& kind, double gpu_ms, double weight_bytes,
                               std::int64_t ncols) {
    auto it = std::find_if(rows_.begin(), rows_.end(),
                           [&](const D3d12DispatchRow& r) { return r.kind == kind; });
    if (it == rows_.end()) {
        rows_.push_back(D3d12DispatchRow{kind});
        it = rows_.end() - 1;
    }
    ++it->count;
    it->gpu_ms += gpu_ms;
    it->weight_bytes += weight_bytes;
    it->columns += static_cast<std::uint64_t>(ncols > 0 ? ncols : 0);
}

std::vector<D3d12DispatchRow> D3d12DispatchProfile::sorted() const {
    std::vector<D3d12DispatchRow> out = rows_;
    std::stable_sort(
        out.begin(), out.end(),
        [](const D3d12DispatchRow& a, const D3d12DispatchRow& b) { return a.gpu_ms > b.gpu_ms; });
    return out;
}

std::string format_d3d12_dispatch_profile(const char* phase, const D3d12DispatchProfile& p) {
    if (p.empty())
        return {};
    std::string out;
    double total = 0.0;
    std::uint64_t timed = 0;
    for (const D3d12DispatchRow& r : p.sorted()) {
        char buf[512];
        std::snprintf(buf, sizeof(buf),
                      "[xllama] d3d12-dispatch phase=%s op=%s type=%s cpg=%d threads=%d cols=%s "
                      "count=%llu gpu_ms=%.3f ms_per=%.4f weight_mib=%.1f columns=%llu\n",
                      phase ? phase : "-", r.kind.op.c_str(), r.kind.type.c_str(),
                      r.kind.columns_per_group, r.kind.threads, r.kind.cols.c_str(),
                      static_cast<unsigned long long>(r.count), r.gpu_ms,
                      r.count ? r.gpu_ms / static_cast<double>(r.count) : 0.0,
                      r.weight_bytes / 1048576.0, static_cast<unsigned long long>(r.columns));
        out += buf;
        total += r.gpu_ms;
        timed += r.count;
    }
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "[xllama] d3d12-dispatch phase=%s summary kinds=%zu timed=%llu missing=%llu "
                  "gpu_ms=%.3f\n",
                  phase ? phase : "-", p.rows().size(), static_cast<unsigned long long>(timed),
                  static_cast<unsigned long long>(p.missing()), total);
    out += buf;
    return out;
}

namespace {

std::optional<int> parse_nonneg_int(std::string_view s, std::size_t i, std::size_t* end) {
    if (i >= s.size() || s[i] < '0' || s[i] > '9')
        return std::nullopt;
    int v = 0;
    std::size_t j = i;
    while (j < s.size() && s[j] >= '0' && s[j] <= '9') {
        const int d = s[j] - '0';
        if (v > (std::numeric_limits<int>::max() - d) / 10)
            return std::nullopt;
        v = v * 10 + d;
        ++j;
    }
    if (end)
        *end = j;
    return v;
}

bool layer_component_end(std::string_view s, std::size_t j) {
    return j == s.size() || s[j] == '.' || s[j] == '#' || s[j] == '/';
}

void write_ms_or_missing(char* buf, std::size_t n, std::uint64_t timed, double ms) {
    if (timed == 0)
        std::snprintf(buf, n, "missing");
    else
        std::snprintf(buf, n, "%.3f", ms);
}

} // namespace

std::optional<int> d3d12_layer_from_name(std::string_view name) {
    std::string_view inner = name;
    const auto hash1 = name.find('#');
    const auto hash2 = name.rfind('#');
    if (hash1 != std::string_view::npos && hash2 != hash1)
        inner = name.substr(hash1 + 1, hash2 - hash1 - 1);

    for (std::size_t i = 0; i + 4 <= inner.size(); ++i) {
        if ((i == 0 || inner[i - 1] == '.' || inner[i - 1] == '/' || inner[i - 1] == '#') &&
            inner.compare(i, 4, "blk.") == 0) {
            std::size_t end = 0;
            const auto v = parse_nonneg_int(inner, i + 4, &end);
            if (v && layer_component_end(inner, end))
                return v;
        }
    }

    const auto dash = inner.rfind('-');
    if (dash != std::string_view::npos && dash + 1 < inner.size()) {
        std::size_t end = 0;
        const auto v = parse_nonneg_int(inner, dash + 1, &end);
        if (v && end == inner.size())
            return v;
    }
    return std::nullopt;
}

std::optional<int> d3d12_layer_from_names(std::string_view tensor_name,
                                          std::string_view node_name) {
    if (const auto layer = d3d12_layer_from_name(tensor_name))
        return layer;
    return d3d12_layer_from_name(node_name);
}

std::string d3d12_layer_label(std::optional<int> layer) {
    if (!layer)
        return kD3d12UnattributedLayer;
    return std::to_string(*layer);
}

void D3d12LayerProfile::add(const D3d12LayerKey& key, const std::optional<double>& gpu_ms,
                            const std::optional<double>& wait_ms) {
    auto it = std::find_if(rows_.begin(), rows_.end(),
                           [&](const D3d12LayerRow& r) { return r.key == key; });
    if (it == rows_.end()) {
        rows_.push_back(D3d12LayerRow{key});
        it = rows_.end() - 1;
    }
    ++it->count;
    if (gpu_ms) {
        ++it->gpu_timed;
        it->gpu_ms += *gpu_ms;
    } else {
        ++it->gpu_missing;
    }
    if (wait_ms) {
        ++it->wait_timed;
        it->wait_ms += *wait_ms;
    } else {
        ++it->wait_missing;
    }
}

std::vector<D3d12LayerRow> D3d12LayerProfile::sorted() const {
    std::vector<D3d12LayerRow> out = rows_;
    std::stable_sort(out.begin(), out.end(), [](const D3d12LayerRow& a, const D3d12LayerRow& b) {
        const bool a_u = !a.key.layer.has_value();
        const bool b_u = !b.key.layer.has_value();
        if (a_u != b_u)
            return b_u; // unattributed last
        if (a.key.layer && b.key.layer && *a.key.layer != *b.key.layer)
            return *a.key.layer < *b.key.layer;
        return a.key.op < b.key.op;
    });
    return out;
}

std::string format_d3d12_layer_profile(const char* phase, const D3d12LayerProfile& p) {
    if (p.empty())
        return {};
    std::string out;
    std::uint64_t timed = 0, missing = 0, wait_timed = 0, wait_missing = 0;
    double gpu_total = 0.0, wait_total = 0.0;
    for (const D3d12LayerRow& r : p.sorted()) {
        char gpu[32], wait[32];
        write_ms_or_missing(gpu, sizeof(gpu), r.gpu_timed, r.gpu_ms);
        write_ms_or_missing(wait, sizeof(wait), r.wait_timed, r.wait_ms);
        char buf[512];
        std::snprintf(buf, sizeof(buf),
                      "[xllama] d3d12-layer phase=%s layer=%s op=%s count=%llu gpu_ms=%s "
                      "wait_ms=%s gpu_missing=%llu wait_missing=%llu\n",
                      phase ? phase : "-", d3d12_layer_label(r.key.layer).c_str(), r.key.op.c_str(),
                      static_cast<unsigned long long>(r.count), gpu, wait,
                      static_cast<unsigned long long>(r.gpu_missing),
                      static_cast<unsigned long long>(r.wait_missing));
        out += buf;
        timed += r.gpu_timed;
        missing += r.gpu_missing;
        wait_timed += r.wait_timed;
        wait_missing += r.wait_missing;
        gpu_total += r.gpu_ms;
        wait_total += r.wait_ms;
    }
    char gpu[32], wait[32];
    write_ms_or_missing(gpu, sizeof(gpu), timed, gpu_total);
    write_ms_or_missing(wait, sizeof(wait), wait_timed, wait_total);
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "[xllama] d3d12-layer phase=%s summary rows=%zu timed=%llu missing=%llu "
                  "gpu_ms=%s wait_ms=%s wait_missing=%llu\n",
                  phase ? phase : "-", p.rows().size(), static_cast<unsigned long long>(timed),
                  static_cast<unsigned long long>(missing), gpu, wait,
                  static_cast<unsigned long long>(wait_missing));
    out += buf;
    return out;
}

std::string format_d3d12_phase(const char* phase, const D3d12PhaseStats& s) {
    char gpu[32];
    if (s.gpu_time_valid())
        std::snprintf(gpu, sizeof(gpu), "%.3f", s.gpu_ms);
    else
        std::snprintf(gpu, sizeof(gpu), "unavailable");
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "[xllama] d3d12-phase phase=%s calls=%llu submissions=%llu dispatches=%llu "
                  "wall_ms=%.3f quant_ms=%.3f record_ms=%.3f exec_wait_ms=%.3f gpu_ms=%s "
                  "ts_valid=%llu ts_missing=%llu\n",
                  phase ? phase : "-", static_cast<unsigned long long>(s.calls),
                  static_cast<unsigned long long>(s.submissions),
                  static_cast<unsigned long long>(s.dispatches), s.wall_ms, s.quant_ms, s.record_ms,
                  s.exec_wait_ms, gpu, static_cast<unsigned long long>(s.ts_valid),
                  static_cast<unsigned long long>(s.ts_missing));
    return buf;
}

std::string format_d3d12_selftest_row(const D3d12SelftestRow& r, const char* host_label) {
    char date_buf[32];
    std::time_t now = std::time(nullptr);
    std::strftime(date_buf, sizeof(date_buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    std::string err = r.error.empty() ? "-" : r.error;
    for (char& ch : err)
        if (ch == ',' || ch == '\n' || ch == '\r')
            ch = ' ';
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s,%d,%d,%d,%.3g,%.4f,%.2f,%d,%d,%s,%s,%s\n", r.type.c_str(),
                  r.n, r.k, r.ncols, r.rel_err, r.gpu_ms, r.packed_gbs, r.ok ? 1 : 0,
                  r.d3d12_ran ? 1 : 0, host_label ? host_label : "unknown", date_buf, err.c_str());
    return buf;
}

} // namespace xllama

    #if !defined(_WIN32)

namespace xllama {

bool ggml_d3d12_register() {
    return false;
}

ggml_backend_buffer_type_t ggml_d3d12_weights_buft() {
    return nullptr;
}

bool ggml_d3d12_take_stats(D3d12PhaseStats* out) {
    if (out)
        *out = D3d12PhaseStats{};
    return false;
}

void run_d3d12_selftest(std::vector<D3d12SelftestRow>* out) {
    if (!out)
        return;
    D3d12SelftestRow r;
    r.type = "-";
    r.error = "d3d12 unavailable on this platform";
    out->push_back(std::move(r));
}

} // namespace xllama

    #else // _WIN32 — the backend itself

        #include <chrono>
        #include <cstdint>
        #include <memory>
        #include <mutex>
        #include <random>

        #include <dxgi1_4.h>

        #include "d3d12_compute.h"
        #include "ggml-alloc.h"
        #include "ggml-backend-impl.h"
        #include "ggml-backend.h"
        #include "ggml_d3d12_add_bias_f32_dxil.h"
        #include "ggml_d3d12_get_rows_q6_k_dxil.h"
        #include "ggml_d3d12_mmv_q4_0_t128_dxil.h"
        #include "ggml_d3d12_mmv_q4_0_t64_dxil.h"
        #include "ggml_d3d12_mmv_q4_k_t128_dxil.h"
        #include "ggml_d3d12_mmv_q4_k_t64_dxil.h"
        #include "ggml_d3d12_mmv_q6_k_t128_dxil.h"
        #include "ggml_d3d12_mmv_q6_k_t64_dxil.h"
        #include "ggml_d3d12_quantize_q8_0_dxil.h"
        #include "ggml_d3d12_quantize_q8_k_dxil.h"
        #include "ggml_d3d12_rope_f32_dxil.h"
        #include "ggml_d3d12_swiglu_f32_dxil.h"
        #include "xllama/d3d12_dyn.h"
        #include "xllama/path_utils.h"
        #include "xllama/platform.h"
        #include "xllama/utf8_utils.h"

namespace xllama {
namespace {

using d3d12c::ComPtr;

enum Pso { kPsoQ40 = 0, kPsoQ4K = 1, kPsoQ6K = 2, kPsoCount = 3 };

int pso_for(ggml_type t) {
    switch (t) {
    case GGML_TYPE_Q4_0:
        return kPsoQ40;
    case GGML_TYPE_Q4_K:
        return kPsoQ4K;
    case GGML_TYPE_Q6_K:
        return kPsoQ6K;
    default:
        return -1;
    }
}

// Process-wide D3D12 state behind the ggml device. One queue, serialised.
struct Gpu {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> cl;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pso[kPsoCount][2]; // [type][0 = 64 threads, 1 = 128]
    ComPtr<ID3D12PipelineState> pso_get_rows;      // Q6_K
    ComPtr<ID3D12PipelineState> pso_add_bias;      // F32 bias ADD
    ComPtr<ID3D12PipelineState> pso_rope;          // F32 ROPE
    // d3d12-bias-add.txt = 1: supports_op accepts the bias ADD, so llama.cpp
    // places biases in D3D12_Weights and keeps them in the projection's split.
    bool bias_add = false;
    bool rope = false;
    // d3d12-gpu-quant.txt = 1: a matmul whose input this submission produced
    // quantizes it on the GPU instead of ending the submission.
    ComPtr<ID3D12PipelineState> pso_quant[2]; // [0 = q8_0, 1 = q8_K]
    bool gpu_quant = false;
    // d3d12-swiglu.txt = 1: SWIGLU (separate gate and up) runs here.
    ComPtr<ID3D12PipelineState> pso_swiglu;
    bool swiglu = false;
    ComPtr<ID3D12QueryHeap> ts;
    ComPtr<ID3D12Resource> ts_rb;
    // #421 per-dispatch timing (d3d12-dispatch-timing.txt): one timestamp
    // before the first dispatch and one after each, grown on demand.
    bool dispatch_timing = false;
    ComPtr<ID3D12QueryHeap> dts;
    ComPtr<ID3D12Resource> dts_rb;
    UINT dts_cap = 0;
    ComPtr<ID3D12Resource> staging;  // weight upload ring (kStagingBytes)
    ComPtr<ID3D12Resource> readback; // get_tensor on weights (kStagingBytes)
    std::uint8_t* staging_ptr = nullptr;
    std::uint8_t* readback_ptr = nullptr;
    // q8 activations for the next submission (CPU-written, GPU-read; grows on
    // demand to the largest single input, ~6 MiB for a 512-token ubatch).
    ComPtr<ID3D12Resource> xq;
    std::uint8_t* xq_ptr = nullptr;
    std::size_t xq_bytes = 0;
    d3d12c::QueueFence fence;
    UINT64 ts_freq = 0;
    LUID luid = {};
    double last_gpu_ms = 0.0;
    // Per-backend-lifetime counters, logged when the backend is freed.
    std::uint64_t n_calls = 0;
    std::uint64_t n_dispatches = 0; // MUL_MAT + GET_ROWS + ADD
    double wall_ms = 0.0;
    double gpu_ms = 0.0;
    // #421: activity since the last ggml_d3d12_take_stats(), guarded by mu.
    D3d12PhaseStats phase;
    std::mutex mu;
    bool ok = false;
    std::string error;
};

// Weight upload / readback ring. Mapped for the process lifetime and counted in
// its working set, so it stays small: 64 MiB each put the first GPU-layer
// smoke at 640 MB peak vs 311 MB on the CPU (D2b); 8 MiB costs a few more
// round trips at load only.
constexpr UINT64 kStagingBytes = 8ull << 20;

// `with_rope` adds u2 (the ROPE cos/sin table, SWIGLU's up input). Without the
// ROPE or SWIGLU knob the signature is the four-parameter one every other
// kernel was built against.
ComPtr<ID3D12RootSignature> make_root_sig(ID3D12Device* device, bool with_rope, std::string* err) {
    D3D12_ROOT_PARAMETER p[5] = {};
    p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    p[0].Constants.ShaderRegister = 0;
    p[0].Constants.Num32BitValues = 8;
    p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; // t0 weights
    p[1].Descriptor.ShaderRegister = 0;
    p[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; // u0 output
    p[2].Descriptor.ShaderRegister = 0;
    p[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; // u1 activations
    p[3].Descriptor.ShaderRegister = 1;
    p[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; // u2 ROPE cos/sin table
    p[4].Descriptor.ShaderRegister = 2;
    for (auto& q : p)
        q.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC desc = {};
    desc.NumParameters = with_rope ? 5 : 4;
    desc.pParameters = p;
    auto serialize = d3d12_dyn::SerializeRootSignature();
    if (!serialize) {
        *err = "D3D12SerializeRootSignature not available";
        return {};
    }
    ComPtr<ID3DBlob> sig, blob_err;
    HRESULT hr = serialize(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &blob_err);
    if (FAILED(hr)) {
        *err = d3d12c::hr_message("D3D12SerializeRootSignature", hr);
        return {};
    }
    ComPtr<ID3D12RootSignature> root;
    hr = device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                     IID_PPV_ARGS(&root));
    if (FAILED(hr)) {
        *err = d3d12c::hr_message("CreateRootSignature", hr);
        return {};
    }
    return root;
}

bool init_gpu(Gpu& g) {
    std::string& err = g.error;
    g.device = d3d12c::create_device(&err);
    if (!g.device)
        return false;
    D3D12_FEATURE_DATA_ARCHITECTURE1 arch = {};
    if (FAILED(g.device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE1, &arch, sizeof(arch))) ||
        !arch.CacheCoherentUMA) {
        // D3D12_Host relies on a CPU-cached GPU-visible heap (D1: cc_uma=1 on Series S).
        err = "adapter is not cache-coherent UMA";
        return false;
    }
    g.luid = g.device->GetAdapterLuid();
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    HRESULT hr = g.device->CreateCommandQueue(&qd, IID_PPV_ARGS(&g.queue));
    if (FAILED(hr)) {
        err = d3d12c::hr_message("CreateCommandQueue", hr);
        return false;
    }
    hr = g.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&g.alloc));
    if (SUCCEEDED(hr))
        hr = g.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g.alloc.Get(), nullptr,
                                         IID_PPV_ARGS(&g.cl));
    if (FAILED(hr)) {
        err = d3d12c::hr_message("CreateCommandList", hr);
        return false;
    }
    g.cl->Close();
    if (!g.fence.init(g.device.Get(), &err))
        return false;
    if (FILE* f = _wfopen(utf8_to_wstring(resolve_local_path("d3d12-rope.txt")).c_str(), L"rb")) {
        char config[16] = {};
        const std::size_t bytes = std::fread(config, 1, sizeof(config), f);
        std::fclose(f);
        g.rope = d3d12_flag_enabled(std::string_view(config, bytes));
    }
    if (FILE* f = _wfopen(utf8_to_wstring(resolve_local_path("d3d12-swiglu.txt")).c_str(), L"rb")) {
        char config[16] = {};
        const std::size_t bytes = std::fread(config, 1, sizeof(config), f);
        std::fclose(f);
        g.swiglu = d3d12_flag_enabled(std::string_view(config, bytes));
    }
    g.root = make_root_sig(g.device.Get(), g.rope || g.swiglu, &err);
    if (!g.root && (g.rope || g.swiglu)) {
        // The opt-in must not take the backend down: fall back to the
        // four-parameter signature every other kernel was built against.
        log_output("[xllama] d3d12: ROPE/SWIGLU unavailable: " + err + "\n");
        g.rope = false;
        g.swiglu = false;
        err.clear();
        g.root = make_root_sig(g.device.Get(), false, &err);
    }
    if (!g.root)
        return false;
    const void* blobs[kPsoCount][2] = {{kGgmlD3d12MmvQ40T64Dxil, kGgmlD3d12MmvQ40T128Dxil},
                                       {kGgmlD3d12MmvQ4KT64Dxil, kGgmlD3d12MmvQ4KT128Dxil},
                                       {kGgmlD3d12MmvQ6KT64Dxil, kGgmlD3d12MmvQ6KT128Dxil}};
    const size_t sizes[kPsoCount][2] = {
        {kGgmlD3d12MmvQ40T64DxilSize, kGgmlD3d12MmvQ40T128DxilSize},
        {kGgmlD3d12MmvQ4KT64DxilSize, kGgmlD3d12MmvQ4KT128DxilSize},
        {kGgmlD3d12MmvQ6KT64DxilSize, kGgmlD3d12MmvQ6KT128DxilSize}};
    for (int i = 0; i < kPsoCount; ++i) {
        for (int wd = 0; wd < 2; ++wd) {
            D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
            pd.pRootSignature = g.root.Get();
            pd.CS.pShaderBytecode = blobs[i][wd];
            pd.CS.BytecodeLength = sizes[i][wd];
            hr = g.device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g.pso[i][wd]));
            if (FAILED(hr)) {
                err = d3d12c::hr_message("CreateComputePipelineState", hr);
                return false;
            }
        }
    }
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = g.root.Get();
        pd.CS.pShaderBytecode = kGgmlD3d12GetRowsQ6KDxil;
        pd.CS.BytecodeLength = kGgmlD3d12GetRowsQ6KDxilSize;
        hr = g.device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g.pso_get_rows));
        if (FAILED(hr)) {
            err = d3d12c::hr_message("CreateComputePipelineState", hr);
            return false;
        }
    }
    if (FILE* f =
            _wfopen(utf8_to_wstring(resolve_local_path("d3d12-bias-add.txt")).c_str(), L"rb")) {
        char config[16] = {};
        const std::size_t bytes = std::fread(config, 1, sizeof(config), f);
        std::fclose(f);
        g.bias_add = d3d12_flag_enabled(std::string_view(config, bytes));
    }
    if (g.bias_add) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = g.root.Get();
        pd.CS.pShaderBytecode = kGgmlD3d12AddBiasF32Dxil;
        pd.CS.BytecodeLength = kGgmlD3d12AddBiasF32DxilSize;
        hr = g.device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g.pso_add_bias));
        if (FAILED(hr)) {
            log_output("[xllama] d3d12: bias ADD unavailable: " +
                       d3d12c::hr_message("CreateComputePipelineState", hr) + "\n");
            g.bias_add = false;
        }
    }
    log_output(std::string("[xllama] d3d12: bias_add=") + (g.bias_add ? "1" : "0") + "\n");
    if (g.rope) { // knob read before the root signature, which it shapes
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = g.root.Get();
        pd.CS.pShaderBytecode = kGgmlD3d12RopeF32Dxil;
        pd.CS.BytecodeLength = kGgmlD3d12RopeF32DxilSize;
        hr = g.device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g.pso_rope));
        if (FAILED(hr)) {
            log_output("[xllama] d3d12: ROPE unavailable: " +
                       d3d12c::hr_message("CreateComputePipelineState", hr) + "\n");
            g.rope = false;
        }
    }
    log_output(std::string("[xllama] d3d12: rope=") + (g.rope ? "1" : "0") + "\n");
    if (FILE* f =
            _wfopen(utf8_to_wstring(resolve_local_path("d3d12-gpu-quant.txt")).c_str(), L"rb")) {
        char config[16] = {};
        const std::size_t bytes = std::fread(config, 1, sizeof(config), f);
        std::fclose(f);
        g.gpu_quant = d3d12_flag_enabled(std::string_view(config, bytes));
    }
    if (g.gpu_quant) {
        const void* qblobs[2] = {kGgmlD3d12QuantizeQ80Dxil, kGgmlD3d12QuantizeQ8KDxil};
        const size_t qsizes[2] = {kGgmlD3d12QuantizeQ80DxilSize, kGgmlD3d12QuantizeQ8KDxilSize};
        for (int i = 0; i < 2 && g.gpu_quant; ++i) {
            D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
            pd.pRootSignature = g.root.Get();
            pd.CS.pShaderBytecode = qblobs[i];
            pd.CS.BytecodeLength = qsizes[i];
            hr = g.device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g.pso_quant[i]));
            if (FAILED(hr)) {
                log_output("[xllama] d3d12: GPU quantize unavailable: " +
                           d3d12c::hr_message("CreateComputePipelineState", hr) + "\n");
                g.gpu_quant = false;
            }
        }
    }
    log_output(std::string("[xllama] d3d12: gpu_quant=") + (g.gpu_quant ? "1" : "0") + "\n");
    if (g.swiglu) { // knob read before the root signature, which it shapes
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
        pd.pRootSignature = g.root.Get();
        pd.CS.pShaderBytecode = kGgmlD3d12SwigluF32Dxil;
        pd.CS.BytecodeLength = kGgmlD3d12SwigluF32DxilSize;
        hr = g.device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&g.pso_swiglu));
        if (FAILED(hr)) {
            log_output("[xllama] d3d12: SWIGLU unavailable: " +
                       d3d12c::hr_message("CreateComputePipelineState", hr) + "\n");
            g.swiglu = false;
        }
    }
    log_output(std::string("[xllama] d3d12: swiglu=") + (g.swiglu ? "1" : "0") + "\n");
    if (SUCCEEDED(g.queue->GetTimestampFrequency(&g.ts_freq)) && g.ts_freq > 0) {
        D3D12_QUERY_HEAP_DESC qhd = {};
        qhd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        qhd.Count = 2;
        g.device->CreateQueryHeap(&qhd, IID_PPV_ARGS(&g.ts));
        g.ts_rb = d3d12c::create_buffer(g.device.Get(), 256, D3D12_HEAP_TYPE_READBACK,
                                        D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
                                        "ts_rb", &err);
        if (!g.ts || !g.ts_rb) {
            g.ts.Reset();
            g.ts_rb.Reset();
            err.clear();
        }
    }
    if (g.ts) {
        if (FILE* f = _wfopen(
                utf8_to_wstring(resolve_local_path("d3d12-dispatch-timing.txt")).c_str(), L"rb")) {
            char config[16] = {};
            const std::size_t bytes = std::fread(config, 1, sizeof(config), f);
            std::fclose(f);
            g.dispatch_timing = d3d12_flag_enabled(std::string_view(config, bytes));
        }
    }
    log_output(std::string("[xllama] d3d12: dispatch_timing=") + (g.dispatch_timing ? "1" : "0") +
               "\n");
    g.staging = d3d12c::create_buffer(g.device.Get(), kStagingBytes, D3D12_HEAP_TYPE_UPLOAD,
                                      D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ,
                                      "staging", &err);
    g.readback = d3d12c::create_buffer(g.device.Get(), kStagingBytes, D3D12_HEAP_TYPE_READBACK,
                                       D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
                                       "readback", &err);
    if (!g.staging || !g.readback)
        return false;
    void* p = nullptr;
    if (FAILED(g.staging->Map(0, nullptr, &p)))
        return false;
    g.staging_ptr = static_cast<std::uint8_t*>(p);
    if (FAILED(g.readback->Map(0, nullptr, &p)))
        return false;
    g.readback_ptr = static_cast<std::uint8_t*>(p);
    return true;
}

Gpu& gpu() {
    static Gpu g;
    static std::once_flag once;
    std::call_once(once, [] { g.ok = init_gpu(g); });
    return g;
}

// Record into the shared list and run it to completion. Caller holds g.mu.
// `stats` (graph submissions only) receives the record and execute+wait times.
template <typename F> bool run_now(Gpu& g, F&& record, D3d12PhaseStats* stats = nullptr) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    g.alloc->Reset();
    g.cl->Reset(g.alloc.Get(), nullptr);
    record(g.cl.Get());
    g.cl->Close();
    const auto t1 = clock::now();
    ID3D12CommandList* lists[] = {g.cl.Get()};
    g.queue->ExecuteCommandLists(1, lists);
    const bool ok = g.fence.signal_and_wait(g.queue.Get(), /*spin=*/true);
    if (stats) {
        stats->record_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats->exec_wait_ms += std::chrono::duration<double, std::milli>(clock::now() - t1).count();
        ++stats->submissions;
    }
    return ok;
}

// --- Buffers ---

struct BufCtx {
    ComPtr<ID3D12Resource> res;
    D3D12_GPU_VIRTUAL_ADDRESS va = 0;
    std::uint8_t* host = nullptr; // mapped pointer (D3D12_Host only)
    bool is_host = false;
};

void* buf_base(ggml_backend_buffer_t b) {
    auto* c = static_cast<BufCtx*>(b->context);
    return c->is_host ? static_cast<void*>(c->host)
                      : reinterpret_cast<void*>(static_cast<std::uintptr_t>(c->va));
}

void buf_free(ggml_backend_buffer_t b) {
    delete static_cast<BufCtx*>(b->context);
}

std::size_t tensor_offset(const ggml_tensor* t) {
    return static_cast<std::size_t>(static_cast<const std::uint8_t*>(t->data) -
                                    static_cast<const std::uint8_t*>(buf_base(t->buffer)));
}

D3D12_GPU_VIRTUAL_ADDRESS tensor_va(const ggml_tensor* t) {
    return static_cast<BufCtx*>(t->buffer->context)->va + tensor_offset(t);
}

// D3D12_Host: plain memory.
void host_memset(ggml_backend_buffer_t, ggml_tensor* t, uint8_t v, size_t off, size_t size) {
    std::memset(static_cast<std::uint8_t*>(t->data) + off, v, size);
}
void host_set(ggml_backend_buffer_t, ggml_tensor* t, const void* data, size_t off, size_t size) {
    std::memcpy(static_cast<std::uint8_t*>(t->data) + off, data, size);
}
void host_get(ggml_backend_buffer_t, const ggml_tensor* t, void* data, size_t off, size_t size) {
    std::memcpy(data, static_cast<const std::uint8_t*>(t->data) + off, size);
}
bool host_cpy(ggml_backend_buffer_t, const ggml_tensor* src, ggml_tensor* dst) {
    if (!ggml_backend_buffer_is_host(src->buffer))
        return false;
    std::memcpy(dst->data, src->data, ggml_nbytes(src));
    return true;
}
void host_clear(ggml_backend_buffer_t b, uint8_t v) {
    std::memset(static_cast<BufCtx*>(b->context)->host, v, b->size);
}

const ggml_backend_buffer_i kHostBufIface = {
    /* .free_buffer   = */ buf_free,
    /* .get_base      = */ buf_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ host_memset,
    /* .set_tensor    = */ host_set,
    /* .get_tensor    = */ host_get,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ host_cpy,
    /* .clear         = */ host_clear,
    /* .reset         = */ nullptr,
};

// D3D12_Weights: DEFAULT heap, filled through the staging ring. The buffer
// stays in COMMON; buffers promote implicitly to COPY_DEST / shader read and
// decay back at the end of each submission.
void weights_write(ggml_backend_buffer_t b, std::size_t dst_off, const void* data, std::size_t size,
                   int fill) {
    Gpu& g = gpu();
    std::lock_guard<std::mutex> lock(g.mu);
    auto* c = static_cast<BufCtx*>(b->context);
    for (std::size_t done = 0; done < size;) {
        const std::size_t chunk = std::min<std::size_t>(size - done, kStagingBytes);
        if (data)
            std::memcpy(g.staging_ptr, static_cast<const std::uint8_t*>(data) + done, chunk);
        else
            std::memset(g.staging_ptr, fill, chunk);
        run_now(g, [&](ID3D12GraphicsCommandList* cl) {
            cl->CopyBufferRegion(c->res.Get(), dst_off + done, g.staging.Get(), 0, chunk);
        });
        done += chunk;
    }
}
void weights_memset(ggml_backend_buffer_t b, ggml_tensor* t, uint8_t v, size_t off, size_t size) {
    weights_write(b, tensor_offset(t) + off, nullptr, size, v);
}
void weights_set(ggml_backend_buffer_t b, ggml_tensor* t, const void* data, size_t off,
                 size_t size) {
    weights_write(b, tensor_offset(t) + off, data, size, 0);
}
void weights_get(ggml_backend_buffer_t b, const ggml_tensor* t, void* data, size_t off,
                 size_t size) {
    Gpu& g = gpu();
    std::lock_guard<std::mutex> lock(g.mu);
    auto* c = static_cast<BufCtx*>(b->context);
    const std::size_t src_off = tensor_offset(t) + off;
    for (std::size_t done = 0; done < size;) {
        const std::size_t chunk = std::min<std::size_t>(size - done, kStagingBytes);
        run_now(g, [&](ID3D12GraphicsCommandList* cl) {
            cl->CopyBufferRegion(g.readback.Get(), 0, c->res.Get(), src_off + done, chunk);
        });
        std::memcpy(static_cast<std::uint8_t*>(data) + done, g.readback_ptr, chunk);
        done += chunk;
    }
}
void weights_clear(ggml_backend_buffer_t b, uint8_t v) {
    weights_write(b, 0, nullptr, b->size, v);
}

const ggml_backend_buffer_i kWeightsBufIface = {
    /* .free_buffer   = */ buf_free,
    /* .get_base      = */ buf_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ weights_memset,
    /* .set_tensor    = */ weights_set,
    /* .get_tensor    = */ weights_get,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ nullptr,
    /* .clear         = */ weights_clear,
    /* .reset         = */ nullptr,
};

// --- Buffer types ---

ggml_backend_dev_t device_ptr();

const char* host_buft_name(ggml_backend_buffer_type_t) {
    return "D3D12_Host";
}
const char* weights_buft_name(ggml_backend_buffer_type_t) {
    return "D3D12_Weights";
}
size_t buft_alignment(ggml_backend_buffer_type_t) {
    return 256;
}
bool buft_is_host_true(ggml_backend_buffer_type_t) {
    return true;
}
// The Q4_0/Q6_K kernels read whole aligned dwords, up to 2 bytes past a
// tensor's last block; root descriptors have no bounds check, so pad.
size_t weights_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor* t) {
    return ggml_nbytes(t) + 16;
}

ggml_backend_buffer_t alloc_buf(ggml_backend_buffer_type_t buft, size_t size, bool host) {
    Gpu& g = gpu();
    if (!g.ok)
        return nullptr;
    size = std::max<size_t>(size, 256);
    auto* c = new BufCtx;
    c->is_host = host;
    std::string err;
    if (host) {
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_CUSTOM;
        hp.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
        hp.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
        c->res = d3d12c::create_buffer_props(
            g.device.Get(), size, hp, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "D3D12_Host", &err);
        void* p = nullptr;
        if (c->res && SUCCEEDED(c->res->Map(0, nullptr, &p)))
            c->host = static_cast<std::uint8_t*>(p);
    } else {
        c->res = d3d12c::create_buffer(g.device.Get(), size, D3D12_HEAP_TYPE_DEFAULT,
                                       D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON,
                                       "D3D12_Weights", &err);
    }
    if (!c->res || (host && !c->host)) {
        char msg[256];
        std::snprintf(msg, sizeof(msg), "[xllama] d3d12: allocation of %zu bytes failed: %s\n",
                      size, err.c_str());
        log_output(msg);
        delete c;
        return nullptr;
    }
    c->va = c->res->GetGPUVirtualAddress();
    if (size >= (1u << 20)) {
        char msg[128];
        std::snprintf(msg, sizeof(msg), "[xllama] d3d12: %s buffer %.1f MiB\n",
                      host ? "D3D12_Host" : "D3D12_Weights", static_cast<double>(size) / 1048576.0);
        log_output(msg);
    }
    return ggml_backend_buffer_init(buft, host ? kHostBufIface : kWeightsBufIface, c, size);
}

ggml_backend_buffer_t host_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    return alloc_buf(buft, size, true);
}
ggml_backend_buffer_t weights_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    return alloc_buf(buft, size, false);
}

ggml_backend_buffer_type kHostBuft = {
    /* .iface = */
    {
        /* .get_name         = */ host_buft_name,
        /* .alloc_buffer     = */ host_alloc,
        /* .alloc_buffer_n   = */ nullptr,
        /* .get_alignment    = */ buft_alignment,
        /* .get_max_size     = */ nullptr,
        /* .get_alloc_size   = */ nullptr,
        /* .get_alloc_size_n = */ nullptr,
        /* .is_host          = */ buft_is_host_true,
    },
    /* .device  = */ nullptr,
    /* .context = */ nullptr,
};
ggml_backend_buffer_type kWeightsBuft = {
    /* .iface = */
    {
        /* .get_name         = */ weights_buft_name,
        /* .alloc_buffer     = */ weights_alloc,
        /* .alloc_buffer_n   = */ nullptr,
        /* .get_alignment    = */ buft_alignment,
        /* .get_max_size     = */ nullptr,
        /* .get_alloc_size   = */ weights_alloc_size,
        /* .get_alloc_size_n = */ nullptr,
        /* .is_host          = */ nullptr,
    },
    /* .device  = */ nullptr,
    /* .context = */ nullptr,
};

bool is_ours(ggml_backend_buffer_type_t buft) {
    return buft == &kHostBuft || buft == &kWeightsBuft;
}

// --- Backend ---

const char* backend_name(ggml_backend_t) {
    return "D3D12";
}
void backend_free(ggml_backend_t b) {
    Gpu& g = gpu();
    {
        std::lock_guard<std::mutex> lock(g.mu);
        char msg[256];
        std::snprintf(msg, sizeof(msg),
                      "[xllama] d3d12: %llu graph_compute calls, %llu dispatches, %.1f ms wall "
                      "(%.1f ms GPU)\n",
                      static_cast<unsigned long long>(g.n_calls),
                      static_cast<unsigned long long>(g.n_dispatches), g.wall_ms, g.gpu_ms);
        log_output(msg);
        g.n_calls = g.n_dispatches = 0;
        g.wall_ms = g.gpu_ms = 0.0;
    }
    delete b;
}

// Grow the q8 activation scratch. Caller holds g.mu and has nothing in flight.
bool ensure_xq(Gpu& g, std::size_t bytes) {
    if (bytes <= g.xq_bytes)
        return true;
    const std::size_t want = std::max<std::size_t>(bytes, std::size_t(1) << 20);
    g.xq.Reset();
    g.xq_ptr = nullptr;
    g.xq_bytes = 0;
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_CUSTOM;
    hp.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
    hp.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
    std::string err;
    g.xq = d3d12c::create_buffer_props(g.device.Get(), want, hp,
                                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "D3D12_Xq", &err);
    void* p = nullptr;
    if (!g.xq || FAILED(g.xq->Map(0, nullptr, &p))) {
        log_output("[xllama] d3d12: q8 activation scratch allocation failed: " + err + "\n");
        g.xq.Reset();
        return false;
    }
    g.xq_ptr = static_cast<std::uint8_t*>(p);
    g.xq_bytes = want;
    return true;
}

// The tensor that owns a node's memory (follows view chains).
const ggml_tensor* base_of(const ggml_tensor* t) {
    while (t->view_src)
        t = t->view_src;
    return t;
}

// One dispatch: a MUL_MAT (u1 = its quantized input) or a GET_ROWS (u1 = its
// row ids), both staged in g.xq, or a bias ADD (u1 = its f32 input, read in
// place from D3D12_Host, or staged in g.xq when it lives elsewhere).
struct Op {
    const ggml_tensor* node;
    std::size_t xq_off;      // staged input in g.xq
    std::uint32_t xq_row;    // bytes per quantized column (MUL_MAT)
    bool staged = true;      // ADD/ROPE: false = u1 is src[0] itself
    std::size_t tab_off = 0; // ROPE: cos/sin table in g.xq
    int n_dims = 0;          // ROPE
    int mode = 0;            // ROPE
    // GPU quantize (d3d12-gpu-quant.txt): node is the matmul it feeds, `src`
    // its F32 input; the q8 output goes to g.xq at xq_off, xq_row per column.
    const ggml_tensor* src = nullptr;
    // SWIGLU: the up input, staged in g.xq at up_off when up_staged.
    bool up_staged = false;
    std::size_t up_off = 0;
};

// Room for `count` per-dispatch timestamps. Caller holds g.mu, nothing in flight.
bool ensure_dts(Gpu& g, UINT count) {
    if (count <= g.dts_cap)
        return true;
    const UINT want = std::max<UINT>(count, 256);
    g.dts.Reset();
    g.dts_rb.Reset();
    g.dts_cap = 0;
    D3D12_QUERY_HEAP_DESC qhd = {};
    qhd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qhd.Count = want;
    std::string err;
    if (FAILED(g.device->CreateQueryHeap(&qhd, IID_PPV_ARGS(&g.dts))))
        return false;
    g.dts_rb =
        d3d12c::create_buffer(g.device.Get(), ((std::size_t(want) * 8) + 255) & ~std::size_t(255),
                              D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                              D3D12_RESOURCE_STATE_COPY_DEST, "dts_rb", &err);
    if (!g.dts_rb) {
        g.dts.Reset();
        return false;
    }
    g.dts_cap = want;
    return true;
}

// The weight a dispatch reads: src[0] for MUL_MAT / GET_ROWS, the bias for ADD.
const ggml_tensor* dispatch_weight(const ggml_tensor* node) {
    return node->op == GGML_OP_ADD ? node->src[1] : node->src[0];
}

const char* dispatch_op_name(const ggml_tensor* node) {
    switch (node->op) {
    case GGML_OP_GET_ROWS:
        return "get_rows";
    case GGML_OP_ADD:
        return "add";
    case GGML_OP_ROPE:
        return "rope";
    case GGML_OP_GLU:
        return "swiglu";
    default:
        return "mul_mat";
    }
}

// Columns of a dispatch for the profile: tokens. A ROPE node is [head_dim,
// heads, tokens], so its ne[1] is the head count.
std::int64_t dispatch_columns(const ggml_tensor* node) {
    return node->op == GGML_OP_ROPE ? node->ne[2] : node->ne[1];
}

// What a dispatch is, for the per-dispatch profile.
D3d12DispatchKind dispatch_kind(const ggml_tensor* node) {
    const ggml_tensor* w = dispatch_weight(node);
    D3d12DispatchKind k;
    k.type = ggml_type_name(w->type);
    k.cols = d3d12_cols_class(dispatch_columns(node));
    if (node->op == GGML_OP_ADD) {
        k.op = "add";
        k.threads = kD3d12AddThreads;
    } else if (node->op == GGML_OP_ROPE) {
        k.op = "rope";
        k.threads = kD3d12RopeThreads;
    } else if (node->op == GGML_OP_GLU) {
        k.op = "swiglu";
        k.threads = kD3d12SwigluThreads;
    } else if (node->op == GGML_OP_GET_ROWS) {
        k.op = "get_rows";
    } else {
        k.op = "mul_mat";
        k.threads = d3d12_mm_threads(w->ne[0]);
    }
    return k;
}

D3d12LayerKey dispatch_layer_key(const ggml_tensor* node) {
    D3d12LayerKey k;
    k.op = dispatch_op_name(node);
    const ggml_tensor* w = dispatch_weight(node);
    const char* weight = w ? w->name : "";
    k.layer = d3d12_layer_from_names(weight, node->name);
    return k;
}

// A GPU quantize is profiled as its own op ("quant"), attributed to the layer
// of the matmul it feeds; every other dispatch by its node.
D3d12DispatchKind op_kind(const Op& op) {
    if (!op.src)
        return dispatch_kind(op.node);
    D3d12DispatchKind k;
    k.op = "quant";
    k.type = ggml_type_name(d3d12_activation_type(op.node->src[0]->type));
    k.cols = d3d12_cols_class(op.src->ne[1]);
    k.threads = kD3d12QuantThreads;
    return k;
}

D3d12LayerKey op_layer_key(const Op& op) {
    D3d12LayerKey k = dispatch_layer_key(op.node);
    if (op.src)
        k.op = "quant";
    return k;
}

// Record and run one submission. Caller holds g.mu.
bool submit_ops(Gpu& g, const std::vector<Op>& ops) {
    const bool ts = g.ts && g.ts_rb;
    const bool dt = ts && g.dispatch_timing && ensure_dts(g, static_cast<UINT>(ops.size() + 1));
    // A UAV barrier before each timestamp waits for the dispatch to finish, so
    // the interval between two timestamps belongs to one dispatch.
    D3D12_RESOURCE_BARRIER uav = {};
    uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uav.UAV.pResource = nullptr;
    double exec_before = 0.0;
    if (dt)
        exec_before = g.phase.exec_wait_ms;
    const bool ran = run_now(
        g,
        [&](ID3D12GraphicsCommandList* cl) {
            cl->SetComputeRootSignature(g.root.Get());
            if (ts)
                cl->EndQuery(g.ts.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
            if (dt)
                cl->EndQuery(g.dts.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
            const D3D12_GPU_VIRTUAL_ADDRESS xq_va = g.xq ? g.xq->GetGPUVirtualAddress() : 0;
            UINT stamp = 0;
            const auto after_dispatch = [&]() {
                if (!dt)
                    return;
                cl->ResourceBarrier(1, &uav);
                cl->EndQuery(g.dts.Get(), D3D12_QUERY_TYPE_TIMESTAMP, ++stamp);
            };
            // Matmuls and GET_ROWS in one submission are independent. A bias
            // ADD reads its projection's output and may hand its input's memory
            // to a later dispatch (gallocr reuse), so it is fenced by a UAV
            // barrier on both sides (dt already puts one after each dispatch).
            bool prev = false, prev_add = false;
            for (const Op& op : ops) {
                const ggml_tensor* node = op.node;
                const ggml_tensor* w = node->src[0];
                const bool is_quant = op.src != nullptr;
                const bool is_rope = !is_quant && node->op == GGML_OP_ROPE;
                const bool is_add = !is_quant && node->op == GGML_OP_ADD;
                const bool is_glu = !is_quant && node->op == GGML_OP_GLU;
                // A quantize reads a tensor an earlier dispatch wrote and its
                // matmul reads the quantize's output: barriers on both sides.
                // SWIGLU reads gate and up from the matmuls before it.
                const bool fenced = is_add || is_rope || is_quant || is_glu;
                if (prev && !dt && (fenced || prev_add))
                    cl->ResourceBarrier(1, &uav);
                prev = true;
                prev_add = fenced;
                if (is_quant) {
                    const ggml_tensor* x = op.src;
                    const ggml_type at = d3d12_activation_type(w->type);
                    const std::int64_t units = d3d12_quant_units(at, x->ne[0]);
                    const std::uint32_t c[8] = {
                        static_cast<std::uint32_t>(units),
                        static_cast<std::uint32_t>(x->nb[1] / sizeof(float)),
                        op.xq_row,
                        0,
                        0,
                        0,
                        0,
                        0};
                    cl->SetPipelineState(g.pso_quant[at == GGML_TYPE_Q8_0 ? 0 : 1].Get());
                    cl->SetComputeRoot32BitConstants(0, 8, c, 0);
                    cl->SetComputeRootUnorderedAccessView(2, xq_va + op.xq_off);
                    cl->SetComputeRootUnorderedAccessView(3, tensor_va(x));
                    cl->Dispatch(
                        static_cast<UINT>((units + kD3d12QuantThreads - 1) / kD3d12QuantThreads),
                        static_cast<UINT>(x->ne[1]), 1);
                    after_dispatch();
                    continue;
                }
                if (is_glu) {
                    const ggml_tensor* ga = node->src[0];
                    const ggml_tensor* up = node->src[1];
                    const std::uint32_t c[8] = {
                        static_cast<std::uint32_t>(node->ne[0]),
                        static_cast<std::uint32_t>(op.staged ? node->ne[0]
                                                             : ga->nb[1] / sizeof(float)),
                        static_cast<std::uint32_t>(op.up_staged ? node->ne[0]
                                                                : up->nb[1] / sizeof(float)),
                        static_cast<std::uint32_t>(node->nb[1] / sizeof(float)),
                        0,
                        0,
                        0,
                        0};
                    cl->SetPipelineState(g.pso_swiglu.Get());
                    cl->SetComputeRoot32BitConstants(0, 8, c, 0);
                    cl->SetComputeRootUnorderedAccessView(2, tensor_va(node));
                    cl->SetComputeRootUnorderedAccessView(3, op.staged ? xq_va + op.xq_off
                                                                       : tensor_va(ga));
                    cl->SetComputeRootUnorderedAccessView(4, op.up_staged ? xq_va + op.up_off
                                                                          : tensor_va(up));
                    cl->Dispatch(static_cast<UINT>((node->ne[0] + kD3d12SwigluThreads - 1) /
                                                   kD3d12SwigluThreads),
                                 static_cast<UINT>(node->ne[1]), 1);
                    after_dispatch();
                    continue;
                }
                if (is_rope) {
                    const ggml_tensor* x = node->src[0];
                    const std::uint32_t nd = static_cast<std::uint32_t>(op.n_dims);
                    const std::uint32_t xr = op.staged ? static_cast<std::uint32_t>(x->ne[0])
                                                       : static_cast<std::uint32_t>(x->nb[1] / 4);
                    const std::uint32_t xt = op.staged ? xr * static_cast<std::uint32_t>(x->ne[1])
                                                       : static_cast<std::uint32_t>(x->nb[2] / 4);
                    const std::uint32_t c[8] = {static_cast<std::uint32_t>(node->ne[0]),
                                                nd,
                                                static_cast<std::uint32_t>(op.mode),
                                                xr,
                                                xt,
                                                static_cast<std::uint32_t>(node->nb[1] / 4),
                                                static_cast<std::uint32_t>(node->nb[2] / 4),
                                                0};
                    cl->SetPipelineState(g.pso_rope.Get());
                    cl->SetComputeRoot32BitConstants(0, 8, c, 0);
                    cl->SetComputeRootUnorderedAccessView(2, tensor_va(node));
                    cl->SetComputeRootUnorderedAccessView(3, op.staged ? xq_va + op.xq_off
                                                                       : tensor_va(x));
                    cl->SetComputeRootUnorderedAccessView(4, xq_va + op.tab_off);
                    const std::int64_t span = d3d12_rope_span(nd, node->ne[0]);
                    cl->Dispatch(
                        static_cast<UINT>((span + kD3d12RopeThreads - 1) / kD3d12RopeThreads),
                        static_cast<UINT>(node->ne[1]), static_cast<UINT>(node->ne[2]));
                    after_dispatch();
                    continue;
                }
                if (is_add) {
                    const ggml_tensor* b = node->src[1];
                    const std::uint32_t c[8] = {
                        static_cast<std::uint32_t>(node->ne[0]),
                        static_cast<std::uint32_t>(op.staged ? node->ne[0]
                                                             : w->nb[1] / sizeof(float)),
                        static_cast<std::uint32_t>(node->nb[1] / sizeof(float)),
                        0,
                        0,
                        0,
                        0,
                        0};
                    cl->SetPipelineState(g.pso_add_bias.Get());
                    cl->SetComputeRoot32BitConstants(0, 8, c, 0);
                    cl->SetComputeRootShaderResourceView(1, tensor_va(b));
                    cl->SetComputeRootUnorderedAccessView(2, tensor_va(node));
                    cl->SetComputeRootUnorderedAccessView(3, op.staged ? xq_va + op.xq_off
                                                                       : tensor_va(w));
                    cl->Dispatch(
                        static_cast<UINT>((node->ne[0] + kD3d12AddThreads - 1) / kD3d12AddThreads),
                        static_cast<UINT>(node->ne[1]), 1);
                    after_dispatch();
                    continue;
                }
                if (node->op == GGML_OP_GET_ROWS) {
                    const std::uint32_t c[8] = {
                        static_cast<std::uint32_t>(node->ne[1]),
                        static_cast<std::uint32_t>(w->ne[0]),
                        static_cast<std::uint32_t>(w->ne[0] / kD3d12Chunk),
                        static_cast<std::uint32_t>(w->nb[1]),
                        0,
                        static_cast<std::uint32_t>(node->nb[1] / sizeof(float)),
                        0,
                        0};
                    cl->SetPipelineState(g.pso_get_rows.Get());
                    cl->SetComputeRoot32BitConstants(0, 8, c, 0);
                    cl->SetComputeRootShaderResourceView(1, tensor_va(w));
                    cl->SetComputeRootUnorderedAccessView(2, tensor_va(node));
                    cl->SetComputeRootUnorderedAccessView(3, xq_va + op.xq_off);
                    cl->Dispatch(static_cast<UINT>(w->ne[0] / kD3d12Chunk),
                                 static_cast<UINT>(node->ne[1]), 1);
                    after_dispatch();
                    continue;
                }
                const D3d12Dispatch d = d3d12_mm_dispatch(node->ne[0], node->ne[1]);
                const std::uint32_t c[8] = {static_cast<std::uint32_t>(w->ne[1]),
                                            static_cast<std::uint32_t>(w->ne[0]),
                                            static_cast<std::uint32_t>(w->ne[0] / kD3d12Chunk),
                                            static_cast<std::uint32_t>(w->nb[1]),
                                            op.xq_row,
                                            static_cast<std::uint32_t>(node->nb[1] / sizeof(float)),
                                            0,
                                            0};
                const int wide = d3d12_mm_threads(w->ne[0]) == kD3d12MmvThreadsLong ? 1 : 0;
                cl->SetPipelineState(g.pso[pso_for(w->type)][wide].Get());
                cl->SetComputeRoot32BitConstants(0, 8, c, 0);
                cl->SetComputeRootShaderResourceView(1, tensor_va(w));
                cl->SetComputeRootUnorderedAccessView(2, tensor_va(node));
                cl->SetComputeRootUnorderedAccessView(3, xq_va + op.xq_off);
                cl->Dispatch(d.groups_x, d.groups_y, 1);
                after_dispatch();
            }
            if (dt)
                cl->ResolveQueryData(g.dts.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0,
                                     static_cast<UINT>(ops.size() + 1), g.dts_rb.Get(), 0);
            if (ts) {
                cl->EndQuery(g.ts.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
                cl->ResolveQueryData(g.ts.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, g.ts_rb.Get(),
                                     0);
            }
        },
        &g.phase);
    if (!ran)
        return false;
    bool ts_ok = false;
    double submission_gpu_ms = 0.0;
    if (ts && g.ts_freq > 0) {
        void* p = nullptr;
        if (SUCCEEDED(g.ts_rb->Map(0, nullptr, &p))) {
            const auto* t = static_cast<const std::uint64_t*>(p);
            const double ms = t[1] > t[0] ? 1000.0 * static_cast<double>(t[1] - t[0]) /
                                                static_cast<double>(g.ts_freq)
                                          : 0.0;
            g.last_gpu_ms += ms;
            ts_ok = t[1] > t[0];
            if (ts_ok) {
                g.phase.gpu_ms += ms;
                submission_gpu_ms = ms;
            }
            g.ts_rb->Unmap(0, nullptr);
        }
    }
    ++(ts_ok ? g.phase.ts_valid : g.phase.ts_missing);
    if (dt) {
        // wait = exec_wait − submission GPU time. Missing GPU timestamps leave
        // wait as missing; it is not filled in from wall time.
        const std::optional<double> wait_share =
            ts_ok && !ops.empty()
                ? std::optional<double>((g.phase.exec_wait_ms - exec_before - submission_gpu_ms) /
                                        static_cast<double>(ops.size()))
                : std::nullopt;
        void* p = nullptr;
        const D3D12_RANGE range = {0, (ops.size() + 1) * sizeof(std::uint64_t)};
        if (SUCCEEDED(g.dts_rb->Map(0, &range, &p))) {
            const auto* t = static_cast<const std::uint64_t*>(p);
            for (std::size_t i = 0; i < ops.size(); ++i) {
                const Op& op = ops[i];
                const ggml_tensor* node = op.node;
                const ggml_tensor* w = dispatch_weight(node);
                const D3d12LayerKey key = op_layer_key(op);
                if (t[i + 1] <= t[i]) {
                    g.phase.dispatch.note_missing();
                    g.phase.layer.add(key, std::nullopt, wait_share);
                    continue;
                }
                const double ms =
                    1000.0 * static_cast<double>(t[i + 1] - t[i]) / static_cast<double>(g.ts_freq);
                const double bytes =
                    op.src ? static_cast<double>(ggml_nbytes(op.src))
                    : node->op == GGML_OP_ADD || node->op == GGML_OP_ROPE || node->op == GGML_OP_GLU
                        ? static_cast<double>(ggml_nbytes(w))
                    : node->op == GGML_OP_GET_ROWS ? static_cast<double>(w->nb[1]) * node->ne[1]
                                                   : static_cast<double>(w->nb[1]) * w->ne[1];
                g.phase.dispatch.add(op_kind(op), ms, bytes, dispatch_columns(node));
                g.phase.layer.add(key, ms, wait_share);
            }
            const D3D12_RANGE none = {0, 0};
            g.dts_rb->Unmap(0, &none);
        } else {
            for (std::size_t i = 0; i < ops.size(); ++i) {
                g.phase.dispatch.note_missing();
                g.phase.layer.add(op_layer_key(ops[i]), std::nullopt, wait_share);
            }
        }
    }
    g.n_dispatches += ops.size();
    g.phase.dispatches += ops.size();
    return true;
}

ggml_status backend_graph_compute(ggml_backend_t, ggml_cgraph* cgraph) {
    Gpu& g = gpu();
    std::vector<const ggml_tensor*> mm;
    for (int i = 0; i < ggml_graph_n_nodes(cgraph); ++i) {
        const ggml_tensor* node = ggml_graph_node(cgraph, i);
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0)
            continue;
        switch (node->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_GET_ROWS:
        case GGML_OP_ADD:
        case GGML_OP_ROPE:
        case GGML_OP_GLU:
            mm.push_back(node);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            break;
        default:
            GGML_ABORT("d3d12: unsupported op %s", ggml_op_desc(node));
        }
    }
    if (mm.empty())
        return GGML_STATUS_SUCCESS;

    std::lock_guard<std::mutex> lock(g.mu);
    const auto t0 = std::chrono::steady_clock::now();
    g.last_gpu_ms = 0.0;
    // Activations are quantized on the CPU before the GPU reads them, so a
    // matmul whose input another matmul of the same batch produces (a LoRA
    // chain) waits for that batch. Matmuls sharing an input (q/k/v, gate/up)
    // share one quantized copy.
    std::vector<Op> ops;
    std::vector<const ggml_tensor*> produced; // outputs of `ops`
    struct Quantized {
        const void* data;
        ggml_type type;
        std::size_t off;
        std::uint32_t row;
        int ncols;
        std::size_t src_bytes; // F32 bytes the copy was made from
    };
    std::vector<Quantized> done;
    struct RopeTable { // a cos/sin table staged in this submission
        const void* pos;
        std::int64_t n;
        D3d12RopeParams p;
        std::size_t off;
    };
    std::vector<RopeTable> tables;
    std::size_t used = 0;
    // A new output may reuse the memory of an input quantized earlier in this
    // submission (gallocr): that quantized copy is stale from here on.
    const auto note_produced = [&](const ggml_tensor* node) {
        produced.push_back(base_of(node));
        const auto* lo = static_cast<const std::uint8_t*>(node->data);
        const auto* hi = lo + ggml_nbytes(node);
        done.erase(std::remove_if(done.begin(), done.end(),
                                  [&](const Quantized& d) {
                                      const auto* a = static_cast<const std::uint8_t*>(d.data);
                                      const auto* b = a + d.src_bytes;
                                      return a < hi && lo < b;
                                  }),
                   done.end());
    };
    auto flush = [&]() -> bool {
        if (ops.empty())
            return true;
        const bool ok = submit_ops(g, ops);
        ops.clear();
        produced.clear();
        done.clear();
        tables.clear();
        used = 0;
        return ok;
    };
    for (const ggml_tensor* node : mm) {
        if (node->op == GGML_OP_ROPE) {
            const ggml_tensor* a = node->src[0];
            const ggml_tensor* pos = node->src[1];
            GGML_ASSERT(a->buffer && node->buffer && is_ours(node->buffer->buft));
            GGML_ASSERT(pos->buffer && ggml_backend_buffer_is_host(pos->buffer));
            if (ggml_nelements(node) == 0)
                continue;
            const std::int32_t* op_params = node->op_params;
            D3d12RopeParams rp;
            rp.n_dims = op_params[1];
            std::memcpy(&rp.freq_base, op_params + 5, sizeof(float));
            std::memcpy(&rp.freq_scale, op_params + 6, sizeof(float));
            std::memcpy(&rp.attn_factor, op_params + 8, sizeof(float));
            const std::int64_t nt = node->ne[2];
            const std::size_t tab_bytes = static_cast<std::size_t>(nt) * rp.n_dims * sizeof(float);
            const bool direct = a->buffer->buft == &kHostBuft;
            std::size_t xbytes = 0;
            if (!direct) {
                GGML_ASSERT(ggml_backend_buffer_is_host(a->buffer));
                xbytes = static_cast<std::size_t>(ggml_nbytes(a));
            }
            // Same positions and rope parameters share one table (q and k of a
            // layer) within this graph evaluation; the next layer's starts anew.
            const auto find_table = [&]() -> const RopeTable* {
                for (const RopeTable& t : tables)
                    if (t.pos == pos->data && t.n == nt && t.p.n_dims == rp.n_dims &&
                        t.p.freq_base == rp.freq_base && t.p.freq_scale == rp.freq_scale &&
                        t.p.attn_factor == rp.attn_factor)
                        return &t;
                return nullptr;
            };
            const auto need_bytes = [&](bool has_table) {
                return ((xbytes + 255) & ~std::size_t(255)) +
                       (has_table ? 0 : ((tab_bytes + 255) & ~std::size_t(255)));
            };
            const RopeTable* tab = find_table();
            if (used + need_bytes(tab != nullptr) > g.xq_bytes) {
                if (!flush())
                    return GGML_STATUS_FAILED;
                tab = nullptr; // flush() dropped the staged tables
            }
            Op o{node, 0, 0, true};
            o.n_dims = rp.n_dims;
            o.mode = op_params[2];
            if (!ensure_xq(g, used + need_bytes(tab != nullptr)))
                return GGML_STATUS_FAILED;
            if (!direct) {
                // Another host buffer: a dense copy (contiguous, so one memcpy).
                std::memcpy(g.xq_ptr + used, a->data, xbytes);
                o.xq_off = used;
                used += (xbytes + 255) & ~std::size_t(255);
            } else {
                o.staged = false;
            }
            if (!tab) {
                const auto* ids = static_cast<const std::int32_t*>(pos->data);
                auto* t = reinterpret_cast<float*>(g.xq_ptr + used);
                for (std::int64_t i = 0; i < nt; ++i)
                    d3d12_rope_cache(rp, ids[i], t + i * rp.n_dims);
                tables.push_back({pos->data, nt, rp, used});
                tab = &tables.back();
                used += (tab_bytes + 255) & ~std::size_t(255);
            }
            o.tab_off = tab->off;
            ops.push_back(o);
            note_produced(node);
            continue;
        }
        if (node->op == GGML_OP_GLU) {
            GGML_ASSERT(node->buffer && is_ours(node->buffer->buft));
            if (ggml_nelements(node) == 0)
                continue;
            // Gate and up are read in place from D3D12_Host; anything in another
            // host buffer is computed already (this batch only writes ours) and
            // gets a dense copy in g.xq.
            Op o{node, 0, 0, false};
            std::size_t need = 0;
            const auto dense = [](const ggml_tensor* t) {
                return static_cast<std::size_t>(t->ne[0]) * sizeof(float) *
                       static_cast<std::size_t>(t->ne[1]);
            };
            for (int i = 0; i < 2; ++i) {
                const ggml_tensor* t = node->src[i];
                GGML_ASSERT(t->buffer && ggml_backend_buffer_is_host(t->buffer));
                if (!is_ours(t->buffer->buft))
                    need += (dense(t) + 255) & ~std::size_t(255);
            }
            if (used + need > g.xq_bytes && !flush())
                return GGML_STATUS_FAILED;
            if (!ensure_xq(g, used + need))
                return GGML_STATUS_FAILED;
            for (int i = 0; i < 2; ++i) {
                const ggml_tensor* t = node->src[i];
                if (is_ours(t->buffer->buft))
                    continue;
                const std::size_t row = static_cast<std::size_t>(t->ne[0]) * sizeof(float);
                for (std::int64_t c = 0; c < t->ne[1]; ++c)
                    std::memcpy(g.xq_ptr + used + c * row,
                                static_cast<const std::uint8_t*>(t->data) + c * t->nb[1], row);
                (i == 0 ? o.staged : o.up_staged) = true;
                (i == 0 ? o.xq_off : o.up_off) = used;
                used += (dense(t) + 255) & ~std::size_t(255);
            }
            ops.push_back(o);
            note_produced(node);
            continue;
        }
        if (node->op == GGML_OP_ADD) {
            const ggml_tensor* a = node->src[0];
            const ggml_tensor* b = node->src[1];
            GGML_ASSERT(b->buffer && b->buffer->buft == &kWeightsBuft);
            GGML_ASSERT(a->buffer && node->buffer && is_ours(node->buffer->buft));
            if (ggml_nelements(node) == 0)
                continue;
            if (is_ours(a->buffer->buft)) { // D3D12_Host: the GPU reads it in place
                ops.push_back({node, 0, 0, false});
            } else {
                // Another host buffer: stage a dense copy. Not produced by this
                // batch (its outputs are all ours), so it is already computed.
                GGML_ASSERT(ggml_backend_buffer_is_host(a->buffer));
                const std::size_t row = static_cast<std::size_t>(a->ne[0]) * sizeof(float);
                const std::size_t bytes = row * static_cast<std::size_t>(a->ne[1]);
                if (used + bytes > g.xq_bytes && !flush())
                    return GGML_STATUS_FAILED;
                if (!ensure_xq(g, used + bytes))
                    return GGML_STATUS_FAILED;
                for (std::int64_t c = 0; c < a->ne[1]; ++c)
                    std::memcpy(g.xq_ptr + used + c * row,
                                static_cast<const std::uint8_t*>(a->data) + c * a->nb[1], row);
                ops.push_back({node, used, 0, true});
                used += (bytes + 255) & ~std::size_t(255);
            }
            note_produced(node);
            continue;
        }
        const ggml_tensor* w = node->src[0];
        const ggml_tensor* x = node->src[1];
        GGML_ASSERT(w->buffer && w->buffer->buft == &kWeightsBuft);
        GGML_ASSERT(x->buffer && ggml_backend_buffer_is_host(x->buffer));
        GGML_ASSERT(node->buffer && is_ours(node->buffer->buft));
        if (ggml_nelements(node) == 0)
            continue; // a ubatch without output rows: nothing to compute
        if (node->op == GGML_OP_GET_ROWS) {
            // Row ids are tiny: stage them next to the quantized activations.
            const std::size_t bytes = static_cast<std::size_t>(x->ne[0]) * sizeof(std::int32_t);
            if (used + bytes > g.xq_bytes && !flush())
                return GGML_STATUS_FAILED;
            if (!ensure_xq(g, used + bytes))
                return GGML_STATUS_FAILED;
            std::memcpy(g.xq_ptr + used, x->data, bytes);
            // The kernel does not bound-check rows; ggml-cpu asserts the same.
            const auto* ids = reinterpret_cast<const std::int32_t*>(g.xq_ptr + used);
            for (std::int64_t i = 0; i < x->ne[0]; ++i)
                GGML_ASSERT(ids[i] >= 0 && ids[i] < w->ne[1]);
            ops.push_back({node, used, 0, true});
            note_produced(node);
            used += (bytes + 255) & ~std::size_t(255);
            continue;
        }
        const ggml_tensor* xb = base_of(x);
        const ggml_type at = d3d12_activation_type(w->type);
        const int k = static_cast<int>(x->ne[0]);
        const int ncols = static_cast<int>(x->ne[1]);
        const std::uint32_t row = static_cast<std::uint32_t>(ggml_row_size(at, k));
        bool from_batch = false;
        for (const ggml_tensor* p : produced)
            if (p == xb) {
                from_batch = true;
                break;
            }
        // An input this submission produces lives in D3D12_Host (every output
        // here does): quantize it on the GPU, or end the submission so the CPU
        // can read it.
        const bool gpu_q =
            from_batch && g.gpu_quant && is_ours(x->buffer->buft) && d3d12_quant_units(at, k) > 0;
        if (from_batch && !gpu_q && !flush())
            return GGML_STATUS_FAILED;
        const Quantized* q = nullptr;
        for (const Quantized& d : done)
            if (d.data == x->data && d.type == at && d.row == row && d.ncols >= ncols)
                q = &d;
        if (!q && gpu_q) {
            const std::size_t bytes = static_cast<std::size_t>(row) * ncols;
            if (used + bytes > g.xq_bytes) {
                // Growing g.xq drops what is staged, so the batch must end first;
                // its input is then computed and the CPU path below applies.
                if (!flush())
                    return GGML_STATUS_FAILED;
            } else {
                Op o{node, used, row, true};
                o.src = x;
                ops.push_back(o);
                done.push_back({x->data, at, used, row, ncols, ggml_nbytes(x)});
                q = &done.back();
                used += (bytes + 255) & ~std::size_t(255);
            }
        }
        if (!q) {
            const std::size_t bytes = static_cast<std::size_t>(row) * ncols;
            if (used + bytes > g.xq_bytes && !flush())
                return GGML_STATUS_FAILED;
            if (!ensure_xq(g, used + bytes))
                return GGML_STATUS_FAILED;
            const auto tq = std::chrono::steady_clock::now();
            d3d12_quantize_activations(w->type, static_cast<const float*>(x->data),
                                       x->nb[1] / sizeof(float), k, ncols, g.xq_ptr + used);
            g.phase.quant_ms +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tq)
                    .count();
            done.push_back({x->data, at, used, row, ncols, ggml_nbytes(x)});
            q = &done.back();
            used += (bytes + 255) & ~std::size_t(255); // root UAV addresses stay aligned
        }
        ops.push_back({node, q->off, q->row, true});
        note_produced(node);
    }
    if (!flush())
        return GGML_STATUS_FAILED;
    const double call_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    ++g.n_calls;
    g.gpu_ms += g.last_gpu_ms;
    g.wall_ms += call_ms;
    ++g.phase.calls;
    g.phase.wall_ms += call_ms;
    return GGML_STATUS_SUCCESS;
}

} // namespace

bool ggml_d3d12_take_stats(D3d12PhaseStats* out) {
    Gpu& g = gpu();
    std::lock_guard<std::mutex> lock(g.mu);
    if (out)
        *out = g.ok ? g.phase : D3d12PhaseStats{};
    g.phase = D3d12PhaseStats{};
    return g.ok;
}

namespace {

ggml_guid_t backend_guid() {
    static ggml_guid guid = {0x78, 0x6c, 0x6c, 0x61, 0x6d, 0x61, 0x2d, 0x64,
                             0x33, 0x64, 0x31, 0x32, 0x2d, 0x76, 0x30, 0x31};
    return &guid;
}

const ggml_backend_i kBackendIface = {
    /* .get_name            = */ backend_name,
    /* .free                = */ backend_free,
    /* .set_tensor_async    = */ nullptr,
    /* .get_tensor_async    = */ nullptr,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async    = */ nullptr,
    /* .synchronize         = */ nullptr, // graph_compute completes before returning
    /* .graph_plan_create   = */ nullptr,
    /* .graph_plan_free     = */ nullptr,
    /* .graph_plan_update   = */ nullptr,
    /* .graph_plan_compute  = */ nullptr,
    /* .graph_compute       = */ backend_graph_compute,
    /* .event_record        = */ nullptr,
    /* .event_wait          = */ nullptr,
    /* .graph_optimize      = */ nullptr,
};

// --- Device ---

const char* dev_name(ggml_backend_dev_t) {
    return "D3D12";
}
const char* dev_description(ggml_backend_dev_t) {
    return "xllama D3D12 compute (Q4_0/Q4_K/Q6_K matmul, Q6_K get_rows, F32 bias add and ROPE)";
}
void dev_memory(ggml_backend_dev_t, size_t* free, size_t* total) {
    *free = *total = 0;
    Gpu& g = gpu();
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter3> adapter;
    if (!g.ok || FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) ||
        FAILED(factory->EnumAdapterByLuid(g.luid, IID_PPV_ARGS(&adapter))))
        return;
    DXGI_QUERY_VIDEO_MEMORY_INFO info = {};
    if (SUCCEEDED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
        *total = static_cast<size_t>(info.Budget);
        *free = info.Budget > info.CurrentUsage
                    ? static_cast<size_t>(info.Budget - info.CurrentUsage)
                    : 0;
    }
}
enum ggml_backend_dev_type dev_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}
void dev_props(ggml_backend_dev_t dev, ggml_backend_dev_props* props) {
    props->name = dev_name(dev);
    props->description = dev_description(dev);
    props->type = dev_type(dev);
    props->device_id = nullptr;
    dev_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {/* async */ false, /* host_buffer */ false, /* buffer_from_host_ptr */ false,
                   /* events */ false, /* mmap_support */ false};
}
ggml_backend_t dev_init_backend(ggml_backend_dev_t dev, const char*) {
    if (!gpu().ok)
        return nullptr;
    return new ggml_backend{backend_guid(), kBackendIface, dev, nullptr};
}
ggml_backend_buffer_type_t dev_buffer_type(ggml_backend_dev_t) {
    return &kHostBuft;
}

bool dev_supports_op(ggml_backend_dev_t, const ggml_tensor* op) {
    switch (op->op) {
    case GGML_OP_NONE:
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE:
        return true;
    case GGML_OP_MUL_MAT: {
        const ggml_tensor* w = op->src[0];
        const ggml_tensor* x = op->src[1];
        D3d12MatmulDesc d;
        d.src0_type = w->type;
        d.ne00 = w->ne[0];
        d.ne01 = w->ne[1];
        d.ne02 = w->ne[2];
        d.ne03 = w->ne[3];
        d.ne10 = x->ne[0];
        d.ne11 = x->ne[1];
        d.ne12 = x->ne[2];
        d.ne13 = x->ne[3];
        d.src1_type = x->type;
        d.dst_type = op->type;
        d.src0_contiguous = ggml_is_contiguous(w);
        d.src1_contiguous = ggml_is_contiguous(x);
        d.src0_in_weight_buffer = w->buffer && w->buffer->buft == &kWeightsBuft;
        const bool ok = d3d12_mm_supported(d);
        // A refused matmul whose weight already sits in D3D12_Weights means a
        // placement the backend then cannot run — log the first few (D2b #309).
        static int logged = 0;
        if (!ok && w->buffer && w->buffer->buft == &kWeightsBuft && logged < 8) {
            ++logged;
            char msg[256];
            std::snprintf(msg, sizeof(msg),
                          "[xllama] d3d12: MUL_MAT refused: %s %lldx%lld x [%lld,%lld,%lld,%lld] "
                          "src1=%s dst=%s\n",
                          ggml_type_name(w->type), static_cast<long long>(w->ne[0]),
                          static_cast<long long>(w->ne[1]), static_cast<long long>(x->ne[0]),
                          static_cast<long long>(x->ne[1]), static_cast<long long>(x->ne[2]),
                          static_cast<long long>(x->ne[3]), ggml_type_name(x->type),
                          ggml_type_name(op->type));
            log_output(msg);
        }
        return ok;
    }
    case GGML_OP_GET_ROWS: {
        const ggml_tensor* w = op->src[0];
        const ggml_tensor* ids = op->src[1];
        D3d12GetRowsDesc d;
        d.src0_type = w->type;
        d.ne00 = w->ne[0];
        d.ne01 = w->ne[1];
        d.ne02 = w->ne[2];
        d.ne03 = w->ne[3];
        d.src1_type = ids->type;
        d.ne10 = ids->ne[0];
        d.ne11 = ids->ne[1];
        d.ne12 = ids->ne[2];
        d.dst_type = op->type;
        d.src0_contiguous = ggml_is_contiguous(w);
        d.src1_contiguous = ggml_is_contiguous(ids);
        d.src0_in_weight_buffer = w->buffer && w->buffer->buft == &kWeightsBuft;
        return d3d12_get_rows_supported(d);
    }
    case GGML_OP_ROPE: {
        if (!gpu().rope)
            return false;
        const ggml_tensor* a = op->src[0];
        const ggml_tensor* pos = op->src[1];
        D3d12RopeDesc d;
        d.src0_type = a->type;
        d.ne0 = a->ne[0];
        d.ne1 = a->ne[1];
        d.ne2 = a->ne[2];
        d.ne3 = a->ne[3];
        d.dst_type = op->type;
        d.same_shape = ggml_are_same_shape(a, op);
        d.src0_contiguous = ggml_is_contiguous(a);
        d.dst_contiguous = ggml_is_contiguous(op);
        d.src0_in_weight_buffer = a->buffer && a->buffer->buft == &kWeightsBuft;
        d.src0_in_non_host_buffer = a->buffer && !ggml_backend_buffer_is_host(a->buffer);
        d.pos_in_non_host_buffer = pos->buffer && !ggml_backend_buffer_is_host(pos->buffer);
        d.pos_type = pos->type;
        d.ne_pos = pos->ne[0] * pos->ne[1] * pos->ne[2] * pos->ne[3];
        d.has_freq_factors = op->src[2] != nullptr;
        d.n_dims = op->op_params[1];
        d.mode = op->op_params[2];
        d.n_offs = op->op_params[15];
        std::memcpy(&d.ext_factor, op->op_params + 7, sizeof(float));
        return d3d12_rope_supported(d);
    }
    case GGML_OP_GLU: {
        if (!gpu().swiglu)
            return false;
        const ggml_tensor* a = op->src[0];
        const ggml_tensor* b = op->src[1];
        D3d12SwigluDesc d;
        d.two_sources = b != nullptr;
        if (!d.two_sources)
            return false;
        d.src0_type = a->type;
        d.src1_type = b->type;
        d.dst_type = op->type;
        d.ne0 = op->ne[0];
        d.ne1 = op->ne[1];
        d.ne2 = op->ne[2];
        d.ne3 = op->ne[3];
        d.glu_op_swiglu = ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU;
        d.swapped = op->op_params[1] != 0;
        d.rows_contiguous =
            a->nb[0] == sizeof(float) && b->nb[0] == sizeof(float) && op->nb[0] == sizeof(float);
        d.same_shapes = ggml_are_same_shape(a, op) && ggml_are_same_shape(b, op);
        // Inputs the GPU reads must be host-visible (D3D12_Host or another CPU
        // buffer), never weights; activations without a buffer yet are the
        // scheduler's (D3D12_Host).
        for (const ggml_tensor* t : {a, b})
            if (t->buffer &&
                (t->buffer->buft == &kWeightsBuft || !ggml_backend_buffer_is_host(t->buffer)))
                return false;
        return d3d12_swiglu_supported(d);
    }
    case GGML_OP_ADD: {
        if (!gpu().bias_add)
            return false;
        const ggml_tensor* a = op->src[0];
        const ggml_tensor* b = op->src[1];
        D3d12AddDesc d;
        d.src0_type = a->type;
        d.ne00 = a->ne[0];
        d.ne01 = a->ne[1];
        d.ne02 = a->ne[2];
        d.ne03 = a->ne[3];
        d.src1_type = b->type;
        d.ne10 = b->ne[0];
        d.ne11 = b->ne[1];
        d.ne12 = b->ne[2];
        d.ne13 = b->ne[3];
        d.dst_type = op->type;
        d.src0_rows_contiguous = a->nb[0] == sizeof(float);
        d.src1_contiguous = ggml_is_contiguous(b);
        d.src1_in_weight_buffer = b->buffer && b->buffer->buft == &kWeightsBuft;
        return d3d12_add_supported(d);
    }
    default:
        return false;
    }
}
bool dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    return is_ours(buft);
}

const ggml_backend_device_i kDeviceIface = {
    /* .get_name             = */ dev_name,
    /* .get_description      = */ dev_description,
    /* .get_memory           = */ dev_memory,
    /* .get_type             = */ dev_type,
    /* .get_props            = */ dev_props,
    /* .init_backend         = */ dev_init_backend,
    /* .get_buffer_type      = */ dev_buffer_type,
    // The default buft doubles as the host buft: llama.cpp then allocates the
    // CPU backend's compute buffer in D3D12_Host, which supports_buft accepts,
    // so the scheduler stops copying every matmul input into a second buffer.
    /* .get_host_buffer_type = */ dev_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ dev_supports_op,
    /* .supports_buft        = */ dev_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

// --- Registry ---

const char* reg_name(ggml_backend_reg_t) {
    return "D3D12";
}
size_t reg_device_count(ggml_backend_reg_t) {
    return 1;
}
ggml_backend_dev_t reg_device(ggml_backend_reg_t, size_t index) {
    GGML_ASSERT(index == 0);
    return device_ptr();
}

// Weights live in the extra buft; llama.cpp lists it after the default one.
ggml_backend_buffer_type_t* dev_extra_bufts(ggml_backend_dev_t) {
    static ggml_backend_buffer_type_t list[] = {&kWeightsBuft, nullptr};
    return list;
}

void* reg_proc_address(ggml_backend_reg_t, const char* name) {
    if (std::strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0)
        return reinterpret_cast<void*>(dev_extra_bufts);
    return nullptr;
}

const ggml_backend_reg_i kRegIface = {
    /* .get_name         = */ reg_name,
    /* .get_device_count = */ reg_device_count,
    /* .get_device       = */ reg_device,
    /* .get_proc_address = */ reg_proc_address,
};

ggml_backend_reg kReg = {
    /* .api_version = */ GGML_BACKEND_API_VERSION,
    /* .iface       = */ kRegIface,
    /* .context     = */ nullptr,
};

ggml_backend_device kDevice = {
    /* .iface   = */ kDeviceIface,
    /* .reg     = */ &kReg,
    /* .context = */ nullptr,
};

ggml_backend_dev_t device_ptr() {
    return &kDevice;
}

} // namespace

bool ggml_d3d12_register() {
    static std::once_flag once;
    static bool registered = false;
    std::call_once(once, [] {
        Gpu& g = gpu();
        if (!g.ok) {
            log_output(("[xllama] d3d12: backend unavailable: " + g.error + "\n").c_str());
            return;
        }
        kHostBuft.device = &kDevice;
        kWeightsBuft.device = &kDevice;
        ggml_backend_register(&kReg);
        registered = true;
    });
    return registered;
}

ggml_backend_buffer_type_t ggml_d3d12_weights_buft() {
    return ggml_d3d12_register() ? &kWeightsBuft : nullptr;
}

// --- Selftest ---

namespace {

constexpr int kSelftestTimedRuns = 5;

struct SelftestCase {
    ggml_type type;
    const char* name;
    int n, k, ncols;
};

// Max |gpu - cpu| / max |cpu| against the CPU backend's own path: the
// vec_dot type's from_float, then ggml's vec_dot (#312, gate A).
double reference_rel_err(ggml_type t, const std::vector<std::uint8_t>& q, std::size_t row_bytes,
                         const std::vector<float>& x, const std::vector<float>& y, int n, int k,
                         int ncols) {
    ggml_cpu_init();
    const auto* wt = ggml_get_type_traits_cpu(t);
    std::vector<std::uint8_t> xq(ggml_row_size(wt->vec_dot_type, k));
    double max_ref = 0.0, max_diff = 0.0;
    for (int c = 0; c < ncols; ++c) {
        ggml_get_type_traits_cpu(wt->vec_dot_type)
            ->from_float(x.data() + static_cast<std::size_t>(c) * k, xq.data(), k);
        for (int r = 0; r < n; ++r) {
            float ref = 0.f;
            wt->vec_dot(k, &ref, 0, q.data() + static_cast<std::size_t>(r) * row_bytes, 0,
                        xq.data(), 0, 1);
            const double got = y[static_cast<std::size_t>(c) * n + r];
            // std::max ignores a NaN operand, which would hide corrupt output.
            if (!std::isfinite(ref) || !std::isfinite(got))
                return std::numeric_limits<double>::infinity();
            max_ref = std::max(max_ref, std::fabs(static_cast<double>(ref)));
            max_diff = std::max(max_diff, std::fabs(static_cast<double>(ref) - got));
        }
    }
    return max_ref > 0.0 ? max_diff / max_ref : max_diff;
}

D3d12SelftestRow run_case(ggml_backend_t backend, const SelftestCase& sc) {
    D3d12SelftestRow row;
    row.type = sc.name;
    row.n = sc.n;
    row.k = sc.k;
    row.ncols = sc.ncols;

    ggml_init_params ip = {};
    ip.mem_size = 8 * ggml_tensor_overhead() + ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context* ctx_w = ggml_init(ip);
    ggml_context* ctx_x = ggml_init(ip);
    ggml_context* ctx_g = ggml_init(ip);
    ggml_tensor* w = ggml_new_tensor_2d(ctx_w, sc.type, sc.k, sc.n);
    ggml_tensor* x = ggml_new_tensor_2d(ctx_x, GGML_TYPE_F32, sc.k, sc.ncols);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, &kWeightsBuft);
    ggml_backend_buffer_t xbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_x, &kHostBuft);
    ggml_gallocr_t galloc = nullptr;

    auto cleanup = [&] {
        if (galloc)
            ggml_gallocr_free(galloc);
        if (wbuf)
            ggml_backend_buffer_free(wbuf);
        if (xbuf)
            ggml_backend_buffer_free(xbuf);
        ggml_free(ctx_g);
        ggml_free(ctx_x);
        ggml_free(ctx_w);
    };
    if (!wbuf || !xbuf) {
        row.error = "buffer allocation failed";
        cleanup();
        return row;
    }

    std::mt19937 rng(1234u + static_cast<unsigned>(sc.n + sc.k + sc.ncols));
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    std::vector<float> wf(static_cast<std::size_t>(sc.n) * sc.k);
    for (float& v : wf)
        v = uni(rng);
    std::vector<std::uint8_t> q(ggml_nbytes(w));
    ggml_quantize_chunk(sc.type, wf.data(), q.data(), 0, sc.n, sc.k, nullptr);
    ggml_backend_tensor_set(w, q.data(), 0, q.size());
    std::vector<float> xf(static_cast<std::size_t>(sc.k) * sc.ncols);
    for (float& v : xf)
        v = uni(rng);
    ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * sizeof(float));

    ggml_tensor* y = ggml_mul_mat(ctx_g, w, x);
    ggml_cgraph* gf = ggml_new_graph(ctx_g);
    ggml_build_forward_expand(gf, y);
    galloc = ggml_gallocr_new(&kHostBuft);
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        row.error = "graph allocation failed";
        cleanup();
        return row;
    }
    // Warm-up, then the median of kSelftestTimedRuns: one sample of a ~9 MB
    // matmul swung from 78 to 109 GB/s across back-to-back selftests (CI
    // 1.6.0.1154, same blob).
    ggml_status st = ggml_backend_graph_compute(backend, gf);
    std::vector<double> ms;
    for (int i = 0; i < kSelftestTimedRuns && st == GGML_STATUS_SUCCESS; ++i) {
        st = ggml_backend_graph_compute(backend, gf);
        ms.push_back(gpu().last_gpu_ms);
    }
    row.d3d12_ran = st == GGML_STATUS_SUCCESS;
    if (!row.d3d12_ran) {
        row.error = "graph_compute failed";
        cleanup();
        return row;
    }
    std::nth_element(ms.begin(), ms.begin() + ms.size() / 2, ms.end());
    row.gpu_ms = ms[ms.size() / 2];
    row.packed_gbs = row.gpu_ms > 0.0 ? static_cast<double>(q.size()) / 1e6 / row.gpu_ms : 0.0;
    std::vector<float> yf(static_cast<std::size_t>(sc.n) * sc.ncols);
    ggml_backend_tensor_get(y, yf.data(), 0, yf.size() * sizeof(float));
    row.rel_err = reference_rel_err(sc.type, q, w->nb[1], xf, yf, sc.n, sc.k, sc.ncols);
    row.ok = row.rel_err <= kD3d12SelftestRelTol;
    if (!row.ok)
        row.error = "mismatch vs the CPU backend (q8 vec_dot)";
    cleanup();
    return row;
}

// GET_ROWS from a Q6_K weight (the tied token_embd path, #309): the GPU must
// equal ggml's dequantizer bit for bit, as the host emulation does.
D3d12SelftestRow run_get_rows_case(ggml_backend_t backend, int n, int k) {
    D3d12SelftestRow row;
    row.type = "get_rows_q6_k";
    row.n = n;
    row.k = k;
    const std::int32_t ids_v[] = {0, 1, n / 2 + 1, n - 1, 7, n / 3};
    const int n_ids = static_cast<int>(sizeof(ids_v) / sizeof(ids_v[0]));
    row.ncols = n_ids;

    ggml_init_params ip = {};
    ip.mem_size = 8 * ggml_tensor_overhead() + ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context* ctx_w = ggml_init(ip);
    ggml_context* ctx_i = ggml_init(ip);
    ggml_context* ctx_g = ggml_init(ip);
    ggml_tensor* w = ggml_new_tensor_2d(ctx_w, GGML_TYPE_Q6_K, k, n);
    ggml_tensor* ids = ggml_new_tensor_1d(ctx_i, GGML_TYPE_I32, n_ids);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, &kWeightsBuft);
    ggml_backend_buffer_t ibuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_i, &kHostBuft);
    ggml_gallocr_t galloc = nullptr;
    auto cleanup = [&] {
        if (galloc)
            ggml_gallocr_free(galloc);
        if (wbuf)
            ggml_backend_buffer_free(wbuf);
        if (ibuf)
            ggml_backend_buffer_free(ibuf);
        ggml_free(ctx_g);
        ggml_free(ctx_i);
        ggml_free(ctx_w);
    };
    if (!wbuf || !ibuf) {
        row.error = "buffer allocation failed";
        cleanup();
        return row;
    }
    std::mt19937 rng(99u);
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    std::vector<float> wf(static_cast<std::size_t>(n) * k);
    for (float& v : wf)
        v = uni(rng);
    std::vector<std::uint8_t> q(ggml_nbytes(w));
    ggml_quantize_chunk(GGML_TYPE_Q6_K, wf.data(), q.data(), 0, n, k, nullptr);
    ggml_backend_tensor_set(w, q.data(), 0, q.size());
    ggml_backend_tensor_set(ids, ids_v, 0, sizeof(ids_v));

    ggml_tensor* y = ggml_get_rows(ctx_g, w, ids);
    ggml_cgraph* gf = ggml_new_graph(ctx_g);
    ggml_build_forward_expand(gf, y);
    galloc = ggml_gallocr_new(&kHostBuft);
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        row.error = "graph allocation failed";
        cleanup();
        return row;
    }
    row.d3d12_ran = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    if (!row.d3d12_ran) {
        row.error = "graph_compute failed";
        cleanup();
        return row;
    }
    row.gpu_ms = gpu().last_gpu_ms;
    std::vector<float> got(static_cast<std::size_t>(k) * n_ids), ref(static_cast<std::size_t>(k));
    ggml_backend_tensor_get(y, got.data(), 0, got.size() * sizeof(float));
    double max_ref = 0.0, max_diff = 0.0;
    for (int i = 0; i < n_ids; ++i) {
        ggml_get_type_traits(GGML_TYPE_Q6_K)
            ->to_float(q.data() + static_cast<std::size_t>(ids_v[i]) * w->nb[1], ref.data(), k);
        for (int j = 0; j < k; ++j) {
            const double actual = got[static_cast<std::size_t>(i) * k + j];
            if (!std::isfinite(ref[j]) || !std::isfinite(actual)) {
                row.rel_err = std::numeric_limits<double>::infinity();
                row.error = "non-finite get_rows reference or output";
                cleanup();
                return row;
            }
            max_ref = std::max(max_ref, std::fabs(static_cast<double>(ref[j])));
            max_diff = std::max(max_diff, std::fabs(static_cast<double>(ref[j]) - actual));
        }
    }
    row.rel_err = max_ref > 0.0 ? max_diff / max_ref : max_diff;
    row.ok = max_diff == 0.0; // bit-exact, like the host emulation
    if (!row.ok)
        row.error = "mismatch vs ggml dequantize_row_q6_K";
    cleanup();
    return row;
}

// "cold_scratch=1" means no op has allocated the staging buffer yet: the op
// about to run is the first one to reach it, and bench-d3d12-selftest.sh gates on it.
void log_cold_scratch(const char* what) {
    Gpu& g = gpu();
    std::lock_guard<std::mutex> lock(g.mu);
    log_output(std::string("[xllama] d3d12be: ") + what + " cold_scratch=" + (g.xq ? "0" : "1") +
               "\n");
}

// Run before the staged cases so a direct ADD cannot rely on an existing scratch buffer.
D3d12SelftestRow run_bias_add_case(ggml_backend_t backend) {
    constexpr int width = 129;
    constexpr int columns = 2;
    D3d12SelftestRow row;
    row.type = "add_f32";
    row.n = width;
    row.ncols = columns;
    const ggml_init_params params = {8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr,
                                     true};
    const std::unique_ptr<ggml_context, decltype(&ggml_free)> weights(ggml_init(params), ggml_free);
    const std::unique_ptr<ggml_context, decltype(&ggml_free)> activations(ggml_init(params),
                                                                          ggml_free);
    if (!weights || !activations) {
        row.error = "context allocation failed";
        return row;
    }
    ggml_tensor* bias = ggml_new_tensor_1d(weights.get(), GGML_TYPE_F32, width);
    ggml_tensor* input = ggml_new_tensor_2d(activations.get(), GGML_TYPE_F32, width, columns);
    ggml_tensor* output = ggml_add(activations.get(), input, bias);
    const std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> bias_buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(weights.get(), &kWeightsBuft),
        ggml_backend_buffer_free);
    const std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> host_buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(activations.get(), &kHostBuft),
        ggml_backend_buffer_free);
    if (!bias_buffer || !host_buffer) {
        row.error = "buffer allocation failed";
        return row;
    }

    float bias_values[width], input_values[width * columns], expected[width * columns];
    for (int i = 0; i < width; ++i)
        bias_values[i] = static_cast<float>(i - 64) * 0.25f;
    for (int i = 0; i < width * columns; ++i) {
        input_values[i] = static_cast<float>(i - width) * 0.125f;
        expected[i] = input_values[i] + bias_values[i % width];
    }
    ggml_backend_tensor_set(bias, bias_values, 0, sizeof(bias_values));
    ggml_backend_tensor_set(input, input_values, 0, sizeof(input_values));
    ggml_cgraph* graph = ggml_new_graph(activations.get());
    ggml_build_forward_expand(graph, output);
    log_cold_scratch("bias ADD");
    row.d3d12_ran = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    if (!row.d3d12_ran) {
        row.error = "graph_compute failed";
        return row;
    }

    row.gpu_ms = gpu().last_gpu_ms;
    float actual[width * columns];
    ggml_backend_tensor_get(output, actual, 0, sizeof(actual));
    row.ok = std::memcmp(actual, expected, sizeof(expected)) == 0;
    if (!row.ok) {
        double max_diff = 0.0;
        double max_ref = 0.0;
        for (int i = 0; i < width * columns; ++i) {
            if (!std::isfinite(actual[i])) {
                max_diff = std::numeric_limits<double>::infinity();
                break;
            }
            max_diff = std::max(max_diff, std::fabs(static_cast<double>(actual[i]) - expected[i]));
            max_ref = std::max(max_ref, std::fabs(static_cast<double>(expected[i])));
        }
        row.rel_err = max_ref > 0.0 ? max_diff / max_ref : max_diff;
        row.error = "mismatch vs F32 reference";
    }
    return row;
}

// ROPE on the GPU against ggml-cpu's ggml_rope_ext (G2: NMSE <= 1e-7, the
// test-backend-ops bound). `staged` puts the activations in a plain CPU buffer so
// the backend copies them into xq; otherwise it reads the D3D12_Host buffer in
// place. n = head_dim, k = heads, ncols = tokens. Max |diff| and bit identity go
// in the note (the error column) whatever the verdict.
D3d12SelftestRow run_rope_case(ggml_backend_t backend, int mode, bool staged, int head_dim,
                               int heads, int tokens, int n_dims) {
    D3d12SelftestRow row;
    row.type = std::string(mode == kD3d12RopeModeNeox ? "rope_neox" : "rope_norm") +
               (staged ? "_staged" : "");
    row.n = head_dim;
    row.k = heads;
    row.ncols = tokens;
    const float freq_base = 1000000.0f, freq_scale = 1.0f, attn_factor = 1.0f;
    const std::size_t count = static_cast<std::size_t>(head_dim) * heads * tokens;

    ggml_init_params ip = {};
    ip.mem_size = 16 * ggml_tensor_overhead() + 2 * ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context* ctx_x = ggml_init(ip);
    ggml_context* ctx_p = ggml_init(ip);
    ggml_context* ctx_g = ggml_init(ip);
    if (!ctx_x || !ctx_p || !ctx_g) {
        row.error = "context allocation failed";
        ggml_free(ctx_g);
        ggml_free(ctx_p);
        ggml_free(ctx_x);
        return row;
    }
    ggml_tensor* x = ggml_new_tensor_3d(ctx_x, GGML_TYPE_F32, head_dim, heads, tokens);
    ggml_tensor* pos = ggml_new_tensor_1d(ctx_p, GGML_TYPE_I32, tokens);
    ggml_backend_buffer_t xbuf = ggml_backend_alloc_ctx_tensors_from_buft(
        ctx_x, staged ? ggml_backend_cpu_buffer_type() : &kHostBuft);
    ggml_backend_buffer_t pbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_p, &kHostBuft);
    ggml_gallocr_t galloc = nullptr;
    auto cleanup = [&] {
        if (galloc)
            ggml_gallocr_free(galloc);
        if (xbuf)
            ggml_backend_buffer_free(xbuf);
        if (pbuf)
            ggml_backend_buffer_free(pbuf);
        ggml_free(ctx_g);
        ggml_free(ctx_p);
        ggml_free(ctx_x);
    };
    if (!xbuf || !pbuf) {
        row.error = "buffer allocation failed";
        cleanup();
        return row;
    }
    std::mt19937 rng(4242u + static_cast<unsigned>(mode * 17 + tokens));
    std::uniform_real_distribution<float> uni(-2.f, 2.f);
    std::vector<float> xf(count);
    for (float& v : xf)
        v = uni(rng);
    std::vector<std::int32_t> pv(static_cast<std::size_t>(tokens));
    for (int i = 0; i < tokens; ++i)
        pv[static_cast<std::size_t>(i)] = 311 + 5 * i; // decode-like positions
    ggml_backend_tensor_set(x, xf.data(), 0, count * sizeof(float));
    ggml_backend_tensor_set(pos, pv.data(), 0, pv.size() * sizeof(std::int32_t));

    ggml_tensor* y = ggml_rope_ext(ctx_g, x, pos, nullptr, n_dims, mode, 0, freq_base, freq_scale,
                                   0.0f, attn_factor, 32.0f, 1.0f);
    ggml_cgraph* gf = ggml_new_graph(ctx_g);
    ggml_build_forward_expand(gf, y);
    galloc = ggml_gallocr_new(&kHostBuft);
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        row.error = "graph allocation failed";
        cleanup();
        return row;
    }
    row.d3d12_ran = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    if (!row.d3d12_ran) {
        row.error = "graph_compute failed";
        cleanup();
        return row;
    }
    row.gpu_ms = gpu().last_gpu_ms;
    std::vector<float> got(count);
    ggml_backend_tensor_get(y, got.data(), 0, count * sizeof(float));

    // Reference: the same ROPE on ggml-cpu.
    ggml_init_params rp = {};
    rp.mem_size = count * sizeof(float) * 2 + (4u << 20);
    ggml_context* rc = ggml_init(rp);
    if (!rc) {
        row.error = "reference context allocation failed";
        cleanup();
        return row;
    }
    ggml_tensor* rx = ggml_new_tensor_3d(rc, GGML_TYPE_F32, head_dim, heads, tokens);
    ggml_tensor* rpos = ggml_new_tensor_1d(rc, GGML_TYPE_I32, tokens);
    std::memcpy(rx->data, xf.data(), count * sizeof(float));
    std::memcpy(rpos->data, pv.data(), pv.size() * sizeof(std::int32_t));
    ggml_tensor* ry = ggml_rope_ext(rc, rx, rpos, nullptr, n_dims, mode, 0, freq_base, freq_scale,
                                    0.0f, attn_factor, 32.0f, 1.0f);
    ggml_cgraph* rg = ggml_new_graph(rc);
    ggml_build_forward_expand(rg, ry);
    const bool ref_ok = ggml_graph_compute_with_ctx(rc, rg, 1) == GGML_STATUS_SUCCESS;
    double num = 0.0, den = 0.0, max_abs = 0.0;
    bool finite = ref_ok, identical = ref_ok;
    if (ref_ok) {
        const auto* ref = static_cast<const float*>(ry->data);
        for (std::size_t i = 0; i < count; ++i) {
            if (!std::isfinite(ref[i]) || !std::isfinite(got[i]))
                finite = false;
            const double e = static_cast<double>(got[i]) - ref[i];
            num += e * e;
            den += static_cast<double>(ref[i]) * ref[i];
            max_abs = std::max(max_abs, std::fabs(e));
            identical = identical && std::memcmp(&got[i], &ref[i], sizeof(float)) == 0;
        }
    }
    ggml_free(rc);
    const double nmse = den > 0.0 ? num / den : num;
    row.rel_err = nmse; // NMSE for the rope rows
    row.ok = ref_ok && finite && nmse <= 1e-7;
    char note[96];
    std::snprintf(note, sizeof(note), "nmse=%.3g max_abs=%.3g bit_identical=%d", nmse, max_abs,
                  identical ? 1 : 0);
    row.error = !ref_ok ? "cpu reference failed" : !finite ? "non-finite rope output" : note;
    cleanup();
    return row;
}

// GPU quantize (d3d12-gpu-quant.txt): y2 = W2 · (W1 · x), so the second
// matmul's input is produced inside the submission. With the knob the backend
// quantizes it on the GPU (one submission); without it, it ends the submission
// and the CPU quantizes (two). The kernel must equal ggml's from_float byte for
// byte, so the two runs must give bit-identical y2. n = W1 rows = W2 width;
// W2 has 1024 rows.
D3d12SelftestRow run_quant_chain_case(ggml_backend_t backend, ggml_type wt, int n, int k,
                                      int ncols) {
    D3d12SelftestRow row;
    row.type = std::string("quant_") + ggml_type_name(d3d12_activation_type(wt));
    row.n = n;
    row.k = k;
    row.ncols = ncols;
    ggml_init_params ip = {};
    ip.mem_size = 8 * ggml_tensor_overhead() + ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context* ctx_w = ggml_init(ip);
    ggml_context* ctx_x = ggml_init(ip);
    ggml_context* ctx_g = ggml_init(ip);
    ggml_backend_buffer_t wbuf = nullptr, xbuf = nullptr;
    ggml_gallocr_t galloc = nullptr;
    auto cleanup = [&] {
        if (galloc)
            ggml_gallocr_free(galloc);
        if (wbuf)
            ggml_backend_buffer_free(wbuf);
        if (xbuf)
            ggml_backend_buffer_free(xbuf);
        ggml_free(ctx_g);
        ggml_free(ctx_x);
        ggml_free(ctx_w);
    };
    if (!ctx_w || !ctx_x || !ctx_g) {
        row.error = "context allocation failed";
        cleanup();
        return row;
    }
    ggml_tensor* w1 = ggml_new_tensor_2d(ctx_w, wt, k, n);
    constexpr int m = 1024;
    ggml_tensor* w2 = ggml_new_tensor_2d(ctx_w, wt, n, m);
    ggml_tensor* x = ggml_new_tensor_2d(ctx_x, GGML_TYPE_F32, k, ncols);
    wbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, &kWeightsBuft);
    xbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_x, &kHostBuft);
    if (!wbuf || !xbuf) {
        row.error = "buffer allocation failed";
        cleanup();
        return row;
    }
    std::mt19937 rng(777u + static_cast<unsigned>(n + k + ncols));
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    for (ggml_tensor* w : {w1, w2}) {
        std::vector<float> wf(static_cast<std::size_t>(ggml_nelements(w)));
        for (float& v : wf)
            v = uni(rng);
        std::vector<std::uint8_t> q(ggml_nbytes(w));
        ggml_quantize_chunk(wt, wf.data(), q.data(), 0, w->ne[1], w->ne[0], nullptr);
        ggml_backend_tensor_set(w, q.data(), 0, q.size());
    }
    std::vector<float> xf(static_cast<std::size_t>(k) * ncols);
    for (float& v : xf)
        v = uni(rng);
    ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * sizeof(float));
    ggml_tensor* y2 = ggml_mul_mat(ctx_g, w2, ggml_mul_mat(ctx_g, w1, x));
    ggml_cgraph* gf = ggml_new_graph(ctx_g);
    ggml_build_forward_expand(gf, y2);
    galloc = ggml_gallocr_new(&kHostBuft);
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        row.error = "graph allocation failed";
        cleanup();
        return row;
    }
    const std::size_t count = static_cast<std::size_t>(m) * ncols;
    const auto run = [&](bool gpu_q, std::vector<float>* y, std::uint64_t* subs) {
        Gpu& g = gpu();
        {
            std::lock_guard<std::mutex> lock(g.mu);
            g.gpu_quant = gpu_q;
        }
        ggml_d3d12_take_stats(nullptr);
        const bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
        D3d12PhaseStats st;
        ggml_d3d12_take_stats(&st);
        *subs = st.submissions;
        y->resize(count);
        if (ok)
            ggml_backend_tensor_get(y2, y->data(), 0, count * sizeof(float));
        return ok;
    };
    std::vector<float> on, off;
    std::uint64_t subs_on = 0, subs_off = 0;
    const bool ran_on = run(true, &on, &subs_on);
    row.gpu_ms = gpu().last_gpu_ms;
    const bool ran_off = run(false, &off, &subs_off);
    {
        Gpu& g = gpu();
        std::lock_guard<std::mutex> lock(g.mu);
        g.gpu_quant = true; // only called with the knob on
    }
    row.d3d12_ran = ran_on && ran_off;
    if (!row.d3d12_ran) {
        row.error = "graph_compute failed";
        cleanup();
        return row;
    }
    double max_diff = 0.0, max_ref = 0.0;
    bool finite = true;
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(on[i]) || !std::isfinite(off[i]))
            finite = false;
        max_diff = std::max(max_diff, std::fabs(static_cast<double>(on[i]) - off[i]));
        max_ref = std::max(max_ref, std::fabs(static_cast<double>(off[i])));
    }
    const bool identical = std::memcmp(on.data(), off.data(), count * sizeof(float)) == 0;
    row.rel_err = max_ref > 0.0 ? max_diff / max_ref : max_diff;
    // One submission with the GPU quantize, two without (the chain's split).
    row.ok = finite && identical && subs_on == 1 && subs_off == 2;
    char note[96];
    std::snprintf(note, sizeof(note), "bit_identical=%d submissions_on=%llu submissions_off=%llu",
                  identical ? 1 : 0, static_cast<unsigned long long>(subs_on),
                  static_cast<unsigned long long>(subs_off));
    row.error = !finite ? "non-finite output" : note;
    cleanup();
    return row;
}

// SWIGLU (d3d12-swiglu.txt) against ggml-cpu's own SWIGLU on the same inputs:
// the kernel reproduces the AVX2 + FMA vector path, so the output must be bit
// identical. `staged` puts gate and up in a plain CPU buffer (copied into xq).
// n = elements per column, ncols = columns. Inputs span the exp slow path.
D3d12SelftestRow run_swiglu_case(ggml_backend_t backend, bool staged, int n, int ncols) {
    D3d12SelftestRow row;
    row.type = staged ? "swiglu_staged" : "swiglu";
    row.n = n;
    row.ncols = ncols;
    const std::size_t count = static_cast<std::size_t>(n) * ncols;
    ggml_init_params ip = {};
    ip.mem_size = 8 * ggml_tensor_overhead() + ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context* ctx_x = ggml_init(ip);
    ggml_context* ctx_g = ggml_init(ip);
    ggml_backend_buffer_t xbuf = nullptr;
    ggml_gallocr_t galloc = nullptr;
    auto cleanup = [&] {
        if (galloc)
            ggml_gallocr_free(galloc);
        if (xbuf)
            ggml_backend_buffer_free(xbuf);
        ggml_free(ctx_g);
        ggml_free(ctx_x);
    };
    if (!ctx_x || !ctx_g) {
        row.error = "context allocation failed";
        cleanup();
        return row;
    }
    ggml_tensor* ga = ggml_new_tensor_2d(ctx_x, GGML_TYPE_F32, n, ncols);
    ggml_tensor* up = ggml_new_tensor_2d(ctx_x, GGML_TYPE_F32, n, ncols);
    xbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_x, staged ? ggml_backend_cpu_buffer_type()
                                                                  : &kHostBuft);
    if (!xbuf) {
        row.error = "buffer allocation failed";
        cleanup();
        return row;
    }
    std::mt19937 rng(99u + static_cast<unsigned>(n + ncols));
    std::normal_distribution<float> nd(0.0f, 3.0f);
    std::uniform_real_distribution<float> wide(-250.0f, 250.0f);
    std::vector<float> gv(count), uv(count);
    for (std::size_t i = 0; i < count; ++i) {
        gv[i] = i % 5 == 0 ? wide(rng) : nd(rng);
        uv[i] = nd(rng);
    }
    ggml_backend_tensor_set(ga, gv.data(), 0, count * sizeof(float));
    ggml_backend_tensor_set(up, uv.data(), 0, count * sizeof(float));
    ggml_tensor* y = ggml_swiglu_split(ctx_g, ga, up);
    ggml_cgraph* gf = ggml_new_graph(ctx_g);
    ggml_build_forward_expand(gf, y);
    galloc = ggml_gallocr_new(&kHostBuft);
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        row.error = "graph allocation failed";
        cleanup();
        return row;
    }
    row.d3d12_ran = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    if (!row.d3d12_ran) {
        row.error = "graph_compute failed";
        cleanup();
        return row;
    }
    row.gpu_ms = gpu().last_gpu_ms;
    std::vector<float> got(count);
    ggml_backend_tensor_get(y, got.data(), 0, count * sizeof(float));

    ggml_init_params rp = {};
    rp.mem_size = count * sizeof(float) * 3 + (4u << 20);
    ggml_context* rc = ggml_init(rp);
    if (!rc) {
        row.error = "reference context allocation failed";
        cleanup();
        return row;
    }
    ggml_tensor* ra = ggml_new_tensor_2d(rc, GGML_TYPE_F32, n, ncols);
    ggml_tensor* rb = ggml_new_tensor_2d(rc, GGML_TYPE_F32, n, ncols);
    std::memcpy(ra->data, gv.data(), count * sizeof(float));
    std::memcpy(rb->data, uv.data(), count * sizeof(float));
    ggml_tensor* ry = ggml_swiglu_split(rc, ra, rb);
    ggml_cgraph* rg = ggml_new_graph(rc);
    ggml_build_forward_expand(rg, ry);
    const bool ref_ok = ggml_graph_compute_with_ctx(rc, rg, 1) == GGML_STATUS_SUCCESS;
    std::size_t diff = 0;
    double max_diff = 0.0, max_ref = 0.0;
    if (ref_ok) {
        const auto* ref = static_cast<const float*>(ry->data);
        for (std::size_t i = 0; i < count; ++i) {
            if (std::memcmp(&got[i], &ref[i], sizeof(float)) != 0)
                ++diff;
            max_diff = std::max(max_diff, std::fabs(static_cast<double>(got[i]) - ref[i]));
            max_ref = std::max(max_ref, std::fabs(static_cast<double>(ref[i])));
        }
    }
    ggml_free(rc);
    row.rel_err = max_ref > 0.0 ? max_diff / max_ref : max_diff;
    row.ok = ref_ok && diff == 0;
    char note[64];
    std::snprintf(note, sizeof(note), "mismatched=%zu", diff);
    row.error = !ref_ok ? "cpu reference failed" : note;
    cleanup();
    return row;
}

// The FFN as llama.cpp builds it: gate and up projections, SWIGLU, down. With
// SWIGLU and the GPU quantize on, it is one submission; the knob-off run of the
// quantize (down's input quantized on the CPU, two submissions) must give the
// same bits.
D3d12SelftestRow run_ffn_chain_case(ggml_backend_t backend, ggml_type wt, int embd, int ff,
                                    int ncols) {
    D3d12SelftestRow row;
    row.type = std::string("ffn_") + ggml_type_name(wt);
    row.n = embd;
    row.k = ff;
    row.ncols = ncols;
    ggml_init_params ip = {};
    ip.mem_size = 16 * ggml_tensor_overhead() + ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context* ctx_w = ggml_init(ip);
    ggml_context* ctx_x = ggml_init(ip);
    ggml_context* ctx_g = ggml_init(ip);
    ggml_backend_buffer_t wbuf = nullptr, xbuf = nullptr;
    ggml_gallocr_t galloc = nullptr;
    auto cleanup = [&] {
        if (galloc)
            ggml_gallocr_free(galloc);
        if (wbuf)
            ggml_backend_buffer_free(wbuf);
        if (xbuf)
            ggml_backend_buffer_free(xbuf);
        ggml_free(ctx_g);
        ggml_free(ctx_x);
        ggml_free(ctx_w);
    };
    if (!ctx_w || !ctx_x || !ctx_g) {
        row.error = "context allocation failed";
        cleanup();
        return row;
    }
    ggml_tensor* wg = ggml_new_tensor_2d(ctx_w, wt, embd, ff);
    ggml_tensor* wu = ggml_new_tensor_2d(ctx_w, wt, embd, ff);
    ggml_tensor* wd = ggml_new_tensor_2d(ctx_w, wt, ff, embd);
    ggml_tensor* x = ggml_new_tensor_2d(ctx_x, GGML_TYPE_F32, embd, ncols);
    wbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, &kWeightsBuft);
    xbuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_x, &kHostBuft);
    if (!wbuf || !xbuf) {
        row.error = "buffer allocation failed";
        cleanup();
        return row;
    }
    std::mt19937 rng(555u + static_cast<unsigned>(embd + ff + ncols));
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    for (ggml_tensor* w : {wg, wu, wd}) {
        std::vector<float> wf(static_cast<std::size_t>(ggml_nelements(w)));
        for (float& v : wf)
            v = uni(rng) * 0.1f;
        std::vector<std::uint8_t> q(ggml_nbytes(w));
        ggml_quantize_chunk(wt, wf.data(), q.data(), 0, w->ne[1], w->ne[0], nullptr);
        ggml_backend_tensor_set(w, q.data(), 0, q.size());
    }
    std::vector<float> xf(static_cast<std::size_t>(embd) * ncols);
    for (float& v : xf)
        v = uni(rng);
    ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * sizeof(float));
    ggml_tensor* y = ggml_mul_mat(
        ctx_g, wd,
        ggml_swiglu_split(ctx_g, ggml_mul_mat(ctx_g, wg, x), ggml_mul_mat(ctx_g, wu, x)));
    ggml_cgraph* gf = ggml_new_graph(ctx_g);
    ggml_build_forward_expand(gf, y);
    galloc = ggml_gallocr_new(&kHostBuft);
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        row.error = "graph allocation failed";
        cleanup();
        return row;
    }
    const std::size_t count = static_cast<std::size_t>(embd) * ncols;
    const auto run = [&](bool gpu_q, std::vector<float>* out_y, std::uint64_t* subs) {
        Gpu& g = gpu();
        {
            std::lock_guard<std::mutex> lock(g.mu);
            g.gpu_quant = gpu_q;
        }
        ggml_d3d12_take_stats(nullptr);
        const bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
        D3d12PhaseStats st;
        ggml_d3d12_take_stats(&st);
        *subs = st.submissions;
        out_y->resize(count);
        if (ok)
            ggml_backend_tensor_get(y, out_y->data(), 0, count * sizeof(float));
        return ok;
    };
    std::vector<float> on, off;
    std::uint64_t subs_on = 0, subs_off = 0;
    const bool ran_on = run(true, &on, &subs_on);
    row.gpu_ms = gpu().last_gpu_ms;
    const bool ran_off = run(false, &off, &subs_off);
    {
        Gpu& g = gpu();
        std::lock_guard<std::mutex> lock(g.mu);
        g.gpu_quant = true; // only called with the knob on
    }
    row.d3d12_ran = ran_on && ran_off;
    if (!row.d3d12_ran) {
        row.error = "graph_compute failed";
        cleanup();
        return row;
    }
    bool finite = true;
    for (std::size_t i = 0; i < count; ++i)
        finite = finite && std::isfinite(on[i]) && std::isfinite(off[i]);
    const bool identical = std::memcmp(on.data(), off.data(), count * sizeof(float)) == 0;
    row.ok = finite && identical && subs_on == 1 && subs_off == 2;
    char note[96];
    std::snprintf(note, sizeof(note), "bit_identical=%d submissions_on=%llu submissions_off=%llu",
                  identical ? 1 : 0, static_cast<unsigned long long>(subs_on),
                  static_cast<unsigned long long>(subs_off));
    row.error = !finite ? "non-finite output" : note;
    cleanup();
    return row;
}

} // namespace

void run_d3d12_selftest(std::vector<D3d12SelftestRow>* out) {
    if (!out)
        return;
    if (!ggml_d3d12_register()) {
        D3d12SelftestRow r;
        r.type = "-";
        r.error = gpu().error.empty() ? "d3d12 backend unavailable" : gpu().error;
        out->push_back(std::move(r));
        return;
    }
    ggml_backend_t backend = dev_init_backend(&kDevice, nullptr);
    if (gpu().bias_add)
        out->push_back(run_bias_add_case(backend));
    // The first ROPE of the process, before any matmul has allocated the staging
    // buffer: its cos/sin table is the first thing staged there.
    if (gpu().rope) {
        log_cold_scratch("rope");
        out->push_back(run_rope_case(backend, kD3d12RopeModeNeox, false, 128, 16, 1, 128));
    }
    // Shapes of the D2 target models; q6_k 2048x11008 has 2-byte-aligned rows.
    const SelftestCase cases[] = {
        {GGML_TYPE_Q4_0, "q4_0", 8192, 2048, 1},   {GGML_TYPE_Q4_0, "q4_0", 2048, 8192, 1},
        {GGML_TYPE_Q4_K, "q4_k", 11008, 2048, 1},  {GGML_TYPE_Q4_K, "q4_k", 2048, 11008, 1},
        {GGML_TYPE_Q6_K, "q6_k", 2048, 11008, 1},  {GGML_TYPE_Q6_K, "q6_k", 65536, 1024, 1},
        {GGML_TYPE_Q4_0, "q4_0", 1024, 1024, 7},   {GGML_TYPE_Q4_K, "q4_k", 1024, 1024, 7},
        {GGML_TYPE_Q6_K, "q6_k", 1024, 1024, 7},   {GGML_TYPE_Q4_0, "q4_0", 1024, 1024, 512},
        {GGML_TYPE_Q4_K, "q4_k", 1024, 1024, 512}, {GGML_TYPE_Q6_K, "q6_k", 1024, 1024, 512},
    };
    for (const auto& sc : cases)
        out->push_back(run_case(backend, sc));
    out->push_back(run_get_rows_case(backend, 151936, 2048)); // Coder-3B tied token_embd
    out->push_back(run_get_rows_case(backend, 65536, 1024));  // LFM2.5-350M
    // ROPE (d3d12-rope.txt): head_dim 128, q (16 heads) and GQA k (2), decode and
    // a 512-token ubatch; NEOX (Qwen2.5) and NORMAL; in-place and staged input.
    for (const int mode : {kD3d12RopeModeNeox, kD3d12RopeModeNormal}) {
        if (!gpu().rope)
            break;
        if (mode != kD3d12RopeModeNeox) // the NEOX decode q row ran first, above
            out->push_back(run_rope_case(backend, mode, false, 128, 16, 1, 128));
        out->push_back(run_rope_case(backend, mode, false, 128, 2, 1, 128));
        out->push_back(run_rope_case(backend, mode, false, 128, 2, 512, 128));
        out->push_back(run_rope_case(backend, mode, true, 128, 16, 7, 128));
        out->push_back(run_rope_case(backend, mode, false, 128, 4, 3, 64)); // n_dims < head_dim
    }
    // GPU quantize (d3d12-gpu-quant.txt): q8_0 (Q4_0) and q8_K (Q4_K, Q6_K
    // share it), decode and a short ubatch, and an 11008-wide input (Coder-3B
    // ffn_down: 172 q8_0 pairs, 43 q8_K blocks).
    if (gpu().gpu_quant) {
        out->push_back(run_quant_chain_case(backend, GGML_TYPE_Q4_0, 2048, 2048, 1));
        out->push_back(run_quant_chain_case(backend, GGML_TYPE_Q4_K, 2048, 2048, 1));
        out->push_back(run_quant_chain_case(backend, GGML_TYPE_Q4_0, 2048, 2048, 7));
        out->push_back(run_quant_chain_case(backend, GGML_TYPE_Q6_K, 2048, 2048, 7));
        out->push_back(run_quant_chain_case(backend, GGML_TYPE_Q4_0, 11008, 2048, 1));
        out->push_back(run_quant_chain_case(backend, GGML_TYPE_Q4_K, 11008, 2048, 1));
    }
    // SWIGLU (d3d12-swiglu.txt): Coder-3B's FFN width, decode and a ubatch, in
    // place and staged; then whole FFN chains when the GPU quantize is on too.
    if (gpu().swiglu) {
        out->push_back(run_swiglu_case(backend, false, 11008, 1));
        out->push_back(run_swiglu_case(backend, false, 11008, 7));
        out->push_back(run_swiglu_case(backend, true, 8192, 3));
        if (gpu().gpu_quant) {
            out->push_back(run_ffn_chain_case(backend, GGML_TYPE_Q4_0, 2048, 8192, 1));
            out->push_back(run_ffn_chain_case(backend, GGML_TYPE_Q4_K, 2048, 11008, 1));
            out->push_back(run_ffn_chain_case(backend, GGML_TYPE_Q6_K, 2048, 8192, 7));
        }
    }
    ggml_backend_free(backend);
}

} // namespace xllama

    #endif // _WIN32

#endif // XLLAMA_USE_LLAMA
