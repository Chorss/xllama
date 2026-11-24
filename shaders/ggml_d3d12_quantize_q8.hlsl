// ggml backend d3d12: quantize F32 activations into the vec_dot type of the
// matmul that reads them, on the GPU (knob d3d12-gpu-quant.txt). Built twice:
//   Q8_K undefined: block_q8_0 (fp16 d, int8 qs[32]), as ggml-cpu's x86 AVX2
//                   quantize_row_q8_0;
//   -D Q8_K=1:      block_q8_K (float d, int8 qs[256], int16 bsums[16]), as
//                   quantize_row_q8_K_ref.
// The output must equal the CPU's byte for byte: the matmul kernels then sum
// the same integers as before. Every step mirrors d3d12_quantize_q8_0_emulate /
// d3d12_quantize_q8_K_emulate (src/bridge/ggml_d3d12.cpp), which the host tests
// compare against ggml:
//   - fp32 mul and add are correctly rounded (D3D11.3 spec §3.1.3.1); the blob
//     is built with -denorm preserve and the products are `precise`, so no
//     flush to zero and no fused multiply-add;
//   - fp32 division is not correctly rounded on D3D12, so div_rn
//     (ggml_d3d12_fp32_rn.hlsli) divides the significands on integers;
//   - HLSL round() rounds half to even, like _mm256_round_ps(_MM_ROUND_NEAREST).
// q8_0: one thread per pair of blocks (68 B, so every store is a whole dword).
// q8_K: one group of 32 threads per block (292 B). SV_GroupID.y is the
// activation column.
// Root signature (src/bridge/ggml_d3d12.cpp): b0 constants, u0 the quantized
// output (the staging buffer), u1 the F32 input.

#define NUM_THREADS 64 // kD3d12QuantThreads (q8_0)

cbuffer Params : register(b0) {
    uint nunits;    // work units per column: k / 64 (q8_0 threads) or k / 256 (q8_K groups)
    uint x_stride;  // floats between input columns
    uint out_row;   // bytes between quantized columns
    uint pad0;
    uint pad1;
    uint pad2;
    uint pad3;
    uint pad4;
};

RWByteAddressBuffer Q : register(u0);
RWByteAddressBuffer X : register(u1);

#include "ggml_d3d12_fp32_rn.hlsli"

float ldx(uint col, uint i) {
    return asfloat(X.Load((col * x_stride + i) * 4u));
}

#ifndef Q8_K

// fp32 → fp16 as F16C, round to nearest even (d3d12_f32_to_f16_rn).
uint f16_rn(uint a) {
    const uint sign = (a >> 16) & 0x8000u;
    const uint e = (a >> 23) & 0xffu;
    uint m = a & 0x7fffffu;
    if (e == 0xffu)
        return sign | 0x7c00u | (m != 0u ? 0x200u | (m >> 13) : 0u);
    const int e16 = int(e) - 127 + 15;
    uint mant;
    uint rem;
    uint half_;
    uint bits;
    if (e16 >= 1) {
        mant = m >> 13;
        rem = m & 0x1fffu;
        half_ = 0x1000u;
        bits = (uint(e16) << 10) + mant;
    } else {
        if (e == 0u)
            return sign;
        m |= 0x800000u;
        const uint s = uint(14 - e16);
        if (s >= 32u)
            return sign;
        mant = m >> s;
        rem = m & ((1u << s) - 1u);
        half_ = 1u << (s - 1u);
        bits = mant;
    }
    if (rem > half_ || (rem == half_ && (mant & 1u) != 0u))
        bits += 1u;
    if (bits >= 0x7c00u)
        bits = 0x7c00u;
    return sign | bits;
}

// _mm256_max_ps per lane: a NaN in either operand yields b.
float max_ps(float a, float b) {
    return a > b ? a : b;
}

float abs_bits(float v) { // _mm256_andnot_ps(-0.f, v)
    return asfloat(asuint(v) & 0x7fffffffu);
}

// One q8_0 block (32 floats from element i0): 2 bytes of fp16 d in the low half
// of `head`, the 32 quant bytes in w[0..7].
void quant_block(uint col, uint i0, out uint head, out uint w[8]) {
    // |x| reduced in the tree of ggml's AVX2 quantize_row_q8_0 with its max
    // semantics (a > b ? a : b), so a NaN lands where it lands on the CPU.
    float m[8];
    [unroll] for (uint l = 0u; l < 8u; ++l)
        m[l] = max_ps(max_ps(max_ps(abs_bits(ldx(col, i0 + l)), abs_bits(ldx(col, i0 + 8u + l))),
                             abs_bits(ldx(col, i0 + 16u + l))),
                      abs_bits(ldx(col, i0 + 24u + l)));
    float m4[4];
    [unroll] for (uint i = 0u; i < 4u; ++i)
        m4[i] = max_ps(m[4u + i], m[i]);
    const uint amax = asuint(max_ps(max_ps(m4[0], m4[2]), max_ps(m4[1], m4[3])));
    head = f16_rn(div_rn(amax, 0x42fe0000u)); // 127.0f
    const float id = amax != 0u ? asfloat(div_rn(0x42fe0000u, amax)) : 0.0f;
    [unroll] for (uint wi = 0u; wi < 8u; ++wi) {
        uint packed = 0u;
        [unroll] for (uint b = 0u; b < 4u; ++b) {
            precise const float v = ldx(col, i0 + wi * 4u + b) * id;
            const float r = round(v);
            // _mm256_cvtps_epi32: NaN / out of range → INT_MIN; packs saturates.
            int qi = abs(r) < 2147483648.0f ? int(r) : int(0x80000000u);
            qi = clamp(qi, -128, 127);
            packed |= (uint(qi) & 0xffu) << (8u * b);
        }
        w[wi] = packed;
    }
}

[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint u = gid.x * NUM_THREADS + tid; // pair of blocks
    if (u >= nunits)
        return;
    const uint col = gid.y;
    uint h0, h1;
    uint a[8], b[8];
    quant_block(col, u * 64u, h0, a);
    quant_block(col, u * 64u + 32u, h1, b);
    // 68 bytes: [d0][a 32 B][d1][b 32 B]. d0 and d1 are 2 bytes, so the quant
    // dwords are shifted by 16 bits.
    const uint o = col * out_row + u * 68u;
    Q.Store(o, h0 | (a[0] << 16));
    [unroll] for (uint i = 1u; i < 8u; ++i)
        Q.Store(o + i * 4u, (a[i - 1u] >> 16) | (a[i] << 16));
    Q.Store(o + 32u, (a[7] >> 16) | (h1 << 16));
    [unroll] for (uint j = 0u; j < 8u; ++j)
        Q.Store(o + 36u + j * 4u, b[j]);
}

#else // Q8_K

// One group of Q8K_THREADS per q8_K block (kD3d12QuantQ8KThreads): thread t
// owns elements 8t..8t+7, which is also its two quant dwords; threads 2j and
// 2j+1 make up bsums[j]. SV_GroupID.x is the block, SV_GroupID.y the column.
#define Q8K_THREADS 32

groupshared float s_mx[Q8K_THREADS];
groupshared uint s_idx[Q8K_THREADS];
groupshared int s_sum[Q8K_THREADS];

[numthreads(Q8K_THREADS, 1, 1)]
void CSMain(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint u = gid.x; // q8_K block
    if (u >= nunits) // whole group: uniform, before any barrier
        return;
    const uint col = gid.y;
    const uint i0 = u * 256u + tid * 8u;
    const uint o = col * out_row + u * 292u;
    // quantize_row_q8_K_ref keeps the first index of the largest |x| (strict >
    // on floats: a NaN never compares greater and is skipped). Each thread
    // finds its own first maximum; the tree keeps the larger |x| and, on a tie,
    // the smaller index, which is the same element in any reduction order.
    float x[8];
    float amax = 0.0f;
    float mx = 0.0f;
    uint idx = 256u; // none above 0 yet
    const uint a = (col * x_stride + i0) * 4u;
    const uint4 lo4 = X.Load4(a);
    const uint4 hi4 = X.Load4(a + 16u);
    x[0] = asfloat(lo4.x);
    x[1] = asfloat(lo4.y);
    x[2] = asfloat(lo4.z);
    x[3] = asfloat(lo4.w);
    x[4] = asfloat(hi4.x);
    x[5] = asfloat(hi4.y);
    x[6] = asfloat(hi4.z);
    x[7] = asfloat(hi4.w);
    [unroll] for (uint j = 0u; j < 8u; ++j) {
        const float ax = abs(x[j]);
        if (ax > amax) {
            amax = ax;
            mx = x[j];
            idx = tid * 8u + j;
        }
    }
    s_mx[tid] = mx;
    s_idx[tid] = idx;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint step = Q8K_THREADS / 2u; step > 0u; step >>= 1u) {
        if (tid < step) {
            const float ba = abs(s_mx[tid + step]);
            const float aa = abs(s_mx[tid]); // |mx| is the lane's amax
            const uint bi = s_idx[tid + step];
            if (ba > aa || (ba == aa && bi < s_idx[tid])) {
                s_mx[tid] = s_mx[tid + step];
                s_idx[tid] = bi;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (s_mx[0] == 0.0f) { // no |x| above 0 (or only NaN)
        Q.Store2(o + 4u + tid * 8u, uint2(0u, 0u));
        if ((tid & 3u) == 0u)
            Q.Store(o + 260u + tid, 0u);
        if (tid == 0u)
            Q.Store(o, 0u);
        return;
    }
    // max is non-zero and not NaN here (the strict > skips NaN); it may be
    // ±inf, which gives iscale = -0 and stores 0 for every quant, as on x86.
    const uint iscale_bits = div_rn(0xc2fe0000u, asuint(s_mx[0])); // -127.0f / max
    const float iscale = asfloat(iscale_bits);
    int sum = 0;
    uint w[2];
    [unroll] for (uint wi = 0u; wi < 2u; ++wi) {
        uint packed = 0u;
        [unroll] for (uint b = 0u; b < 4u; ++b) {
            precise const float fv = iscale * x[wi * 4u + b];
            int v = 0;
            // nearest_int; a non-finite fv stores 0, as on x86.
            if (((asuint(fv) >> 23) & 0xffu) != 0xffu) {
                precise const float t = fv + 12582912.0f;
                v = int(asuint(t) & 0x007fffffu) - 0x00400000;
            }
            v = min(127, v);
            sum += int(uint(v) << 24) >> 24; // the int8 the CPU stores
            packed |= (uint(v) & 0xffu) << (8u * b);
        }
        w[wi] = packed;
    }
    Q.Store2(o + 4u + tid * 8u, uint2(w[0], w[1]));
    s_sum[tid] = sum;
    GroupMemoryBarrierWithGroupSync();
    // bsums[j] = threads 2j + 2j+1; a dword holds bsums[j], bsums[j+1] (j even),
    // so thread t = 2j with t % 4 == 0 stores it at byte 260 + 2j = 260 + t.
    if ((tid & 3u) == 0u) {
        const int lo = s_sum[tid] + s_sum[tid + 1u];
        const int hi = s_sum[tid + 2u] + s_sum[tid + 3u];
        Q.Store(o + 260u + tid, (uint(lo) & 0xffffu) | (uint(hi) << 16));
    }
    if (tid == 0u)
        Q.Store(o, div_rn(0x3f800000u, iscale_bits)); // d = 1 / iscale
}

#endif
