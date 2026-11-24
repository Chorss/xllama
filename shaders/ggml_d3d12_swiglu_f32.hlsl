// ggml backend d3d12: SWIGLU on F32 activations (knob d3d12-swiglu.txt),
//   Y[c * y_stride + i] = silu(G[c * g_stride + i]) * U[c * u_stride + i],
//   silu(x) = x / (1 + ggml_v_expf(-x)),
// equal bit for bit to ggml-cpu's x86 AVX2 + FMA ggml_vec_swiglu_f32. Every
// step mirrors d3d12_swiglu_emulate (src/bridge/ggml_d3d12.cpp), which the host
// tests compare against ggml:
//   - fp32 mul/add/sub are correctly rounded (D3D11.3 spec §3.1.3.1); the blob
//     is built with -denorm preserve and every expression is `precise`, so
//     nothing is fused or flushed;
//   - _mm256_fmadd_ps is a single rounding, which HLSL cannot express for fp32:
//     fma_rn computes it on integers (a 96-bit window in three dwords);
//   - _mm256_div_ps is correctly rounded, D3D12 division is not: div_rn.
// One thread per element; SV_GroupID.y is the activation column. Each thread
// reads both inputs before writing, so Y may alias either input.
// Root signature (src/bridge/ggml_d3d12.cpp): b0 constants, u0 Y, u1 G (gate),
// u2 U (up) — the ROPE root signature's third UAV.

#define NUM_THREADS 256 // kD3d12SwigluThreads

cbuffer Params : register(b0) {
    uint n;        // elements per column (ne0)
    uint g_stride; // floats between gate columns
    uint u_stride; // floats between up columns
    uint y_stride; // floats between output columns
    uint pad0;
    uint pad1;
    uint pad2;
    uint pad3;
};

RWStructuredBuffer<float> Y : register(u0);
RWByteAddressBuffer G : register(u1);
RWByteAddressBuffer U : register(u2);

#include "ggml_d3d12_fp32_rn.hlsli"

float fmadd(float a, float b, float c) {
    return asfloat(fma_rn(asuint(a), asuint(b), asuint(c)));
}

// ggml_v_expf (AVX2 + FMA), one lane (v_expf_lane).
float v_expf_lane(float x) {
    const float r = 12582912.0f; // 0x1.8p23f
    const float z = fmadd(x, asfloat(0x3fb8aa3bu), r); // 0x1.715476p+0f
    precise const float n = z - r;
    const float b = fmadd(-n, asfloat(0x35bfbe8eu), fmadd(-n, asfloat(0x3f317200u), x));
    const uint e = asuint(z) << 23;
    const float k = asfloat(e + 0x3f800000u);
    const bool c = abs(n) > 126.0f;
    precise const float u = b * b;
    precise const float lin = asfloat(0x3f7ffff6u) * b; // 0x1.ffffecp-1f
    const float j = fmadd(fmadd(fmadd(asfloat(0x3c072010u), b, asfloat(0x3d2b9f17u)), u,
                                fmadd(asfloat(0x3e2aaf33u), b, asfloat(0x3efffedbu))),
                          u, lin);
    if (!c)
        return fmadd(j, k, k);
    const uint g = n <= 0.0f ? 0x82000000u : 0u;
    const float s1 = asfloat(g + 0x7f000000u);
    const float s2 = asfloat(e - g);
    if (abs(n) > 192.0f) {
        precise const float sq = s1 * s1;
        return sq;
    }
    precise const float res = fmadd(s2, j, s2) * s1;
    return res;
}

[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint i = gid.x * NUM_THREADS + tid;
    if (i >= n)
        return;
    const uint col = gid.y;
    const float x = asfloat(G.Load((col * g_stride + i) * 4u));
    const float up = asfloat(U.Load((col * u_stride + i) * 4u));
    precise const float neg_x = 0.0f - x;
    precise const float den = 1.0f + v_expf_lane(neg_x);
    const float silu = asfloat(div_rn(asuint(x), asuint(den)));
    precise const float y = silu * up;
    Y[col * y_stride + i] = y;
}
