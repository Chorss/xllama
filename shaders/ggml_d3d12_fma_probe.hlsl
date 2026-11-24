// ggml backend d3d12: cost probe for a correctly rounded fp32 fma (knob
// d3d12-rmsnorm-probe.txt, selftest only; no graph op uses it). Each thread
// runs a chain of nine dependent fmas, the length of SWIGLU's longest chain,
// so the time is the chain's latency:
//   v = fma(a, b, v); a = b; b = v_old; ...
// Built three times:
//   default:  fma_rn, the integer emulation SWIGLU uses (ggml_d3d12_fp32_rn.hlsli);
//   -D F64=1: fma_f64, the double-precision emulation (d3d12_f32_fma_f64);
//   -D MAD=1: a * b + c in two roundings, a timing floor, not exact.
// The host compares every result with std::fmaf's chain (MAD rows excepted).
// Root signature (src/bridge/ggml_d3d12.cpp): b0 constants, u0 the results,
// u1 the inputs (a, b, v per thread).

#define NUM_THREADS 256

cbuffer Params : register(b0) {
    uint n; // threads with work
    uint pad0;
    uint pad1;
    uint pad2;
    uint pad3;
    uint pad4;
    uint pad5;
    uint pad6;
};

RWByteAddressBuffer Out : register(u0);
RWByteAddressBuffer In : register(u1);

#include "ggml_d3d12_fp32_rn.hlsli"

#if defined(F64)
// d3d12_f32_fma_f64, step for step.
uint fma_f64(uint a, uint b, uint c) {
    precise const double p = (double)asfloat(a) * (double)asfloat(b);
    precise const double cd = (double)asfloat(c);
    precise const double s = p + cd;
    const float r = (float)s;
    const uint rb = asuint(r);
    const bool r_inf = (rb & 0x7fffffffu) == 0x7f800000u;
    const bool r_neg = (rb & 0x80000000u) != 0u;
    // An overflow to ±inf stands for ±2^128, the grid point after FLT_MAX.
    const double rd = r_inf ? (r_neg ? asdouble(0u, 0xc7f00000u) : asdouble(0u, 0x47f00000u))
                            : (double)r;
    uint slo, shi;
    asuint(s, slo, shi);
    if (((shi >> 20) & 0x7ffu) == 0x7ffu || rd == s)
        return rb;
    precise const double bb = s - p;
    precise const double t1 = s - bb;
    precise const double e1 = p - t1;
    precise const double e2 = cd - bb;
    precise const double err = e1 + e2;
    const bool up = s > rd;
    double nd;
    if (r_inf)
        nd = (double)asfloat((rb & 0x80000000u) | 0x7f7fffffu); // ±FLT_MAX
    else if (r == 0.0f)
        nd = up ? asdouble(0u, 0x36a00000u) : asdouble(0u, 0xb6a00000u); // ±2^-149
    else
        nd = (double)asfloat((r > 0.0f) == up ? rb + 1u : rb - 1u);
    precise const double mid = (rd + nd) * 0.5;
    if (s != mid || err == 0.0)
        return rb;
    return (err > 0.0) == up ? asuint((float)nd) : rb;
}
#define FMA(a, b, c) asfloat(fma_f64(asuint(a), asuint(b), asuint(c)))
#elif defined(MAD)
float mad2(float a, float b, float c) {
    precise const float p = a * b;
    precise const float s = p + c;
    return s;
}
#define FMA(a, b, c) mad2(a, b, c)
#else
#define FMA(a, b, c) asfloat(fma_rn(asuint(a), asuint(b), asuint(c)))
#endif

[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    const uint i = dtid.x;
    if (i >= n)
        return;
    const uint3 in3 = In.Load3(i * 12u);
    float a = asfloat(in3.x);
    float b = asfloat(in3.y);
    float v = asfloat(in3.z);
    [unroll] for (uint k = 0u; k < 9u; ++k) {
        const float t = FMA(a, b, v);
        a = b;
        b = v;
        v = t;
    }
    Out.Store(i * 4u, asuint(v));
}
