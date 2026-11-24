// ggml backend d3d12: cost probe for a bit-exact RMS_NORM (knob
// d3d12-rmsnorm-probe.txt, selftest only; no graph op uses it).
// ggml-cpu's ggml_compute_forward_rms_norm_f32 sums the float products x*x
// into a double, one after another; every step rounds, so the order is part of
// the result and one thread must do the sum. The probe times that serial part
// at a model's width: the group loads the row and forms the products (each
// correctly rounded, `precise`, -denorm preserve), thread 0 sums them in double
// precision from groupshared memory and stores the double's bits.
// Root signature (src/bridge/ggml_d3d12.cpp): b0 constants, u0 the output
// (two dwords per row), u1 the F32 input.

#define NUM_THREADS 64
#define MAX_N 4096

cbuffer Params : register(b0) {
    uint n;        // elements per row (<= MAX_N)
    uint x_stride; // floats between input rows
    uint pad0;
    uint pad1;
    uint pad2;
    uint pad3;
    uint pad4;
    uint pad5;
};

RWByteAddressBuffer Out : register(u0);
RWByteAddressBuffer X : register(u1);

groupshared float sq[MAX_N];

[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint row = gid.y;
    const uint m = min(n, MAX_N); // the host refuses wider rows; never index past sq
    for (uint i = tid; i < m; i += NUM_THREADS) {
        const float x = asfloat(X.Load((row * x_stride + i) * 4u));
        precise const float p = x * x;
        sq[i] = p;
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid != 0u)
        return;
    precise double s = 0.0;
    for (uint j = 0u; j < m; ++j)
        s = s + (double)sq[j];
    uint lo, hi;
    asuint(s, lo, hi);
    Out.Store2(row * 8u, uint2(lo, hi));
}
