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

// --- 96-bit unsigned integers, x = lowest dword ---

uint w96(uint3 a, int i) {
    return i == 0 ? a.x : (i == 1 ? a.y : a.z);
}

bool is_zero96(uint3 a) {
    return (a.x | a.y | a.z) == 0u;
}

uint3 add96(uint3 a, uint3 b) {
    uint3 r;
    const uint s0 = a.x + b.x;
    const uint c0 = s0 < a.x ? 1u : 0u;
    const uint s1 = a.y + b.y;
    const uint c1a = s1 < a.y ? 1u : 0u;
    const uint s1c = s1 + c0;
    const uint c1b = s1c < s1 ? 1u : 0u;
    const uint s2 = a.z + b.z + (c1a | c1b);
    r = uint3(s0, s1c, s2);
    return r;
}

uint3 sub96(uint3 a, uint3 b) { // a >= b
    const uint d0 = a.x - b.x;
    const uint b0 = a.x < b.x ? 1u : 0u;
    const uint d1 = a.y - b.y;
    const uint b1a = a.y < b.y ? 1u : 0u;
    const uint d1b = d1 - b0;
    const uint b1b = d1 < b0 ? 1u : 0u;
    const uint d2 = a.z - b.z - (b1a | b1b);
    return uint3(d0, d1b, d2);
}

bool ge96(uint3 a, uint3 b) {
    if (a.z != b.z)
        return a.z > b.z;
    if (a.y != b.y)
        return a.y > b.y;
    return a.x >= b.x;
}

uint3 shl96(uint3 a, int s) { // 0 <= s < 96
    const int ws = s >> 5;
    const uint bs = uint(s & 31);
    uint r[3];
    [unroll] for (int i = 2; i >= 0; --i) {
        const int j = i - ws;
        uint v = 0u;
        if (j >= 0) {
            v = w96(a, j) << bs;
            if (bs != 0u && j >= 1)
                v |= w96(a, j - 1) >> (32u - bs);
        }
        r[i] = v;
    }
    return uint3(r[0], r[1], r[2]);
}

uint3 shr96(uint3 a, int s, inout uint sticky) { // s >= 0
    if (s >= 96) {
        sticky |= is_zero96(a) ? 0u : 1u;
        return uint3(0u, 0u, 0u);
    }
    const int ws = s >> 5;
    const uint bs = uint(s & 31);
    uint lost = 0u;
    [unroll] for (int k = 0; k < 3; ++k)
        if (k < ws)
            lost |= w96(a, k);
    if (bs != 0u)
        lost |= w96(a, ws) & ((1u << bs) - 1u);
    uint r[3];
    [unroll] for (int i = 0; i < 3; ++i) {
        const int j = i + ws;
        uint v = 0u;
        if (j <= 2) {
            v = w96(a, j) >> bs;
            if (bs != 0u && j + 1 <= 2)
                v |= w96(a, j + 1) << (32u - bs);
        }
        r[i] = v;
    }
    sticky |= lost != 0u ? 1u : 0u;
    return uint3(r[0], r[1], r[2]);
}

int top96(uint3 a) {
    if (a.z != 0u)
        return 64 + int(firstbithigh(a.z));
    if (a.y != 0u)
        return 32 + int(firstbithigh(a.y));
    if (a.x != 0u)
        return int(firstbithigh(a.x));
    return -1;
}

// Correctly rounded a * b + c on fp32 bits (d3d12_f32_fma_rn).
uint fma_rn(uint a, uint b, uint c) {
    const uint sp = (a ^ b) & 0x80000000u;
    const uint sc = c & 0x80000000u;
    const uint ea = (a >> 23) & 0xffu;
    const uint eb = (b >> 23) & 0xffu;
    const uint ec = (c >> 23) & 0xffu;
    uint ma = a & 0x7fffffu;
    uint mb = b & 0x7fffffu;
    uint mc = c & 0x7fffffu;
    const bool a_nan = ea == 0xffu && ma != 0u;
    const bool b_nan = eb == 0xffu && mb != 0u;
    const bool c_nan = ec == 0xffu && mc != 0u;
    const bool a_inf = ea == 0xffu && ma == 0u;
    const bool b_inf = eb == 0xffu && mb == 0u;
    const bool c_inf = ec == 0xffu && mc == 0u;
    const bool a_zero = ea == 0u && ma == 0u;
    const bool b_zero = eb == 0u && mb == 0u;
    const bool c_zero = ec == 0u && mc == 0u;
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
            return sp & sc;
        return c;
    }
    const int xa = (ea == 0u ? 1 : int(ea)) - 150;
    const int xb = (eb == 0u ? 1 : int(eb)) - 150;
    const int xc = (ec == 0u ? 1 : int(ec)) - 150;
    if (ea != 0u)
        ma |= 0x800000u;
    if (eb != 0u)
        mb |= 0x800000u;
    if (ec != 0u)
        mc |= 0x800000u;
    const uint al = ma & 0xfffu;
    const uint ah = ma >> 12;
    const uint bl = mb & 0xfffu;
    const uint bh = mb >> 12;
    uint3 prod = add96(add96(shl96(uint3(ah * bh, 0u, 0u), 24),
                             shl96(uint3(ah * bl + al * bh, 0u, 0u), 12)),
                       uint3(al * bl, 0u, 0u));
    const int xp = xa + xb;
    uint3 cm = uint3(mc, 0u, 0u);
    const int tp = xp + top96(prod);
    const int tc = c_zero ? tp : xc + top96(cm);
    const int hi = max(tp, tc);
    int lsb = c_zero ? xp : min(xp, xc);
    if (hi - lsb > 93)
        lsb = hi - 93;
    uint st_p = 0u;
    uint st_c = 0u;
    prod = xp >= lsb ? shl96(prod, xp - lsb) : shr96(prod, lsb - xp, st_p);
    if (c_zero)
        cm = uint3(0u, 0u, 0u);
    else
        cm = xc >= lsb ? shl96(cm, xc - lsb) : shr96(cm, lsb - xc, st_c);
    prod.x |= st_p;
    cm.x |= st_c;
    uint3 sum;
    uint sign;
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
        return 0u;
    const int t = top96(sum);
    const int e = lsb + t;
    if (e > 127)
        return sign | 0x7f800000u;
    const int out_lsb = e <= -126 ? -149 : e - 23;
    const int sh = out_lsb - lsb;
    uint mant;
    uint rnd = 0u;
    uint sticky = 0u;
    if (sh <= 0) {
        mant = shl96(sum, -sh).x;
    } else {
        const uint3 r = shr96(sum, sh - 1, sticky);
        rnd = r.x & 1u;
        mant = r.x >> 1;
    }
    if (rnd != 0u && (sticky != 0u || (mant & 1u) != 0u))
        mant += 1u;
    const uint bits = e <= -126 ? mant : (uint(e + 126) << 23) + mant;
    return sign | min(bits, 0x7f800000u);
}

// Correctly rounded a / b on fp32 bits (d3d12_f32_div_rn).
uint div_rn(uint a, uint b) {
    const uint sign = (a ^ b) & 0x80000000u;
    const uint ea = (a >> 23) & 0xffu;
    const uint eb = (b >> 23) & 0xffu;
    uint ma = a & 0x7fffffu;
    uint mb = b & 0x7fffffu;
    const bool a_nan = ea == 0xffu && ma != 0u;
    const bool b_nan = eb == 0xffu && mb != 0u;
    const bool a_inf = ea == 0xffu && ma == 0u;
    const bool b_inf = eb == 0xffu && mb == 0u;
    const bool a_zero = ea == 0u && ma == 0u;
    const bool b_zero = eb == 0u && mb == 0u;
    if (a_nan || b_nan || (a_inf && b_inf) || (a_zero && b_zero))
        return 0x7fc00000u;
    if (a_inf || b_zero)
        return sign | 0x7f800000u;
    if (b_inf || a_zero)
        return sign;
    int xa = int(ea);
    int xb = int(eb);
    if (ea == 0u) {
        xa = 1;
        while ((ma & 0x800000u) == 0u) {
            ma <<= 1;
            xa -= 1;
        }
    } else {
        ma |= 0x800000u;
    }
    if (eb == 0u) {
        xb = 1;
        while ((mb & 0x800000u) == 0u) {
            mb <<= 1;
            xb -= 1;
        }
    } else {
        mb |= 0x800000u;
    }
    int e = xa - xb + 127;
    if (ma < mb) {
        ma <<= 1;
        e -= 1;
    }
    uint q = 0u;
    uint r = ma;
    [unroll] for (int i = 0; i < 26; ++i) {
        q <<= 1;
        if (r >= mb) {
            r -= mb;
            q |= 1u;
        }
        r <<= 1;
    }
    uint sticky = r != 0u ? 1u : 0u;
    if (e >= 255)
        return sign | 0x7f800000u;
    const bool subnormal = e <= 0;
    if (subnormal) {
        const uint sh = uint(1 - e);
        if (sh > 26u) {
            sticky |= q != 0u ? 1u : 0u;
            q = 0u;
        } else {
            sticky |= (q & ((1u << sh) - 1u)) != 0u ? 1u : 0u;
            q >>= sh;
        }
    }
    uint mant = q >> 2;
    const uint rb = (q >> 1) & 1u;
    sticky |= q & 1u;
    if (rb != 0u && (sticky != 0u || (mant & 1u) != 0u))
        mant += 1u;
    return sign | (subnormal ? mant : (uint(e - 1) << 23) + mant);
}

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
