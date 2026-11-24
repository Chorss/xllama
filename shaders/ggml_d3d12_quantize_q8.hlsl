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
//   - fp32 division is not correctly rounded on D3D12, so div_rn divides the
//     significands on integers;
//   - HLSL round() rounds half to even, like _mm256_round_ps(_MM_ROUND_NEAREST).
// One thread per pair of q8_0 blocks (68 B, so every store is a whole dword)
// or per q8_K block (292 B); SV_GroupID.y is the activation column.
// Root signature (src/bridge/ggml_d3d12.cpp): b0 constants, u0 the quantized
// output (the staging buffer), u1 the F32 input.

#define NUM_THREADS 64 // kD3d12QuantThreads

cbuffer Params : register(b0) {
    uint nunits;    // threads of work per column: k / 64 (q8_0) or k / 256 (q8_K)
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
    const uint rnd = (q >> 1) & 1u;
    sticky |= q & 1u;
    if (rnd != 0u && (sticky != 0u || (mant & 1u) != 0u))
        mant += 1u;
    return sign | (subnormal ? mant : (uint(e - 1) << 23) + mant);
}

float ldx(uint col, uint i) {
    return asfloat(X.Load((col * x_stride + i) * 4u));
}

#ifndef Q8_K

// fp32 → fp16, round to nearest even (d3d12_f32_to_f16_rn); d is finite, >= 0.
uint f16_rn(uint a) {
    const uint sign = (a >> 16) & 0x8000u;
    const uint e = (a >> 23) & 0xffu;
    uint m = a & 0x7fffffu;
    if (e == 0xffu)
        return sign | (m != 0u ? 0x7e00u : 0x7c00u);
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

// One q8_0 block (32 floats from element i0): 2 bytes of fp16 d in the low half
// of `head`, the 32 quant bytes in w[0..7].
void quant_block(uint col, uint i0, out uint head, out uint w[8]) {
    uint amax = 0u;
    for (uint j = 0u; j < 32u; ++j)
        amax = max(amax, asuint(ldx(col, i0 + j)) & 0x7fffffffu);
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

[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint u = gid.x * NUM_THREADS + tid; // one q8_K block
    if (u >= nunits)
        return;
    const uint col = gid.y;
    const uint i0 = u * 256u;
    const uint o = col * out_row + u * 292u;
    // First index of the largest |x| (strict >, as quantize_row_q8_K_ref).
    uint amax = 0u;
    float mx = 0.0f;
    for (uint j = 0u; j < 256u; ++j) {
        const float x = ldx(col, i0 + j);
        const uint ax = asuint(x) & 0x7fffffffu;
        if (ax > amax) {
            amax = ax;
            mx = x;
        }
    }
    if (amax == 0u) {
        for (uint z = 0u; z < 73u; ++z)
            Q.Store(o + z * 4u, 0u);
        return;
    }
    const float iscale = asfloat(div_rn(0xc2fe0000u, asuint(mx))); // -127.0f / max
    int prev = 0;
    for (uint g = 0u; g < 16u; ++g) {
        int sum = 0;
        uint w[4];
        [unroll] for (uint wi = 0u; wi < 4u; ++wi) {
            uint packed = 0u;
            [unroll] for (uint b = 0u; b < 4u; ++b) {
                precise const float fv = iscale * ldx(col, i0 + g * 16u + wi * 4u + b);
                int v = 0;
                // nearest_int; a non-finite fv stores 0, as on x86.
                if (((asuint(fv) >> 23) & 0xffu) != 0xffu) {
                    precise const float t = fv + 12582912.0f;
                    v = int(asuint(t) & 0x007fffffu) - 0x00400000;
                }
                v = min(127, v);
                const int q8 = int(uint(v) << 24) >> 24; // the int8 the CPU stores
                sum += q8;
                packed |= (uint(v) & 0xffu) << (8u * b);
            }
            w[wi] = packed;
        }
        [unroll] for (uint s = 0u; s < 4u; ++s)
            Q.Store(o + 4u + g * 16u + s * 4u, w[s]);
        // bsums are int16 pairs: store each dword once both halves are known.
        if ((g & 1u) == 0u)
            prev = sum;
        else
            Q.Store(o + 260u + (g - 1u) * 2u, (uint(prev) & 0xffffu) | (uint(sum) << 16));
    }
    Q.Store(o, div_rn(0x3f800000u, asuint(iscale))); // d = 1 / iscale
}

#endif
