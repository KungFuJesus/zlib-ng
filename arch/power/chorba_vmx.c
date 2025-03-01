#if !defined(WITHOUT_CHORBA) && defined(PPC_VMX)

#include <altivec.h>
#include <assert.h>
#include "zbuild.h"
#include "zendian.h"
#include "crc32_braid_p.h"
#include "crc32_braid_tbl.h"
#include "crc32.h"
#include "arch/generic/generic_functions.h"

#define vmx_zero()  (vec_splat_u32(0))

#define DO_FULL_ROUND(in, o0, o1, o2, o3, o4, o5) do {\
    vector unsigned int splatter = {17, 23, 19, 20}; \
    o0 = vec_sl(in, vec_splat(splatter, 0)); \
    o1 = vec_xor(vec_sr(in, vec_splat_u32(15)), vec_sl(in, vec_splat(splatter, 1))); \
    o2 = vec_xor(vec_sr(in, vec_splat_u32(9)), vec_sl(in, vec_splat(splatter, 2))); \
    o3 = vec_sr(in, vec_splat_u32(13)); \
    o4 = vec_sl(in, vec_splat_u32(12)); \
    o5 = vec_sr(in, vec_splat(splatter, 3)); \
} while (0);

/* Fun, these aren't endianness independent */
#if BYTE_ORDER == BIG_ENDIAN
#   define VEC_SRO(a, b) vec_sro(a, b)
#   define VEC_SLO(a, b) vec_slo(a, b)
#   define VEC_SLD(a, b, c) vec_sld(a, b, c)
#else
#   define VEC_SRO(a, b) vec_slo(a, b)
#   define VEC_SLO(a, b) vec_sro(a, b)
#   define VEC_SLD(a, b, c) vec_sld(b, a, 16-c)
#endif

Z_FORCEINLINE vector unsigned int bswap_vmx(vector unsigned int a) {
    const vector unsigned char swap_vect = {
         3,  2,  1,  0,
         7,  6,  5,  4,
        11, 10,  9,  8,
        15, 14, 13, 12
    };

    return vec_perm(a, a, swap_vect);
}

extern uint32_t crc32_braid_base(uint32_t c, const uint8_t *buf, size_t len);
extern uint32_t crc32_chorba_small_nondestructive_32bit(uint32_t crc, const uint32_t* buf, size_t len);

Z_FORCEINLINE uint32_t chorba_small_nondestructive_vmx(uint32_t crc, const uint32_t* buf, size_t len) {
    const uint32_t* input = buf;
    ALIGNED_(16) uint32_t final[20] = {0};

    vector unsigned int next1234 = { crc, 0, 0, 0 };
    crc = 0;

    vector unsigned int next5678 = vmx_zero();
    vector unsigned int next910xx = next5678;
    const vector unsigned int mask910 = { UINT32_MAX, UINT32_MAX, 0, 0 };
    size_t i = 0;

    for(; i + 120 < len; i += 80) {
        vector unsigned int in1234, in5678, in910xx;
        vector unsigned int abcd1, efgh1, ijxx1;
        vector unsigned int abcd2, efgh2, ijxx2;
        vector unsigned int abcd3, efgh3, ijxx3;
        vector unsigned int abcd4, efgh4, ijxx4;
        vector unsigned int abcd6, efgh6, ijxx6;
        vector unsigned int abcd7, efgh7, ijxx7;
        vector unsigned char thirty_two = vec_sl(vec_splat_u8(8), vec_splat_u8(2));
        vector unsigned char sixty_four = vec_add(thirty_two, thirty_two);
        vector unsigned char ninety_six = vec_add(sixty_four, thirty_two);

        in1234 = vec_ld(i, input);
        in5678 = vec_ld(i+16, input);
        in910xx = vec_ld(i+32, input);

#if BYTE_ORDER == BIG_ENDIAN
        in1234 = bswap_vmx(in1234);
        in5678 = bswap_vmx(in5678);
        in910xx = bswap_vmx(in910xx);
#endif
        vector unsigned int nextin12 = in910xx; 

        in1234 = vec_xor(in1234, next1234);

        DO_FULL_ROUND(in1234, abcd1, abcd2, abcd3, abcd4, abcd6, abcd7);

        /*
        in5 ^= next5 ^ a1;
        in6 ^= next6 ^ b1 ^ a2;
        in7 ^= next7 ^ c1 ^ b2 ^ a3;
        in8 ^= next8 ^ d1 ^ c2 ^ b3 ^ a4;
        */

        in5678 = vec_xor(in5678, next5678);
        vector unsigned int abc2 = VEC_SRO(abcd2, thirty_two);
        abc2 = vec_xor(abc2, abcd1);
        in5678 = vec_xor(in5678, abc2);
        vector unsigned int ab3 = VEC_SRO(abcd3, sixty_four);
        vector unsigned int a4 = VEC_SRO(abcd4, ninety_six);
        in5678 = vec_xor(in5678, vec_xor(ab3, a4));

        DO_FULL_ROUND(in5678, efgh1, efgh2, efgh3, efgh4, efgh6, efgh7); 

        /*
        e1 = (in5 << 17);
        e2 = (in5 >> 15) ^ (in5 << 23);
        e3 = (in5 >> 9) ^ (in5 << 19);
        e4 = (in5 >> 13);
        e6 = (in5 << 12);
        e7 = (in5 >> 20);

        f1 = (in6 << 17);
        f2 = (in6 >> 15) ^ (in6 << 23);
        f3 = (in6 >> 9) ^ (in6 << 19);
        f4 = (in6 >> 13);
        f6 = (in6 << 12);
        f7 = (in6 >> 20);

        g1 = (in7 << 17);
        g2 = (in7 >> 15) ^ (in7 << 23);
        g3 = (in7 >> 9) ^ (in7 << 19);
        g4 = (in7 >> 13);
        g6 = (in7 << 12);
        g7 = (in7 >> 20);

        h1 = (in8 << 17);
        h2 = (in8 >> 15) ^ (in8 << 23);
        h3 = (in8 >> 9) ^ (in8 << 19);
        h4 = (in8 >> 13);
        h6 = (in8 << 12);
        h7 = (in8 >> 20);
        */

        vector unsigned int j_mask = { UINT32_MAX, 0, 0, 0};

        /*in9 ^= next9 ^ b4 ^ c3 ^ d2 ^ e1;
        in10 ^= next10^c4  ^ d3 ^ e2 ^ f1 ^ a6;*/
        in910xx = vec_xor(in910xx, next910xx);
        vector unsigned int bcd4 = VEC_SLO(abcd4, thirty_two);
        vector unsigned int cd3 = VEC_SLO(abcd3, sixty_four);
        vector unsigned int de2 = VEC_SLD(abcd2, efgh2, 12);
        vector unsigned int a6 = VEC_SRO(vec_and(abcd6, j_mask), thirty_two);
        /* The last two lanes of this vector are unused */
        in910xx = vec_xor(vec_xor(in910xx, de2), vec_xor(bcd4, cd3));
        in910xx = vec_xor(in910xx, vec_xor(efgh1, a6));

        DO_FULL_ROUND(in910xx, ijxx1, ijxx2, ijxx3, ijxx4, ijxx6, ijxx7);

        /*
        i1 = (in9 << 17);
        i2 = (in9 >> 15) ^ (in9 << 23);
        i3 = (in9 >> 9) ^ (in9 << 19);
        i4 = (in9 >> 13);
        i6 = (in9 << 12);
        i7 = (in9 >> 20);

        j1 = (in10 << 17);
        j2 = (in10 >> 15) ^ (in10 << 23);
        j3 = (in10 >> 9) ^ (in10 << 19);
        j4 = (in10 >> 13);
        j6 = (in10 << 12);
        j7 = (in10 >> 20);
        */

        /*
        out1 = a7 ^ b6 ^ d4 ^ e3 ^ f2 ^ g1;
        out2 = b7 ^ c6 ^ e4 ^ f3 ^ g2 ^ h1;
        out3 = c7 ^ d6 ^ f4 ^ g3 ^ h2 ^ i1;
        out4 = d7 ^ e6 ^ g4 ^ h3 ^ i2 ^ j1;

        out5 = e7 ^ f6 ^ h4 ^ i3 ^ j2;
        out6 = f7 ^ g6 ^ i4 ^ j3;
        out7 = g7 ^ h6 ^ j4;
        out8 = h7 ^ i6;

        out9 = i7 ^ j6;
        out10 = j7;

        next1 = out1;
        next2 = out2;
        next3 = out3;
        next4 = out4;
        next5 = out5;
        next6 = out6;
        next7 = out7;
        next8 = out8;
        next9 = out9;
        next10 = out10;
        */

        vector unsigned int bcde6 = VEC_SLD(abcd6, efgh6, 4);
        vector unsigned int defg4 = VEC_SLD(abcd4, efgh4, 12);
        vector unsigned int fghi2 = VEC_SLD(efgh2, ijxx2, 4);
        vector unsigned int ghij1 = VEC_SLD(efgh1, ijxx1, 8);

        next1234 = vec_xor(vec_xor(abcd7, bcde6), vec_xor(defg4, efgh3));
        next1234 = vec_xor(next1234, vec_xor(fghi2, ghij1));

        ijxx4 = vec_and(ijxx4, mask910);
        ijxx3 = vec_and(ijxx3, mask910);
        vector unsigned int fghi6 = VEC_SLD(efgh6, ijxx6, 4);
        vector unsigned int hijx4 = VEC_SLD(efgh4, ijxx4, 12);
        next5678 = vec_xor(vec_xor(efgh7, fghi6), vec_xor(hijx4, ijxx3));
        /* Yes, this is an unholy way to get this out... There are worse */
        vector unsigned int j2 = vec_and(VEC_SLO(ijxx2, thirty_two), j_mask);
        vector unsigned int j6 = vec_and(VEC_SLO(ijxx6, thirty_two), j_mask);
        next5678 = vec_xor(next5678, j2);
        next910xx = vec_xor(ijxx7, j6);

        /* Now repeat this, but with some extra permutations to get the
         * same operands. This is unrolling but for the sake of alignment,
         * not neccessarily speed */
        vector unsigned int in3456, in78910;
        in3456 = vec_ld(i+48, input);
        in78910 = vec_ld(i+64, input);

#if BYTE_ORDER == BIG_ENDIAN
        in3456 = bswap_vmx(in3456);
        in78910 = bswap_vmx(in78910);
#endif

        in1234 = VEC_SLD(nextin12, in3456, 8);
        in5678 = VEC_SLD(in3456, in78910, 8);
        in910xx = VEC_SLO(in78910, sixty_four);

        in1234 = vec_xor(in1234, next1234);
        DO_FULL_ROUND(in1234, abcd1, abcd2, abcd3, abcd4, abcd6, abcd7);

        in5678 = vec_xor(in5678, next5678);
        abc2 = VEC_SRO(abcd2, thirty_two);
        abc2 = vec_xor(abc2, abcd1);
        in5678 = vec_xor(in5678, abc2);
        ab3 = VEC_SRO(abcd3, sixty_four);
        a4 = VEC_SRO(abcd4, ninety_six);
        in5678 = vec_xor(in5678, vec_xor(ab3, a4));
        DO_FULL_ROUND(in5678, efgh1, efgh2, efgh3, efgh4, efgh6, efgh7); 

        in910xx = vec_xor(in910xx, next910xx);
        bcd4 = VEC_SLO(abcd4, thirty_two);
        cd3 = VEC_SLO(abcd3, sixty_four);
        de2 = VEC_SLD(abcd2, efgh2, 12);
        a6 = VEC_SRO(vec_and(abcd6, j_mask), thirty_two);
        in910xx = vec_xor(vec_xor(in910xx, de2), vec_xor(bcd4, cd3));
        in910xx = vec_xor(in910xx, vec_xor(efgh1, a6));
        DO_FULL_ROUND(in910xx, ijxx1, ijxx2, ijxx3, ijxx4, ijxx6, ijxx7);

        bcde6 = VEC_SLD(abcd6, efgh6, 4);
        defg4 = VEC_SLD(abcd4, efgh4, 12);
        fghi2 = VEC_SLD(efgh2, ijxx2, 4);
        ghij1 = VEC_SLD(efgh1, ijxx1, 8);
        next1234 = vec_xor(vec_xor(abcd7, bcde6), vec_xor(defg4, efgh3));
        next1234 = vec_xor(next1234, vec_xor(fghi2, ghij1));

        ijxx4 = vec_and(ijxx4, mask910);
        ijxx3 = vec_and(ijxx3, mask910);
        fghi6 = VEC_SLD(efgh6, ijxx6, 4);
        hijx4 = VEC_SLD(efgh4, ijxx4, 12);
        next5678 = vec_xor(vec_xor(efgh7, fghi6), vec_xor(hijx4, ijxx3));
        j2 = vec_and(VEC_SLO(ijxx2, thirty_two), j_mask);
        j6 = vec_and(VEC_SLO(ijxx6, thirty_two), j_mask);
        next5678 = vec_xor(next5678, j2);
        next910xx = vec_xor(ijxx7, j6);
    }

    if (len > i + 40) {
        vector unsigned int in1234, in5678, in910xx;
        vector unsigned int abcd1, efgh1, ijxx1;
        vector unsigned int abcd2, efgh2, ijxx2;
        vector unsigned int abcd3, efgh3, ijxx3;
        vector unsigned int abcd4, efgh4, ijxx4;
        vector unsigned int abcd6, efgh6, ijxx6;
        vector unsigned int abcd7, efgh7, ijxx7;
        vector unsigned char thirty_two = vec_sl(vec_splat_u8(8), vec_splat_u8(2));
        vector unsigned char sixty_four = vec_add(thirty_two, thirty_two);
        vector unsigned char ninety_six = vec_add(sixty_four, thirty_two);

        in1234 = vec_ld(i, input);
        in5678 = vec_ld(i+16, input);
        in910xx = vec_ld(i+32, input);

#if BYTE_ORDER == BIG_ENDIAN
        in1234 = bswap_vmx(in1234);
        in5678 = bswap_vmx(in5678);
        in910xx = bswap_vmx(in910xx);
#endif

        in1234 = vec_xor(in1234, next1234);
        DO_FULL_ROUND(in1234, abcd1, abcd2, abcd3, abcd4, abcd6, abcd7);
        in5678 = vec_xor(in5678, next5678);
        vector unsigned int abc2 = VEC_SRO(abcd2, thirty_two);
        abc2 = vec_xor(abc2, abcd1);
        in5678 = vec_xor(in5678, abc2);
        vector unsigned int ab3 = VEC_SRO(abcd3, sixty_four);
        vector unsigned int a4 = VEC_SRO(abcd4, ninety_six);
        in5678 = vec_xor(in5678, vec_xor(ab3, a4));
        DO_FULL_ROUND(in5678, efgh1, efgh2, efgh3, efgh4, efgh6, efgh7); 
        vector unsigned int j_mask = { UINT32_MAX, 0, 0, 0};
        in910xx = vec_xor(in910xx, next910xx);
        vector unsigned int bcd4 = VEC_SLO(abcd4, thirty_two);
        vector unsigned int cd3 = VEC_SLO(abcd3, sixty_four);
        vector unsigned int de2 = VEC_SLD(abcd2, efgh2, 12);
        vector unsigned int a6 = VEC_SRO(vec_and(abcd6, j_mask), thirty_two);
        in910xx = vec_xor(vec_xor(in910xx, de2), vec_xor(bcd4, cd3));
        in910xx = vec_xor(in910xx, vec_xor(efgh1, a6));
        DO_FULL_ROUND(in910xx, ijxx1, ijxx2, ijxx3, ijxx4, ijxx6, ijxx7);
        vector unsigned int bcde6 = VEC_SLD(abcd6, efgh6, 4);
        vector unsigned int defg4 = VEC_SLD(abcd4, efgh4, 12);
        vector unsigned int fghi2 = VEC_SLD(efgh2, ijxx2, 4);
        vector unsigned int ghij1 = VEC_SLD(efgh1, ijxx1, 8);
        next1234 = vec_xor(vec_xor(abcd7, bcde6), vec_xor(defg4, efgh3));
        next1234 = vec_xor(next1234, vec_xor(fghi2, ghij1));
        ijxx4 = vec_and(ijxx4, mask910);
        ijxx3 = vec_and(ijxx3, mask910);
        vector unsigned int fghi6 = VEC_SLD(efgh6, ijxx6, 4);
        vector unsigned int hijx4 = VEC_SLD(efgh4, ijxx4, 12);
        next5678 = vec_xor(vec_xor(efgh7, fghi6), vec_xor(hijx4, ijxx3));
        vector unsigned int j2 = vec_and(VEC_SLO(ijxx2, thirty_two), j_mask);
        vector unsigned int j6 = vec_and(VEC_SLO(ijxx6, thirty_two), j_mask);
        next5678 = vec_xor(next5678, j2);
        next910xx = vec_xor(ijxx7, j6);
        i += 40;
    }


    next910xx = vec_and(next910xx, mask910);

#if BYTE_ORDER == BIG_ENDIAN
    next1234 = bswap_vmx(next1234);
    next5678 = bswap_vmx(next5678);
    next910xx = bswap_vmx(next910xx);
#endif

    ALIGNED_(16) uint32_t nexts[12];
    vec_st(next1234, 0, nexts);
    vec_st(next5678, 16, nexts);
    vec_st(next910xx, 32, nexts);

    size_t copy_dist = len - i;
    uint8_t* __restrict src = (uint8_t*)input + i;
    uint8_t* __restrict dst = (uint8_t*)final;

    while (copy_dist--) {
       *dst++ = *src++;
    }

    final[0] ^= nexts[0];
    final[1] ^= nexts[1];
    final[2] ^= nexts[2];
    final[3] ^= nexts[3];
    final[4] ^= nexts[4];
    final[5] ^= nexts[5];
    final[6] ^= nexts[6];
    final[7] ^= nexts[7];
    final[8] ^= nexts[8];
    final[9] ^= nexts[9];

    crc = crc32_braid_internal(crc, (uint8_t*) final, len-i);

    return crc;
}

Z_INTERNAL uint32_t crc32_chorba_vmx(uint32_t crc, const uint8_t *buf, size_t len) {
    uint32_t c;
    uint32_t* aligned_buf;
    size_t aligned_len;

    c = (~crc) & 0xffffffff;
    unsigned long algn_diff = ((uintptr_t)16 - ((uintptr_t)buf & 15)) & 15;
    if (algn_diff < len) {
        if (algn_diff) {
            c = crc32_braid_internal(c, buf, algn_diff);
        }
        aligned_buf = (uint32_t*) (buf + algn_diff);
        aligned_len = len - algn_diff;
        if(aligned_len > CHORBA_LARGE_THRESHOLD) {
            c = crc32_chorba_118960_nondestructive(c, (z_word_t*) aligned_buf, aligned_len);
        } else if (aligned_len > CHORBA_SMALL_THRESHOLD_32BIT) {
#   if OPTIMAL_CMP == 64
            /* At the moment, this is _marginally_ faster than the VMX 32 bit version. The
             * wider scalar GPRs end up just barely winning out. The data dependency stalls
             * in the loop of this function are the bottleneck. As soon as we have a "braided"
             * version of this it should be faster */
            c = crc32_chorba_small_nondestructive(c, (uint64_t*) aligned_buf, aligned_len);
#   else
            c = chorba_small_nondestructive_vmx(c, aligned_buf, aligned_len);
#   endif
        } else {
            c = crc32_braid_internal(c, (uint8_t*) aligned_buf, aligned_len);
        }
    }
    else {
        c = crc32_braid_internal(c, buf, len);
    }

    /* Return the CRC, post-conditioned. */
    return c ^ 0xffffffff;
}
#endif
