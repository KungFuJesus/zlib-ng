/* adler32_avx512_vnni_tpl.h -- adler32 avx512 VNNI vectorized function templates
 * Copyright (C) 2022 Adam Stylinski
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "../../zbuild.h"
#include "../../adler32_p.h"
#include "../../cpu_features.h"
#include "../../fallback_builtins.h"
#include <immintrin.h>
#include "adler32_avx512_p.h"
#include "../../adler32_fold.h"

#ifdef COPY
Z_INTERNAL uint32_t adler32_fold_copy_avx512_vnni(uint32_t adler, uint8_t *dst, const uint8_t *src, size_t len) {
#else
Z_INTERNAL uint32_t adler32_avx512_vnni(uint32_t adler, const uint8_t *src, size_t len) {
#endif

    uint32_t adler0, adler1;
    adler1 = (adler >> 16) & 0xffff;
    adler0 = adler & 0xffff; 

    if (src == NULL) return 1L;
    if (len == 0) return adler;

rem_peel:
    if (len < 64) {
        /* This handles the remaining copies, just call normal adler checksum after this */
#ifdef COPY
        __mmask64 storemask = (0xFFFFFFFFFFFFFFFFUL >> (64 - len));
        __m512i copy_vec = _mm512_maskz_loadu_epi8(storemask, src);
        _mm512_mask_storeu_epi8(dst, storemask, copy_vec);
#endif

#ifdef X86_AVX2_ADLER32
        return adler32_avx2(adler, src, len);
#elif defined(X86_SSSE3_ADLER32)
        return adler32_ssse3(adler, src, len);
#else
        return adler32_len_16(adler0, src, len, adler1); 
#endif
    }

    const __m512i dot2v = _mm512_set_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                          20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
                                          38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                                          56, 57, 58, 59, 60, 61, 62, 63, 64);

    const __m512i zero = _mm512_setzero_si512();
    __m512i vs1, vs2;

    while (len >= 64) {
        vs1 = _mm512_zextsi128_si512(_mm_cvtsi32_si128(adler0));
        vs2 = _mm512_zextsi128_si512(_mm_cvtsi32_si128(adler1));
        size_t k = MIN(len, NMAX);
        k -= k % 64;
        len -= k;
        __m512i vs1_0 = vs1;
        __m512i vs3 = _mm512_setzero_si512();
        /* We might get a tad bit more ILP here if we sum to a second register in the loop */
        __m512i vs2_1 = _mm512_setzero_si512();
        __m512i vbuf0, vbuf1;

        /* Remainder peeling */
        if (k % 128) {
            vbuf1 = _mm512_loadu_si512(src);
#ifdef COPY
            _mm512_storeu_si512(dst, vbuf1);
            dst += 64;
#endif
            src += 64;
            k -= 64;

            __m512i vs1_sad = _mm512_sad_epu8(vbuf1, zero);
            vs1 = _mm512_add_epi32(vs1, vs1_sad);
            vs3 = _mm512_add_epi32(vs3, vs1_0);
            vs2 = _mm512_dpbusd_epi32(vs2, vbuf1, dot2v);
            vs1_0 = vs1;
        }

        /* Manually unrolled this loop by 2 for an decent amount of ILP */
        while (k >= 128) {
            /*
               vs1 = adler + sum(c[i])
               vs2 = sum2 + 64 vs1 + sum( (64-i+1) c[i] )
            */
            vbuf0 = _mm512_loadu_si512(src);
            vbuf1 = _mm512_loadu_si512(src + 64);
#ifdef COPY
            _mm512_storeu_si512(dst, vbuf0);
            _mm512_storeu_si512(dst + 64, vbuf1);
            dst += 128;
#endif
            src += 128;
            k -= 128;

            __m512i vs1_sad = _mm512_sad_epu8(vbuf0, zero);
            vs1 = _mm512_add_epi32(vs1, vs1_sad);
            vs3 = _mm512_add_epi32(vs3, vs1_0);
            /* multiply-add, resulting in 16 ints. Fuse with sum stage from prior versions, as we now have the dp
             * instructions to eliminate them */
            vs2 = _mm512_dpbusd_epi32(vs2, vbuf0, dot2v);

            vs3 = _mm512_add_epi32(vs3, vs1);
            vs1_sad = _mm512_sad_epu8(vbuf1, zero);
            vs1 = _mm512_add_epi32(vs1, vs1_sad);
            vs2_1 = _mm512_dpbusd_epi32(vs2_1, vbuf1, dot2v);
            vs1_0 = vs1;
        }

        vs3 = _mm512_slli_epi32(vs3, 6);
        vs2 = _mm512_add_epi32(vs2, vs3);
        vs2 = _mm512_add_epi32(vs2, vs2_1);

        adler0 = partial_hsum(vs1) % BASE;
        adler1 = _mm512_reduce_add_epu32(vs2) % BASE;
    }

    adler = adler0 | (adler1 << 16);

    /* Process tail (len < 64). */
    if (len) {
        goto rem_peel;
    }

    return adler; 
}
