/* chunkset_avx512.c -- AVX512 inline functions to copy small data chunks.
 * For conditions of distribution and use, see copyright notice in zlib.h
 */
#include "zbuild.h"

#ifdef X86_AVX512_VBMI

#include "avx2_tables.h"
#include <immintrin.h>
#include "x86_intrins.h"

typedef __m256i chunk_t;
typedef __m128i halfchunk_t;
typedef __mmask32 mask_t;
typedef __mmask16 halfmask_t;

#define HAVE_CHUNKMEMSET_2
#define HAVE_CHUNKMEMSET_4
#define HAVE_CHUNKMEMSET_8
#define HAVE_CHUNKMEMSET_16
#define HAVE_CHUNKMEMSET_1
#define HAVE_CHUNK_MAG
#define HAVE_HALF_CHUNK
#define HAVE_MASKED_READWRITE
#define HAVE_CHUNKCOPY
#define HAVE_HALFCHUNKCOPY

//#define HAVE_CHUNKMEMSET

#define CHUNKMEMSET      chunkmemset_avx512_vbmi

static inline halfmask_t gen_half_mask(unsigned len) {
   return (halfmask_t)_bzhi_u32(0xFFFF, len); 
}

static inline mask_t gen_mask(unsigned len) {
   return (mask_t)_bzhi_u32(0xFFFFFFFF, len); 
}

static inline void chunkmemset_2(uint8_t *from, chunk_t *chunk) {
    int16_t tmp;
    memcpy(&tmp, from, sizeof(tmp));
    *chunk = _mm256_set1_epi16(tmp);
}

static inline void chunkmemset_4(uint8_t *from, chunk_t *chunk) {
    int32_t tmp;
    memcpy(&tmp, from, sizeof(tmp));
    *chunk = _mm256_set1_epi32(tmp);
}

static inline void chunkmemset_8(uint8_t *from, chunk_t *chunk) {
    int64_t tmp;
    memcpy(&tmp, from, sizeof(tmp));
    *chunk = _mm256_set1_epi64x(tmp);
}

static inline void chunkmemset_16(uint8_t *from, chunk_t *chunk) {
    *chunk = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)from));
}

static inline void loadchunk(uint8_t const *s, chunk_t *chunk) {
    *chunk = _mm256_loadu_si256((__m256i *)s);
}

static inline void storechunk(uint8_t *out, chunk_t *chunk) {
    _mm256_storeu_si256((__m256i *)out, *chunk);
}

static inline void storechunk_mask(uint8_t *out, mask_t mask, chunk_t *chunk) {
    _mm256_mask_storeu_epi8(out, mask, *chunk);
}

#if 0
static inline void* memcpy_erms(void* dst, void const* src, size_t len) {
    __asm__ volatile ("rep movsb" : "+D" (dst), "+S" (src), "+c" (len));
    return dst;
}

static inline uint8_t* CHUNKCOPY(uint8_t *out, uint8_t const *from, unsigned len) {
    Assert(len > 0, "chunkcopy should never have a length 0");

    memcpy_erms(out, from, len);
    out += len;

    return out;
}

static inline uint8_t* CHUNKMEMSET(uint8_t *out, uint8_t *from, unsigned len) {
    return CHUNKCOPY(out, from, len);
}
#endif

#if 1
static inline uint8_t* CHUNKCOPY(uint8_t *out, uint8_t const *from, unsigned len) {
    Assert(len > 0, "chunkcopy should never have a length 0");

    unsigned rem = len % sizeof(chunk_t);
    mask_t rem_mask = gen_mask(rem);

    /* Since this is only ever called if dist >= a chunk, we don't need a masked load */
    chunk_t chunk;
    loadchunk(from, &chunk);
    _mm256_mask_storeu_epi8(out, rem_mask, chunk);
    out += rem;
    from += rem;
    len -= rem;

    while (len) {
        loadchunk(from, &chunk);
        storechunk(out, &chunk);
        out += sizeof(chunk_t);
        from += sizeof(chunk_t);
        len -= sizeof(chunk_t);
    }

    return out;
}
#endif

static inline chunk_t GET_CHUNK_MAG(uint8_t *buf, uint32_t *chunk_rem, uint32_t dist) {
    lut_rem_pair lut_rem = perm_idx_lut[dist - 3];
    *chunk_rem = lut_rem.remval;

    /* While it would be cheaper to do a non-masked load, as masked loads of 8 bit values are
     * a bit more expensive than other types, we have to do a masked load as the avx512 versions of this
     * this will be called when the "from" buffer has < a chunk's worth of legal bounds */
    mask_t loadmask = gen_mask(dist);

    if (dist < 16) {
        __m128i in_vec = _mm_maskz_loadu_epi8((halfmask_t)loadmask, buf);
        /* It turns out icelake and above can do two of these 128 bit lane shuffles in a given cycle.
         * combine that the ability to do loads on multiple ports and it seems that 128bit operations
         * here are the winner */
        __m128i perm_vec0 = _mm_load_si128((__m128i*)(permute_table + lut_rem.idx));
        __m128i perm_vec1 = _mm_load_si128((__m128i*)(permute_table + lut_rem.idx + 16));
        __m128i shuf0 = _mm_shuffle_epi8(in_vec, perm_vec0);
        __m128i shuf1 = _mm_shuffle_epi8(in_vec, perm_vec1);
        return _mm256_inserti64x2(_mm256_castsi128_si256(shuf0), shuf1, 1);
    } else {
        /* The table is not explicitly set up for full permutations, as the front half the vector, if dist > 16, is
         * unaltered and implicitly sequential. However, we can take advantage of the single instruction, faster
         * variant of permute here. To do this, we must specify a blend mask to keep the original elements unperturbed. The
         * latter elements will be permuted according to the latter half of the permute vector. This is significantly cheaper
         * than building up a modified permutation vector where the first half is a normal sequence.  */
        __m256i in_vec = _mm256_maskz_loadu_epi8(loadmask, buf);
        mask_t shuf_mask = 0xFFFF0000;
        __m256i perm_vec = _mm256_loadu_si256((__m256i*)(permute_table + lut_rem.idx - 16));
        return _mm256_mask_permutexvar_epi8(in_vec, shuf_mask, perm_vec, in_vec);
    }
}

static inline void loadhalfchunk(uint8_t const *s, halfchunk_t *chunk) {
    *chunk = _mm_loadu_si128((__m128i *)s);
}

static inline void storehalfchunk(uint8_t *out, halfchunk_t *chunk) {
    _mm_storeu_si128((__m128i *)out, *chunk);
}

static inline chunk_t halfchunk2whole(halfchunk_t *chunk) {
    /* We zero extend mostly to appease some memory sanitizers. These bytes are ultimately
     * unlikely to be actually written or read from */
    return _mm256_zextsi128_si256(*chunk);
}

static inline halfchunk_t GET_HALFCHUNK_MAG(uint8_t *buf, uint32_t *chunk_rem, uint32_t dist) {
    lut_rem_pair lut_rem = perm_idx_lut[dist - 3];
    __m128i perm_vec, ret_vec;
    halfmask_t load_mask = gen_half_mask(dist);
    ret_vec = _mm_maskz_loadu_epi8(load_mask, buf);
    *chunk_rem = half_rem_vals[dist - 3];

    perm_vec = _mm_load_si128((__m128i*)(permute_table + lut_rem.idx));
    ret_vec = _mm_shuffle_epi8(ret_vec, perm_vec);

    return ret_vec;
}

static inline uint8_t* HALFCHUNKCOPY(uint8_t *out, uint8_t const *from, unsigned len) {
    Assert(len > 0, "chunkcopy should never have a length 0");

    unsigned rem = len % sizeof(halfchunk_t);
    halfmask_t rem_mask = gen_half_mask(rem);

    /* Since this is only ever called if dist >= a chunk, we don't need a masked load */
    halfchunk_t chunk;
    loadhalfchunk(from, &chunk);
    _mm_mask_storeu_epi8(out, rem_mask, chunk);
    out += rem;
    from += rem;
    len -= rem;

    while (len) {
        loadhalfchunk(from, &chunk);
        storehalfchunk(out, &chunk);
        out += sizeof(halfchunk_t);
        from += sizeof(halfchunk_t);
        len -= sizeof(halfchunk_t);
    }

    return out;
}

#define CHUNKSIZE        chunksize_avx512_vbmi
#define CHUNKUNROLL      chunkunroll_avx512_vbmi
#define CHUNKMEMSET_SAFE chunkmemset_safe_avx512_vbmi

#include "chunkset_tpl.h"

#define INFLATE_FAST     inflate_fast_avx512_vbmi

#include "inffast_tpl.h"

#endif
