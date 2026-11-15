/* slide_hash_swar.c -- slide hash table SWAR implementation
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zbuild.h"
#include "arch_functions.h"

#ifdef SLIDE_HASH_SWAR_FALLBACK

#include "deflate.h"
#include <stdint.h>

/* Compile-time determination for 32-bit vs 64-bit native register width */
#if defined(UINTPTR_MAX) && UINTPTR_MAX > 0xFFFFFFFF
typedef uint64_t slide_word_t;
#define SLIDE_SWAR_MSB 0x8000800080008000ULL
#define SLIDE_SWAR_REP 0x0001000100010001ULL
#else
typedef uint32_t slide_word_t;
#define SLIDE_SWAR_MSB 0x80008000U
#define SLIDE_SWAR_REP 0x00010001U
#endif

/* Number of Pos (16-bit) elements packed into a single native register word */
#define SLIDE_ELEMS_PER_WORD (sizeof(slide_word_t) / sizeof(Pos))

Z_FORCEINLINE static slide_word_t slide_load_word(Pos *p) {
    slide_word_t v;
    memcpy(&v, HINT_ALIGNED(p, sizeof(slide_word_t)), sizeof(v));
    return v;
}

Z_FORCEINLINE static void slide_store_word(Pos *p, slide_word_t v) {
    memcpy(HINT_ALIGNED(p, sizeof(slide_word_t)), &v, sizeof(v));
}

/* Subtract 32768 with unsigned saturation across parallel 16-bit lanes.
 * Entries below 32768 become 0 and entries at or above 32768 clear bit 15. */
Z_FORCEINLINE static slide_word_t slide_swar_32k(slide_word_t v) {
    slide_word_t m = v & SLIDE_SWAR_MSB;
    return v & (m - (m >> 15));
}

/* Subtract wsize < 32768 with unsigned saturation across parallel 16-bit lanes.
 * Setting bit 15 before subtracting prevents borrows between adjacent lanes
 * and leaves bit 15 of t set iff the low 15 bits of the lane are >= wsize. */
Z_FORCEINLINE static slide_word_t slide_swar_lt32k(slide_word_t v, slide_word_t sw) {
    slide_word_t t = (v | SLIDE_SWAR_MSB) - sw;
    slide_word_t ge = (v | t) & SLIDE_SWAR_MSB;
    return t & ((ge - (ge >> 15)) | (v & SLIDE_SWAR_MSB));
}

/* Slide the hash table when sliding the window down, batch_size entries at a time
 * when the table is 8-byte aligned, as guaranteed by alloc_deflate.
 *
 * fill_window slides by W_BUF_SIZE(w_size) - w_size, which never exceeds
 * 32768, so larger values are only reachable from tests and simply take the
 * plain loop below.
 *
 * Not inlined, otherwise Clang duplicates and unrolls the loops at each of the
 * three call sites and roughly doubles the code size.
 */
Z_NOINLINE static void slide_hash_swar_chain(Pos *table, uint32_t entries, Pos wsize) {
    size_t align_mask = sizeof(slide_word_t) - 1;

    if (((uintptr_t)table & align_mask) == 0) {
        Pos *q = (Pos *)HINT_ALIGNED(table, sizeof(slide_word_t));
        uint32_t batch_size = SLIDE_ELEMS_PER_WORD * 2;

        if (wsize == 32768) {
            while (entries >= batch_size) {
                slide_word_t v0 = slide_load_word(q);
                slide_word_t v1 = slide_load_word(q + SLIDE_ELEMS_PER_WORD);
                slide_store_word(q, slide_swar_32k(v0));
                slide_store_word(q + SLIDE_ELEMS_PER_WORD, slide_swar_32k(v1));
                q += batch_size;
                entries -= batch_size;
            }
        } else if (wsize < 32768) {
            slide_word_t sw = (slide_word_t)wsize * SLIDE_SWAR_REP;
            while (entries >= batch_size) {
                slide_word_t v0 = slide_load_word(q);
                slide_word_t v1 = slide_load_word(q + SLIDE_ELEMS_PER_WORD);
                slide_store_word(q, slide_swar_lt32k(v0, sw));
                slide_store_word(q + SLIDE_ELEMS_PER_WORD, slide_swar_lt32k(v1, sw));
                q += batch_size;
                entries -= batch_size;
            }
        }
        table = q;
    }

    Pos *q = table;
    for (uint32_t i = 0; i < entries; i++) {
        Pos m = *q;
        *q++ = (m >= wsize) ? m - wsize : 0;
    }
    /* If entries is not on any hash chain, prev[entries] is garbage but
     * its value will never be used.
     */
}

Z_INTERNAL void slide_hash_swar(deflate_state *s) {
    Assert(s->slide_len <= UINT16_MAX, "slide_len should fit in uint16_t");
    Pos slide = (Pos)s->slide_len;

    slide_hash_swar_chain(s->head, HASH_SIZE, slide);
    slide_hash_swar_chain(s->prev, s->w_size, slide);
}

Z_INTERNAL void slide_hash_head_swar(deflate_state *s) {
    Assert(s->slide_len <= UINT16_MAX, "slide_len should fit in uint16_t");
    Pos slide = (Pos)s->slide_len;

    slide_hash_swar_chain(s->head, HASH_SIZE, slide);
}

#endif /* SLIDE_HASH_SWAR_FALLBACK */
