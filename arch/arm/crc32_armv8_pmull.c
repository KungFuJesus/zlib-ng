/* crc32_armv8_pmull.c -- ARMv8 CRC32 using PMULL for parallel folding
 * Copyright (C) 2025 Peter Cawley
 *   https://github.com/corsix/fast-crc32
 * For conditions of distribution and use, see copyright notice in zlib.h
 *
 * This uses a hybrid approach: 4-way parallel scalar CRC32 instructions
 * combined with 3 PMULL vector lanes for folding, processing 112 bytes/iter.
 */

#if defined(ARM_PMULL)
#include "zbuild.h"
#include "zutil.h"
#include "acle_intrins.h"
#include "crc32.h"
#include "crc32_armv8_pmull.h"

/* Carryless multiply low 64 bits, XOR with c: (a[0] * b[0]) ^ c */
static inline uint64x2_t clmul_lo_e(uint64x2_t a, uint64x2_t b, uint64x2_t c) {
    uint64x2_t r;
    __asm("pmull %0.1q, %2.1d, %3.1d\neor %0.16b, %0.16b, %1.16b\n"
          : "=w"(r), "+w"(c) : "w"(a), "w"(b));
    return r;
}

/* Carryless multiply high 64 bits, XOR with c: (a[1] * b[1]) ^ c */
static inline uint64x2_t clmul_hi_e(uint64x2_t a, uint64x2_t b, uint64x2_t c) {
    uint64x2_t r;
    __asm("pmull2 %0.1q, %2.2d, %3.2d\neor %0.16b, %0.16b, %1.16b\n"
          : "=w"(r), "+w"(c) : "w"(a), "w"(b));
    return r;
}

Z_INTERNAL Z_TARGET_PMULL uint32_t crc32_armv8_pmull(uint32_t crc, const uint8_t *buf, size_t len) {
    uint32_t crc0 = ~crc;

    /* Align to 8-byte boundary */
    for (; len && ((uintptr_t)buf & 7); --len)
        crc0 = __crc32b(crc0, *buf++);

    /* Align to 16-byte boundary */
    if (((uintptr_t)buf & 8) && len >= 8) {
        crc0 = __crc32d(crc0, *(const uint64_t*)buf);
        buf += 8;
        len -= 8;
    }

    /* Large buffer path: 4-way scalar CRC + 3-way PMULL folding (112 bytes/iter) */
    if (len >= 112) {
        const uint8_t *end = buf + len;
        size_t blk = len / 112;               /* Number of 112-byte blocks */
        size_t klen = blk * 16;                  /* Scalar stride per CRC lane */
        const uint8_t *buf2 = buf + klen * 4;    /* Vector data starts after scalar lanes */
        const uint8_t *limit = buf + klen - 32;
        uint32_t crc1 = 0, crc2 = 0, crc3 = 0;
        uint64x2_t vc0, vc1, vc2, vc3;
        uint64_t vc;

        /* Load first 3 vector chunks (48 bytes) */
        uint64x2_t x0 = vld1q_u64((const uint64_t*)buf2), y0;
        uint64x2_t x1 = vld1q_u64((const uint64_t*)(buf2 + 16)), y1;
        uint64x2_t x2 = vld1q_u64((const uint64_t*)(buf2 + 32)), y2;
        uint64x2_t k;
        /* k = {x^48 mod P, x^48+64 mod P} for 48-byte fold */
        { static const uint64_t ALIGNED_(16) k_[] = {0x3db1ecdc, 0xaf449247}; k = vld1q_u64(k_); }
        buf2 += 48;

        /* Main loop: fold vectors + 4-way parallel scalar CRC */
        while (buf <= limit) {
            /* Fold 3 vector lanes */
            y0 = clmul_lo_e(x0, k, vld1q_u64((const uint64_t*)buf2));
            x0 = clmul_hi_e(x0, k, y0);
            y1 = clmul_lo_e(x1, k, vld1q_u64((const uint64_t*)(buf2 + 16)));
            x1 = clmul_hi_e(x1, k, y1);
            y2 = clmul_lo_e(x2, k, vld1q_u64((const uint64_t*)(buf2 + 32)));
            x2 = clmul_hi_e(x2, k, y2);

            /* 4-way parallel scalar CRC (16 bytes each) */
            crc0 = __crc32d(crc0, *(const uint64_t*)buf);
            crc1 = __crc32d(crc1, *(const uint64_t*)(buf + klen));
            crc2 = __crc32d(crc2, *(const uint64_t*)(buf + klen * 2));
            crc3 = __crc32d(crc3, *(const uint64_t*)(buf + klen * 3));
            crc0 = __crc32d(crc0, *(const uint64_t*)(buf + 8));
            crc1 = __crc32d(crc1, *(const uint64_t*)(buf + klen + 8));
            crc2 = __crc32d(crc2, *(const uint64_t*)(buf + klen * 2 + 8));
            crc3 = __crc32d(crc3, *(const uint64_t*)(buf + klen * 3 + 8));
            buf += 16;
            buf2 += 48;
        }

        /* Reduce 3 vectors to 1: x0 = fold(x0, x1), then x0 = fold(x0, x2) */
        { static const uint64_t ALIGNED_(16) k_[] = {0xae689191, 0xccaa009e}; k = vld1q_u64(k_); }
        y0 = clmul_lo_e(x0, k, x1);
        x0 = clmul_hi_e(x0, k, y0);
        x1 = x2;
        y0 = clmul_lo_e(x0, k, x1);
        x0 = clmul_hi_e(x0, k, y0);

        /* Process final scalar chunk */
        crc0 = __crc32d(crc0, *(const uint64_t*)buf);
        crc1 = __crc32d(crc1, *(const uint64_t*)(buf + klen));
        crc2 = __crc32d(crc2, *(const uint64_t*)(buf + klen * 2));
        crc3 = __crc32d(crc3, *(const uint64_t*)(buf + klen * 3));
        crc0 = __crc32d(crc0, *(const uint64_t*)(buf + 8));
        crc1 = __crc32d(crc1, *(const uint64_t*)(buf + klen + 8));
        crc2 = __crc32d(crc2, *(const uint64_t*)(buf + klen * 2 + 8));
        crc3 = __crc32d(crc3, *(const uint64_t*)(buf + klen * 3 + 8));

        /* Shift and combine 4 scalar CRCs */
        vc0 = crc_shift(crc0, klen * 3 + blk * 48);
        vc1 = crc_shift(crc1, klen * 2 + blk * 48);
        vc2 = crc_shift(crc2, klen + blk * 48);
        vc3 = crc_shift(crc3, blk * 48);
        vc = vgetq_lane_u64(veorq_u64(veorq_u64(vc0, vc1), veorq_u64(vc2, vc3)), 0);

        /* Final reduction: 128-bit vector + scalar CRCs -> 32-bit */
        crc0 = __crc32d(0, vgetq_lane_u64(x0, 0));
        crc0 = __crc32d(crc0, vc ^ vgetq_lane_u64(x0, 1));
        buf = buf2;
        len = end - buf;
    }

    /* Medium buffer path: 2-way PMULL folding (32 bytes/iter) */
    if (len >= 32) {
        uint64x2_t x0 = vld1q_u64((const uint64_t*)buf), y0;
        uint64x2_t x1 = vld1q_u64((const uint64_t*)(buf + 16)), y1;
        uint64x2_t k;
        /* k = {x^32 mod P, x^32+64 mod P} for 32-byte fold */
        { static const uint64_t ALIGNED_(16) k_[] = {0xf1da05aa, 0x81256527}; k = vld1q_u64(k_); }
        x0 = veorq_u64((uint64x2_t){crc0, 0}, x0);  /* Mix in current CRC */
        buf += 32;
        len -= 32;

        /* Fold 32 bytes at a time */
        while (len >= 32) {
            y0 = clmul_lo_e(x0, k, vld1q_u64((const uint64_t*)buf));
            x0 = clmul_hi_e(x0, k, y0);
            y1 = clmul_lo_e(x1, k, vld1q_u64((const uint64_t*)(buf + 16)));
            x1 = clmul_hi_e(x1, k, y1);
            buf += 32;
            len -= 32;
        }

        /* Reduce 2 vectors to 1 */
        { static const uint64_t ALIGNED_(16) k_[] = {0xae689191, 0xccaa009e}; k = vld1q_u64(k_); }
        y0 = clmul_lo_e(x0, k, x1);
        x0 = clmul_hi_e(x0, k, y0);

        /* Final reduction: 128-bit -> 32-bit */
        crc0 = __crc32d(0, vgetq_lane_u64(x0, 0));
        crc0 = __crc32d(crc0, vgetq_lane_u64(x0, 1));
    }

    /* Process remaining 8-byte chunks */
    for (; len >= 8; buf += 8, len -= 8)
        crc0 = __crc32d(crc0, *(const uint64_t*)buf);

    /* Process remaining bytes */
    for (; len; --len)
        crc0 = __crc32b(crc0, *buf++);

    return ~crc0;
}

#endif
