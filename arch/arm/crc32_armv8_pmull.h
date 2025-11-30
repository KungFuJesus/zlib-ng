/* crc32_armv8_pmull.h -- Shared functions for hardware accelerated CRC-32 using ARMv8 PMULL
 * Copyright (C) 2025 Peter Cawley
 *   https://github.com/corsix/fast-crc32
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zbuild.h"

#include "acle_intrins.h"
#include "neon_intrins.h"

/* Carryless multiply low 64 bits: a[0] * b[0] */
static inline uint64x2_t clmul_lo(uint64x2_t a, uint64x2_t b) {
    return vreinterpretq_u64_p128(vmull_p64(
        vget_lane_p64(vreinterpret_p64_u64(vget_low_u64(a)), 0),
        vget_lane_p64(vreinterpret_p64_u64(vget_low_u64(b)), 0)));
}

/* Carryless multiply high 64 bits: a[1] * b[1] */
static inline uint64x2_t clmul_hi(uint64x2_t a, uint64x2_t b) {
    return vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(a), vreinterpretq_p64_u64(b)));
}

/* Carryless multiply of two 32-bit scalars: a * b (returns 64-bit result in 128-bit vector) */
static inline uint64x2_t clmul_scalar(uint32_t a, uint32_t b) {
  return vreinterpretq_u64_p128(vmull_p64((poly64_t)a, (poly64_t)b));
}

/* Compute x^n mod P (CRC-32 polynomial) in log(n) time, where P = 0x104c11db7 */
static uint32_t xnmodp(uint64_t n) {
  uint64_t stack = ~(uint64_t)1;
  uint32_t acc, low;
  for (; n > 191; n = (n >> 1) - 16) {
    stack = (stack << 1) + (n & 1);
  }
  stack = ~stack;
  acc = ((uint32_t)0x80000000) >> (n & 31);
  for (n >>= 5; n; --n) {
    acc = __crc32w(acc, 0);
  }
  while ((low = stack & 1), stack >>= 1) {
    poly8x8_t x = vreinterpret_p8_u64(vmov_n_u64(acc));
    uint64_t y = vgetq_lane_u64(vreinterpretq_u64_p16(vmull_p8(x, x)), 0);
    acc = __crc32d(0, y << low);
  }
  return acc;
}

/* Shift CRC forward by nbytes: equivalent to appending nbytes of zeros to the data stream */
static inline uint64x2_t crc_shift(uint32_t crc, size_t nbytes) {
  return clmul_scalar(crc, xnmodp(nbytes * 8 - 33));
}
