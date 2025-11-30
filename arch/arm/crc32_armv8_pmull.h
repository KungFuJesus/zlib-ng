/* crc32_armv8_pmull.h -- Shared functions for hardware accelerated CRC-32 using ARMv8 PMULL
 * Copyright (C) 2025 Peter Cawley
 *   https://github.com/corsix/fast-crc32
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zbuild.h"

#include <arm_acle.h>
#include <arm_neon.h>

static inline uint64x2_t clmul_scalar(uint32_t a, uint32_t b) {
  uint64x2_t r;
  __asm("pmull %0.1q, %1.1d, %2.1d\n" : "=w"(r) : "w"(vmovq_n_u64(a)), "w"(vmovq_n_u64(b)));
  return r;
}

static uint32_t xnmodp(uint64_t n) /* x^n mod P, in log(n) time */ {
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

static inline uint64x2_t crc_shift(uint32_t crc, size_t nbytes) {
  return clmul_scalar(crc, xnmodp(nbytes * 8 - 33));
}
