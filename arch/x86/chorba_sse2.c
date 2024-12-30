#ifdef X86_CHORBA_SSE_CRC

#include "zbuild.h"
#include "crc32_braid_p.h"
#include "crc32_braid_tbl.h"
#include <emmintrin.h>
#include <smmintrin.h>

#define bitbuffersizebytes (16 * 1024 * sizeof(z_word_t))
#define bitbuffersizezwords (bitbuffersizebytes / sizeof(z_word_t))
#define bitbuffersizeqwords (bitbuffersizebytes / sizeof(uint64_t))

extern uint32_t crc32_braid_base(uint32_t c, const uint8_t *buf, size_t len);

#define READ_NEXT(in, off, a, b) do { \
        a = _mm_load_si128((__m128i*)(in + off / sizeof(uint64_t))); \
        b = _mm_load_si128((__m128i*)(in + off / sizeof(uint64_t) + 2)); \
        } while (0);

#define NEXT_ROUND(invec, a, b, c, d) do { \
        a = _mm_xor_si128(_mm_slli_epi64(invec, 17), _mm_slli_epi64(invec, 55)); \
        b = _mm_xor_si128(_mm_xor_si128(_mm_srli_epi64(invec, 47), _mm_srli_epi64(invec, 9)), _mm_slli_epi64(invec, 19)); \
        c = _mm_xor_si128(_mm_srli_epi64(invec, 45), _mm_slli_epi64(invec, 44)); \
        d  = _mm_srli_epi64(invec, 20); \
        } while (0);
    

Z_INTERNAL uint32_t chorba_118960_nondestructive_sse(uint32_t crc, const z_word_t* input, size_t len) {
    ALIGNED_(16) z_word_t bitbuffer[bitbuffersizezwords];
    const uint8_t* bitbufferbytes = (const uint8_t*) bitbuffer;

    size_t i = 0;

    z_word_t next1 = crc;
    z_word_t next2 = 0;
    z_word_t next3 = 0;
    z_word_t next4 = 0;
    z_word_t next5 = 0;
    z_word_t next6 = 0;
    z_word_t next7 = 0;
    z_word_t next8 = 0;
    z_word_t next9 = 0;
    z_word_t next10 = 0;
    z_word_t next11 = 0;
    z_word_t next12 = 0;
    z_word_t next13 = 0;
    z_word_t next14 = 0;
    z_word_t next15 = 0;
    z_word_t next16 = 0;
    z_word_t next17 = 0;
    z_word_t next18 = 0;
    z_word_t next19 = 0;
    z_word_t next20 = 0;
    z_word_t next21 = 0;
    z_word_t next22 = 0;
    crc = 0;

    // do a first pass to zero out bitbuffer
    for(; i < (14848 * sizeof(z_word_t)); i += (32 * sizeof(z_word_t))) {
        z_word_t in1, in2, in3, in4, in5, in6, in7, in8;
        z_word_t in9, in10, in11, in12, in13, in14, in15, in16;
        z_word_t in17, in18, in19, in20, in21, in22, in23, in24;
        z_word_t in25, in26, in27, in28, in29, in30, in31, in32;
        int outoffset1 = ((i / sizeof(z_word_t)) + 14848) % bitbuffersizezwords;
        int outoffset2 = ((i / sizeof(z_word_t)) + 14880) % bitbuffersizezwords;

        in1 = input[i / sizeof(z_word_t) + 0] ^ next1;
        in2 = input[i / sizeof(z_word_t) + 1] ^ next2;
        in3 = input[i / sizeof(z_word_t) + 2] ^ next3;
        in4 = input[i / sizeof(z_word_t) + 3] ^ next4;
        in5 = input[i / sizeof(z_word_t) + 4] ^ next5;
        in6 = input[i / sizeof(z_word_t) + 5] ^ next6;
        in7 = input[i / sizeof(z_word_t) + 6] ^ next7;
        in8 = input[i / sizeof(z_word_t) + 7] ^ next8 ^ in1;
        in9 = input[i / sizeof(z_word_t) + 8] ^ next9 ^ in2;
        in10 = input[i / sizeof(z_word_t) + 9] ^ next10 ^ in3;
        in11 = input[i / sizeof(z_word_t) + 10] ^ next11 ^ in4;
        in12 = input[i / sizeof(z_word_t) + 11] ^ next12 ^ in1 ^ in5;
        in13 = input[i / sizeof(z_word_t) + 12] ^ next13 ^ in2 ^ in6;
        in14 = input[i / sizeof(z_word_t) + 13] ^ next14 ^ in3 ^ in7;
        in15 = input[i / sizeof(z_word_t) + 14] ^ next15 ^ in4 ^ in8;
        in16 = input[i / sizeof(z_word_t) + 15] ^ next16 ^ in5 ^ in9;
        in17 = input[i / sizeof(z_word_t) + 16] ^ next17 ^ in6 ^ in10;
        in18 = input[i / sizeof(z_word_t) + 17] ^ next18 ^ in7 ^ in11;
        in19 = input[i / sizeof(z_word_t) + 18] ^ next19 ^ in8 ^ in12;
        in20 = input[i / sizeof(z_word_t) + 19] ^ next20 ^ in9 ^ in13;
        in21 = input[i / sizeof(z_word_t) + 20] ^ next21 ^ in10 ^ in14;
        in22 = input[i / sizeof(z_word_t) + 21] ^ next22 ^ in11 ^ in15;
        in23 = input[i / sizeof(z_word_t) + 22] ^ in1 ^ in12 ^ in16;
        in24 = input[i / sizeof(z_word_t) + 23] ^ in2 ^ in13 ^ in17;
        in25 = input[i / sizeof(z_word_t) + 24] ^ in3 ^ in14 ^ in18;
        in26 = input[i / sizeof(z_word_t) + 25] ^ in4 ^ in15 ^ in19;
        in27 = input[i / sizeof(z_word_t) + 26] ^ in5 ^ in16 ^ in20;
        in28 = input[i / sizeof(z_word_t) + 27] ^ in6 ^ in17 ^ in21;
        in29 = input[i / sizeof(z_word_t) + 28] ^ in7 ^ in18 ^ in22;
        in30 = input[i / sizeof(z_word_t) + 29] ^ in8 ^ in19 ^ in23;
        in31 = input[i / sizeof(z_word_t) + 30] ^ in9 ^ in20 ^ in24;
        in32 = input[i / sizeof(z_word_t) + 31] ^ in10 ^ in21 ^ in25;

        next1 = in11 ^ in22 ^ in26;
        next2 = in12 ^ in23 ^ in27;
        next3 = in13 ^ in24 ^ in28;
        next4 = in14 ^ in25 ^ in29;
        next5 = in15 ^ in26 ^ in30;
        next6 = in16 ^ in27 ^ in31;
        next7 = in17 ^ in28 ^ in32;
        next8 = in18 ^ in29;
        next9 = in19 ^ in30;
        next10 = in20 ^ in31;
        next11 = in21 ^ in32;
        next12 = in22;
        next13 = in23;
        next14 = in24;
        next15 = in25;
        next16 = in26;
        next17 = in27;
        next18 = in28;
        next19 = in29;
        next20 = in30;
        next21 = in31;
        next22 = in32;

        bitbuffer[outoffset1 + 22] = in1;
        bitbuffer[outoffset1 + 23] = in2;
        bitbuffer[outoffset1 + 24] = in3;
        bitbuffer[outoffset1 + 25] = in4;
        bitbuffer[outoffset1 + 26] = in5;
        bitbuffer[outoffset1 + 27] = in6;
        bitbuffer[outoffset1 + 28] = in7;
        bitbuffer[outoffset1 + 29] = in8;
        bitbuffer[outoffset1 + 30] = in9;
        bitbuffer[outoffset1 + 31] = in10;
        bitbuffer[outoffset2 + 0] = in11;
        bitbuffer[outoffset2 + 1] = in12;
        bitbuffer[outoffset2 + 2] = in13;
        bitbuffer[outoffset2 + 3] = in14;
        bitbuffer[outoffset2 + 4] = in15;
        bitbuffer[outoffset2 + 5] = in16;
        bitbuffer[outoffset2 + 6] = in17;
        bitbuffer[outoffset2 + 7] = in18;
        bitbuffer[outoffset2 + 8] = in19;
        bitbuffer[outoffset2 + 9] = in20;
        bitbuffer[outoffset2 + 10] = in21;
        bitbuffer[outoffset2 + 11] = in22;
        bitbuffer[outoffset2 + 12] = in23;
        bitbuffer[outoffset2 + 13] = in24;
        bitbuffer[outoffset2 + 14] = in25;
        bitbuffer[outoffset2 + 15] = in26;
        bitbuffer[outoffset2 + 16] = in27;
        bitbuffer[outoffset2 + 17] = in28;
        bitbuffer[outoffset2 + 18] = in29;
        bitbuffer[outoffset2 + 19] = in30;
        bitbuffer[outoffset2 + 20] = in31;
        bitbuffer[outoffset2 + 21] = in32;
    }

    // one intermediate pass where we pull half the values
    for(; i < (14880 * sizeof(z_word_t)); i += (32 * sizeof(z_word_t))) {
        z_word_t in1, in2, in3, in4, in5, in6, in7, in8;
        z_word_t in9, in10, in11, in12, in13, in14, in15, in16;
        z_word_t in17, in18, in19, in20, in21, in22, in23, in24;
        z_word_t in25, in26, in27, in28, in29, in30, in31, in32;
        int inoffset = (i / sizeof(z_word_t)) % bitbuffersizezwords;
        int outoffset1 = ((i / sizeof(z_word_t)) + 14848) % bitbuffersizezwords;
        int outoffset2 = ((i / sizeof(z_word_t)) + 14880) % bitbuffersizezwords;

        in1 = input[i / sizeof(z_word_t) + 0] ^ next1;
        in2 = input[i / sizeof(z_word_t) + 1] ^ next2;
        in3 = input[i / sizeof(z_word_t) + 2] ^ next3;
        in4 = input[i / sizeof(z_word_t) + 3] ^ next4;
        in5 = input[i / sizeof(z_word_t) + 4] ^ next5;
        in6 = input[i / sizeof(z_word_t) + 5] ^ next6;
        in7 = input[i / sizeof(z_word_t) + 6] ^ next7;
        in8 = input[i / sizeof(z_word_t) + 7] ^ next8 ^ in1;
        in9 = input[i / sizeof(z_word_t) + 8] ^ next9 ^ in2;
        in10 = input[i / sizeof(z_word_t) + 9] ^ next10 ^ in3;
        in11 = input[i / sizeof(z_word_t) + 10] ^ next11 ^ in4;
        in12 = input[i / sizeof(z_word_t) + 11] ^ next12 ^ in1 ^ in5;
        in13 = input[i / sizeof(z_word_t) + 12] ^ next13 ^ in2 ^ in6;
        in14 = input[i / sizeof(z_word_t) + 13] ^ next14 ^ in3 ^ in7;
        in15 = input[i / sizeof(z_word_t) + 14] ^ next15 ^ in4 ^ in8;
        in16 = input[i / sizeof(z_word_t) + 15] ^ next16 ^ in5 ^ in9;
        in17 = input[i / sizeof(z_word_t) + 16] ^ next17 ^ in6 ^ in10;
        in18 = input[i / sizeof(z_word_t) + 17] ^ next18 ^ in7 ^ in11;
        in19 = input[i / sizeof(z_word_t) + 18] ^ next19 ^ in8 ^ in12;
        in20 = input[i / sizeof(z_word_t) + 19] ^ next20 ^ in9 ^ in13;
        in21 = input[i / sizeof(z_word_t) + 20] ^ next21 ^ in10 ^ in14;
        in22 = input[i / sizeof(z_word_t) + 21] ^ next22 ^ in11 ^ in15;
        in23 = input[i / sizeof(z_word_t) + 22] ^ in1 ^ in12 ^ in16 ^ bitbuffer[inoffset + 22];
        in24 = input[i / sizeof(z_word_t) + 23] ^ in2 ^ in13 ^ in17 ^ bitbuffer[inoffset + 23];
        in25 = input[i / sizeof(z_word_t) + 24] ^ in3 ^ in14 ^ in18 ^ bitbuffer[inoffset + 24];
        in26 = input[i / sizeof(z_word_t) + 25] ^ in4 ^ in15 ^ in19 ^ bitbuffer[inoffset + 25];
        in27 = input[i / sizeof(z_word_t) + 26] ^ in5 ^ in16 ^ in20 ^ bitbuffer[inoffset + 26];
        in28 = input[i / sizeof(z_word_t) + 27] ^ in6 ^ in17 ^ in21 ^ bitbuffer[inoffset + 27];
        in29 = input[i / sizeof(z_word_t) + 28] ^ in7 ^ in18 ^ in22 ^ bitbuffer[inoffset + 28];
        in30 = input[i / sizeof(z_word_t) + 29] ^ in8 ^ in19 ^ in23 ^ bitbuffer[inoffset + 29];
        in31 = input[i / sizeof(z_word_t) + 30] ^ in9 ^ in20 ^ in24 ^ bitbuffer[inoffset + 30];
        in32 = input[i / sizeof(z_word_t) + 31] ^ in10 ^ in21 ^ in25 ^ bitbuffer[inoffset + 31];

        next1 = in11 ^ in22 ^ in26;
        next2 = in12 ^ in23 ^ in27;
        next3 = in13 ^ in24 ^ in28;
        next4 = in14 ^ in25 ^ in29;
        next5 = in15 ^ in26 ^ in30;
        next6 = in16 ^ in27 ^ in31;
        next7 = in17 ^ in28 ^ in32;
        next8 = in18 ^ in29;
        next9 = in19 ^ in30;
        next10 = in20 ^ in31;
        next11 = in21 ^ in32;
        next12 = in22;
        next13 = in23;
        next14 = in24;
        next15 = in25;
        next16 = in26;
        next17 = in27;
        next18 = in28;
        next19 = in29;
        next20 = in30;
        next21 = in31;
        next22 = in32;

        bitbuffer[outoffset1 + 22] = in1;
        bitbuffer[outoffset1 + 23] = in2;
        bitbuffer[outoffset1 + 24] = in3;
        bitbuffer[outoffset1 + 25] = in4;
        bitbuffer[outoffset1 + 26] = in5;
        bitbuffer[outoffset1 + 27] = in6;
        bitbuffer[outoffset1 + 28] = in7;
        bitbuffer[outoffset1 + 29] = in8;
        bitbuffer[outoffset1 + 30] = in9;
        bitbuffer[outoffset1 + 31] = in10;
        bitbuffer[outoffset2 + 0] = in11;
        bitbuffer[outoffset2 + 1] = in12;
        bitbuffer[outoffset2 + 2] = in13;
        bitbuffer[outoffset2 + 3] = in14;
        bitbuffer[outoffset2 + 4] = in15;
        bitbuffer[outoffset2 + 5] = in16;
        bitbuffer[outoffset2 + 6] = in17;
        bitbuffer[outoffset2 + 7] = in18;
        bitbuffer[outoffset2 + 8] = in19;
        bitbuffer[outoffset2 + 9] = in20;
        bitbuffer[outoffset2 + 10] = in21;
        bitbuffer[outoffset2 + 11] = in22;
        bitbuffer[outoffset2 + 12] = in23;
        bitbuffer[outoffset2 + 13] = in24;
        bitbuffer[outoffset2 + 14] = in25;
        bitbuffer[outoffset2 + 15] = in26;
        bitbuffer[outoffset2 + 16] = in27;
        bitbuffer[outoffset2 + 17] = in28;
        bitbuffer[outoffset2 + 18] = in29;
        bitbuffer[outoffset2 + 19] = in30;
        bitbuffer[outoffset2 + 20] = in31;
        bitbuffer[outoffset2 + 21] = in32;
    }

    for(; (i + (14870 + 64) * sizeof(z_word_t)) < len; i += (32 * sizeof(z_word_t))) {
        z_word_t in1, in2, in3, in4, in5, in6, in7, in8;
        z_word_t in9, in10, in11, in12, in13, in14, in15, in16;
        z_word_t in17, in18, in19, in20, in21, in22, in23, in24;
        z_word_t in25, in26, in27, in28, in29, in30, in31, in32;
        int inoffset = (i / sizeof(z_word_t)) % bitbuffersizezwords;
        int outoffset1 = ((i / sizeof(z_word_t)) + 14848) % bitbuffersizezwords;
        int outoffset2 = ((i / sizeof(z_word_t)) + 14880) % bitbuffersizezwords;

        in1 = input[i / sizeof(z_word_t) + 0] ^ next1 ^ bitbuffer[inoffset + 0];
        in2 = input[i / sizeof(z_word_t) + 1] ^ next2 ^ bitbuffer[inoffset + 1];
        in3 = input[i / sizeof(z_word_t) + 2] ^ next3 ^ bitbuffer[inoffset + 2];
        in4 = input[i / sizeof(z_word_t) + 3] ^ next4 ^ bitbuffer[inoffset + 3];
        in5 = input[i / sizeof(z_word_t) + 4] ^ next5 ^ bitbuffer[inoffset + 4];
        in6 = input[i / sizeof(z_word_t) + 5] ^ next6 ^ bitbuffer[inoffset + 5];
        in7 = input[i / sizeof(z_word_t) + 6] ^ next7 ^ bitbuffer[inoffset + 6];
        in8 = input[i / sizeof(z_word_t) + 7] ^ next8 ^ in1 ^ bitbuffer[inoffset + 7];
        in9 = input[i / sizeof(z_word_t) + 8] ^ next9 ^ in2 ^ bitbuffer[inoffset + 8];
        in10 = input[i / sizeof(z_word_t) + 9] ^ next10 ^ in3 ^ bitbuffer[inoffset + 9];
        in11 = input[i / sizeof(z_word_t) + 10] ^ next11 ^ in4 ^ bitbuffer[inoffset + 10];
        in12 = input[i / sizeof(z_word_t) + 11] ^ next12 ^ in1 ^ in5 ^ bitbuffer[inoffset + 11];
        in13 = input[i / sizeof(z_word_t) + 12] ^ next13 ^ in2 ^ in6 ^ bitbuffer[inoffset + 12];
        in14 = input[i / sizeof(z_word_t) + 13] ^ next14 ^ in3 ^ in7 ^ bitbuffer[inoffset + 13];
        in15 = input[i / sizeof(z_word_t) + 14] ^ next15 ^ in4 ^ in8 ^ bitbuffer[inoffset + 14];
        in16 = input[i / sizeof(z_word_t) + 15] ^ next16 ^ in5 ^ in9 ^ bitbuffer[inoffset + 15];
        in17 = input[i / sizeof(z_word_t) + 16] ^ next17 ^ in6 ^ in10 ^ bitbuffer[inoffset + 16];
        in18 = input[i / sizeof(z_word_t) + 17] ^ next18 ^ in7 ^ in11 ^ bitbuffer[inoffset + 17];
        in19 = input[i / sizeof(z_word_t) + 18] ^ next19 ^ in8 ^ in12 ^ bitbuffer[inoffset + 18];
        in20 = input[i / sizeof(z_word_t) + 19] ^ next20 ^ in9 ^ in13 ^ bitbuffer[inoffset + 19];
        in21 = input[i / sizeof(z_word_t) + 20] ^ next21 ^ in10 ^ in14 ^ bitbuffer[inoffset + 20];
        in22 = input[i / sizeof(z_word_t) + 21] ^ next22 ^ in11 ^ in15 ^ bitbuffer[inoffset + 21];
        in23 = input[i / sizeof(z_word_t) + 22] ^ in1 ^ in12 ^ in16 ^ bitbuffer[inoffset + 22];
        in24 = input[i / sizeof(z_word_t) + 23] ^ in2 ^ in13 ^ in17 ^ bitbuffer[inoffset + 23];
        in25 = input[i / sizeof(z_word_t) + 24] ^ in3 ^ in14 ^ in18 ^ bitbuffer[inoffset + 24];
        in26 = input[i / sizeof(z_word_t) + 25] ^ in4 ^ in15 ^ in19 ^ bitbuffer[inoffset + 25];
        in27 = input[i / sizeof(z_word_t) + 26] ^ in5 ^ in16 ^ in20 ^ bitbuffer[inoffset + 26];
        in28 = input[i / sizeof(z_word_t) + 27] ^ in6 ^ in17 ^ in21 ^ bitbuffer[inoffset + 27];
        in29 = input[i / sizeof(z_word_t) + 28] ^ in7 ^ in18 ^ in22 ^ bitbuffer[inoffset + 28];
        in30 = input[i / sizeof(z_word_t) + 29] ^ in8 ^ in19 ^ in23 ^ bitbuffer[inoffset + 29];
        in31 = input[i / sizeof(z_word_t) + 30] ^ in9 ^ in20 ^ in24 ^ bitbuffer[inoffset + 30];
        in32 = input[i / sizeof(z_word_t) + 31] ^ in10 ^ in21 ^ in25 ^ bitbuffer[inoffset + 31];

        next1 = in11 ^ in22 ^ in26;
        next2 = in12 ^ in23 ^ in27;
        next3 = in13 ^ in24 ^ in28;
        next4 = in14 ^ in25 ^ in29;
        next5 = in15 ^ in26 ^ in30;
        next6 = in16 ^ in27 ^ in31;
        next7 = in17 ^ in28 ^ in32;
        next8 = in18 ^ in29;
        next9 = in19 ^ in30;
        next10 = in20 ^ in31;
        next11 = in21 ^ in32;
        next12 = in22;
        next13 = in23;
        next14 = in24;
        next15 = in25;
        next16 = in26;
        next17 = in27;
        next18 = in28;
        next19 = in29;
        next20 = in30;
        next21 = in31;
        next22 = in32;

        bitbuffer[outoffset1 + 22] = in1;
        bitbuffer[outoffset1 + 23] = in2;
        bitbuffer[outoffset1 + 24] = in3;
        bitbuffer[outoffset1 + 25] = in4;
        bitbuffer[outoffset1 + 26] = in5;
        bitbuffer[outoffset1 + 27] = in6;
        bitbuffer[outoffset1 + 28] = in7;
        bitbuffer[outoffset1 + 29] = in8;
        bitbuffer[outoffset1 + 30] = in9;
        bitbuffer[outoffset1 + 31] = in10;
        bitbuffer[outoffset2 + 0] = in11;
        bitbuffer[outoffset2 + 1] = in12;
        bitbuffer[outoffset2 + 2] = in13;
        bitbuffer[outoffset2 + 3] = in14;
        bitbuffer[outoffset2 + 4] = in15;
        bitbuffer[outoffset2 + 5] = in16;
        bitbuffer[outoffset2 + 6] = in17;
        bitbuffer[outoffset2 + 7] = in18;
        bitbuffer[outoffset2 + 8] = in19;
        bitbuffer[outoffset2 + 9] = in20;
        bitbuffer[outoffset2 + 10] = in21;
        bitbuffer[outoffset2 + 11] = in22;
        bitbuffer[outoffset2 + 12] = in23;
        bitbuffer[outoffset2 + 13] = in24;
        bitbuffer[outoffset2 + 14] = in25;
        bitbuffer[outoffset2 + 15] = in26;
        bitbuffer[outoffset2 + 16] = in27;
        bitbuffer[outoffset2 + 17] = in28;
        bitbuffer[outoffset2 + 18] = in29;
        bitbuffer[outoffset2 + 19] = in30;
        bitbuffer[outoffset2 + 20] = in31;
        bitbuffer[outoffset2 + 21] = in32;
    }

    bitbuffer[(i / sizeof(z_word_t) + 0) % bitbuffersizezwords] ^= next1;
    bitbuffer[(i / sizeof(z_word_t) + 1) % bitbuffersizezwords] ^= next2;
    bitbuffer[(i / sizeof(z_word_t) + 2) % bitbuffersizezwords] ^= next3;
    bitbuffer[(i / sizeof(z_word_t) + 3) % bitbuffersizezwords] ^= next4;
    bitbuffer[(i / sizeof(z_word_t) + 4) % bitbuffersizezwords] ^= next5;
    bitbuffer[(i / sizeof(z_word_t) + 5) % bitbuffersizezwords] ^= next6;
    bitbuffer[(i / sizeof(z_word_t) + 6) % bitbuffersizezwords] ^= next7;
    bitbuffer[(i / sizeof(z_word_t) + 7) % bitbuffersizezwords] ^= next8;
    bitbuffer[(i / sizeof(z_word_t) + 8) % bitbuffersizezwords] ^= next9;
    bitbuffer[(i / sizeof(z_word_t) + 9) % bitbuffersizezwords] ^= next10;
    bitbuffer[(i / sizeof(z_word_t) + 10) % bitbuffersizezwords] ^= next11;
    bitbuffer[(i / sizeof(z_word_t) + 11) % bitbuffersizezwords] ^= next12;
    bitbuffer[(i / sizeof(z_word_t) + 12) % bitbuffersizezwords] ^= next13;
    bitbuffer[(i / sizeof(z_word_t) + 13) % bitbuffersizezwords] ^= next14;
    bitbuffer[(i / sizeof(z_word_t) + 14) % bitbuffersizezwords] ^= next15;
    bitbuffer[(i / sizeof(z_word_t) + 15) % bitbuffersizezwords] ^= next16;
    bitbuffer[(i / sizeof(z_word_t) + 16) % bitbuffersizezwords] ^= next17;
    bitbuffer[(i / sizeof(z_word_t) + 17) % bitbuffersizezwords] ^= next18;
    bitbuffer[(i / sizeof(z_word_t) + 18) % bitbuffersizezwords] ^= next19;
    bitbuffer[(i / sizeof(z_word_t) + 19) % bitbuffersizezwords] ^= next20;
    bitbuffer[(i / sizeof(z_word_t) + 20) % bitbuffersizezwords] ^= next21;
    bitbuffer[(i / sizeof(z_word_t) + 21) % bitbuffersizezwords] ^= next22;

    for (int j = 14870; j < 14870 + 60; j++) {
        bitbuffer[(j + (i / sizeof(z_word_t))) % bitbuffersizezwords] = 0;
    }

    uint64_t next1_64 = 0;
    uint64_t next2_64 = 0;
    uint64_t next3_64 = 0;
    uint64_t next4_64 = 0;
    uint64_t next5_64 = 0;
    uint64_t final[9] = {0};

    for(; (i + 72 < len); i += 32) {
        uint64_t in1;
        uint64_t in2;
        uint64_t in3;
        uint64_t in4;
        uint64_t a1, a2, a3, a4;
        uint64_t b1, b2, b3, b4;
        uint64_t c1, c2, c3, c4;
        uint64_t d1, d2, d3, d4;

        uint64_t out1;
        uint64_t out2;
        uint64_t out3;
        uint64_t out4;
        uint64_t out5;

        in1 = input[i / sizeof(z_word_t)] ^ bitbuffer[(i / sizeof(uint64_t)) % bitbuffersizeqwords];
        in2 = input[(i + 8) / sizeof(z_word_t)] ^ bitbuffer[(i / sizeof(uint64_t) + 1) % bitbuffersizeqwords];

        in1 ^= next1_64;
        in2 ^= next2_64;

        a1 = (in1 << 17) ^ (in1 << 55);
        a2 = (in1 >> 47) ^ (in1 >> 9) ^ (in1 << 19);
        a3 = (in1 >> 45) ^ (in1 << 44);
        a4 = (in1 >> 20);
        
        b1 = (in2 << 17) ^ (in2 << 55);
        b2 = (in2 >> 47) ^ (in2 >> 9) ^ (in2 << 19);
        b3 = (in2 >> 45) ^ (in2 << 44);
        b4 = (in2 >> 20);

        in3 = input[(i + 16) / sizeof(z_word_t)] ^ bitbuffer[(i / sizeof(uint64_t) + 2) % bitbuffersizeqwords];
        in4 = input[(i + 24) / sizeof(z_word_t)] ^ bitbuffer[(i / sizeof(uint64_t) + 3) % bitbuffersizeqwords];

        in3 ^= next3_64 ^ a1;
        in4 ^= next4_64 ^ a2 ^ b1;

        c1 = (in3 << 17) ^ (in3 << 55);
        c2 = (in3 >> 47) ^ (in3 >> 9) ^ (in3 << 19);
        c3 = (in3 >> 45) ^ (in3 << 44);
        c4 = (in3 >> 20);
        
        d1 = (in4 << 17) ^ (in4 << 55);
        d2 = (in4 >> 47) ^ (in4 >> 9) ^ (in4 << 19);
        d3 = (in4 >> 45) ^ (in4 << 44);
        d4 = (in4 >> 20);

        out1 = a3 ^ b2 ^ c1;
        out2 = a4 ^ b3 ^ c2 ^ d1;
        out3 = b4 ^ c3 ^ d2;
        out4 = c4 ^ d3;
        out5 = d4;

        next1_64 = next5_64 ^ out1;
        next2_64 = out2;
        next3_64 = out3;
        next4_64 = out4;
        next5_64 = out5;

    }

    memcpy(final, input+(i / sizeof(uint64_t)), len-i);
    final[0] ^= next1_64;
    final[1] ^= next2_64;
    final[2] ^= next3_64;
    final[3] ^= next4_64;
    final[4] ^= next5_64;

    uint8_t* final_bytes = (uint8_t*) final;

    for(size_t j = 0; j < (len-i); j++) {
        crc = crc_table[(crc ^ final_bytes[j] ^ bitbufferbytes[(j+i) % bitbuffersizebytes]) & 0xff] ^ (crc >> 8);
    }

    return crc;

}

Z_INTERNAL uint32_t chorba_small_nondestructive_sse(uint32_t crc, const uint64_t* buf, size_t len) {
    const uint64_t* input = buf;
    uint64_t final[9] = {0};
    uint64_t next1 = crc;
    crc = 0;
    uint64_t next2 = 0;
    uint64_t next3 = 0;
    uint64_t next4 = 0;
    uint64_t next5 = 0;

    __m128i next12 = _mm_cvtsi64x_si128(next1);
    __m128i next34 = _mm_setzero_si128();
    __m128i next56 = _mm_setzero_si128();

    __m128i ab1, ab2, ab3, ab4, cd1, cd2, cd3, cd4;

    size_t i = 0;

    /* This is weird, doing for vs while drops 10% off the exec time */
    for(; (i + 256 + 40 + 32 + 32) < len; i += 32) {
        __m128i in1in2, in3in4;

        /*
        uint64_t chorba1 = input[i / sizeof(uint64_t)];
        uint64_t chorba2 = input[i / sizeof(uint64_t) + 1];
        uint64_t chorba3 = input[i / sizeof(uint64_t) + 2];
        uint64_t chorba4 = input[i / sizeof(uint64_t) + 3];
        uint64_t chorba5 = input[i / sizeof(uint64_t) + 4];
        uint64_t chorba6 = input[i / sizeof(uint64_t) + 5];
        uint64_t chorba7 = input[i / sizeof(uint64_t) + 6];
        uint64_t chorba8 = input[i / sizeof(uint64_t) + 7];
        */

        const uint64_t *inputPtr = input + (i / sizeof(uint64_t));
        const __m128i *inputPtr128 = (__m128i*)inputPtr;
        __m128i chorba12 = _mm_load_si128(inputPtr128++);
        __m128i chorba34 = _mm_load_si128(inputPtr128++);
        __m128i chorba56 = _mm_load_si128(inputPtr128++);
        __m128i chorba78 = _mm_load_si128(inputPtr128++);

        chorba12 = _mm_xor_si128(chorba12, next12);
        chorba34 = _mm_xor_si128(chorba34, next34);
        chorba56 = _mm_xor_si128(chorba56, next56);
        chorba78 = _mm_xor_si128(chorba78, chorba12);
        __m128i chorba45 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(chorba34), _mm_castsi128_pd(chorba56), 1));
        __m128i chorba23 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(chorba12),
                                                           _mm_castsi128_pd(chorba34), 1));
        /*
        chorba1 ^= next1;
        chorba2 ^= next2;
        chorba3 ^= next3;
        chorba4 ^= next4;
        chorba5 ^= next5;
        chorba7 ^= chorba1;
        chorba8 ^= chorba2;
        */
        i += 8 * 8;

        /* 0-3 */
        /*in1 = input[i / sizeof(uint64_t)];
        in2 = input[i / sizeof(uint64_t) + 1];*/
        READ_NEXT(input, i, in1in2, in3in4);
        __m128i chorba34xor = _mm_xor_si128(chorba34, _mm_unpacklo_epi64(_mm_setzero_si128(), chorba12));
        in1in2 = _mm_xor_si128(in1in2, chorba34xor);
        /*
        in1 ^= chorba3;
        in2 ^= chorba4 ^ chorba1;
        */

        NEXT_ROUND(in1in2, ab1, ab2, ab3, ab4);
        /*
        a1 = (in1 << 17) ^ (in1 << 55);
        a2 = (in1 >> 47) ^ (in1 >> 9) ^ (in1 << 19);
        a3 = (in1 >> 45) ^ (in1 << 44);
        a4 = (in1 >> 20);
        
        b1 = (in2 << 17) ^ (in2 << 55);
        b2 = (in2 >> 47) ^ (in2 >> 9) ^ (in2 << 19);
        b3 = (in2 >> 45) ^ (in2 << 44);
        b4 = (in2 >> 20);

        */

        in3in4 = _mm_xor_si128(in3in4, ab1);
        /* _hopefully_ we don't get a huge domain switching penalty for this. This seems to be the best sequence */
        __m128i chorba56xor = _mm_xor_si128(chorba56, _mm_unpacklo_epi64(_mm_setzero_si128(), ab2)); 

        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(chorba56xor, chorba23));
        in3in4 = _mm_xor_si128(in3in4, chorba12);

        NEXT_ROUND(in3in4, cd1, cd2, cd3, cd4);
        /*
        in3 = input[i / sizeof(uint64_t) + 2];
        in4 = input[i / sizeof(uint64_t) + 3];
        in3 ^= a1 ^ chorba5 ^ chorba2 ^ chorba1;
        in4 ^= b1 ^a2 ^ chorba6 ^ chorba3 ^ chorba2;

        c1 = (in3 << 17) ^ (in3 << 55);
        c2 = (in3 >> 47) ^ (in3 >> 9) ^ (in3 << 19);
        c3 = (in3 >> 45) ^ (in3 << 44);
        c4 = (in3 >> 20);
        
        d1 = (in4 << 17) ^ (in4 << 55);
        d2 = (in4 >> 47) ^ (in4 >> 9) ^ (in4 << 19);
        d3 = (in4 >> 45) ^ (in4 << 44);
        d4 = (in4 >> 20);
        */

        __m128i b2c2 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab2), _mm_castsi128_pd(cd2), 1)); 
        __m128i a4_ = _mm_unpacklo_epi64(_mm_setzero_si128(), ab4);
        a4_ = _mm_xor_si128(b2c2, a4_);
        next12 = _mm_xor_si128(ab3, a4_);
        next12 = _mm_xor_si128(next12, cd1);

        __m128i d2_ = _mm_unpackhi_epi64(cd2, _mm_setzero_si128());
        __m128i b4c4 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab4), _mm_castsi128_pd(cd4), 1));

        /*out1 = a3 ^ b2 ^ c1;
        out2 = b3 ^ c2 ^ d1 ^ a4;*/
        next34 = _mm_xor_si128(cd3, _mm_xor_si128(b4c4, d2_));
        next56 = _mm_unpackhi_epi64(cd4, _mm_setzero_si128());

        //out3 = b4 ^ c3 ^ d2;
        //out4 = c4 ^ d3;

        //out5 = d4;

        /*
        next1 = out1;
        next2 = out2;
        next3 = out3;
        next4 = out4;
        next5 = out5;
        */

        i += 32;

        /* 4-7 */
        /*in1 = input[i / sizeof(uint64_t)];
        in2 = input[i / sizeof(uint64_t) + 1];*/
        READ_NEXT(input, i, in1in2, in3in4);

        in1in2 = _mm_xor_si128(in1in2, next12);
        in1in2 = _mm_xor_si128(in1in2, chorba78);
        in1in2 = _mm_xor_si128(in1in2, chorba45);
        in1in2 = _mm_xor_si128(in1in2, chorba34);

        /*
        in1 ^= next1 ^ chorba7 ^ chorba4 ^ chorba3;
        in2 ^= next2 ^ chorba8 ^ chorba5 ^ chorba4;
        */

        /*
        a1 = (in1 << 17) ^ (in1 << 55);
        a2 = (in1 >> 47) ^ (in1 >> 9) ^ (in1 << 19);
        a3 = (in1 >> 45) ^ (in1 << 44);
        a4 = (in1 >> 20);
        
        b1 = (in2 << 17) ^ (in2 << 55);
        b2 = (in2 >> 47) ^ (in2 >> 9) ^ (in2 << 19);
        b3 = (in2 >> 45) ^ (in2 << 44);
        b4 = (in2 >> 20);
        */

        NEXT_ROUND(in1in2, ab1, ab2, ab3, ab4);

        /*
        in3 = input[i / sizeof(uint64_t) + 2];
        in4 = input[i / sizeof(uint64_t) + 3];

        in3 ^= next3 ^ a1 ^ chorba6 ^ chorba5;
        in4 ^= next4 ^ b1 ^ a2  ^ chorba7 ^ chorba6;
        */
        in3in4 = _mm_xor_si128(in3in4, next34);
        in3in4 = _mm_xor_si128(in3in4, ab1);
        in3in4 = _mm_xor_si128(in3in4, chorba56);
        __m128i chorba67 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(chorba56), _mm_castsi128_pd(chorba78), 1));
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(chorba67, _mm_unpacklo_epi64(_mm_setzero_si128(), ab2)));

        /*
        c1 = (in3 << 17) ^ (in3 << 55);
        c2 = (in3 >> 47) ^ (in3 >> 9) ^ (in3 << 19);
        c3 = (in3 >> 45) ^ (in3 << 44);
        c4 = (in3 >> 20);
        
        d1 = (in4 << 17) ^ (in4 << 55);
        d2 = (in4 >> 47) ^ (in4 >> 9) ^ (in4 << 19);
        d3 = (in4 >> 45) ^ (in4 << 44);
        d4 = (in4 >> 20);
        */

        NEXT_ROUND(in3in4, cd1, cd2, cd3, cd4);

        ///*
        b2c2 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab2), _mm_castsi128_pd(cd2), 1)); 
        a4_ = _mm_unpacklo_epi64(_mm_setzero_si128(), ab4);
        a4_ = _mm_xor_si128(b2c2, a4_);
        next12 = _mm_xor_si128(ab3, cd1);

        next12 = _mm_xor_si128(next12, a4_);
        next12 = _mm_xor_si128(next12, next56);
        b4c4 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab4), _mm_castsi128_pd(cd4), 1)); 
        next34 = _mm_xor_si128(b4c4, cd3);
        d2_ = _mm_unpackhi_epi64(cd2, _mm_setzero_si128());
        next34 = _mm_xor_si128(next34, d2_);
        next56 = _mm_unpackhi_epi64(cd4, _mm_setzero_si128());
        //*/

        /*
        out1 = a3 ^ b2 ^ c1;
        out2 = b3 ^ c2 ^ d1 ^ a4;
        out3 = b4 ^ c3 ^ d2;
        out4 = c4 ^ d3;
        out5 = d4;

        next1 = next5 ^ out1;
        next2 = out2;
        next3 = out3;
        next4 = out4;
        next5 = out5;
        */

        i += 32;

        /* 8-11 */
        /*
        in1 = input[i / sizeof(uint64_t)];
        in2 = input[i / sizeof(uint64_t) + 1];
        in1 ^= next1 ^ chorba8 ^ chorba7 ^ chorba1;
        in2 ^= next2 ^ chorba8 ^ chorba2;
        */

        READ_NEXT(input, i, in1in2, in3in4);

        __m128i chorba80 = _mm_unpackhi_epi64(chorba78, _mm_setzero_si128());
        __m128i next12_chorba12 = _mm_xor_si128(next12, chorba12);
        in1in2 = _mm_xor_si128(in1in2, chorba80);
        in1in2 = _mm_xor_si128(in1in2, chorba78);
        in1in2 = _mm_xor_si128(in1in2, next12_chorba12);

        NEXT_ROUND(in1in2, ab1, ab2, ab3, ab4);

        /*
        a1 = (in1 << 17) ^ (in1 << 55);
        a2 = (in1 >> 47) ^ (in1 >> 9) ^ (in1 << 19);
        a3 = (in1 >> 45) ^ (in1 << 44);
        a4 = (in1 >> 20);
        
        b1 = (in2 << 17) ^ (in2 << 55);
        b2 = (in2 >> 47) ^ (in2 >> 9) ^ (in2 << 19);
        b3 = (in2 >> 45) ^ (in2 << 44);
        b4 = (in2 >> 20);
        */

        /*in3 = input[i / sizeof(uint64_t) + 2];
        in4 = input[i / sizeof(uint64_t) + 3];*/
        in3in4 = _mm_xor_si128(next34, in3in4);
        in3in4 = _mm_xor_si128(in3in4, ab1);
        __m128i a2_ = _mm_unpacklo_epi64(_mm_setzero_si128(), ab2);
        in3in4 = _mm_xor_si128(in3in4, chorba34);
        in3in4 = _mm_xor_si128(in3in4, a2_);

        /*
        in3 ^= next3 ^ a1 ^ chorba3;
        in4 ^= next4 ^ a2 ^ b1 ^ chorba4;

        c1 = (in3 << 17) ^ (in3 << 55);
        c2 = (in3 >> 47) ^ (in3 >> 9) ^ (in3 << 19);
        c3 = (in3 >> 45) ^ (in3 << 44);
        c4 = (in3 >> 20);
        
        d1 = (in4 << 17) ^ (in4 << 55);
        d2 = (in4 >> 47) ^ (in4 >> 9) ^ (in4 << 19);
        d3 = (in4 >> 45) ^ (in4 << 44);
        d4 = (in4 >> 20);
        */


        NEXT_ROUND(in3in4, cd1, cd2, cd3, cd4);

        a4_ = _mm_unpacklo_epi64(next56, ab4);
        next12 = _mm_xor_si128(a4_, ab3);
        next12 = _mm_xor_si128(next12, cd1);
        b2c2 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab2), _mm_castsi128_pd(cd2), 1)); 
        b4c4 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab4), _mm_castsi128_pd(cd4), 1)); 
        d2_ = _mm_unpackhi_epi64(cd2, _mm_setzero_si128());
        next12 = _mm_xor_si128(next12, b2c2);
        next34 = _mm_xor_si128(b4c4, cd3);
        next34 = _mm_xor_si128(next34, d2_);
        next56 = _mm_unpackhi_epi64(cd4, _mm_setzero_si128());

        /*
        out1 =      a3 ^ b2 ^ c1;
        out2 = a4 ^ b3 ^ c2 ^ d1;
        out3 = b4 ^ c3 ^ d2;
        out4 = c4 ^ d3;
        out5 = d4;

        next1 = next5 ^ out1;
        next2 = out2;
        next3 = out3;
        next4 = out4;
        next5 = out5;
        */

        i += 32;

        /* 12-15 */
        /*
        in1 = input[i / sizeof(uint64_t)];
        in2 = input[i / sizeof(uint64_t) + 1];
        */
        READ_NEXT(input, i, in1in2, in3in4);
        in1in2 = _mm_xor_si128(in1in2, next12);
        __m128i chorb56xorchorb12 = _mm_xor_si128(chorba56, chorba12);
        in1in2 = _mm_xor_si128(in1in2, chorb56xorchorb12);
        __m128i chorb1_ = _mm_unpacklo_epi64(_mm_setzero_si128(), chorba12);
        in1in2 = _mm_xor_si128(in1in2, chorb1_);


        /*
        in1 ^= next1 ^ chorba5 ^ chorba1;
        in2 ^= next2 ^ chorba6 ^ chorba2 ^ chorba1;

        a1 = (in1 << 17) ^ (in1 << 55);
        a2 = (in1 >> 47) ^ (in1 >> 9) ^ (in1 << 19);
        a3 = (in1 >> 45) ^ (in1 << 44);
        a4 = (in1 >> 20);
        
        b1 = (in2 << 17) ^ (in2 << 55);
        b2 = (in2 >> 47) ^ (in2 >> 9) ^ (in2 << 19);
        b3 = (in2 >> 45) ^ (in2 << 44);
        b4 = (in2 >> 20);
        */

        NEXT_ROUND(in1in2, ab1, ab2, ab3, ab4);

        /*
        in3 = input[i / sizeof(uint64_t) + 2];
        in4 = input[i / sizeof(uint64_t) + 3];
        in3 ^= next3 ^ a1 ^ chorba7 ^ chorba3 ^ chorba2 ^ chorba1;
        in4 ^= next4 ^ a2 ^ b1 ^ chorba8 ^ chorba4 ^ chorba3 ^ chorba2;
        */

        in3in4 = _mm_xor_si128(next34, in3in4);
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(ab1, chorba78));
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(chorba34, chorba12));
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(chorba23, _mm_unpacklo_epi64(_mm_setzero_si128(), ab2)));
        NEXT_ROUND(in3in4, cd1, cd2, cd3, cd4);

        /*

        c1 = (in3 << 17) ^ (in3 << 55);
        c2 = (in3 >> 47) ^ (in3 >> 9) ^ (in3 << 19);
        c3 = (in3 >> 45) ^ (in3 << 44);
        c4 = (in3 >> 20);
        
        d1 = (in4 << 17) ^ (in4 << 55);
        d2 = (in4 >> 47) ^ (in4 >> 9) ^ (in4 << 19);
        d3 = (in4 >> 45) ^ (in4 << 44);
        d4 = (in4 >> 20);
        */

        ///*
        a4_ = _mm_unpacklo_epi64(next56, ab4);
        next12 = _mm_xor_si128(_mm_xor_si128(a4_, ab3), cd1); 
        b2c2 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab2), _mm_castsi128_pd(cd2), 1)); 
        b4c4 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab4), _mm_castsi128_pd(cd4), 1)); 
        d2_ = _mm_unpackhi_epi64(cd2, _mm_setzero_si128());
        next12 = _mm_xor_si128(next12, b2c2);
        next34 = _mm_xor_si128(b4c4, cd3);
        next34 = _mm_xor_si128(next34, d2_);
        next56 = _mm_unpackhi_epi64(cd4, _mm_setzero_si128());
        //*/

        /*
        out1 =      a3 ^ b2 ^ c1;
        out2 = a4 ^ b3 ^ c2 ^ d1;
        out3 = b4 ^ c3 ^ d2;
        out4 = c4 ^ d3;
        out5 = d4;

        next1 = next5 ^ out1;
        next2 = out2;
        next3 = out3;
        next4 = out4;
        next5 = out5;
        */
        
        i += 32;

        /* 16-19 */
        /*
        in1 = input[i / sizeof(uint64_t)];
        in2 = input[i / sizeof(uint64_t) + 1];
        in1 ^= next1 ^ chorba5 ^ chorba4 ^ chorba3 ^ chorba1;
        in2 ^= next2 ^ chorba6 ^ chorba5 ^ chorba4 ^ chorba1 ^ chorba2;
        */
        ///*
        READ_NEXT(input, i, in1in2, in3in4);
        __m128i chorba1_ = _mm_unpacklo_epi64(_mm_setzero_si128(), chorba12);
        in1in2 = _mm_xor_si128(_mm_xor_si128(next12, in1in2), _mm_xor_si128(chorba56, chorba45));
        in1in2 = _mm_xor_si128(in1in2, _mm_xor_si128(chorba12, chorba34));
        in1in2 = _mm_xor_si128(chorba1_, in1in2);

        NEXT_ROUND(in1in2, ab1, ab2, ab3, ab4);
        //*/

        /*
        a1 = (in1 << 17) ^ (in1 << 55);
        a2 = (in1 >> 47) ^ (in1 >> 9) ^ (in1 << 19);
        a3 = (in1 >> 45) ^ (in1 << 44);
        a4 = (in1 >> 20);
        
        b1 = (in2 << 17) ^ (in2 << 55);
        b2 = (in2 >> 47) ^ (in2 >> 9) ^ (in2 << 19);
        b3 = (in2 >> 45) ^ (in2 << 44);
        b4 = (in2 >> 20);
        */

        /*
        in3 = input[i / sizeof(uint64_t) + 2];
        in4 = input[i / sizeof(uint64_t) + 3];
        */
        ///*
        a2_ = _mm_unpacklo_epi64(_mm_setzero_si128(), ab2);
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(ab1, chorba78));
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(chorba56, chorba34));
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(chorba23, chorba67));
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(chorba1_, a2_));
        in3in4 = _mm_xor_si128(in3in4, next34);
        //*/
        /*
        in3 ^= next3 ^ a1 ^ chorba7 ^ chorba6 ^ chorba5 ^ chorba2 ^ chorba3;
        in4 ^= next4 ^ a2 ^ b1 ^ chorba8 ^ chorba7 ^ chorba6 ^ chorba3 ^ chorba4 ^ chorba1;
        */
        NEXT_ROUND(in3in4, cd1, cd2, cd3, cd4);

        /*
        c1 = (in3 << 17) ^ (in3 << 55);
        c2 = (in3 >> 47) ^ (in3 >> 9) ^ (in3 << 19);
        c3 = (in3 >> 45) ^ (in3 << 44);
        c4 = (in3 >> 20);
        
        d1 = (in4 << 17) ^ (in4 << 55);
        d2 = (in4 >> 47) ^ (in4 >> 9) ^ (in4 << 19);
        d3 = (in4 >> 45) ^ (in4 << 44);
        d4 = (in4 >> 20);
        */

        a4_ = _mm_unpacklo_epi64(next56, ab4);
        next12 = _mm_xor_si128(_mm_xor_si128(a4_, ab3), cd1); 
        b2c2 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab2), _mm_castsi128_pd(cd2), 1)); 
        b4c4 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab4), _mm_castsi128_pd(cd4), 1)); 
        d2_ = _mm_unpackhi_epi64(cd2, _mm_setzero_si128());
        next12 = _mm_xor_si128(next12, b2c2);
        next34 = _mm_xor_si128(b4c4, cd3);
        next34 = _mm_xor_si128(next34, d2_);
        next56 = _mm_unpackhi_epi64(cd4, _mm_setzero_si128());

        /*
        out1 =      a3 ^ b2 ^ c1;
        out2 = a4 ^ b3 ^ c2 ^ d1;
        out3 = b4 ^ c3 ^ d2;
        out4 = c4 ^ d3;
        out5 = d4;

        next1 = next5 ^ out1;
        next2 = out2;
        next3 = out3;
        next4 = out4;
        next5 = out5;
        */

        i += 32;

        /* 20-23 */
        /*
        in1 = input[i / sizeof(uint64_t)];
        in2 = input[i / sizeof(uint64_t) + 1];
        in1 ^= next1 ^ chorba8 ^ chorba7 ^ chorba4 ^ chorba5 ^ chorba2 ^ chorba1;
        in2 ^= next2 ^ chorba8 ^ chorba5 ^ chorba6 ^ chorba3 ^ chorba2;
        */

        READ_NEXT(input, i, in1in2, in3in4);
        in1in2 = _mm_xor_si128(in1in2, _mm_xor_si128(next12, chorba78));
        in1in2 = _mm_xor_si128(in1in2, _mm_xor_si128(chorba45, chorba56));
        in1in2 = _mm_xor_si128(in1in2, _mm_xor_si128(chorba23, chorba12));
        in1in2 = _mm_xor_si128(in1in2, chorba80);
        NEXT_ROUND(in1in2, ab1, ab2, ab3, ab4);

        /*
        a1 = (in1 << 17) ^ (in1 << 55);
        a2 = (in1 >> 47) ^ (in1 >> 9) ^ (in1 << 19);
        a3 = (in1 >> 45) ^ (in1 << 44);
        a4 = (in1 >> 20);
        
        b1 = (in2 << 17) ^ (in2 << 55);
        b2 = (in2 >> 47) ^ (in2 >> 9) ^ (in2 << 19);
        b3 = (in2 >> 45) ^ (in2 << 44);
        b4 = (in2 >> 20);
        */

        /*
        in3 = input[i / sizeof(uint64_t) + 2];
        in4 = input[i / sizeof(uint64_t) + 3];
        in3 ^= next3 ^ a1 ^ chorba7 ^ chorba6 ^ chorba4 ^ chorba3 ^ chorba1;
        in4 ^= next4 ^ a2 ^ b1 ^ chorba8 ^ chorba7 ^ chorba5 ^ chorba4 ^ chorba2 ^ chorba1;
        */
        a2_ = _mm_unpacklo_epi64(_mm_setzero_si128(), ab2);
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(next34, ab1));
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(chorba78, chorba67));
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(chorba45, chorba34));
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(chorba1_, a2_));
        in3in4 = _mm_xor_si128(in3in4, chorba12);
        NEXT_ROUND(in3in4, cd1, cd2, cd3, cd4);

        /*
        c1 = (in3 << 17) ^ (in3 << 55);
        c2 = (in3 >> 47) ^ (in3 >> 9) ^ (in3 << 19);
        c3 = (in3 >> 45) ^ (in3 << 44);
        c4 = (in3 >> 20);
        
        d1 = (in4 << 17) ^ (in4 << 55);
        d2 = (in4 >> 47) ^ (in4 >> 9) ^ (in4 << 19);
        d3 = (in4 >> 45) ^ (in4 << 44);
        d4 = (in4 >> 20);
        */

        /*
        out1 =      a3 ^ b2 ^ c1;
        out2 = a4 ^ b3 ^ c2 ^ d1;
        out3 = b4 ^ c3 ^ d2;
        out4 = c4 ^ d3;
        out5 = d4;

        next1 = next5 ^ out1;
        next2 = out2;
        next3 = out3;
        next4 = out4;
        next5 = out5;
        */

        a4_ = _mm_unpacklo_epi64(next56, ab4);
        next12 = _mm_xor_si128(_mm_xor_si128(a4_, ab3), cd1); 
        b2c2 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab2), _mm_castsi128_pd(cd2), 1)); 
        b4c4 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab4), _mm_castsi128_pd(cd4), 1)); 
        d2_ = _mm_unpackhi_epi64(cd2, _mm_setzero_si128());
        next12 = _mm_xor_si128(next12, b2c2);
        next34 = _mm_xor_si128(b4c4, cd3);
        next34 = _mm_xor_si128(next34, d2_);
        next56 = _mm_unpackhi_epi64(cd4, _mm_setzero_si128());

        i += 32;

        /* 24-27 */
        /*
        in1 = input[i / sizeof(uint64_t)];
        in2 = input[i / sizeof(uint64_t) + 1];
        in1 ^= next1 ^ chorba8 ^ chorba6 ^ chorba5 ^ chorba3 ^ chorba2 ^ chorba1;
        in2 ^= next2 ^ chorba7 ^ chorba6 ^ chorba4 ^ chorba3 ^ chorba2;
        */

        READ_NEXT(input, i, in1in2, in3in4);
        in1in2 = _mm_xor_si128(in1in2, _mm_xor_si128(next12, chorba67));
        in1in2 = _mm_xor_si128(in1in2, _mm_xor_si128(chorba56, chorba34));
        in1in2 = _mm_xor_si128(in1in2, _mm_xor_si128(chorba23, chorba12));
        in1in2 = _mm_xor_si128(in1in2, chorba80);
        NEXT_ROUND(in1in2, ab1, ab2, ab3, ab4);

        /*
        a1 = (in1 << 17) ^ (in1 << 55);
        a2 = (in1 >> 47) ^ (in1 >> 9) ^ (in1 << 19);
        a3 = (in1 >> 45) ^ (in1 << 44);
        a4 = (in1 >> 20);
        
        b1 = (in2 << 17) ^ (in2 << 55);
        b2 = (in2 >> 47) ^ (in2 >> 9) ^ (in2 << 19);
        b3 = (in2 >> 45) ^ (in2 << 44);
        b4 = (in2 >> 20);
        */

        /*in3 = input[i / sizeof(uint64_t) + 2];
        in4 = input[i / sizeof(uint64_t) + 3];
        in3 ^= next3 ^ a1 ^ chorba8 ^ chorba7 ^ chorba5 ^ chorba4 ^ chorba3;
        in4 ^= next4 ^ a2 ^ b1 ^ chorba8 ^ chorba6 ^ chorba5 ^ chorba4;

        c1 = (in3 << 17) ^ (in3 << 55);
        c2 = (in3 >> 47) ^ (in3 >> 9) ^ (in3 << 19);
        c3 = (in3 >> 45) ^ (in3 << 44);
        c4 = (in3 >> 20);
        
        d1 = (in4 << 17) ^ (in4 << 55);
        d2 = (in4 >> 47) ^ (in4 >> 9) ^ (in4 << 19);
        d3 = (in4 >> 45) ^ (in4 << 44);
        d4 = (in4 >> 20);
        */
        a2_ = _mm_unpacklo_epi64(_mm_setzero_si128(), ab2);
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(next34, ab1));
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(chorba78, chorba56));
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(chorba45, chorba34));
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(chorba80, a2_));
        NEXT_ROUND(in3in4, cd1, cd2, cd3, cd4);

        a4_ = _mm_unpacklo_epi64(next56, ab4);
        next12 = _mm_xor_si128(_mm_xor_si128(a4_, ab3), cd1); 
        b2c2 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab2), _mm_castsi128_pd(cd2), 1)); 
        b4c4 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab4), _mm_castsi128_pd(cd4), 1)); 
        d2_ = _mm_unpackhi_epi64(cd2, _mm_setzero_si128());
        next12 = _mm_xor_si128(next12, b2c2);
        next34 = _mm_xor_si128(b4c4, cd3);
        next34 = _mm_xor_si128(next34, d2_);
        next56 = _mm_unpackhi_epi64(cd4, _mm_setzero_si128());

        /*
        out1 =      a3 ^ b2 ^ c1;
        out2 = a4 ^ b3 ^ c2 ^ d1;
        out3 = b4 ^ c3 ^ d2;
        out4 = c4 ^ d3;
        out5 = d4;

        next1 = next5 ^ out1;
        next2 = out2;
        next3 = out3;
        next4 = out4;
        next5 = out5;
        */
        i += 32;

        /* 28-31 */
        /*
        in1 = input[i / sizeof(uint64_t)];
        in2 = input[i / sizeof(uint64_t) + 1];
        in1 ^= next1 ^ chorba7 ^ chorba6 ^ chorba5;
        in2 ^= next2 ^ chorba8 ^ chorba7 ^ chorba6;
        */
        READ_NEXT(input, i, in1in2, in3in4);
        in1in2 = _mm_xor_si128(in1in2, _mm_xor_si128(next12, chorba78));
        in1in2 = _mm_xor_si128(in1in2, _mm_xor_si128(chorba67, chorba56));
        NEXT_ROUND(in1in2, ab1, ab2, ab3, ab4);

        /*
        a1 = (in1 << 17) ^ (in1 << 55);
        a2 = (in1 >> 47) ^ (in1 >> 9) ^ (in1 << 19);
        a3 = (in1 >> 45) ^ (in1 << 44);
        a4 = (in1 >> 20);
        
        b1 = (in2 << 17) ^ (in2 << 55);
        b2 = (in2 >> 47) ^ (in2 >> 9) ^ (in2 << 19);
        b3 = (in2 >> 45) ^ (in2 << 44);
        b4 = (in2 >> 20);
        */

        /*
        in3 = input[i / sizeof(uint64_t) + 2];
        in4 = input[i / sizeof(uint64_t) + 3];
        in3 ^= next3 ^ a1 ^ chorba8 ^ chorba7;
        in4 ^= next4 ^ a2 ^ b1 ^ chorba8;

        c1 = (in3 << 17) ^ (in3 << 55);
        c2 = (in3 >> 47) ^ (in3 >> 9) ^ (in3 << 19);
        c3 = (in3 >> 45) ^ (in3 << 44);
        c4 = (in3 >> 20);
        
        d1 = (in4 << 17) ^ (in4 << 55);
        d2 = (in4 >> 47) ^ (in4 >> 9) ^ (in4 << 19);
        d3 = (in4 >> 45) ^ (in4 << 44);
        d4 = (in4 >> 20);
        */
        a2_ = _mm_unpacklo_epi64(_mm_setzero_si128(), ab2);
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(next34, ab1));
        in3in4 = _mm_xor_si128(in3in4, _mm_xor_si128(chorba78, chorba80));
        in3in4 = _mm_xor_si128(a2_, in3in4);
        NEXT_ROUND(in3in4, cd1, cd2, cd3, cd4);

        /*
        out1 =      a3 ^ b2 ^ c1;
        out2 = a4 ^ b3 ^ c2 ^ d1;
        out3 = b4 ^ c3 ^ d2;
        out4 = c4 ^ d3;
        out5 = d4;
        */

        /*
        next1 = next5 ^ out1;
        next2 = out2;
        next3 = out3;
        next4 = out4;
        next5 = out5;
        */
        
        a4_ = _mm_unpacklo_epi64(next56, ab4);
        next12 = _mm_xor_si128(_mm_xor_si128(a4_, ab3), cd1); 
        b2c2 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab2), _mm_castsi128_pd(cd2), 1)); 
        b4c4 = _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(ab4), _mm_castsi128_pd(cd4), 1)); 
        d2_ = _mm_unpackhi_epi64(cd2, _mm_setzero_si128());
        next12 = _mm_xor_si128(next12, b2c2);
        next34 = _mm_xor_si128(b4c4, cd3);
        next34 = _mm_xor_si128(next34, d2_);
        next56 = _mm_unpackhi_epi64(cd4, _mm_setzero_si128());
    }

    next1 = _mm_cvtsi128_si64(next12);
    next2 = _mm_cvtsi128_si64(_mm_unpackhi_epi64(next12, next12));
    next3 = _mm_cvtsi128_si64(next34);
    next4 = _mm_cvtsi128_si64(_mm_unpackhi_epi64(next34, next34));
    next5 = _mm_cvtsi128_si64(next56);

    for(; (i + 40 + 32) < len; i += 32) {
        uint64_t in1;
        uint64_t in2;
        uint64_t in3;
        uint64_t in4;
        uint64_t a1, a2, a3, a4;
        uint64_t b1, b2, b3, b4;
        uint64_t c1, c2, c3, c4;
        uint64_t d1, d2, d3, d4;

        uint64_t out1;
        uint64_t out2;
        uint64_t out3;
        uint64_t out4;
        uint64_t out5;

        in1 = input[i / sizeof(uint64_t)];
        in2 = input[i / sizeof(uint64_t) + 1];
        in1 ^=next1;
        in2 ^=next2;

        a1 = (in1 << 17) ^ (in1 << 55);
        a2 = (in1 >> 47) ^ (in1 >> 9) ^ (in1 << 19);
        a3 = (in1 >> 45) ^ (in1 << 44);
        a4 = (in1 >> 20);
        
        b1 = (in2 << 17) ^ (in2 << 55);
        b2 = (in2 >> 47) ^ (in2 >> 9) ^ (in2 << 19);
        b3 = (in2 >> 45) ^ (in2 << 44);
        b4 = (in2 >> 20);

        in3 = input[i / sizeof(uint64_t) + 2];
        in4 = input[i / sizeof(uint64_t) + 3];
        in3 ^= next3 ^ a1;
        in4 ^= next4 ^ a2 ^ b1;

        c1 = (in3 << 17) ^ (in3 << 55);
        c2 = (in3 >> 47) ^ (in3 >> 9) ^ (in3 << 19);
        c3 = (in3 >> 45) ^ (in3 << 44);
        c4 = (in3 >> 20);
        
        d1 = (in4 << 17) ^ (in4 << 55);
        d2 = (in4 >> 47) ^ (in4 >> 9) ^ (in4 << 19);
        d3 = (in4 >> 45) ^ (in4 << 44);
        d4 = (in4 >> 20);

        out1 = a3 ^ b2 ^ c1;
        out2 = a4 ^ b3 ^ c2 ^ d1;
        out3 = b4 ^ c3 ^ d2;
        out4 = c4 ^ d3;
        out5 = d4;

        next1 = next5 ^ out1;
        next2 = out2;
        next3 = out3;
        next4 = out4;
        next5 = out5;
    }

    memcpy(final, input+(i / sizeof(uint64_t)), len-i);
    final[0] ^= next1;
    final[1] ^= next2;
    final[2] ^= next3;
    final[3] ^= next4;
    final[4] ^= next5;

    crc = crc32_braid_base(crc, (uint8_t*) final, len-i);

    return crc;
}

Z_INTERNAL uint32_t PREFIX(crc32_chorba_sse)(uint32_t crc, const uint8_t *buf, size_t len) {
    uint32_t c;
    uint64_t* aligned_buf;
    size_t aligned_len;

    c = (~crc) & 0xffffffff;
    unsigned long algn_diff = ((uintptr_t)16 - ((uintptr_t)buf & 0x10)) & 0x10;
    if (algn_diff < len) {
        if (algn_diff) {
            c = crc32_braid_base(c, buf, algn_diff);
        }
        aligned_buf = (uint64_t*) (buf + algn_diff);
        aligned_len = len - algn_diff;
        if(aligned_len > (sizeof(z_word_t) * 64) * 1024)
            c = chorba_118960_nondestructive_sse(c, (z_word_t*) aligned_buf, aligned_len);
#if W == 8
        else if (aligned_len > 72)
            c = chorba_small_nondestructive_sse(c, aligned_buf, aligned_len);
#endif
        else {
            c = crc32_braid_base(c, (uint8_t*) aligned_buf, aligned_len);
        }
    }
    else {
        c = crc32_braid_base(c, buf, len);
    }

    /* Return the CRC, post-conditioned. */
    return c ^ 0xffffffff;
}
#endif
