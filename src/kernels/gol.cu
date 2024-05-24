#include "constants.h"

#define SIMULATED_ROWS (blockDim.y * WORK_PER_THREAD - 2 * STEP_SIZE)
#define HEIGHT (gridDim.y * SIMULATED_ROWS + 2 * STEP_SIZE)

#define CLIP_TOP_LY ((STEP_SIZE + WORK_PER_THREAD - 1) / WORK_PER_THREAD - 1)
#define CLIP_TOP_OFFSET (STEP_SIZE - CLIP_TOP_LY * WORK_PER_THREAD)
#define CLIP_BOTTOM_LY (blockDim.y - 1 - CLIP_TOP_LY)
#define CLIP_BOTTOM_OFFSET (WORK_PER_THREAD + 1 - CLIP_TOP_OFFSET)

inline __device__ unsigned int sub_step(const unsigned int a0,
                                const unsigned int a1,
                                const unsigned int a2,
                                const unsigned int a3,
                                const unsigned int a4,
                                const unsigned int a5,
                                const unsigned int a6,
                                const unsigned int a7,
                                const unsigned int top_xor,
                                const unsigned int bottom_xor,
                                const unsigned int top_maj,
                                const unsigned int bottom_maj,
                                unsigned int center) {
    unsigned int a8, a9, aA, b0, b1, b2, magic0, magic1, magic2, garbage;

    // stage 1
    asm("lop3.and.b32 %0|%1, %2, %3, %4, 0b10010110, 1;" : "=r"(garbage), "=r"(aA) : "r"(top_xor), "r"(a4), "r"(a3));
    asm("lop3.and.b32 %0|%1, %2, %3, %4, 0b11101000, 1;" : "=r"(garbage), "=r"(b2) : "r"(top_xor), "r"(a4), "r"(a3));

    // magic stage dreamt up by an insane SAT-solver
    asm("lop3.and.b32 %0|%1, %2, %3, %4, 0b00111110, 1;" : "=r"(garbage), "=r"(magic0) : "r"(bottom_xor), "r"(aA), "r"(center));
    asm("lop3.and.b32 %0|%1, %2, %3, %4, 0b01011011, 1;" : "=r"(garbage), "=r"(magic1) : "r"(magic0), "r"(center), "r"(b2));
    asm("lop3.and.b32 %0|%1, %2, %3, %4, 0b10010001, 1;" : "=r"(garbage), "=r"(magic2) : "r"(magic1), "r"(bottom_maj), "r"(top_maj));
    asm("lop3.and.b32 %0|%1, %2, %3, %4, 0b01011000, 1;" : "=r"(garbage), "=r"(center) : "r"(magic2), "r"(magic0), "r"(magic1));

    return center;
}

inline __device__ uint4 load_uint4(uint4 *address) {
    unsigned int x;
    unsigned int y;
    unsigned int z;
    unsigned int w;

    asm("ld.global.v4.u32 {%0, %1, %2, %3}, [%4], 1;" : "=r"(garbage), "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(address));

    return make_uint4(x,y,z,w);
}

// destination = left >> 16 | right << 16
inline __device__ void permute(unsigned int *destination, unsigned int left, unsigned int right) {
    asm("prmt.b32 %0, %1, %2, 0x1076, 1;" : "=r"(garbage), "=r"(*destination) : "r"(left), "r"(right));
}

extern "C" __global__ void
step(const unsigned int *field, unsigned int *new_field, const unsigned int steps) {
    const size_t py = threadIdx.y * WORK_PER_THREAD;
    const size_t i = (blockIdx.x + 1) * HEIGHT + blockIdx.y * SIMULATED_ROWS + py;

    unsigned int left[WORK_PER_THREAD + 2];
    unsigned int right[WORK_PER_THREAD + 2];

    #pragma unroll
    for (size_t row = 0; row < WORK_PER_THREAD / 4; row++) {
        uint4 l = load_uint4((uint4 *) &field[i + row * 4 - HEIGHT]);
        uint4 m = load_uint4((uint4 *) &field[i + row * 4]);
        uint4 r = load_uint4((uint4 *) &field[i + row * 4 + HEIGHT]);

        permute(&left[row * 4 + 1], l.x, m.x);
        permute(&right[row * 4 + 1], m.x, r.x);

        permute(&left[row * 4 + 2], l.y, m.y);
        permute(&right[row * 4 + 2], m.y, r.y);

        permute(&left[row * 4 + 3], l.z, m.z);
        permute(&right[row * 4 + 3], m.z, r.z);

        permute(&left[row * 4 + 4], l.w, m.w);
        permute(&right[row * 4 + 4], m.w, r.w);
    }

    for (size_t step = 0; step < steps; step++) {
        unsigned int result_left[WORK_PER_THREAD];
        unsigned int result_right[WORK_PER_THREAD];

        // Clip top boundary.
        if (blockIdx.y == 0 && threadIdx.y == CLIP_TOP_LY) {
            left[CLIP_TOP_OFFSET] = 0;
            right[CLIP_TOP_OFFSET] = 0;
        }

        // Clip bottom boundary.
        if (blockIdx.y == gridDim.y - 1 && threadIdx.y == CLIP_BOTTOM_LY) {
            left[CLIP_BOTTOM_OFFSET] = 0;
            right[CLIP_BOTTOM_OFFSET] = 0;
        }

        // Clip left boundary.
        if (blockIdx.x == 0) {
            for (size_t row = 0; row < WORK_PER_THREAD; row++) {
                left[row + 1] &= 0x0000FFFF;
            }
        }

        // Clip right boundary.
        if (blockIdx.x == gridDim.x - 1) {
            for (size_t row = 0; row < WORK_PER_THREAD; row++) {
                right[row + 1] &= 0xFFFF0000;
            }
        }

        left[0] = __shfl_up_sync(-1, left[WORK_PER_THREAD], 1);
        right[0] = __shfl_up_sync(-1, right[WORK_PER_THREAD], 1);
        left[WORK_PER_THREAD + 1] = __shfl_down_sync(-1, left[1], 1);
        right[WORK_PER_THREAD + 1] = __shfl_down_sync(-1, right[1], 1);

        unsigned int left_top_xor;
        unsigned int left_mid_xor;
        unsigned int left_top_maj;
        unsigned int left_mid_maj;
        unsigned int right_top_xor;
        unsigned int right_mid_xor;
        unsigned int right_top_maj;
        unsigned int right_mid_maj;

        #pragma unroll
        for (size_t row = 0; row < WORK_PER_THREAD; row++) {
            size_t ly2 = row + 1;

            // left
            {
                // top: left mid right
                const unsigned int a0 = left[ly2 - 1] >> 1;
                const unsigned int a1 = left[ly2 - 1];
                const unsigned int a2 = __funnelshift_l(right[ly2 - 1], left[ly2 - 1], 1);

                // middle: left right
                const unsigned int a3 = left[ly2] >> 1;
                const unsigned int a4 = __funnelshift_l(right[ly2], left[ly2], 1);

                // bottom: left mid right
                const unsigned int a5 = left[ly2 + 1] >> 1;
                const unsigned int a6 = left[ly2 + 1];
                const unsigned int a7 = __funnelshift_l(right[ly2 + 1], left[ly2 + 1], 1);

                if (row == 0) {
                    asm("lop3.and.b32 %0|%1, %2, %3, %4, 0b10010110, 1;" : "=r"(garbage), "=r"(left_top_xor) : "r"(a2), "r"(a1), "r"(a0));
                    asm("lop3.and.b32 %0|%1, %2, %3, %4, 0b10010110, 1;" : "=r"(garbage), "=r"(left_mid_xor) : "r"(a4), "r"(a3), "r"(left[ly2]));
                    asm("lop3.and.b32 %0|%1, %2, %3, %4, 0b11101000, 1;" : "=r"(garbage), "=r"(left_top_maj) : "r"(a2), "r"(a1), "r"(a0));
                    asm("lop3.and.b32 %0|%1, %2, %3, %4, 0b11101000, 1;" : "=r"(garbage), "=r"(left_mid_maj) : "r"(a4), "r"(a3), "r"(left[ly2]));
                }

                unsigned int left_bottom_xor;
                unsigned int left_bottom_maj;
                asm("lop3.and.b32 %0|%1, %2, %3, %4, 0b10010110, 1;" : "=r"(garbage), "=r"(left_bottom_xor) : "r"(a7), "r"(a6), "r"(a5));
                asm("lop3.and.b32 %0|%1, %2, %3, %4, 0b11101000, 1;" : "=r"(garbage), "=r"(left_bottom_maj) : "r"(a7), "r"(a6), "r"(a5));


                result_left[row] = sub_step(a0, a1, a2, a3, a4, a5, a6, a7, left_top_xor, left_bottom_xor, left_top_maj, left_bottom_maj, left[ly2]);
                left_top_xor = left_mid_xor;
                left_mid_xor = left_bottom_xor;
                left_top_maj = left_mid_maj;
                left_mid_maj = left_bottom_maj;
            }

            //right
            {
                // top: left mid right
                const unsigned int a0 = __funnelshift_r(right[ly2 - 1], left[ly2 - 1], 1);
                const unsigned int a1 = right[ly2 - 1];
                const unsigned int a2 = right[ly2 - 1] << 1;

                // middle: left right
                const unsigned int a3 = __funnelshift_r(right[ly2], left[ly2], 1);
                const unsigned int a4 = right[ly2] << 1;

                // bottom: left mid right
                const unsigned int a5 = __funnelshift_r(right[ly2 + 1], left[ly2 + 1], 1);
                const unsigned int a6 = right[ly2 + 1];
                const unsigned int a7 = right[ly2 + 1] << 1;

                if (row == 0) {
                    asm("lop3.and.b32 %0|%1, %2, %3, %4, 0b10010110, 1;" : "=r"(garbage), "=r"(right_top_xor) : "r"(a2), "r"(a1), "r"(a0));
                    asm("lop3.and.b32 %0|%1, %2, %3, %4, 0b10010110, 1;" : "=r"(garbage), "=r"(right_mid_xor) : "r"(a4), "r"(a3), "r"(right[ly2]));
                    asm("lop3.and.b32 %0|%1, %2, %3, %4, 0b11101000, 1;" : "=r"(garbage), "=r"(right_top_maj) : "r"(a2), "r"(a1), "r"(a0));
                    asm("lop3.and.b32 %0|%1, %2, %3, %4, 0b11101000, 1;" : "=r"(garbage), "=r"(right_mid_maj) : "r"(a4), "r"(a3), "r"(right[ly2]));
                }

                unsigned int right_bottom_xor;
                unsigned int right_bottom_maj;
                asm("lop3.and.b32 %0|%1, %2, %3, %4, 0b10010110, 1;" : "=r"(garbage), "=r"(right_bottom_xor) : "r"(a7), "r"(a6), "r"(a5));
                asm("lop3.and.b32 %0|%1, %2, %3, %4, 0b11101000, 1;" : "=r"(garbage), "=r"(right_bottom_maj) : "r"(a7), "r"(a6), "r"(a5));

                result_right[row] = sub_step(a0, a1, a2, a3, a4, a5, a6, a7, right_top_xor, right_bottom_xor, right_top_maj, right_bottom_maj,  right[ly2]);
                right_top_xor = right_mid_xor;
                right_mid_xor = right_bottom_xor;
                right_top_maj = right_mid_maj;
                right_mid_maj = right_bottom_maj;
            }
        }

        #pragma unroll
        for (size_t row = 0; row < WORK_PER_THREAD; row++) {
            left[row + 1] = result_left[row];
            right[row + 1] = result_right[row];
        }
    }

    #pragma unroll
    for (size_t row = 0; row < WORK_PER_THREAD; row++) {
        if (py + row >= STEP_SIZE && py + row < blockDim.y * WORK_PER_THREAD - STEP_SIZE) {
            permute(&new_field[i + row], left[row + 1], right[row + 1]);
        }
    }
}
