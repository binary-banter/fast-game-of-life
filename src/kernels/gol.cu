#include "constants.h"

#define SIMULATED_ROWS (blockDim.y * WORK_PER_THREAD - 2 * STEP_SIZE)
#define HEIGHT (gridDim.y * SIMULATED_ROWS + 2 * STEP_SIZE)

#define CLIP_TOP_LY ((STEP_SIZE + WORK_PER_THREAD - 1) / WORK_PER_THREAD - 1)
#define CLIP_TOP_OFFSET (STEP_SIZE - CLIP_TOP_LY * WORK_PER_THREAD)
#define CLIP_BOTTOM_LY (blockDim.y - 1 - CLIP_TOP_LY)
#define CLIP_BOTTOM_OFFSET (WORK_PER_THREAD + 1 - CLIP_TOP_OFFSET)

__device__ unsigned int sub_step(const unsigned int a0,
                                const unsigned int a1,
                                const unsigned int a2,
                                const unsigned int a3,
                                const unsigned int a4,
                                const unsigned int a5,
                                const unsigned int a6,
                                const unsigned int a7,
                                unsigned int center) {
    // stage 0
    unsigned int a8;
    asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(a8) : "r"(a2), "r"(a1), "r"(a0));
    unsigned int b0;
    asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(b0) : "r"(a2), "r"(a1), "r"(a0));
    unsigned int a9;
    asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(a9) : "r"(a5), "r"(a4), "r"(a3));
    unsigned int b1;
    asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(b1) : "r"(a5), "r"(a4), "r"(a3));

    // stage 1
    unsigned int aA;
    asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(aA) : "r"(a8), "r"(a7), "r"(a6));
    unsigned int b2;
    asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(b2) : "r"(a8), "r"(a7), "r"(a6));

    // magic stage dreamt up by an insane SAT-solver
    unsigned int magic0;
    asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(a9), "r"(aA), "r"(center));
    unsigned int magic1;
    asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(center), "r"(b2));
    unsigned int magic2;
    asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(b1), "r"(b0));
    asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(center) : "r"(magic2), "r"(magic0), "r"(magic1));

    return center;
}

// destination = left >> 16 | right << 16
inline __device__ void permute(unsigned int *destination, unsigned int left, unsigned int right) {
    asm("prmt.b32 %0, %1, %2, 0x1076;" : "=r"(*destination) : "r"(left), "r"(right));
}

extern "C" __global__ void
step(const unsigned int *field, unsigned int *new_field, const unsigned int steps) {
    const size_t py = threadIdx.y * WORK_PER_THREAD;
    const size_t i = (blockIdx.x + 1) * HEIGHT + blockIdx.y * SIMULATED_ROWS + py;

    unsigned int left[WORK_PER_THREAD + 2];
    unsigned int right[WORK_PER_THREAD + 2];

    for (size_t row = 0; row < WORK_PER_THREAD; row++) {
        unsigned int col_l = field[i + row - HEIGHT];
        unsigned int col_m = field[i + row];
        unsigned int col_r = field[i + row + HEIGHT];

        permute(&left[row + 1], col_l, col_m);
        permute(&right[row + 1], col_m, col_r);
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

        for (size_t row = 0; row < WORK_PER_THREAD; row++) {
            size_t ly2 = row + 1;

            // left
            {
                // top: left mid right
                const unsigned int a0 = left[ly2 - 1] >> 1;
                const unsigned int a1 = left[ly2 - 1];
                const unsigned int a2 = (left[ly2 - 1] << 1) | (right[ly2 - 1] >> 31);

                // middle: left right
                const unsigned int a3 = left[ly2] >> 1;
                const unsigned int a4 = (left[ly2] << 1) | (right[ly2] >> 31);

                // bottom: left mid right
                const unsigned int a5 = left[ly2 + 1] >> 1;
                const unsigned int a6 = left[ly2 + 1];
                const unsigned int a7 = (left[ly2 + 1] << 1) | (right[ly2 + 1] >> 31);

                result_left[row] = sub_step(a0, a1, a2, a3, a4, a5, a6, a7, left[ly2]);
            }

            //right
            {
                // top: left mid right
                const unsigned int a0 = (right[ly2 - 1] >> 1) | (left[ly2 - 1] << 31);
                const unsigned int a1 = right[ly2 - 1];
                const unsigned int a2 = right[ly2 - 1] << 1;

                // middle: left right
                const unsigned int a3 = (right[ly2] >> 1) | (left[ly2] << 31);
                const unsigned int a4 = right[ly2] << 1;

                // bottom: left mid right
                const unsigned int a5 = (right[ly2 + 1] >> 1) | (left[ly2 + 1] << 31);
                const unsigned int a6 = right[ly2 + 1];
                const unsigned int a7 = right[ly2 + 1] << 1;

                result_right[row] = sub_step(a0, a1, a2, a3, a4, a5, a6, a7, right[ly2]);
            }
        }

        for (size_t row = 0; row < WORK_PER_THREAD; row++) {
            left[row + 1] = result_left[row];
            right[row + 1] = result_right[row];
        }
    }

    for (size_t row = 0; row < WORK_PER_THREAD; row++) {
        if (py + row >= STEP_SIZE && py + row < blockDim.y * WORK_PER_THREAD - STEP_SIZE) {
            permute(&new_field[i + row], left[row + 1], right[row + 1]);
        }
    }
}
