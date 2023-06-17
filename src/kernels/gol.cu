#include "constants.h"
#define SIMULATION_SIZE (WORK_GROUP_SIZE * WORK_PER_THREAD - 2 * PADDING_Y)

__device__ unsigned int substep(const unsigned int a0,
                                const unsigned int a1,
                                const unsigned int a2,
                                const unsigned int a3,
                                const unsigned int a4,
                                const unsigned int a5,
                                const unsigned int a6,
                                const unsigned int a7,
                                unsigned int center) {
    // stage 0
    const unsigned int ta0 = a0 ^ a1;
    const unsigned int a8 = ta0 ^ a2;
    const unsigned int b0 = (a0 & a1) | (ta0 & a2);

    const unsigned int ta3 = a3 ^ a4;
    const unsigned int a9 = ta3 ^ a5;
    const unsigned int b1 = (a3 & a4) | (ta3 & a5);

    const unsigned int aA = a6 ^ a7;
    const unsigned int b2 = a6 & a7;

    // stage 1
    const unsigned int ta8 = a8 ^ a9;
    const unsigned int aB = ta8 ^ aA;
    const unsigned int b3 = (a8 & a9) | (ta8 & aA);

    const unsigned int tb0 = b0 ^ b1;
    const unsigned int b4 = tb0 ^ b2;
    const unsigned int c0 = (b0 & b1) | (tb0 & b2);

    center |= aB;
    center &= (b3 ^ b4);
    center &= ~c0;

    return center;
}

extern "C" __global__ void step(const unsigned int *field, unsigned int *new_field, const unsigned int height, const unsigned int steps) {
    const size_t y = blockIdx.y * SIMULATION_SIZE + threadIdx.y;
    const size_t x = blockIdx.x + PADDING_X;
    const size_t ly = threadIdx.y;
    const size_t py = ly*WORK_PER_THREAD;
    const size_t i = x * height + y-ly+py;

    unsigned int left[WORK_PER_THREAD + 2];
    unsigned int right[WORK_PER_THREAD + 2];

    for (size_t row = 0; row < WORK_PER_THREAD; row++) {
        unsigned int col_l = field[i + row - height];
        unsigned int col_m = field[i + row];
        unsigned int col_r = field[i + row + height];

        left[row + 1] = (col_l << 16) | (col_m >> 16);
        right[row + 1] = (col_m << 16) | (col_r >> 16);
    }

    for (size_t step = 0; step < steps; step++) {
        unsigned int result_left[WORK_PER_THREAD];
        unsigned int result_right[WORK_PER_THREAD];

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

                result_left[row] = substep(a0, a1, a2, a3, a4, a5, a6, a7, left[ly2]);
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

                result_right[row] = substep(a0, a1, a2, a3, a4, a5, a6, a7, right[ly2]);
            }
        }

        for(size_t row = 0; row < WORK_PER_THREAD; row++) {
            left[row + 1] = result_left[row];
            right[row + 1] = result_right[row];
        }
    }

    for(size_t row = 0; row < WORK_PER_THREAD; row++) {
        if(py + row >= PADDING_Y && py + row < WORK_GROUP_SIZE * WORK_PER_THREAD - PADDING_Y) {
            new_field[i + row] = (left[row + 1] << 16) | (right[row + 1] >> 16);
        }
    }
}
