#define SIMULATION_SIZE (WORK_GROUP_SIZE * WORK_PER_THREAD - 2 * PADDING_Y)

uint substep(const uint a0,
             const uint a1,
             const uint a2,
             const uint a3,
             const uint a4,
             const uint a5,
             const uint a6,
             const uint a7,
             uint center) {
    // stage 0
    const uint ta0 = a0 ^ a1;
    const uint a8 = ta0 ^ a2;
    const uint b0 = (a0 & a1) | (ta0 & a2);

    const uint ta3 = a3 ^ a4;
    const uint a9 = ta3 ^ a5;
    const uint b1 = (a3 & a4) | (ta3 & a5);

    const uint aA = a6 ^ a7;
    const uint b2 = a6 & a7;

    // stage 1
    const uint ta8 = a8 ^ a9;
    const uint aB = ta8 ^ aA;
    const uint b3 = (a8 & a9) | (ta8 & aA);

    const uint tb0 = b0 ^ b1;
    const uint b4 = tb0 ^ b2;
    const uint c0 = (b0 & b1) | (tb0 & b2);

    center |= aB;
    center &= (b3 ^ b4);
    center &= ~c0;

    return center;
}

kernel void step(const global uint* field, global uint* new_field, const uint height, const uint steps)
{
    const size_t y = get_group_id(0) * SIMULATION_SIZE + get_local_id(0);
    const size_t x = get_global_id(1);
    const size_t ly = get_local_id(0);
    const size_t py = ly*WORK_PER_THREAD;
    const size_t i = x * height + y-ly+py;


    local uint left[WORK_GROUP_SIZE * WORK_PER_THREAD];
    local uint right[WORK_GROUP_SIZE * WORK_PER_THREAD];

    // Boundary masks: the halo halves of edge windows and the global padding
    // rows lie outside the universe and must stay dead on every substep.
    // Without this, intermediate generations inside a multi-step launch can
    // give birth into the padding and leak back into the visible universe.
    // Masks are step-invariant, so they are precomputed per row.
    const size_t first_x = get_global_offset(1);
    const size_t last_x = first_x + get_global_size(1) - 1;
    const uint left_mask = (x == first_x) ? 0x0000FFFFu : 0xFFFFFFFFu;
    const uint right_mask = (x == last_x) ? 0xFFFF0000u : 0xFFFFFFFFu;
    const size_t core_rows = get_num_groups(0) * SIMULATION_SIZE;
    uint row_left_mask[WORK_PER_THREAD];
    uint row_right_mask[WORK_PER_THREAD];
    for(uint row = 0; row < WORK_PER_THREAD; row++) {
        const size_t buffer_row = get_group_id(0) * SIMULATION_SIZE + py + row;
        const uint row_mask =
            (buffer_row < PADDING_Y || buffer_row >= PADDING_Y + core_rows) ? 0u : 0xFFFFFFFFu;
        row_left_mask[row] = row_mask & left_mask;
        row_right_mask[row] = row_mask & right_mask;
    }

    for(uint row = 0; row < WORK_PER_THREAD; row++){
        uint col_l = field[i+row - height];
        uint col_m = field[i+row];
        uint col_r = field[i+row + height];

        left[py + row] = (col_l << 16) | (col_m >> 16);
        right[py + row] = (col_m << 16) | (col_r >> 16);
    }

    for(uint step = 0; step < steps; step++) {
        barrier(CLK_LOCAL_MEM_FENCE);

        uint result_left[WORK_PER_THREAD];
        uint result_right[WORK_PER_THREAD];

        for(uint row = 0; row < WORK_PER_THREAD; row++){
            uint ly2 = py + row;

            // left
            {
                // top: left mid right
                const uint a0 = left[ly2-1] >> 1;
                const uint a1 = left[ly2-1];
                const uint a2 = (left[ly2-1] << 1) | (right[ly2-1] >> 31);

                // middle: left right
                const uint a3 = left[ly2] >> 1;
                const uint a4 = (left[ly2] << 1) | (right[ly2] >> 31);

                // bottom: left mid right
                const uint a5 = left[ly2+1] >> 1;
                const uint a6 = left[ly2+1];
                const uint a7 = (left[ly2+1] << 1) | (right[ly2+1] >> 31);

                result_left[row] = substep(a0,a1,a2,a3,a4,a5,a6,a7,left[ly2]);
            }

            //right
            {
                // top: left mid right
                const uint a0 = (right[ly2-1] >> 1) | (left[ly2-1] << 31);
                const uint a1 = right[ly2-1];
                const uint a2 = right[ly2-1] << 1;

                // middle: left right
                const uint a3 = (right[ly2] >> 1) | (left[ly2] << 31);
                const uint a4 = right[ly2] << 1;

                // bottom: left mid right
                const uint a5 = (right[ly2+1] >> 1) | (left[ly2+1] << 31);
                const uint a6 = right[ly2+1];
                const uint a7 = right[ly2+1] << 1;

                result_right[row] = substep(a0,a1,a2,a3,a4,a5,a6,a7,right[ly2]);
            }

            result_left[row] &= row_left_mask[row];
            result_right[row] &= row_right_mask[row];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        for(uint row = 0; row < WORK_PER_THREAD; row++) {
            left[py + row] = result_left[row];
            right[py + row] = result_right[row];
        }
    }



    for(uint row = 0; row < WORK_PER_THREAD; row++) {
        if(py + row >= PADDING_Y && py + row < WORK_GROUP_SIZE * WORK_PER_THREAD - PADDING_Y) {
            new_field[i + row] = (left[py + row] << 16) | (right[py + row] >> 16);
        }
    }
}