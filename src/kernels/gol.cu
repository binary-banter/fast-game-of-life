extern "C" __global__ void step(const unsigned int* field, unsigned int* new_field) {
    const size_t y = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t x = blockIdx.y * blockDim.y;
    const size_t i = x * gridDim.x + y + 16;

    unsigned int result = field[i];

    // left: top mid bottom
    const unsigned int a0 = field[i - gridDim.x - 1];
    const unsigned int a1 = field[i - gridDim.x    ];
    const unsigned int a2 = field[i - gridDim.x + 1];

    // middle: top bottom
    const unsigned int a3 = field[i - 1];
    const unsigned int a4 = field[i + 1];

    // right: top mid bottom
    const unsigned int a5 = field[i + gridDim.x - 1];
    const unsigned int a6 = field[i + gridDim.x    ];
    const unsigned int a7 = field[i + gridDim.x + 1];

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

    result |= aB;
    result &= (b3 ^ b4);
    result &= ~c0;

    new_field[i] = result;
}
