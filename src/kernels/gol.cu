extern "C" __global__ void step(unsigned char* a) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = x;

    a[i] += 1;
}
