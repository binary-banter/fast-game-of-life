extern "C" __global__ void add(unsigned int num, unsigned int *a, unsigned int  *b, unsigned int  *result) {
    for (unsigned int  i = 0; i < num; i++) {
        result[i] = a[i] + b[i];
    }
}