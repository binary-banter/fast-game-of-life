 __kernel void add(uint num, __global uint *a, __global uint *b, __global uint *result) {
    for (uint i = 0; i < num; i++) {
      result[i] = a[i] + b[i];
    }
}