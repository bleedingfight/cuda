#include <cuda_runtime.h>
__global__ void sumMatrix(float *a, float *b, int nx, int ny);
__global__ void axy(float *x, float alpha, float b, int N);
__global__ void axy(float *x, float *y, float b, int N);