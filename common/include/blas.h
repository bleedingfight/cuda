#include <cuda_runtime.h>

typedef struct matrix {
  int width;
  int height;
  float *elements;
} Matrix;
void MatMul(const Matrix, const Matrix, Matrix);
__global__ void sumMatrix(float *a, float *b, int nx, int ny);
__global__ void axy(float *x, float alpha, float b, int N);
__global__ void axy(float *x, float *y, float b, int N);
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C);
