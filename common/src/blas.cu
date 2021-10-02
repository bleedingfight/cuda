#include "blas.h"
#define BLOCK_SIZE 16
__global__ void sumMatrix(float *a, float *b, int nx, int ny) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int idy = threadIdx.x + blockDim.y * blockIdx.y;
  int index = idy * nx + idx;
  if (index < nx && index < ny)
    a[index] = a[index] + b[index];
}
__global__ void axy(float *x, float alpha, float b, int N) {
  /**
   * @brief $\vec{x} = \alpha\cdot \vec{x}+b$
   *
   */
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < N)
    x[idx] = x[idx] * alpha + b;
}
__global__ void axy(float *x, float *y, float b, int N) {
  /**
   * @brief $\vec{x} = \vec{x}\cdot\vec{y}+b$
   *
   */
  int idx = threadIdx.x + blockDim.x * blockDim.x;
  if (idx < N)
    x[idx] = x[idx] * y[idx] + b;
}
__global__ void fill_kernel(int N, float ALPHA, float *X, int INCX) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= N)
    return;
  X[index * INCX] = ALPHA;
}
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
  float Cvalue = 0;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  for (int e = 0; e < A.width; ++e)
    Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
  C.elements[row * C.width + col] = Cvalue;
}
void MatMul(const Matrix A, const Matrix B, Matrix C) {
  Matrix d_A;
  d_A.width = A.width;
  d_A.height = A.height;
  size_t size = A.width * A.height * sizeof(float);
  cudaMalloc(&d_A.elements, size);
  cudaMemcpy(&d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
  Matrix d_B;
  d_B.width = B.width;
  d_B.height = B.height;
  size = B.width * B.height * sizeof(float);
  cudaMalloc(&d_B.elements, size);
  cudaMemcpy(&d_B.elements, A.elements, size, cudaMemcpyHostToDevice);
  Matrix d_C;
  d_C.width = C.width;
  d_C.height = C.height;
  size = C.width * C.height * sizeof(float);
  cudaMalloc(&d_C.elements, size);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
  MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
  cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
  cudaFree(d_A.elements);
  cudaFree(d_B.elements);
  cudaFree(d_C.elements);
}
