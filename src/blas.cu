#include "blas.h"
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
__global__ void axy(float *x, float *y, float b, int N){
  /**
   * @brief $\vec{x} = \vec{x}\cdot\vec{y}+b$
   * 
   */
  int idx = threadIdx.x+blockDim.x * blockDim.x;
  if(idx<N)
      x[idx] = x[idx]*y[idx]+b;
}