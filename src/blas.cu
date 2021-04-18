#include "blas.h"
__global__ void sumMatrix(float *a, float *b, int nx, int ny) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.x + blockDim.y * blockIdx.y;
    int index = idy * nx + idx;
    // printf("==> (%d,%d) threadidx:%d index:%d Current x:%.2f,y:%.2f\n",idx,idy,threadIdx.x,index,a[index],b[index]);
    if (index < nx && index < ny)
      a[index] = a[index] + b[index];
  }