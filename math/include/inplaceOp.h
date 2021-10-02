// __device__ void inplaceAdd(int* v,int* y,int* data);
#include <cuda_runtime.h>
__global__ void inplaceMatrixAdd(int* a,int*b,int N);