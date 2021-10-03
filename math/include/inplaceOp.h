// __device__ void inplaceAdd(int* v,int* y,int* data);
#include <cuda_runtime.h>
__global__ void inplaceMatrixAdd(int* a,int*b,int N);
template<typename T>
__global__ void add_vector(T* a,T*b,T n){
    int idx = threadIdx.x;
    if(idx<n)
        a[idx] += b[idx];

}
// CUDA kernel to add elements of two arrays
template<typename T>
__global__ void add(int n, T *x, T *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}