#include "inplaceOp.h"
#include <cuda_runtime.h>
#include <iostream>

int main() {
  using namespace std;
  int n = 32;
  int data_a[n] = {1};
  int data_b[n] = {2};
  int *d_a, *d_b;
  
  cudaMalloc(&d_a, n * sizeof(int));
  cudaMalloc(&d_b, n * sizeof(int));

  cudaMemcpy(d_a, &data_a, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &data_b, n * sizeof(int), cudaMemcpyHostToDevice);

  inplaceMatrixAdd<<<1, n>>>(d_a, d_b, n);
  cudaMemcpy(data_a, d_a,n*sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++)
    cout << data_a[i] << endl;
}