#include "blas.h"
#include <stdio.h>
#include "utils.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

static __inline__ void modify (cublasHandle_t handle, float *m, int ldm, int n, int p, int q, float alpha, float beta){
    cublasSscal (handle, n-q, &alpha, &m[IDX2C(p,q,ldm)], ldm);
    cublasSscal (handle, ldm-p, &beta, &m[IDX2C(p,q,ldm)], 1);
}

int main (void){
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    int i, j;
    float* devPtrA;
    float* a = 0;
    int M = 6,N=5;
    a = (float *)malloc (M * N * sizeof (*a));
    if (!a) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }
    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
            a[IDX2C(i,j,M)] = (float)(i * N + j + 1);
        }
    }
    cudaStat = cudaMalloc ((void**)&devPtrA, M*N*sizeof(*a));
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    stat = cublasSetMatrix (M, N, sizeof(*a), a, M, devPtrA, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    modify (handle, devPtrA, M, N, 1, 2, 16.0f, 12.0f);
    stat = cublasGetMatrix (M, N, sizeof(*a), devPtrA, M, a, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    cudaFree (devPtrA);
    cublasDestroy(handle);
    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
            printf ("%7.0f", a[IDX2C(i,j,M)]);
        }
        printf ("\n");
    }
    free(a);
    return EXIT_SUCCESS;
}
// int main() {
//     const int dev = 0;
//     showDevice(dev);
//     // cudaDeviceProp deviceProp;
//     // cudaGetDeviceProperties(&deviceProp, dev);
  
//     // int nx = 1 << 5;
//     // int ny = 1 << 5;
  
//     // int nxy = nx * ny;
//     // int nBytes = nxy * sizeof(float);
  
//     // float *h_a, *h_b, *hostRef, *gpuRef;
  
//     // h_a = (float *)malloc(nBytes);
//     // h_b = (float *)malloc(nBytes);
  
//     // hostRef = (float *)malloc(nBytes);
//     // gpuRef = (float *)malloc(nBytes);
  
//     // initData(h_a, nx, 1.0f);
//     // initData(h_b, ny, 2.0f);
  
//     // memset(hostRef, 0, nBytes);
//     // memset(gpuRef, 0, nBytes);
  
//     // float *d_a, *d_b;
//     // cudaMalloc((void **)&d_a, nBytes);
//     // cudaMalloc((void **)&d_b, nBytes);
  
//     // cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
//     // cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);
  
//     // int dimx = 32;
//     // int dimy = 32;
//     // dim3 block(dimx, dimy);
//     // dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
//     // printf("grid:(%d,%d),Block:(%d,%d)",grid.x,grid.y,block.x,block.y);
  
//     // sumMatrix<<<grid, block>>>(d_a, d_b, nx, ny);
//     // cudaMemcpy(gpuRef, d_a, nBytes, cudaMemcpyDeviceToHost);
//     // //check_data(gpuRef, 10);
  
//     // cudaFree(d_a);
//     // cudaFree(d_b);
//     // free(h_a);
//     // free(h_b);
//   }