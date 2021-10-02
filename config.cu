#include <iostream>
#include <cuda_runtime.h>
#include <cuda_occupancy.h>
#include "blas.h"
using namespace std;
__global__ void MyKernel(int *d,int *a,int *b){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    d[idx] = a[idx]*b[idx];
}
int main(){
    int numBlocks;
    int blockSize = 32;
    int device;
    cudaDeviceProp prop;
    int activeWarps;
    int maxWarps;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,sumMatrix,blockSize,1);
    activeWarps = numBlocks*blockSize/prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor/prop.warpSize;
    cout<<"计算线程束: "<<activeWarps<<" 最大线程束:"<<maxWarps<<"\n";
    std::cout<<"Occupancy: "<<static_cast<double>(activeWarps)/maxWarps*100<<" %s\n";
    return 0;
}