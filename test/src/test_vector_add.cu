#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include "inplaceOp.h"
#include <gtest/gtest.h>
using namespace std;

int main(int argc,char* argv[]) {
    testing::InitGoogleTest(&argc,argv);
    int dev = 0;
    cudaSetDevice(dev);
    int nElems = 32;
    int size = nElems*sizeof(int);
    int *d_a,*d_b;
    int *result = new int[nElems];
    int* h_a = new int[nElems];
    int* h_b = new int[nElems];
    fill(h_a,h_a+nElems,1);
    fill(h_b,h_b+nElems,2);

    cudaMalloc((float**) &d_a,size);
    cudaMalloc((float**) &d_b,size);

    cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,size,cudaMemcpyHostToDevice);

    dim3 block(nElems);
    dim3 grid (nElems/block.x);

    add_vector<<<grid,block>>>(d_a,d_b,nElems);
    
    
    cudaMemcpy(result,d_a,size,cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaDeviceReset();
    cout<<result[2]<<endl;
    EXPECT_EQ(result[2],3);
    cudaFree(d_a);
    cudaFree(d_b);
    // cudaDeviceDe
    

    delete [] h_a;
    delete [] h_b;
    delete [] result;
    return RUN_ALL_TESTS();
}
