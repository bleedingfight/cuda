#include "inplaceOp.h"
__global__ void inplaceMatrixAdd(int* a,int*b,int N){
    int idx = threadIdx.x;
    a[idx]+=b[idx];
}
