#include <iostream>
#include <math.h>
using namespace std;
__global__ void add(int n,float* a,float* b){
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	int stride = blockDim.x*gridDim.x;
	for(int i=index;i<n;i+=stride)
		a[i] = a[i]+b[i];
}
int main(void){
	int N=1<<20;
	float *x,*y;
	cudaMallocManaged(&x,N,sizeof(float)*N);
	cudaMallocManaged(&y,N,sizeof(float)*N);
	for(int i=0;i<N;i++){
		x[i] = 1.f;
		y[i] = 2.f;
	}
	int blockSize = 256;
	int numBlocks = (N+blockSize-1)/blockSize;
	add<<<numBlocks,blockSize>>>(N,x,y);
	cudaDeviceSynchronize();
	float maxError = 0.0f;
	for(int i=0;i<N;i++)
		maxError = fmax(maxError, fabs(y[i]-3.0f));
	cudaFree(x);
	cudaFree(y);
	return 0;


}
