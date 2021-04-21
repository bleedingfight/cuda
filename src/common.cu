#include "common.h"
void showDevice(const int devID) {
  cudaDeviceProp deviceProp;
  cudaError_t error_id = cudaGetDeviceProperties(&deviceProp, devID);
  if (error_id != cudaSuccess) {
    printf("CUDA Device Count Return %d Error:%s\n", (int)error_id,
           cudaGetErrorString(error_id));
    exit(EXIT_FAILURE);
  }
  int driverVersion, runtimeVersion = 0;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);
  printf("Current Device Name:%s\n", deviceProp.name);
  printf("CUDA Driver Version:%d.%d Runtime:%d.%d\n", driverVersion / 1000,
         (driverVersion % 100 / 10), runtimeVersion / 1000,
         (runtimeVersion % 100) / 10);
  printf("  Total amount of global memory:                 %.2f GBytes (%llu "
         "bytes)\n",
         (float)deviceProp.totalGlobalMem / pow(1024.0, 3),
         (unsigned long long)deviceProp.totalGlobalMem);
  printf("  GPU Clock rate:                                %.0f MHz (%0.2f "
         "GHz)\n",
         deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
  printf("  Memory Clock rate:                             %.0f Mhz\n",
         deviceProp.memoryClockRate * 1e-3f);
  printf("  Memory Bus Width:                              %d-bit\n",
         deviceProp.memoryBusWidth);

  if (deviceProp.l2CacheSize) {
    printf("  L2 Cache Size:                                 %d kb\n",
           deviceProp.l2CacheSize / 1024);
  }

  printf("  Max Texture Dimension Size (x,y,z)             1D=(%d), "
         "2D=(%d,%d), 3D=(%d,%d,%d)\n",
         deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
         deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
         deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
  printf("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, "
         "2D=(%d,%d) x %d\n",
         deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
         deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
         deviceProp.maxTexture2DLayered[2]);
  printf("  Total amount of constant memory:               %lu bytes\n",
         deviceProp.totalConstMem);
  printf("  Total amount of shared memory per block:       %lu bytes\n",
         deviceProp.sharedMemPerBlock);
  printf("  Total number of registers available per block: %d\n",
         deviceProp.regsPerBlock);
  printf("  Warp size:                                     %d\n",
         deviceProp.warpSize);
  printf("  Maximum number of threads per multiprocessor:  %d\n",
         deviceProp.maxThreadsPerMultiProcessor);
  printf("  Maximum number of threads per block:           %d\n",
         deviceProp.maxThreadsPerBlock);
  printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
         deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
         deviceProp.maxThreadsDim[2]);
  printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
         deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
         deviceProp.maxGridSize[2]);
  printf("  Maximum memory pitch:                          %lu bytes\n",
         deviceProp.memPitch);

  exit(EXIT_SUCCESS);
}
