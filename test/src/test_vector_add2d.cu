#include "inplaceOp.h"
#include <gtest/gtest.h>
#include <iostream>
#include <math.h>
int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  int N = 1 << 20;
  float *x, *y;

  // Allocate Unified Memory -- accessible from CPU or GPU
  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Launch kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i] - 3.0f));
  EXPECT_EQ(maxError, 0);
  // Free memory
  cudaFree(x);
  cudaFree(y);
  return RUN_ALL_TESTS();

  return 0;
}
