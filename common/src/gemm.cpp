// /*
//  * @Author: bleedingfight
//  * @Date: 2021-04-21 21:50:39
//  * @LastEditTime: 2021-04-21 22:06:16
//  * @LastEditors: Please set LastEditors
//  * @Description: In User Settings Edit
//  * @FilePath: /cuda/src/gemm.cpp
//  */
// float norm(float *veca, float *vecb, int n) {
//   /**
//    * @file gemm.cpp
//    * @brief 向量内积
//    * @author bleedingfight
//    * @version 1.0
//    * @date 2021-04-23
//    */
//
//   float res = 0;
//   for (int i = 0; i < n; i++)
//     res += veca[i] * vecb[i];
//   return res;
// }
// void normalize_cpu(float *x, float *mean, float *variance, int batch,
//                    int filters, int spatial) {
//   /**
//    * @file gemm.cpp
//    * @brief 实现BN操作，
//    * @author bleedingfight
//    * @version 1.0
//    * @date 2021-04-23
//    */
//
//   int b, f, i;
//   for (b = 0; b < batch; ++b) {
//     for (f = 0; f < filters; ++f) {
//       for (i = 0; i < spatial; ++i) {
//         int index = b * filters * spatial + f * spatial + i;
//         x[index] = (x[index] - mean[f]) / (sqrt(variance[f] + .00001f));
//       }
//     }
//   }
// }
// void gemm_nt(int M, int N, int K, float ALPHA, float *A, int lda, float *B,
//              int ldb, float *C, int ldc) {
//   /**
//    * @file gemm.cpp
//    * @brief 实现
//    * @author bleedingfight
//    * @version 1.0
//    * @date 2021-04-23
//    * M：C的行数，因为这里A没有做转置换操作，因此这里A的行数是M
//    * N：C的列数，因为这里B也没有做转置操作，因此这里B的列数是N
//    * K：这里都没有转置，因此K代表A的列数，B的行数
//    * lda: 不转置时该变量是A的列数，因此A的列数是lda
//    * ldb: 不转置时该变量时B的行数，因此B的行数是ldb
//    * ldc: C的列数
//    */
//
//   int i, j, k;
//   for (i = 0; i < M; ++i) {
//     for (j = 0; j < N; ++j) {
//       PUT_IN_REGISTER float sum = 0;
//       for (k = 0; k < K; ++k) {
//         sum += ALPHA * A[i * lda + k] * B[j * ldb + k];
//       }
//       C[i * ldc + j] += sum;
//     }
//   }
// }
