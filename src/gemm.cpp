/*
 * @Author: bleedingfight
 * @Date: 2021-04-21 21:50:39
 * @LastEditTime: 2021-04-21 22:06:16
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /cuda/src/gemm.cpp
 */
float norm(float *veca, float *vecb, int n) {
    
  float res = 0;
  for (int i = 0; i < n; i++)
    res += veca[i] * vecb[i];
  return res;
}
