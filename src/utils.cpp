
#include "utils.h"
void initData(float *f, int size, float value) {
  for (int i = 0; i < size; i++)
    *(f + i) = value;
}
void check_data(float *a, int n) {
  for (int i = 0; i < n; i++)
    printf("Current :%.3f\n", *(a + i));
}