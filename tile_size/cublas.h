#pragma once
#include <cublas_v2.h>
void cuBLAS_MV(const half *A, const half *x, float *y, const int nrow,
               const int ncol);

void cuBLAS_MV(const half *A, const half *x, half *y, const int nrow,
               const int ncol);

void cuBLAS_MV(const float *A, const float *x, float *y, const int nrow,
               const int ncol);

__global__ void to_float(const half *__restrict__ src, float *__restrict__ dst,
                         int len);

__global__ void to_half(const float *__restrict__ src, half *__restrict__ dst,
                        int len);