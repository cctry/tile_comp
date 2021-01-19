#include "cublas.h"
#include "range.hpp"
#include "utils.h"
using namespace culib;
void cuBLAS_mmul(const half *A, const half *B, float *C, const int M,
                 const int N, const int K, const bool trans_A,
                 const bool trans_B) {
    static cublasHandle_t cublasHandle;
    static bool first_time = true;
    const static float alpha = 1.0f;
    const static float beta = 0.0f;
    if (first_time) {
        first_time = false;
        cublasCreate(&cublasHandle);
    }
    const auto OP_A = trans_A ? CUBLAS_OP_T : CUBLAS_OP_N;
    const auto OP_B = trans_B ? CUBLAS_OP_T : CUBLAS_OP_N;
    const auto lda = trans_A ? M : K;
    const auto ldb = trans_B ? K : N;
    const auto ldc = N;
    cublasGemmEx(cublasHandle, OP_B, OP_A, N, M, K, &alpha, B, CUDA_R_16F, ldb,
                 A, CUDA_R_16F, lda, &beta, C, CUDA_R_32F, ldc, CUDA_R_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void cuBLAS_mmul(const half *A, const half *B, half *C, const int M,
                 const int N, const int K, const bool trans_A,
                 const bool trans_B) {
    static cublasHandle_t cublasHandle;
    static bool first_time = true;
    const static half alpha = half_one;
    const static half beta = half_zero;
    if (first_time) {
        first_time = false;
        cublasCreate(&cublasHandle);
    }
    const auto OP_A = trans_A ? CUBLAS_OP_T : CUBLAS_OP_N;
    const auto OP_B = trans_B ? CUBLAS_OP_T : CUBLAS_OP_N;
    const auto lda = trans_A ? M : K;
    const auto ldb = trans_B ? K : N;
    const auto ldc = N;
    cublasHgemm(cublasHandle, OP_B, OP_A, N, M, K, &alpha, B, ldb, A, lda,
                &beta, C, ldc);
}

void cuBLAS_MV(const half *A, const half *x, float *y, const int nrow,
               const int ncol) {
    cuBLAS_mmul(A, x, y, nrow, 1, ncol, false, true);
}

void cuBLAS_MV(const half *A, const half *x, half *y, const int nrow,
               const int ncol) {
    cuBLAS_mmul(A, x, y, nrow, 1, ncol, false, true);
}

void cuBLAS_MV(const float *A, const float *x, float *y, const int nrow,
               const int ncol) {
    static cublasHandle_t cublasHandle;
    static bool first_time = true;
    const static float alpha = 1.0f;
    const static float beta = 0.0f;
    if (first_time) {
        first_time = false;
        cublasCreate(&cublasHandle);
    }
    cublasSgemv(cublasHandle, CUBLAS_OP_N, nrow, ncol, &alpha, A, nrow, x, 1,
                &beta, y, 1);
}

__global__ void to_float(const half *__restrict__ src, float *__restrict__ dst,
                         int len) {
    for (auto i : grid_stride_range(0, len))
        dst[i] = __half2float(src[i]);
}

__global__ void to_half(const float *__restrict__ src, half *__restrict__ dst,
                        int len) {
    for (auto i : grid_stride_range(0, len))
        dst[i] = __float2half_rn(src[i]);
}