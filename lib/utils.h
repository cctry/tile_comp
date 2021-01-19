#pragma once
#include <cstdio>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <cuda_runtime_api.h>

#ifdef __CUDACC__
#define DEVICE_CALLABLE __host__ __device__
#else
#define DEVICE_CALLABLE
#endif

#ifndef __NVCC__
#define __device__
#define __global__
#endif

#define cudaChk(stat)                                                          \
    { culib::util::cudaErrCheck_((stat), __FILE__, __LINE__); }
#define cudaSMEMCheck(prop, size)                                              \
    { assert(size <= prop.sharedMemPerBlock); }
#define half_one (__half_raw{.x = 0x3c00})
#define half_zero (__half_raw{.x = 0})
#define grid_tid (blockDim.x * blockIdx.x + threadIdx.x)
namespace culib {
namespace util {
void cuda_free_safe(void *p);
void cudaErrCheck_(cudaError_t stat, const char *file, int line);
void cuda_chk_launch(int num_thd, int num_blk, size_t smem);
int cuda_num_thd(int num);
__device__ unsigned dynamic_smem_size();
template <typename T>
__device__ T warp_scan(T val, const int logic_id,
                       const uint32_t mask = 0xFFFFFFFF) {
    T temp;
    const auto size = __popc(mask);
#pragma unroll
    for (int i = 1; i < size; i *= 2) {
        temp = __shfl_up_sync(mask, val, i);
        if (logic_id >= i)
            val += temp;
    }
    return val;
}
} // namespace util
} // namespace culib
