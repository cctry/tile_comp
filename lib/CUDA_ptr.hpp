#pragma once
#include "CUDA_ptr.cuh"
#include "utils.h"
#include <cassert>
#include <vector>
namespace culib {
template <typename T> class CUDA_ptr {
  private:
    T *ptr;

  public:
    size_t size;
    CUDA_ptr() = delete;
    CUDA_ptr(const size_t _size) : size(_size) {
        cudaChk(cudaMalloc((void **)&ptr, sizeof(T) * size));
        cudaChk(cudaMemset(ptr, 0, sizeof(T) * size));
    }
    CUDA_ptr(const T *_data, const size_t _size) : size(_size) {
        cudaChk(cudaMalloc((void **)&ptr, sizeof(T) * size));
        cudaChk(
            cudaMemcpy(ptr, _data, sizeof(T) * size, cudaMemcpyHostToDevice));
    }
    CUDA_ptr(const std::vector<T> &vec) {
        size = vec.size();
        cudaChk(cudaMalloc((void **)&ptr, sizeof(T) * size));
        cudaChk(cudaMemcpy(ptr, vec.data(), sizeof(T) * size,
                           cudaMemcpyHostToDevice));
    }
    CUDA_ptr(const size_t _size, const T val) : size(_size) {
        cudaChk(cudaMalloc((void **)&ptr, sizeof(T) * (size + size % 2)));
        this->set(val);
    }
    CUDA_ptr(const CUDA_ptr<T> &src) {
        if (this != &src) {
            size = src.size;
            cudaChk(cudaMalloc((void **)&ptr, sizeof(T) * src.size));
            cudaChk(cudaMemcpy(ptr, src.ptr, sizeof(T) * size,
                               cudaMemcpyDeviceToDevice));
        }
    }
    CUDA_ptr(CUDA_ptr<T> &&src) {
        if (this != &src) {
            size = src.size;
            assert(src.ptr != nullptr);
            ptr = src.ptr;
            src.ptr = nullptr;
        }
    }
    void clear() { cudaChk(cudaMemset(ptr, 0, size * sizeof(T))); }
    __device__ __host__ T *get() { return ptr; }
    __device__ __host__ T *get() const { return ptr; }
    __device__ T &operator[](const unsigned int i) { return ptr[i]; }
    __device__ const T &operator[](const unsigned int i) const {
        return ptr[i];
    }
    void dump(T *dst) const {
        cudaChk(cudaMemcpy(dst, ptr, sizeof(T) * size, cudaMemcpyDeviceToHost));
    }
    void set(const T val) {
        using namespace _private;
        int num_thd = 1024, num_blk = 256;
        // gridsize(num_blk, num_thd, __kernel_setVal<T>);
        __kernel_setVal<T><<<num_blk, num_thd>>>(ptr, size, val);
    }
    ~CUDA_ptr() { util::cuda_free_safe(ptr); }
};

} // namespace culib
