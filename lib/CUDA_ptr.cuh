#pragma once
#include "range.hpp"
namespace _private {
template <typename V>
__global__ void __kernel_setVal(V *_ptr, const size_t _size, const V val) {
    for (auto i : culib::grid_stride_range(size_t(0), _size))
        _ptr[i] = val;
}

template <>
__global__ void __kernel_setVal<half>(half *_ptr, const size_t _size,
                                      const half val) {
    auto ptr = reinterpret_cast<half2 *>(_ptr);
    for (auto i : culib::grid_stride_range(size_t(0), _size / 2))
        ptr[i] = make_half2(val, val);
}

template <typename F> void gridsize(int *num_blk, int *num_thd, F kernel) {
    cudaOccupancyMaxPotentialBlockSize(num_blk, num_thd, kernel, 0, 0);
}
} // namespace _private