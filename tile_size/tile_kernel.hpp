#pragma once
#include "mma.hpp"
template <int _TileR, int _TileC>
__global__ void
__kernel_mv_tile(const half *__restrict__ mat, const half *__restrict__ vec,
                 float *__restrict__ res, const int nrow, const int ncol) {}

template <int _TileR> struct frag_t {};
template <> struct frag_t<16> { using type = culib::mma::mma_t<16, 16, 16>; };
template <> struct frag_t<32> { using type = culib::mma::mma_t<32, 8, 16>; };

template <int _TileR, int _TileC>
__device__ void __kernel_mv_normal_impl(const half *__restrict__ mat,
                                        const half *__restrict__ vec,
                                        float *__restrict__ res, const int nrow,
                                        const int ncol,
                                        half *__restrict__ smem) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;
    typename frag_t<_TileR>::type::a_t<wmma::col_major> a_frag;
    typename frag_t<_TileR>::type::b_t<wmma::col_major> b_frag;
    typename frag_t<_TileR>::type::c_t<float> c_frag;
    for (auto t_row = blockIdx.x; t_row < (nrow / _TileR); t_row += gridDim.x) {
        wmma::fill_fragment(c_frag, 0.0f);
        for (int i = warp_id; i < (ncol / _TileC); i += num_warps) {
            auto mat_p = &mat[ncol * _TileR * t_row + i * _TileR * _TileC];
            wmma::load_matrix_sync(a_frag, mat_p, _TileR);
            if (lane_id < _TileC)
                smem[512 * warp_id + lane_id] = vec[i * _TileC + lane_id];
            wmma::load_matrix_sync(b_frag, &smem[512 * warp_id], _TileC);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
        auto smem_f = reinterpret_cast<float *>(smem);
        wmma::store_matrix_sync(&smem_f[warp_id * _TileR], c_frag,
                                _TileR * num_warps, wmma::mem_col_major);
        __syncthreads();
        if (threadIdx.x < _TileR) {
            float r = 0.0f;
            for (int i = threadIdx.x; i < num_warps * _TileR; i += _TileR) {
                r += smem_f[i];
            }
            res[t_row * _TileR + threadIdx.x] = r;
        }
        __syncthreads();
    }
}

template <>
__global__ void __kernel_mv_tile<16, 16>(const half *__restrict__ mat,
                                         const half *__restrict__ vec,
                                         float *__restrict__ res,
                                         const int nrow, const int ncol) {
    extern __shared__ half smem[];
    __kernel_mv_normal_impl<16, 16>(mat, vec, res, nrow, ncol, smem);
}

template <>
__global__ void __kernel_mv_tile<32, 16>(const half *__restrict__ mat,
                                         const half *__restrict__ vec,
                                         float *__restrict__ res,
                                         const int nrow, const int ncol) {
    extern __shared__ half smem[];
    __kernel_mv_normal_impl<32, 16>(mat, vec, res, nrow, ncol, smem);}
