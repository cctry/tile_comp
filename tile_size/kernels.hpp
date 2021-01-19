#pragma once
#include "mma.hpp"
#include "range.hpp"
using namespace culib;
template <int _TileR, int _TileC>
__global__ void
__kernel_mv(const half *__restrict__ mat, const half *__restrict__ vec,
            float *__restrict__ res, const int nrow, const int ncol) {}

template <>
__global__ void
__kernel_mv<16, 16>(const half *__restrict__ mat, const half *__restrict__ vec,
                    float *__restrict__ res, const int nrow, const int ncol) {
    extern __shared__ half smem[]; // num_warps * 16 * 16 * 4
    auto mat_p = reinterpret_cast<float *>(&smem[512 * (threadIdx.x >> 5)]);
    auto vec_p = reinterpret_cast<half *>(mat_p);
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    using frag_t = mma::mma_t<16, 16, 16>;
    frag_t::a_t<wmma::row_major> a_frag;
    frag_t::b_t<wmma::col_major> b_frag;
    frag_t::c_t<float> c_frag;
    for (auto row = blockIdx.x * 16; row < nrow; row += 16 * gridDim.x) {
        wmma::fill_fragment(c_frag, 0.0f);
        for (auto col = warp_id * 16; col < ncol; col += blockDim.x / 2) {
            if (lane_id < 16)
                vec_p[lane_id] = vec[col + lane_id];
            auto tile = &mat[row * ncol + col];
            wmma::load_matrix_sync(a_frag, tile, ncol);
            wmma::load_matrix_sync(b_frag, vec_p, 16);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(mat_p, c_frag, 16, wmma::mem_col_major);
        __syncthreads();
        if (threadIdx.x < 16) {
            auto smem_f = reinterpret_cast<float *>(smem);
            float r = 0;
            for (int i = threadIdx.x; i < 8 * blockDim.x; i += 256)
                r += smem_f[i];
            res[row + threadIdx.x] = r;
        }
        __syncthreads();
    }
}

// assume matrix is col-major
template <>
__global__ void
__kernel_mv<32, 16>(const half *__restrict__ mat, const half *__restrict__ vec,
                    float *__restrict__ res, const int nrow, const int ncol) {
    extern __shared__ half smem[]; // num_warps * 32 * 8 * 4
    auto mat_p = reinterpret_cast<float *>(&smem[512 * (threadIdx.x >> 5)]);
    auto vec_p = reinterpret_cast<half *>(mat_p);
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    using frag_t = mma::mma_t<32, 8, 16>;
    frag_t::a_t<wmma::col_major> a_frag;
    frag_t::b_t<wmma::col_major> b_frag;
    frag_t::c_t<float> c_frag;
    for (auto row = blockIdx.x * 32; row < nrow; row += 32 * gridDim.x) {
        wmma::fill_fragment(c_frag, 0.0f);
        for (auto col = warp_id * 16; col < ncol; col += blockDim.x / 2) {
            if (lane_id < 16)
                vec_p[lane_id] = vec[col + lane_id];
            auto tile = &mat[col * nrow + row];
            wmma::load_matrix_sync(a_frag, tile, nrow);
            wmma::load_matrix_sync(b_frag, vec_p, 16);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(mat_p, c_frag, 32, wmma::mem_col_major);
        __syncthreads();
        if (threadIdx.x < 32) {
            auto smem_f = reinterpret_cast<float *>(smem);
            float r = 0;
            for (int i = threadIdx.x; i < 8 * blockDim.x; i += 256)
                r += smem_f[i];
            res[row + threadIdx.x] = r;
        }
        __syncthreads();
    }
}

template <>
__global__ void
__kernel_mv<15, 32>(const half *__restrict__ mat, const half *__restrict__ vec,
                    float *__restrict__ res, const int nrow, const int ncol) {
    extern __shared__ half smem[]; // num_warp * 16 * 16 * 2 * 2
    auto mat1 = &smem[512 * (threadIdx.x >> 5)];
    auto mat2 = &mat1[16];
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    // first half load first 16 element put in mat2
    auto vec_p = (lane_id < 16) ? mat2 : mat1;
    using frag_t = mma::mma_t<16, 16, 16>;
    frag_t::a_t<wmma::row_major> a_frag;
    frag_t::b_t<wmma::col_major> b_frag;
    frag_t::c_t<float> c_frag;
    for (auto row = blockIdx.x * 15; row < nrow; row += 15 * gridDim.x) {
        wmma::fill_fragment(c_frag, 0.0f);
        for (auto col = warp_id * 32; col < ncol; col += blockDim.x) {
            vec_p[lane_id % 16] = vec[col + lane_id];
            auto tile = &mat[row * ncol + col];
#pragma unroll
            for (auto r = 1; r < 16; r++) {
                mat1[r * 32 + lane_id] = tile[(r - 1) * ncol + lane_id];
            }
            wmma::load_matrix_sync(a_frag, mat1, 32);
            wmma::load_matrix_sync(b_frag, mat2, 32);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        auto res_mat = reinterpret_cast<float *>(mat1);
        wmma::store_matrix_sync(res_mat, c_frag, 16, wmma::mem_row_major);
        // first col and first row
        __syncthreads();
        float r = 0;
        if (threadIdx.x < 30) {
            const auto stride = threadIdx.x < 15 ? 1 : 16;
            auto src_idx = (1 + (threadIdx.x % 15)) * stride;
            auto smem_f = reinterpret_cast<float *>(smem);
            for (int i = 0; i < 8 * blockDim.x; i += 256) {
                r += smem_f[src_idx + i];
                // __syncwarp(__activemask());
            }
            r += __shfl_sync(__activemask(), r, threadIdx.x + 15);
            if (threadIdx.x < 15) {
                res[row + threadIdx.x] = r;
            }
        }
        __syncthreads();
    }
}

template <>
__global__ void
__kernel_mv<38, 16>(const half *__restrict__ mat, const half *__restrict__ vec,
                    float *__restrict__ res, const int nrow, const int ncol) {
    extern __shared__ half smem[]; // num_warp * ((16 * 8) + (16 * 32)) * 2
    const int lane_id = threadIdx.x % 32;
    const int gwarp_id = grid_tid >> 5;
    const auto mat_a = &smem[40 * 16 * (threadIdx.x >> 5)];
    const auto mat_b = &mat_a[16 * 32];
    const auto vec_p = (lane_id < 16) ? mat_a : mat_b;
    using frag_t = mma::mma_t<32, 8, 16>;
    frag_t::a_t<wmma::col_major> a_frag;
    frag_t::b_t<wmma::col_major> b_frag;
    frag_t::c_t<float> c_frag;
    const auto total_warps = (blockDim.x * gridDim.x) / 32;
    for (auto tile_r : range(gwarp_id, nrow / 38).step(total_warps)) {
        const auto row = tile_r * 38;
        wmma::fill_fragment(c_frag, 0);
        for (auto col : range(0, ncol).step(16)) {
            const auto tile = &mat[col * nrow + row];
            const auto stride = (lane_id < 16) ? 32 : 1;
            vec_p[(lane_id % 16) * stride] = vec[col + (lane_id % 16)];
            if (lane_id < 31) {
#pragma unroll
                for (int i = 0; i < 16; i++) {
                    mat_a[i * 32 + lane_id + 1] = tile[i * nrow + lane_id];
                }
            }
            if (lane_id < 28) {
                const int idx = lane_id % 7;
#pragma unroll 4
                for (int i = lane_id / 7; i < 16; i += 4) {
                    mat_b[idx * 16 + i + 16] = tile[i * nrow + idx + 32];
                }
            }
            wmma::load_matrix_sync(a_frag, mat_a, 32);
            wmma::load_matrix_sync(b_frag, mat_b, 16);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        const auto mat_c = reinterpret_cast<float *>(mat_a);
        wmma::store_matrix_sync(mat_c, c_frag, 32, wmma::mem_col_major);
        // first row is the first 31 elements
        // first col is the second 7 elements
        if (lane_id < 31)
            res[row + lane_id] = mat_c[1 + lane_id];
        if (lane_id < 7)
            res[row + 31 + lane_id] = mat_c[lane_id * 32 + 32];
    }
}

// assume matrix is col-major
template <>
__global__ void
__kernel_mv<30, 16>(const half *__restrict__ mat, const half *__restrict__ vec,
                    float *__restrict__ res, const int nrow, const int ncol) {
    extern __shared__ half smem[]; // num_warp * 512
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    auto temp1 = &smem[512 * (threadIdx.x >> 5)];
    auto temp2 = &temp1[16];
    // last col
    const auto vec_p = (lane_id < 16) ? temp1 : &temp2[15];
    using frag_t = mma::mma_t<16, 16, 16>;
    frag_t::a_t<wmma::col_major> a_frag;
    frag_t::b_t<wmma::row_major> b_frag;
    frag_t::c_t<float> c_frag;
    for (auto row = blockIdx.x * 30; row < nrow; row += gridDim.x * 30) {
        wmma::fill_fragment(c_frag, 0.0f);
        for (auto col = warp_id * 16; col < ncol; col += blockDim.x / 2) {
            vec_p[lane_id % 16 * 32] = vec[col + (lane_id % 16)];
            auto tile = &mat[col * nrow + row];
            auto dst = &temp1[1];
            if (lane_id < 30) {
#pragma unroll
                for (int i = 0; i < 16; i++) {
                    dst[i * 32 + lane_id] = tile[i * nrow + lane_id];
                }
            }
            wmma::load_matrix_sync(a_frag, temp1, 32);
            wmma::load_matrix_sync(b_frag, temp2, 32);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        auto res_temp = reinterpret_cast<float *>(temp1);
        wmma::store_matrix_sync(res_temp, c_frag, 16, wmma::mem_col_major);
        // last row is the first 15 elements
        // first col is the second 15 elements
        __syncthreads();
        if (threadIdx.x < 30) {
            auto smem_f = reinterpret_cast<float *>(smem);
            float r = 0;
            const auto idx = threadIdx.x % 15;
            const auto src_idx = (threadIdx.x < 15) ? 241 + idx : idx * 16;
            for (int i = 0; i < blockDim.x * 8; i += 256) {
                r += smem_f[src_idx + i];
            }
            res[row + threadIdx.x] = r;
        }
        __syncthreads();
    }
}

template <int _TileR, int _TileC> auto smem_config() {
    return []() {};
}

template <> auto smem_config<38, 16>() {
    return [](int n) { return (n / 32) * 40 * 16 * sizeof(half); };
}

template <> auto smem_config<32, 16>() {
    return [](int n) { return (n / 32) * 32 * 8 * sizeof(float); };
}

template <> auto smem_config<30, 16>() {
    return [](int n) { return (n / 32) * 16 * 16 * sizeof(float); };
}

template <> auto smem_config<16, 16>() {
    return [](int n) { return (n / 32) * 16 * 16 * sizeof(float); };
}

template <> auto smem_config<15, 32>() {
    return [](int n) { return (n / 32) * 16 * 16 * sizeof(float); };
}
