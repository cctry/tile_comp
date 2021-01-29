#include "CUDA_ptr.hpp"
#include "cublas.h"
#include "kernels.hpp"
#include "mma.hpp"
#include "range.hpp"
#include "tile_kernel.hpp"
#include "utils.h"
#include <bits/stdc++.h>
using namespace culib;
using namespace std::chrono;
cudaDeviceProp prop;

template <typename _Action, typename _Prep>
double wtime(const int times, _Action action, _Prep prep) {
    double time = 0.0;
    for (int i : range(0, times)) {
        prep();
        auto start = high_resolution_clock::now();
        action();
        auto end = high_resolution_clock::now();
        time += duration_cast<microseconds>(end - start).count() * 1.0;
    }
    return time / times;
}
constexpr float eps = 1;
bool verify(const std::vector<float> &vec, const float ref) {
    return std::all_of(vec.begin(), vec.end(),
                       [ref](float v) { return abs(v - ref) < eps; });
}

template <int _TileR, int _TileC>
double test(const CUDA_ptr<half> &mat, const CUDA_ptr<half> &vec,
            const int nrow, const int ncol) {
    if (nrow % _TileR)
        puts("Bad nrow");
    if (ncol % _TileC)
        puts("Bad ncol");
    CUDA_ptr<float> d_res(nrow);
    int num_blk = 800, num_thd = 1024;
    // cudaChk(cudaOccupancyMaxPotentialBlockSizeVariableSMem(
    //     &num_blk, &num_thd, __kernel_mv<_TileR, _TileC>,
    //     smem_config<_TileR, _TileC>(), 8*32));
    auto f = smem_config<_TileR, _TileC>();
    __kernel_mv<_TileR, _TileC><<<num_blk, num_thd, f(num_thd)>>>(
        mat.get(), vec.get(), d_res.get(), nrow, ncol);
    std::vector<float> h_res(nrow);
    d_res.dump(h_res.data());
    if (!verify(h_res, ncol / 2)) {
        printf("Wrong %d,%d\n", _TileR, _TileC);
        for (int idx : indices(h_res)) {
            if (abs(h_res[idx] - ncol / 2) > eps) {
                printf("%d:\t%f\n", idx, h_res[idx]);
                break;
            }
        }
        return -1;
    }
#ifdef TIME
    return wtime(
        10,
        [&]() {
            __kernel_mv<_TileR, _TileC><<<num_blk, num_thd, f(num_thd)>>>(
                mat.get(), vec.get(), d_res.get(), nrow, ncol);
            cudaChk(cudaDeviceSynchronize());
        },
        [&]() { d_res.clear(); });
#else
    return 1;
#endif
}

void cublasHF(const CUDA_ptr<half> &mat, const CUDA_ptr<half> &vec,
              const int nrow, const int ncol) {
    CUDA_ptr<float> d_res(nrow);
    cuBLAS_MV(mat.get(), vec.get(), d_res.get(), nrow, ncol);
    cudaChk(cudaDeviceSynchronize());
#ifdef TIME
    double time = wtime(
        10,
        [&]() {
            cuBLAS_MV(mat.get(), vec.get(), d_res.get(), nrow, ncol);
            cudaChk(cudaDeviceSynchronize());
        },
        [&]() { d_res.clear(); });
    std::cout << "cublasHF\t" << time << " us" << std::endl;
#endif
    std::vector<float> h_res(nrow);
    d_res.dump(h_res.data());
    if (!verify(h_res, ncol / 2))
        puts("cublasHF wrong");
}

void cublasHH(const CUDA_ptr<half> &mat, const CUDA_ptr<half> &vec,
              const int nrow, const int ncol) {
    CUDA_ptr<float> d_res(nrow);
    CUDA_ptr<half> d_res_h(nrow);
    int num_thd, num_blk;
    cudaOccupancyMaxPotentialBlockSize(&num_blk, &num_thd, to_float, 0, 0);
    cuBLAS_MV(mat.get(), vec.get(), d_res_h.get(), nrow, ncol);
    to_float<<<num_blk, num_thd>>>(d_res_h.get(), d_res.get(), d_res_h.size);
    cudaChk(cudaDeviceSynchronize());
#ifdef TIME
    double time = wtime(
        10,
        [&]() {
            cuBLAS_MV(mat.get(), vec.get(), d_res_h.get(), nrow, ncol);
            to_float<<<num_blk, num_thd>>>(d_res_h.get(), d_res.get(),
                                           d_res_h.size);
            cudaChk(cudaDeviceSynchronize());
        },
        [&]() {
            d_res_h.clear();
            d_res.clear();
        });
    std::cout << "cublasHH\t" << time << " us" << std::endl;
#endif
    std::vector<float> h_res(nrow);
    d_res.dump(h_res.data());
    if (!verify(h_res, ncol / 2))
        puts("cublasHH wrong");
}

void cublasFF(const CUDA_ptr<half> &mat, const CUDA_ptr<half> &vec,
              const int nrow, const int ncol) {
    CUDA_ptr<float> mat_f(nrow * ncol), vec_f(nrow), d_res(nrow);
    int num_thd, num_blk;
    cudaOccupancyMaxPotentialBlockSize(&num_blk, &num_thd, to_half, 0, 0);
    to_float<<<num_blk, num_thd>>>(mat.get(), mat_f.get(), nrow * ncol);
    to_float<<<num_blk, num_thd>>>(vec.get(), vec_f.get(), nrow);
    cuBLAS_MV(mat_f.get(), vec_f.get(), d_res.get(), nrow, ncol);
    cudaChk(cudaDeviceSynchronize());
#ifdef TIME
    double time = wtime(
        10,
        [&]() {
            cuBLAS_MV(mat_f.get(), vec_f.get(), d_res.get(), nrow, ncol);
            cudaChk(cudaDeviceSynchronize());
        },
        [&]() { d_res.clear(); });
    std::cout << "cublasFF\t" << time << " us" << std::endl;
#endif
    std::vector<float> h_res(nrow);
    d_res.dump(h_res.data());
    if (!verify(h_res, ncol / 2))
        puts("cublasFF wrong");
}

template <int CSIZE>
__global__ void MV(const half *__restrict__ mat, const half *__restrict__ vec,
                   float *__restrict__ res, const int nrow, const int ncol) {
    __shared__ half smem[CSIZE];
    const auto lane_id = threadIdx.x % 32;
    const auto total_warps = (gridDim.x * blockDim.x) / 32;
    for (int row = grid_tid / 32; row < nrow; row += total_warps) {
        auto row_p = &mat[row * ncol];
        half val = half_zero;
        for (int i = 0; i < ncol; i += CSIZE) {
            for (int idx = threadIdx.x; idx < CSIZE; idx += blockDim.x)
                smem[idx] = vec[i + idx];
            __syncthreads();
            for (int k = lane_id; k < CSIZE; k += 32) {
                val += row_p[i + k] * smem[k];
            }
            __syncthreads();
        }
        constexpr auto mask = 0xffffffff;
        val += __shfl_down_sync(mask, val, 16);
        val += __shfl_down_sync(mask, val, 8);
        val += __shfl_down_sync(mask, val, 4);
        val += __shfl_down_sync(mask, val, 2);
        val += __shfl_down_sync(mask, val, 1);
        if (lane_id == 0)
            res[row] = __half2float(val);
    }
}
void my_kernel(const CUDA_ptr<half> &mat, const CUDA_ptr<half> &vec,
               const int nrow, const int ncol) {
    constexpr int csize = 32 * 16;
    CUDA_ptr<float> d_res(nrow);
    if (ncol % csize) {
        puts("Wrong CSIZE");
        return;
    }
    int num_blk, num_thd;
    cudaChk(cudaOccupancyMaxPotentialBlockSize(&num_blk, &num_thd, MV<csize>,
                                               csize * sizeof(half), 0));
    MV<csize>
        <<<num_blk, num_thd>>>(mat.get(), vec.get(), d_res.get(), nrow, ncol);
    cudaChk(cudaDeviceSynchronize());
#ifdef TIME
    double time = wtime(
        10,
        [&]() {
            MV<csize><<<num_blk, num_thd>>>(mat.get(), vec.get(), d_res.get(),
                                            nrow, ncol);
            cudaChk(cudaDeviceSynchronize());
        },
        [&]() { d_res.clear(); });
    std::cout << "my_kernel\t" << time << " us" << std::endl;
#endif
    std::vector<float> h_res(nrow);
    d_res.dump(h_res.data());
    if (!verify(h_res, ncol / 2)) {
        for (int i : indices(h_res)) {
            if (abs(h_res[i] - ncol / 2) > 1) {
                printf("error at %d:\t%f\n", i, h_res[i]);
            }
        }
        puts("my_kernel wrong");
    }
}

template <int _TileR, int _TileC>
void test_tile(const CUDA_ptr<half> &mat, const CUDA_ptr<half> &vec,
               const int nrow, const int ncol) {
    if (nrow % _TileR)
        puts("Bad nrow");
    if (ncol % _TileC)
        puts("Bad ncol");
    CUDA_ptr<float> d_res(nrow);
    int num_blk = 1024, num_thd = 64, smem_size;
    auto f = [](int n) { return (n / 32) * 256 * sizeof(float); };
    cudaChk(cudaOccupancyMaxPotentialBlockSizeVariableSMem(
        &num_blk, &num_thd, __kernel_mv_tile<_TileR, _TileC>, f));
    smem_size = f(num_thd);
    // printf("<<<%d, %d, %d>>>\t%d\n", num_blk, num_thd, smem_size);
    __kernel_mv_tile<_TileR, _TileC><<<num_blk, num_thd, smem_size>>>(
        mat.get(), vec.get(), d_res.get(), nrow, ncol);
    std::vector<float> h_res(nrow);
    d_res.dump(h_res.data());
    if (!verify(h_res, ncol / 2)) {
        puts("Tensor Wrong");
        for (auto i : indices(h_res)) {
            auto val = h_res[i];
            if (abs(val - ncol / 2) > 0.5) {
                std::cout << "error at " << i << ":\t" << val << std::endl;
                return;
            }
        }
    }
#ifdef TIME
    auto time = wtime(
        10,
        [&]() {
            __kernel_mv_tile<_TileR, _TileC><<<num_blk, num_thd, smem_size>>>(
                mat.get(), vec.get(), d_res.get(), nrow, ncol);
            cudaChk(cudaDeviceSynchronize());
        },
        [&]() { d_res.clear(); });
    printf("Limit(%d,%d)\t%lf us\n", _TileR, _TileC, time);
#endif
}

#define TEST(r, c)                                                             \
    printf("(%d,%d):\t%lf us\n", (r), (c),                                     \
           test<(r), (c)>(d_mat, d_vec, nrow, ncol))

int main(int ac, char **av) {
    const int device = 1;
    cudaSetDevice(device);
    cudaGetDeviceProperties(&prop, device);
    // if (ac < 2) {
    //     puts("Usage ./a.out nrow ncol");
    //     exit(1);
    // } // 18240 12800
    // const int nrow = atoi(av[1]);
    // const int ncol = atoi(av[2]);
    const int nrow = 18240;
    const int ncol = 12800;
    CUDA_ptr<half> d_vec(ncol, __float2half_rn(0.5)),
        d_mat(nrow * ncol, half_one);
    TEST(16, 16);
    TEST(32, 16);
    TEST(15, 32);
    TEST(30, 16);
    // TEST(38, 16);
    cublasHH(d_mat, d_vec, nrow, ncol);
    cublasHF(d_mat, d_vec, nrow, ncol);
    cublasFF(d_mat, d_vec, nrow, ncol);
    my_kernel(d_mat, d_vec, nrow, ncol);
    test_tile<32,16>(d_mat, d_vec, nrow, ncol);
    test_tile<16,16>(d_mat, d_vec, nrow, ncol);
}
