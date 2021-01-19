#include "utils.h"
#include <algorithm>
#include <cassert>
namespace culib {
namespace util {

struct AutoExec {
    template <typename Callable> AutoExec(Callable &&callable) { callable(); }
};

struct {
    cudaDeviceProp prop;
    AutoExec ae{[this] {
        int device = 0;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&prop, device);
    }};

} cuda_helper;

void cuda_chk_launch(int num_thd, int num_blk, size_t smem) {
    assert(num_thd <= cuda_helper.prop.maxThreadsPerBlock && "num_thd error");
    assert(num_blk <= 65535 && "num_blk error");
    assert(cuda_helper.prop.sharedMemPerBlock >= smem && "smem error");
}

int cuda_num_thd(int num) {
    return std::min(cuda_helper.prop.maxThreadsPerBlock, num);
}

void cuda_free_safe(void *p) { cudaChk(cudaFree(p)); }

__device__ unsigned dynamic_smem_size() {
    unsigned ret;
    asm volatile("mov.u32 %0, %dynamic_smem_size;" : "=r"(ret));
    return ret;
}

void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat),
                file, line);
        exit(1);
    }
}
} // namespace util
} // namespace culib