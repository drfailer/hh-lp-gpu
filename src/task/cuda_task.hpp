#ifndef TASK_CUDA_TASK
#define TASK_CUDA_TASK
#include "../model/data/cuda_data.hpp"
#include "../tools/gpu.hpp"
#include <cstddef>
#include <hedgehog/hedgehog.h>

template <size_t Sep, typename... Types>
class CUDATask : public hh::AbstractCUDATask<Sep, Types...> {
  protected:
    CUDA cuda_;

    using hh::AbstractCUDATask<Sep, Types...>::AbstractCUDATask;

  private:
    void initializeCuda() override {
        CUDNN_CHECK(cudnnCreate(&cuda_.cudnn_handle));
        CUDNN_CHECK(cudnnSetStream(cuda_.cudnn_handle, this->stream()));
        CUBLAS_CHECK(cublasCreate_v2(&cuda_.cublas_handle));
        CUBLAS_CHECK(cublasSetStream_v2(cuda_.cublas_handle, this->stream()));
    }

    void shutdownCuda() override {
        CUDNN_CHECK(cudnnDestroy(cuda_.cudnn_handle));
        CUBLAS_CHECK(cublasDestroy_v2(cuda_.cublas_handle));
    }
};

#endif
