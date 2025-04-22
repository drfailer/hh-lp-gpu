#ifndef TASK_OPTIMIZER_OPTIMIZER_TASK_H
#define TASK_OPTIMIZER_OPTIMIZER_TASK_H
#include "../../data/opt_layer_data.hpp"
#include "../../types.hpp"
#include <hedgehog/hedgehog.h>

#define OptimizerTaskIn OptLayerData<ftype>
#define OptimizerTaskOut OptLayerData<ftype>
#define OptimizerTaskIO 1, OptimizerTaskIn, OptimizerTaskOut

class OptimizerTask : public hh::AbstractCUDATask<OptimizerTaskIO> {
  public:
    OptimizerTask(std::string const &name, size_t nb_threads,
                  cudnnHandle_t cudnn_handle, cublasHandle_t cublas_handle)
        : hh::AbstractCUDATask<OptimizerTaskIO>(name, nb_threads),
          cudnn_handle_(cudnn_handle), cublas_handle_(cublas_handle) {}

  protected:
    cudnnHandle_t cudnn() { return cudnn_handle_; }
    cublasHandle_t cublas() { return cublas_handle_; }

  private:
    cudnnHandle_t cudnn_handle_;
    cublasHandle_t cublas_handle_;
};

#endif
