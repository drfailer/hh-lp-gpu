#ifndef TASK_LAYER_TASK_H
#define TASK_LAYER_TASK_H
#include "../data/bwd_data.hpp"
#include "../data/fwd_data.hpp"
#include "../types.hpp"
#include <hedgehog/hedgehog.h>

#define LayerTaskIn FwdData<ftype>, BwdData<ftype>
#define LayerTaskOut FwdData<ftype>, BwdData<ftype>
#define LayerTaskType 2, LayerTaskIn, LayerTaskOut

class LayerTask : public hh::AbstractCUDATask<LayerTaskType> {
  public:
    LayerTask(std::string const &name, cudnnHandle_t cudnn_handle,
              cublasHandle_t cublas_handle, size_t layer_idx)
        : hh::AbstractCUDATask<LayerTaskType>(name, 1),
          cudnn_handle_(cudnn_handle), cublas_handle_(cublas_handle),
          layer_idx_(layer_idx) {}

  public:
    cudnnHandle_t cudnn() { return cudnn_handle_; }
    cublasHandle_t cublas() { return cublas_handle_; }
    size_t layer_idx() const { return layer_idx_; }

  private:
    cudnnHandle_t cudnn_handle_;
    cublasHandle_t cublas_handle_;
    size_t layer_idx_;
};

#endif
