#ifndef TASK_LAYER_TASK_H
#define TASK_LAYER_TASK_H
#include "../data/bwd_data.hpp"
#include "../data/fwd_data.hpp"
#include "../data/init_data.hpp"
#include "../data/update_data.hpp"
#include "../types.hpp"
#include <hedgehog/hedgehog.h>

#define LayerTaskIn                                                            \
    InitData<ftype>, FwdData<ftype>, BwdData<ftype>, UpdateData<ftype>
#define LayerTaskOut                                                           \
    InitData<ftype>, FwdData<ftype>, BwdData<ftype>, UpdateData<ftype>
#define LayerTaskIO 4, LayerTaskIn, LayerTaskOut

class LayerTask : public hh::AbstractCUDATask<LayerTaskIO> {
  public:
    LayerTask(std::string const &name, cudnnHandle_t cudnn_handle,
              cublasHandle_t cublas_handle, size_t idx,
              LayerDimentions const &dims)
        : hh::AbstractCUDATask<LayerTaskIO>(name, 1),
          cudnn_handle_(cudnn_handle), cublas_handle_(cublas_handle), idx_(idx),
          dims_(dims) {}

  protected:
    cudnnHandle_t cudnn() { return cudnn_handle_; }
    cublasHandle_t cublas() { return cublas_handle_; }
    size_t idx() const { return idx_; }
    LayerDimentions const &dims() const { return dims_; }

  private:
    cudnnHandle_t cudnn_handle_;
    cublasHandle_t cublas_handle_;
    size_t idx_;
    LayerDimentions dims_;
};

#endif
