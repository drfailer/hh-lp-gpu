#ifndef TASK_LOSS_LOSS_TASK_H
#define TASK_LOSS_LOSS_TASK_H
#include "../../data/loss_bwd_data.hpp"
#include "../../data/bwd_data.hpp"
#include "../../data/loss_fwd_data.hpp"
#include "../../types.hpp"
#include <hedgehog/hedgehog.h>

#define LossTaskIn LossFwdData<ftype>, LossBwdData<ftype>
#define LossTaskOut LossFwdData<ftype>, BwdData<ftype>
#define LossTaskIO 2, LossTaskIn, LossTaskOut

class LossTask : public hh::AbstractCUDATask<LossTaskIO> {
  public:
    LossTask(std::string const &name, cudnnHandle_t cudnn_handle,
             cublasHandle_t cublas_handle)
        : hh::AbstractCUDATask<LossTaskIO>(name, 1),
          cudnn_handle_(cudnn_handle), cublas_handle_(cublas_handle) {}

    virtual void init(NetworkState<ftype> &state) = 0;

  protected:
    cudnnHandle_t cudnn() { return cudnn_handle_; }
    cublasHandle_t cublas() { return cublas_handle_; }

  private:
    cudnnHandle_t cudnn_handle_;
    cublasHandle_t cublas_handle_;
};

#endif
