#ifndef TASK_LOSS_LOSS_TASK_H
#define TASK_LOSS_LOSS_TASK_H
#include "../../data/init_data.hpp"
#include "../../data/loss_bwd_data.hpp"
#include "../../data/loss_fwd_data.hpp"
#include "../../types.hpp"
#include <hedgehog/hedgehog.h>

#define LossTaskIn                                                             \
    InitData<ftype, InitStatus::Init>, LossFwdData<ftype>, LossBwdData<ftype>
#define LossTaskOut                                                            \
    InitData<ftype, InitStatus::Done>, LossFwdData<ftype>, LossBwdData<ftype>
#define LossTaskIO 3, LossTaskIn, LossTaskOut

class LossTask : public hh::AbstractCUDATask<LossTaskIO> {
  public:
    LossTask(std::string const &name = "LossTask")
        : hh::AbstractCUDATask<LossTaskIO>(name, 1) {}

  protected:
    cudnnHandle_t cudnn() { return cudnn_handle_; }
    cublasHandle_t cublas() { return cublas_handle_; }

  private:
    cudnnHandle_t cudnn_handle_;
    cublasHandle_t cublas_handle_;
};

#endif
