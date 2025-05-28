#ifndef MODEL_LOSS_QUADRATIC_LOSS_H
#define MODEL_LOSS_QUADRATIC_LOSS_H
#include "../../tools/gpu.hpp"
#include "../../types.hpp"
#include "loss.hpp"
#include <cudnn.h>
#include <cudnn_graph.h>
#include <cudnn_ops.h>
#include <log.h/log.h>

class QuadraticLoss : public Loss<ftype> {
  public:
    QuadraticLoss(cudnnHandle_t cudnn_handle) : cudnn_handle_(cudnn_handle) {
        CUDNN_CHECK(cudnnCreateOpTensorDescriptor(&addition_));
        CUDNN_CHECK(cudnnSetOpTensorDescriptor(addition_, CUDNN_OP_TENSOR_ADD,
                                               CUDNN_DATA_FLOAT,
                                               CUDNN_NOT_PROPAGATE_NAN));
    }

    ~QuadraticLoss() { cudnnDestroyOpTensorDescriptor(addition_); }

  public:
    Tensor<ftype> *fwd(LossState<ftype> &state, Tensor<ftype> *model_output,
                       Tensor<ftype> *ground_truth) override {
        INFO_GRP("QuadraticLossTask FWD", INFO_GRP_LAYER_TASK);
        ERROR("unimplemented");
        exit(1);
        // ftype diff = ground_truth - output;
        // return 0.5 * diff * diff;
        return nullptr;
    }

    Tensor<ftype> *bwd(LossState<ftype> &state, Tensor<ftype> *model_output,
                       Tensor<ftype> *ground_truth) override {
        INFO_GRP("QuadraticLossTask BWD", INFO_GRP_LAYER_TASK);
        // return output - ground_truth;
        ftype alpha1 = 1, alpha2 = -1, beta = 0;

        CUDNN_CHECK(cudnnOpTensor(
            cudnn_handle_, addition_, &alpha1, model_output->descriptor(),
            model_output->data(), &alpha2, ground_truth->descriptor(),
            ground_truth->data(), &beta, state.tensor->descriptor(),
            state.tensor->data()));
        return state.tensor;
    }

  private:
    cudnnHandle_t cudnn_handle_ = nullptr;
    cudnnOpTensorDescriptor_t addition_;
};

#endif
