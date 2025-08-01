#ifndef MODEL_LOSS_QUADRATIC_LOSS_H
#define MODEL_LOSS_QUADRATIC_LOSS_H
#include "../../tools/gpu.hpp"
#include "../../tools/tensor/tensor.hpp"
#include "../../types.hpp"
#include "loss.hpp"
#include <cudnn.h>
#include <cudnn_graph.h>
#include <cudnn_ops.h>

class QuadraticLoss : public Loss<ftype> {
  public:
    QuadraticLoss() {
        CUDNN_CHECK(cudnnCreateOpTensorDescriptor(&addition_));
        CUDNN_CHECK(cudnnSetOpTensorDescriptor(addition_, CUDNN_OP_TENSOR_ADD,
                                               CUDNN_DATA_TYPE,
                                               CUDNN_NOT_PROPAGATE_NAN));
    }

    ~QuadraticLoss() { cudnnDestroyOpTensorDescriptor(addition_); }

  public:
    tensor::Tensor<ftype> const &
    fwd(CUDA cuda_data, LossState<ftype> &state,
        tensor::Tensor<ftype> const &model_output,
        tensor::Tensor<ftype> const &ground_truth) override {
        INFO_GRP("QuadraticLossTask FWD", INFO_GRP_LAYER_TASK);
        ERROR("unimplemented");
        exit(1);
        // ftype diff = ground_truth - output;
        // return 0.5 * diff * diff;
        return state.tensor;
    }

    tensor::Tensor<ftype> const &
    bwd(CUDA cuda_data, LossState<ftype> &state,
        tensor::Tensor<ftype> const &model_output,
        tensor::Tensor<ftype> const &ground_truth) override {
        INFO_GRP("QuadraticLossTask BWD", INFO_GRP_LAYER_TASK);
        // return output - ground_truth;
        ftype alpha1 = 1, alpha2 = -1, beta = 0;

        CUDNN_CHECK(cudnnOpTensor(cuda_data.cudnn_handle, addition_, &alpha1,
                                  model_output.desc(), model_output.data(),
                                  &alpha2, ground_truth.desc(),
                                  ground_truth.data(), &beta,
                                  state.tensor.desc(), state.tensor.data()));
        return state.tensor;
    }

  private:
    cudnnOpTensorDescriptor_t addition_;
};

#endif
