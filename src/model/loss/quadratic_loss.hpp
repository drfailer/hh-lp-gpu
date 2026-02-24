#ifndef MODEL_LOSS_QUADRATIC_LOSS_H
#define MODEL_LOSS_QUADRATIC_LOSS_H
#include "../../tools/gpu.hpp"
#include "../../tools/tensor/tensors.hpp"
#include "../../types.hpp"
#include "loss.hpp"
#include <cudnn.h>
#include <cudnn_graph.h>
#include <cudnn_ops.h>

class QuadraticLoss : public Loss {
  public:
    QuadraticLoss() {
        CUDNN_CHECK(cudnnCreateOpTensorDescriptor(&addition_));
        CUDNN_CHECK(cudnnSetOpTensorDescriptor(addition_, CUDNN_OP_TENSOR_ADD,
                                               CUDNN_DATA_TYPE,
                                               CUDNN_NOT_PROPAGATE_NAN));
    }

    ~QuadraticLoss() { cudnnDestroyOpTensorDescriptor(addition_); }

  public:
    void fwd(CUDA cuda, LossFwdIn const &in, LossFwdOut const &out) override {
        INFO_GRP("QuadraticLossTask FWD", INFO_GRP_LAYER_TASK);
        ERROR("unimplemented");
        exit(1);
        // ftype diff = ground_truth - output;
        // return 0.5 * diff * diff;
    }

    void bwd(CUDA cuda, LossBwdIn const &in, LossBwdOut const &out) override {
        INFO_GRP("QuadraticLossTask BWD", INFO_GRP_LAYER_TASK);
        // return output - ground_truth;
        ftype alpha1 = 1, alpha2 = -1, beta = 0;

        CUDNN_CHECK(cudnnOpTensor(cuda.cudnn_handle, addition_, &alpha1,
                                  in.y_pred.desc(), in.y_pred.data(), &alpha2,
                                  in.y_true.desc(), in.y_true.data(), &beta,
                                  out.dy.desc(), out.dy.data()));
    }

  private:
    cudnnOpTensorDescriptor_t addition_;
};

#endif
