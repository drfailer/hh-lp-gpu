#ifndef MODEL_LOSS_QUADRATIC_LOSS_H
#define MODEL_LOSS_QUADRATIC_LOSS_H
#include "../../types.hpp"
#include "loss.hpp"
#include <cudnn_ops.h>
#include <log.h/log.h>
#include <cudnn.h>
#include <cudnn_graph.h>

class QuadraticLoss : public Loss<ftype> {
  public:
    QuadraticLoss(cudnnHandle_t cudnn_handle) : cudnn_handle_(cudnn_handle) {}

    ~QuadraticLoss() {
        cudnnDestroyOpTensorDescriptor(addition_);
        cudnnDestroyTensorDescriptor(bwd_graph.model_output_tensor);
        cudnnDestroyTensorDescriptor(bwd_graph.ground_truth_tensor);
        cudnnDestroyTensorDescriptor(bwd_graph.error_tensor);
    }

  public:
    void init(int64_t size) override {
        // model output tensor
        cudnnCreateTensorDescriptor(&bwd_graph.model_output_tensor);
        cudnnSetTensor4dDescriptor(bwd_graph.model_output_tensor,
                                   CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1,
                                   size, 1);

        // ground truth tensor
        cudnnCreateTensorDescriptor(&bwd_graph.ground_truth_tensor);
        cudnnSetTensor4dDescriptor(bwd_graph.ground_truth_tensor,
                                   CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1,
                                   size, 1);
        // result tensor (error)
        cudnnCreateTensorDescriptor(&bwd_graph.error_tensor);
        cudnnSetTensor4dDescriptor(bwd_graph.error_tensor,
                                   CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1,
                                   size, 1);

        // operation tensor
        cudnnCreateOpTensorDescriptor(&addition_);
        cudnnSetOpTensorDescriptor(addition_, CUDNN_OP_TENSOR_ADD,
                                   CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN);
    }

    void fwd(ftype *model_output, ftype *ground_truth, ftype *result) override {
        INFO_GRP("QuadraticLossTask FWD", INFO_GRP_LAYER_TASK);
        ERROR("unimplemented");
        exit(1);
        // ftype diff = ground_truth - output;
        // return 0.5 * diff * diff;
    }

    void bwd(ftype *model_output, ftype *ground_truth, ftype *result) override {
        INFO_GRP("QuadraticLossTask BWD", INFO_GRP_LAYER_TASK);
        // return output - ground_truth;
        ftype alpha1 = 1, alpha2 = -1, beta = 0;

        // TODO: add CUDNN_CHECK
        cudnnOpTensor(cudnn_handle_, addition_, &alpha1,
                      bwd_graph.model_output_tensor, model_output, &alpha2,
                      bwd_graph.ground_truth_tensor, ground_truth, &beta,
                      bwd_graph.error_tensor, result);
    }

  private:
    struct {
        cudnnTensorDescriptor_t model_output_tensor;
        cudnnTensorDescriptor_t ground_truth_tensor;
        cudnnTensorDescriptor_t error_tensor;
    } bwd_graph;
    cudnnHandle_t cudnn_handle_ = nullptr;
    cudnnOpTensorDescriptor_t addition_;
};

#endif
