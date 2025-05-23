#ifndef MODEL_OPTIMIZER_SGD_OPTIMIZER_H
#define MODEL_OPTIMIZER_SGD_OPTIMIZER_H
#include "../../tools/gpu.hpp"
#include "../../types.hpp"
#include "optimizer.hpp"
#include <cudnn.h>
#include <cudnn_graph.h>
#include <cudnn_ops.h>
#include <log.h/log.h>

class SGDOptimizer : public Optimizer<ftype> {
  public:
    SGDOptimizer(cudnnHandle_t cudnn_handle) : cudnn_handle_(cudnn_handle) {}

    void optimize(layer_state_t<ftype> const &state,
                  ftype learning_rate) override {
        INFO_GRP("Optimizer", INFO_GRP_LAYER_TASK);

        // params = params - learning_rate * gradients

        ftype alpha = -learning_rate, beta = 1;

        if (state.parameters && state.parameters->weights) {
            CUDNN_CHECK(cudnnAddTensor(cudnn_handle_, &alpha,
                                       state.gradients->weights->descriptor(),
                                       state.gradients->weights->data(), &beta,
                                       state.parameters->weights->descriptor(),
                                       state.parameters->weights->data()));
        }

        if (state.parameters && state.parameters->biases) {
            CUDNN_CHECK(cudnnAddTensor(cudnn_handle_, &alpha,
                                       state.gradients->biases->descriptor(),
                                       state.gradients->biases->data(), &beta,
                                       state.parameters->biases->descriptor(),
                                       state.parameters->biases->data()));
        }
    }

    std::shared_ptr<Optimizer<ftype>> create() const override {
        return std::make_shared<SGDOptimizer>(cudnn_handle_);
    }

  private:
    cudnnHandle_t cudnn_handle_ = nullptr;
};

#endif
