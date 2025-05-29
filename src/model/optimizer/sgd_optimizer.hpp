#ifndef MODEL_OPTIMIZER_SGD_OPTIMIZER_H
#define MODEL_OPTIMIZER_SGD_OPTIMIZER_H
#include "../../tools/gpu.hpp"
#include "../../types.hpp"
#include "optimizer.hpp"
#include <cudnn.h>
#include <cudnn_graph.h>
#include <cudnn_ops.h>
#include <log.h/log.h>

struct SGDOptimizer : Optimizer<ftype> {
    ftype learning_rate;

    SGDOptimizer(ftype learning_rate) : learning_rate(learning_rate) {}

    void optimize(cuda_data_t cuda_data,
                  LayerState<ftype> const &state) override {
        INFO_GRP("Optimizer", INFO_GRP_LAYER_TASK);

        // params = params - learning_rate * gradients

        ftype alpha = -learning_rate, beta = 1;

        if (state.parameters.weights) {
            CUDNN_CHECK(cudnnAddTensor(cuda_data.cudnn_handle, &alpha,
                                       state.gradients.weights->descriptor(),
                                       state.gradients.weights->data(), &beta,
                                       state.parameters.weights->descriptor(),
                                       state.parameters.weights->data()));
        }

        if (state.parameters.biases) {
            CUDNN_CHECK(cudnnAddTensor(cuda_data.cudnn_handle, &alpha,
                                       state.gradients.biases->descriptor(),
                                       state.gradients.biases->data(), &beta,
                                       state.parameters.biases->descriptor(),
                                       state.parameters.biases->data()));
        }
    }

    std::shared_ptr<Optimizer<ftype>> create() const override {
        return std::make_shared<SGDOptimizer>(learning_rate);
    }
};

#endif
