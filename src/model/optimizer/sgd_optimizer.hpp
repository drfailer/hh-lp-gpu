#ifndef MODEL_OPTIMIZER_SGD_OPTIMIZER_H
#define MODEL_OPTIMIZER_SGD_OPTIMIZER_H
#include "../../tools/gpu.hpp"
#include "../../tools/tensor/tensor.hpp"
#include "../../types.hpp"
#include "optimizer.hpp"
#include <cudnn.h>
#include <cudnn_graph.h>
#include <cudnn_ops.h>

struct SGDOptimizer : Optimizer<ftype> {
    ftype learning_rate;

    SGDOptimizer(ftype learning_rate) : learning_rate(learning_rate) {}

    void optimize(cuda_data_t cuda_data, LayerState<ftype> &state) override {
        INFO_GRP("Optimizer", INFO_GRP_LAYER_TASK);

        // params = params - learning_rate * gradients

        ftype alpha = -learning_rate, beta = 1;

        if (!state.parameters.weights.empty()) {
            CUDNN_CHECK(cudnnAddTensor(cuda_data.cudnn_handle, &alpha,
                                       state.gradients.weights.desc(),
                                       state.gradients.weights.data(), &beta,
                                       state.parameters.weights.desc(),
                                       state.parameters.weights.data()));
        }

        if (!state.parameters.biases.empty()) {
            CUDNN_CHECK(cudnnAddTensor(cuda_data.cudnn_handle, &alpha,
                                       state.gradients.biases.desc(),
                                       state.gradients.biases.data(), &beta,
                                       state.parameters.biases.desc(),
                                       state.parameters.biases.data()));
        }
    }

    std::shared_ptr<Optimizer<ftype>> copy() const override {
        return std::make_shared<SGDOptimizer>(learning_rate);
    }
};

#endif
