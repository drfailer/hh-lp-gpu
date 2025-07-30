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

    void optimize(cuda_data_t cuda_data, LayerData<ftype> &state) override {
        INFO_GRP("Optimizer", INFO_GRP_LAYER_TASK);

        // params = params - learning_rate * gradients

        ftype alpha = -learning_rate, beta = 1;

        if (!state.w.empty()) {
            CUDNN_CHECK(cudnnAddTensor(cuda_data.cudnn_handle, &alpha,
                                       state.dw.desc(),
                                       state.dw.data(), &beta,
                                       state.w.desc(),
                                       state.w.data()));
        }

        if (!state.b.empty()) {
            CUDNN_CHECK(cudnnAddTensor(cuda_data.cudnn_handle, &alpha,
                                       state.db.desc(),
                                       state.db.data(), &beta,
                                       state.b.desc(),
                                       state.b.data()));
        }
    }

    std::shared_ptr<Optimizer<ftype>> copy() const override {
        return std::make_shared<SGDOptimizer>(learning_rate);
    }
};

#endif
