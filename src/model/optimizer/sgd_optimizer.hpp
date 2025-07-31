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

    void optimize(cuda_data_t cuda, LayerData<ftype> &data) override {
        INFO_GRP("Optimizer", INFO_GRP_LAYER_TASK);

        // params = params - learning_rate * gradients

        ftype alpha = -learning_rate, beta = 1;

        if (!data.w.empty() && data.w.size() > 0) {
            CUDNN_CHECK(cudnnAddTensor(cuda.cudnn_handle, &alpha,
                                       data.dw.desc(), data.dw.data(), &beta,
                                       data.w.desc(), data.w.data()));
        }

        if (!data.b.empty() && data.b.size() > 0) {
            CUDNN_CHECK(cudnnAddTensor(cuda.cudnn_handle, &alpha,
                                       data.db.desc(), data.db.data(), &beta,
                                       data.b.desc(), data.b.data()));
        }
    }

    std::shared_ptr<Optimizer<ftype>> copy() const override {
        return std::make_shared<SGDOptimizer>(learning_rate);
    }
};

#endif
