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

    void optimize(CUDA cuda, OptimizerIn const &in, OptimizerOut const &out) override {
        INFO_GRP("Optimizer", INFO_GRP_LAYER_TASK);

        // params = params - learning_rate * gradients

        ftype alpha = -learning_rate, beta = 1;

        if (!out.w.empty() && out.w.size() > 0) {
            CUDNN_CHECK(cudnnAddTensor(cuda.cudnn_handle, &alpha,
                                       in.dw.desc(), in.dw.data(), &beta,
                                       out.w.desc(), out.w.data()));
        }

        if (!out.b.empty() && out.b.size() > 0) {
            CUDNN_CHECK(cudnnAddTensor(cuda.cudnn_handle, &alpha,
                                       in.db.desc(), in.db.data(), &beta,
                                       out.b.desc(), out.b.data()));
        }
    }

    std::shared_ptr<Optimizer<ftype>> copy() const override {
        return std::make_shared<SGDOptimizer>(learning_rate);
    }
};

#endif
