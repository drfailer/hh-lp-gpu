#ifndef MODEL_OPTIMIZER_OPTIMIZER_H
#define MODEL_OPTIMIZER_OPTIMIZER_H
#include "../../model/data/cuda_data.hpp"
#include "../../model/data/layer_data.hpp"
#include <memory>

using OptimizerInitData = LayerData<ftype>;

struct OptimizerIn {
    tensor::Tensor &dw;
    tensor::Tensor &db;
};

struct OptimizerOut {
    tensor::Tensor &w;
    tensor::Tensor &b;
};

template <typename T> struct Optimizer {
    virtual void init(CUDA cuda, OptimizerInitData const &data) {}
    virtual void optimize(CUDA cuda, OptimizerIn const &in, OptimizerOut const &out) = 0;
    virtual std::shared_ptr<Optimizer<T>> copy() const = 0;
};

#endif
