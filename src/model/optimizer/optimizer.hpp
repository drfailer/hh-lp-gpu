#ifndef MODEL_OPTIMIZER_OPTIMIZER_H
#define MODEL_OPTIMIZER_OPTIMIZER_H
#include "../../model/data/cuda_data.hpp"
#include "../../model/data/layer_data.hpp"
#include <memory>

template <typename T> struct Optimizer {
    virtual void optimize(CUDA cuda_data, LayerData<T> &state) = 0;
    virtual std::shared_ptr<Optimizer<T>> copy() const = 0;
};

#endif
