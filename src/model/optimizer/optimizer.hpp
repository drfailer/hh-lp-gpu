#ifndef MODEL_OPTIMIZER_OPTIMIZER_H
#define MODEL_OPTIMIZER_OPTIMIZER_H
#include "../../model/data/cuda_data.hpp"
#include "../../model/data/layer_state.hpp"
#include <memory>

template <typename T> struct Optimizer {
    virtual void optimize(cuda_data_t cuda_data, LayerState<T> &state) = 0;
    virtual std::shared_ptr<Optimizer<T>> copy() const = 0;
};

#endif
