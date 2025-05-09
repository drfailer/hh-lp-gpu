#ifndef MODEL_OPTIMIZER_OPTIMIZER_H
#define MODEL_OPTIMIZER_OPTIMIZER_H
#include "../../model/data/layer_state.hpp"
#include <memory>

template <typename T>
struct Optimizer {
    virtual void init(LayerState<T> const &state) = 0;
    virtual void optimize(LayerState<T> const &state, T learning_rate) = 0;
    virtual std::shared_ptr<Optimizer<T>> copy() const = 0;
};

#endif
