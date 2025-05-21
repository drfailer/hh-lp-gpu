#ifndef MODEL_OPTIMIZER_OPTIMIZER_H
#define MODEL_OPTIMIZER_OPTIMIZER_H
#include "../../model/data/layer_state.hpp"
#include "../../model/data/shape.hpp"
#include <memory>

template <typename T> struct Optimizer {
    virtual std::shared_ptr<Optimizer<T>>
    create(shape_t const &shape) const = 0;
    virtual void optimize(LayerState<T> const &state, T learning_rate) = 0;
};

#endif
