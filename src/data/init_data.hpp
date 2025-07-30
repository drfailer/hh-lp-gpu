#ifndef DATA_INIT_DATA
#define DATA_INIT_DATA
#include "../model/data/nn_state.hpp"
#include <memory>

enum class InitTarget {
    Network,
    Layer,
    Loss,
    Optimizer,
};

template <typename T, InitTarget target = InitTarget::Network> struct InitData {
    std::shared_ptr<NNState<T>> states;
    tensor::dims_t input_dims;
};

#endif
