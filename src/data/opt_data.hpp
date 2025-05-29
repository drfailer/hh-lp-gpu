#ifndef DATA_OPT_DATA_H
#define DATA_OPT_DATA_H
#include "../model/data/nn_state.hpp"
#include <memory>

template <typename T> struct OptData {
    std::shared_ptr<NNState<T>> states;
};

#endif
