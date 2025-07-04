#ifndef DATA_BWD_DATA_H
#define DATA_BWD_DATA_H
#include "../model/data/nn_state.hpp"
#include <memory>

template <typename T> struct BwdData {
    std::shared_ptr<NNState<T>> states;
    Tensor<T> *error;
};

#endif
