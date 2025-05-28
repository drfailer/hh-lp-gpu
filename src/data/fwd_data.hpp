#ifndef DATA_FWD_DATA_H
#define DATA_FWD_DATA_H
#include "../model/data/nn_state.hpp"
#include <memory>

template <typename T> struct FwdData {
    std::shared_ptr<NNState<T>> states;
    Tensor<T> *input;
};

#endif
