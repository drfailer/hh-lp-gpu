#ifndef DATA_INFERENCE_DATA_H
#define DATA_INFERENCE_DATA_H
#include "../model/data/network_state.hpp"

template <typename T>
struct InferenceData {
    NetworkState<T> &states;
    Tensor<T> *input;
};

#endif
