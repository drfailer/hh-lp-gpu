#ifndef DATA_INFERENCE_DATA_H
#define DATA_INFERENCE_DATA_H
#include "layer_state.hpp"

template <typename T>
struct InferenceData {
    NetworkState<T> &states;
    T *input;
};

#endif
