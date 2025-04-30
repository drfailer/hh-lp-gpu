#ifndef DATA_INFERENCE_DATA_H
#define DATA_INFERENCE_DATA_H
#include "../model/data/network_state.hpp"

template <typename T>
struct InferenceData {
    NetworkState<T> &states;
    T *input;
};

#warning "we should have the input and the result here (will facilitate the evaluation)"

#endif
