#ifndef DATA_INFERENCE_DATA_H
#define DATA_INFERENCE_DATA_H
#include "../model/data/nn_state.hpp"
#include <memory>

template <typename T>
struct InferenceData {
    std::shared_ptr<NNState<T>> states;
    Tensor<T> *input;
};

#endif
