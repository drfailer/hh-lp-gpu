#ifndef DATA_PREDICTION_DATA_H
#define DATA_PREDICTION_DATA_H
#include "../model/data/nn_state.hpp"
#include <memory>

template <typename T>
struct PredictionData {
    std::shared_ptr<NNState<T>> states;
    Tensor<T> *input;
};

#endif
