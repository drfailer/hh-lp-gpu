#ifndef DATA_PREDICTION_DATA_H
#define DATA_PREDICTION_DATA_H
#include "../model/data/network_data.hpp"
#include <memory>

template <typename T>
struct PredictionData {
    std::shared_ptr<NetworkData<T>> states;
    tensor::Tensor *input;
};

#endif
