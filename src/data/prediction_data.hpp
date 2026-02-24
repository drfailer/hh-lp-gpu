#ifndef DATA_PREDICTION_DATA_H
#define DATA_PREDICTION_DATA_H
#include "../model/data/network_data.hpp"
#include <memory>

struct PredictionData {
    std::shared_ptr<NetworkData> states;
    tensor::Tensor *input;
};

#endif
