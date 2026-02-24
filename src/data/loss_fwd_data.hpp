#ifndef DATA_LOSS_FWD_DATA_H
#define DATA_LOSS_FWD_DATA_H
#include "../model/data/network_data.hpp"
#include <memory>

struct LossFwdData {
    std::shared_ptr<NetworkData> states;
    tensor::Tensor *input;
    tensor::Tensor *ground_truth;
    tensor::Tensor *result;
};

#endif
