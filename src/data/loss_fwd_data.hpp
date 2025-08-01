#ifndef DATA_LOSS_FWD_DATA_H
#define DATA_LOSS_FWD_DATA_H
#include "../model/data/network_data.hpp"
#include <memory>

template <typename T> struct LossFwdData {
    std::shared_ptr<NetworkData<T>> states;
    tensor::Tensor<T> *input;
    tensor::Tensor<T> *ground_truth;
    tensor::Tensor<T> *result;
};

#endif
