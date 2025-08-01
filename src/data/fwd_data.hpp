#ifndef DATA_FWD_DATA_H
#define DATA_FWD_DATA_H
#include "../model/data/network_data.hpp"
#include <memory>

template <typename T> struct FwdData {
    std::shared_ptr<NetworkData<T>> network_data;
    tensor::Tensor<T> const *input;
};

#endif
