#ifndef DATA_BWD_DATA_H
#define DATA_BWD_DATA_H
#include "../model/data/network_data.hpp"
#include <memory>

template <typename T> struct BwdData {
    std::shared_ptr<NetworkData<T>> network_data;
    tensor::Tensor<T> const *error;
};

#endif
