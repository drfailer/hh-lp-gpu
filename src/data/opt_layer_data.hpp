#ifndef DATA_OPT_LAYER_DATA_H
#define DATA_OPT_LAYER_DATA_H
#include "../model/data/network_data.hpp"
#include <memory>

template <typename T> struct OptLayerData {
    std::shared_ptr<NetworkData<T>> state;
    size_t idx;
};

#endif
