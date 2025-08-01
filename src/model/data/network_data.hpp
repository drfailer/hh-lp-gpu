#ifndef MODEL_DATA_NETWORK_DATA
#define MODEL_DATA_NETWORK_DATA
#include "layer_data.hpp"
#include "loss_data.hpp"

template <typename T>
struct NetworkData {
    std::vector<LayerData<T>> layers_datas = {};
    LossData<T> loss;
};

#endif
