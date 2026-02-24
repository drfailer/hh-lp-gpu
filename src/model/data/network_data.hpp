#ifndef MODEL_DATA_NETWORK_DATA
#define MODEL_DATA_NETWORK_DATA
#include "layer_data.hpp"
#include "loss_data.hpp"

struct NetworkData {
    NetworkData(size_t nb_layers) : layers_datas(nb_layers) {}

    std::vector<LayerData> layers_datas = {};
    LossData loss;
};

#endif
