#ifndef DATA_LOSS_BWD_DATA_H
#define DATA_LOSS_BWD_DATA_H
#include "../model/data/network_data.hpp"
#include <memory>

struct LossBwdData {
    std::shared_ptr<NetworkData> states;
    tensor::Tensor const *y_pred;
    tensor::Tensor const *y_true;
};

#endif
