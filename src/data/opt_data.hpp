#ifndef DATA_OPT_DATA_H
#define DATA_OPT_DATA_H
#include "../model/data/network_data.hpp"
#include <memory>

struct OptData {
    std::shared_ptr<NetworkData> states;
};

#endif
