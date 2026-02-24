#ifndef DATA_OPT_LAYER_DATA_H
#define DATA_OPT_LAYER_DATA_H
#include "../model/data/network_data.hpp"
#include <memory>

struct OptLayerData {
    std::shared_ptr<NetworkData> state;
    size_t idx;

    hh::comm::Package pack() {
        // printf("OptLayerData::pack()\n");
        return hh::comm::Package{
            .data = {
                hh::comm::Buffer{
                    (char*)&this->idx, sizeof(this->idx),
                },
            },
        };
    }
    void unpack(hh::comm::Package) {
        // printf("OptLayerData::unpack()\n");
    }
    hh::comm::Package package() {
        // printf("OptLayerData::package()\n");
        return hh::comm::Package{
            .data = {
                hh::comm::Buffer{
                    (char*)&this->idx, sizeof(this->idx),
                },
            },
        };
    }
};

#endif
