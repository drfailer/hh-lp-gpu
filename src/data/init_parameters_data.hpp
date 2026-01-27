#ifndef DATA_INIT_PARAMETERS_DATA
#define DATA_INIT_PARAMETERS_DATA
#include "../model/data/network_data.hpp"
#include "init_data.hpp"
#include <memory>

template <typename T, InitTarget target = InitTarget::Network>
struct InitParametersData {
    std::shared_ptr<NetworkData<T>> states;

    // the communicator needs these functions (it was not design for
    // transfering no data, maybe this is a featcure that should be added
    // because it would be useful for signaling with data types)
    char buf[1];
    hh::comm::Package pack() {
        return hh::comm::Package{.data = { hh::comm::Buffer{buf, sizeof(buf)} } };
    }
    void unpack(hh::comm::Package) {}
    hh::comm::Package package() {
        return hh::comm::Package{.data = { hh::comm::Buffer{buf, sizeof(buf)} } };
    }
};

#endif
