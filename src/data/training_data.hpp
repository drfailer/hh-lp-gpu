#ifndef DATA_TRAINING_DATA_H
#define DATA_TRAINING_DATA_H
#include "../model/data/network_data.hpp"
#include "data_set.hpp"
#include <memory>

template <typename T> struct TrainingData {
    std::shared_ptr<NetworkData<T>> states;
    // TODO:
    // the data set should be const, and its data should be
    // allocated on the CPU (which is not the case for now).
    // When creating the FwdData, the data should be copied
    // from the data set to the tensor of the fwd data (which
    // will be allcoated on the GPU and not const to allow
    // deserialization when using shards).
    DataSet<T> &data_set;
    size_t epochs;
};

#endif
