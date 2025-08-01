#ifndef DATA_TRAINING_DATA_H
#define DATA_TRAINING_DATA_H
#include "../model/data/network_data.hpp"
#include "data_set.hpp"
#include <memory>

template <typename T> struct TrainingData {
    std::shared_ptr<NetworkData<T>> states;
    DataSet<T> const &data_set;
    size_t epochs;
};

#endif
