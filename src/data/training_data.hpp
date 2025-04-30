#ifndef DATA_TRAINING_DATA_H
#define DATA_TRAINING_DATA_H
#include "data_set.hpp"
#include "../model/data/network_state.hpp"

template <typename T>
struct TrainingData {
    NetworkState<T> &states;
    DataSet<T> data_set;
    T learning_rate;
    size_t epochs;
};

#endif
