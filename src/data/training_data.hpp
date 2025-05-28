#ifndef DATA_TRAINING_DATA_H
#define DATA_TRAINING_DATA_H
#include "../model/data/nn_state.hpp"
#include "data_set.hpp"
#include <memory>

template <typename T> struct TrainingData {
    std::shared_ptr<NNState<T>> states;
    DataSet<T> data_set;
    T learning_rate;
    size_t epochs;
};

#endif
