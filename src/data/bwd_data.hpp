#ifndef DATA_BWD_DATA_H
#define DATA_BWD_DATA_H
#include "model.hpp"

template <typename T>
struct BwdData {
    Model<T> &model;
    T *input_gpu;
};

#endif
