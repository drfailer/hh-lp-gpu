#ifndef DATA_FWD_DATA_H
#define DATA_FWD_DATA_H
#include "model.hpp"

template <typename T>
struct FwdData {
    Model<T> &model;
    T *input_gpu;
};

#endif
