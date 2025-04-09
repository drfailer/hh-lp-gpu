#ifndef DATA_FWD_DATA_H
#define DATA_FWD_DATA_H
#include "layer.hpp"

template <typename T>
struct FwdInputData {
    Layer<T> layer;
    T *input_gpu;
    T *z_gpu;
    T *act_gpu;
};

template <typename T>
struct FwdOutputData {
    T *z_gpu;
    T *act_gpu;
};

#endif
