#ifndef DATA_BWD_DATA_H
#define DATA_BWD_DATA_H
#include "layer.hpp"

template <typename T>
struct BwdInputData {
    Layer<T> layer;
    T *z;
    T *act;
    T *err;
};

template <typename T>
struct BwdOutputData {
    T *grad_w;
    T *grad_b;
};

#endif
