#ifndef DATA_FWD_DATA_H
#define DATA_FWD_DATA_H
#include "layer.hpp"

template <typename T>
struct FwdData {
    std::vector<Layer<T>>::iterator layer;
    T *input_gpu;
    T *output_gpu;
};

#endif
