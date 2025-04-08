#ifndef DATA_MODEL_H
#define DATA_MODEL_H
#include "layer.hpp"

template <typename T>
struct Model {
    std::vector<Layer<T>> layers;
};

#endif
