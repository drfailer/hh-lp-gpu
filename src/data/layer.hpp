#ifndef DATA_LAYER_H
#define DATA_LAYER_H
#include <cstddef>
#include <vector>

template <typename T>
struct Layer {
    T *weights_cpu = nullptr;
    T *biases_cpu = nullptr;
    T *weights_gpu = nullptr;
    T *biases_gpu = nullptr;
    std::vector<size_t> dims;
};

#endif
