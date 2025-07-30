#ifndef DATA_DATA_SET_H
#define DATA_DATA_SET_H
#include <vector>
#include <cuda_runtime_api.h>
#include "../tools/tensor/tensor.hpp"

template <typename T>
struct Data {
    tensor::Tensor<T> input;
    tensor::Tensor<T> ground_truth;
};

template <typename T>
struct DataSet {
    std::vector<Data<T>> datas;
};

#endif
