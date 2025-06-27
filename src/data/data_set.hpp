#ifndef DATA_DATA_SET_H
#define DATA_DATA_SET_H
#include <vector>
#include <cuda_runtime_api.h>
#include "../model/data/tensor.hpp"

template <typename T>
struct Data {
    Tensor<T> input;
    Tensor<T> ground_truth;
};

template <typename T>
struct DataSet {
    std::vector<Data<T>> datas;
};

#endif
