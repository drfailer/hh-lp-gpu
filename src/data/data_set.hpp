#ifndef DATA_DATA_SET_H
#define DATA_DATA_SET_H
#include <vector>
#include <cuda_runtime_api.h>
#include "../model/data/tensor.hpp"

template <typename T>
struct Data {
    Tensor<T> *input = nullptr;
    Tensor<T> *ground_truth = nullptr;
};

template <typename T>
struct DataSet {
    std::vector<Data<T>> datas;
};

template <typename T>
void destroy_data_set(DataSet<T> &data_set) {
    for (auto &data : data_set.datas) {
        delete data.input;
        delete data.ground_truth;
    }
}

#endif
