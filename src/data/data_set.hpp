#ifndef DATA_DATA_SET_H
#define DATA_DATA_SET_H
#include <vector>
#include <cuda_runtime_api.h>

template <typename T>
struct Data {
    T *input = nullptr;
    T *ground_truth = nullptr;
};

template <typename T>
struct DataSet {
    std::vector<Data<T>> datas;
};

template <typename T>
void destroy_data_set(DataSet<T> &data_set) {
    for (auto &data : data_set.datas) {
        cudaFree(data.input);
        cudaFree(data.ground_truth);
    }
}

#endif
