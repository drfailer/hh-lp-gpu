#ifndef DATA_DATA_SET_H
#define DATA_DATA_SET_H
#include <vector>

template <typename T>
struct Data {
    T *input = nullptr;
    T *ground_truth = nullptr;
    size_t batch_size = 0;
};

template <typename T>
struct DataSet {
    std::vector<Data<T>> datas;
};

#endif
