#ifndef DATA_DATA_SET_H
#define DATA_DATA_SET_H
#include <vector>
#include <cuda_runtime_api.h>
#include "../tools/tensor/tensors.hpp"

struct Data {
    tensor::Tensor input;
    tensor::Tensor ground_truth;
};

struct DataSet {
    std::vector<Data> datas;
};

#endif
