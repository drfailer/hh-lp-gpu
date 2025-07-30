#ifndef TOOLS_TENSOR_TENSOR_DATA
#define TOOLS_TENSOR_TENSOR_DATA
#include "../../types.hpp"
#include "../gpu.hpp"
#include <array>
#include <cudnn.h>

namespace tensor {

using dims_t = std::array<int, 4>;
using desc_t = cudnnTensorDescriptor_t;

inline dims_t default_strides(dims_t const &dims) {
    return {dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1};
}

template <typename T> struct TensorData {
    T *data = nullptr;
    size_t size = 0;
    dims_t dims = {0};
    dims_t strides = {0};
    desc_t desc = nullptr;

    TensorData(dims_t const &dims, dims_t const &strides)
        : size(dims[0] * dims[1] * dims[2] * dims[3]), dims(dims),
          strides(strides) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));

        if (size == 0)
            return;
        CUDA_CHECK(alloc_gpu(&data, size));
        cudnnSetTensorNdDescriptor(desc, CUDNN_DATA_TYPE, dims.size(),
                                   dims.data(), strides.data());
    }
    TensorData(dims_t const &dims) : TensorData(dims, default_strides(dims)) {}
    TensorData(int b, int c, int h, int w) : TensorData(dims_t{b, c, h, w}) {}

    TensorData(TensorData const &) = delete;
    TensorData const &operator=(TensorData const &) = delete;

    TensorData(TensorData &&other) {
        std::swap(this->data, other.data);
        std::swap(this->size, other.size);
        std::swap(this->dims, other.dims);
        std::swap(this->strides, other.strides);
        std::swap(this->desc, other.desc);
    }

    ~TensorData() {
        cudaFree(data);
        cudnnDestroyTensorDescriptor(desc);
    }

    void reshape(dims_t const &dims, dims_t const &strides) {
        cudaFree(this->data);
        this->dims = dims;
        this->strides = strides;
        this->size = dims[0] * dims[1] * dims[2] * dims[3];
        CUDA_CHECK(alloc_gpu(&data, size));
        cudnnSetTensorNdDescriptor(desc, CUDNN_DATA_TYPE, dims.size(),
                                   dims.data(), strides.data());
    }

    void reshape(dims_t const &dims) {
        this->reshape(dims, default_strides(dims));
    }

    void reshape(int b, int c, int h, int w) {
        this->reshape(dims_t{b, c, h, w});
    }
};

} // end namespace tensor

#endif
