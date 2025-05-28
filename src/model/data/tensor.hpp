#ifndef MODEL_DATA_TENSOR
#define MODEL_DATA_TENSOR
#include "../../tools/gpu.hpp"
#include <array>
#include <cstdint>
#include <cudnn_ops.h>

using vec_t = std::array<int64_t, 4>;

template <typename T> class Tensor {
  public:
    Tensor(vec_t const &dims = {0}, vec_t const &strides = {0})
        : dims_(dims), strides_(strides) {
        CUDA_CHECK(alloc_gpu(&data_, dims[0] * dims[1] * dims[2] * dims[3]));
        cudnnCreateTensorDescriptor(&descriptor_);
        cudnnSetTensor4dDescriptorEx(descriptor_, CUDNN_DATA_FLOAT, dims[0],
                                     dims[1], dims[2], dims[3], strides[0],
                                     strides[1], strides[2], strides[3]);
    }

    Tensor(Tensor const &) = delete;
    Tensor const &operator=(Tensor const &) = delete;

    Tensor(Tensor &&other) {
        std::swap(this->data_, other.data_);
        std::swap(this->dims_, other.dims_);
        std::swap(this->strides_, other.strides_);
        std::swap(this->descriptor_, other.descriptor_);
    }

    ~Tensor() {
        cudaFree(data_);
        cudnnDestroyTensorDescriptor(descriptor_);
    }

    T const *data() const { return data_; }
    T *data() { return data_; }
    vec_t const &dims() const { return dims_; }
    vec_t const &strides() const { return strides_; }
    cudnnTensorDescriptor_t descriptor() const { return descriptor_; }

    /*
     * Note: we assume that this function will always be called with a complete
     * new shape, so we reallocated automatically
     *
     * This function can also be used as an initializer when using the default
     * constructor.
     */
    void reshape(vec_t const &dims, vec_t const &strides) {
        cudaFree(data_);
        dims_ = dims;
        strides_ = strides;
        CUDA_CHECK(alloc_gpu(&data_, dims[0] * dims[1] * dims[2] * dims[3]));
        cudnnSetTensor4dDescriptorEx(descriptor_, CUDNN_DATA_FLOAT, dims[0],
                                     dims[1], dims[2], dims[3], strides[0],
                                     strides[1], strides[2], strides[3]);
    }

  private:
    T *data_ = nullptr;
    vec_t dims_ = {};
    vec_t strides_ = {};
    cudnnTensorDescriptor_t descriptor_ = nullptr;
};

#endif
