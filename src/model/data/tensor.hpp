#ifndef MODEL_DATA_TENSOR
#define MODEL_DATA_TENSOR
#include "../../tools/gpu.hpp"
#include "../../types.hpp"
#include "dims.hpp"
#include <array>
#include <cstdio>
#include <cudnn_ops.h>

using vec_t = std::array<int, 4>;

template <typename T> class Tensor {
  public:
    Tensor(vec_t const &dims = {0}, vec_t const &strides = {0})
        : dims_(dims), strides_(strides) {
        data_size_ = dims[0] * dims[1] * dims[2] * dims[3];
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&descriptor_));

        if (data_size_ == 0)
            return;
        CUDA_CHECK(alloc_gpu(&data_, data_size_));
        CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
            descriptor_, CUDNN_DATA_TYPE, dims[0], dims[1], dims[2], dims[3],
            strides[0], strides[1], strides[2], strides[3]));
    }

    Tensor(Tensor const &) = delete;
    Tensor const &operator=(Tensor const &) = delete;

    Tensor(Tensor &&other) {
        std::swap(this->data_, other.data_);
        std::swap(this->dims_, other.dims_);
        std::swap(this->strides_, other.strides_);
        std::swap(this->descriptor_, other.descriptor_);
        std::swap(this->data_size_, other.data_size_);
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
    size_t data_size() const { return data_size_; }

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
        data_size_ = dims[0] * dims[1] * dims[2] * dims[3];
        CUDA_CHECK(alloc_gpu(&data_, data_size_));
        CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
            descriptor_, CUDNN_DATA_TYPE, dims[0], dims[1], dims[2], dims[3],
            strides[0], strides[1], strides[2], strides[3]));
    }

    void reshape(vec_t const &dims) {
        reshape(dims,
                {dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1});
    }

  public:
    // assums that the host array has the proper size
    auto from_host(T *host) {
        return memcpy_host_to_gpu(data_, host,
                                  dims_[0] * dims_[1] * dims_[2] * dims_[3]);
    }

    // assums that the host array has the proper size
    auto to_host(T *host) {
        return memcpy_gpu_to_host(host, data_,
                                  dims_[0] * dims_[1] * dims_[2] * dims_[3]);
    }

  private:
    T *data_ = nullptr;
    vec_t dims_ = {};
    vec_t strides_ = {};
    cudnnTensorDescriptor_t descriptor_ = nullptr;
    size_t data_size_;
};

template <typename T>
Tensor<T> *create_tensor(vec_t const &dims, vec_t const &strides) {
    return new Tensor<T>(dims, strides);
}

template <typename T> Tensor<T> *create_tensor(vec_t const &dims) {
    return create_tensor<T>(
        dims, {dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1});
}

template <typename T>
Tensor<T> *create_tensor_from_dims(tensor_dims_t const &dims) {
    return create_tensor<T>({dims.n, dims.c, dims.h, dims.w});
}

#define print_tensor_descriptor(desc)                                          \
    {                                                                          \
        int n, c, h, w;                                                        \
        int ns, cs, hs, ws;                                                    \
        cudnnDataType_t data_type;                                             \
        cudnnGetTensor4dDescriptor(desc, &data_type, &n, &c, &h, &w, &ns, &cs, \
                                   &hs, &ws);                                  \
        printf(#desc ": [%d, %d, %d, %d]%d : (%d, %d, %d, %d)\n", n, c, h, w,    \
               data_type, ns, cs, hs, ws);                                     \
    }

#endif
