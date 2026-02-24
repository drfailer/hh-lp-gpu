#ifndef SRC_TOOLS_TENSOR_TENSOR_BASE
#define SRC_TOOLS_TENSOR_TENSOR_BASE
#include "../../tools/gpu.hpp"
#include "../../types.hpp"
#include "tensor_shape.hpp"
#include <cstdio>
#include <cudnn_ops.h>
#include <memory>
#include <cudnn.h>
#include <exception>
#include <type_traits>

namespace tensor {

using desc_t = cudnnTensorDescriptor_t;
using dtype_t = cudnnDataType_t;

inline size_t tensor_dtype_size(dtype_t dtype) {
    switch (dtype) {
    case CUDNN_DATA_FLOAT: return sizeof(float); break;
    case CUDNN_DATA_DOUBLE: return sizeof(double); break;
    }
    return 0;
}

inline desc_t descriptor_from_shape(TensorShape const &shape, dtype_t dtype) {
    desc_t desc = nullptr;;
    if (shape.size() == 0)
        return desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(desc, dtype,
                                           shape.dims.size(), shape.dims.data(),
                                           shape.strides.data()));
    return desc;
}

template <typename T>
class TensorBase {
  public:
    // constructors ////////////////////////////////////////////////////////////

    TensorBase() = default;

    TensorBase(TensorShape const &shape, dtype_t dtype)
        : shape_(shape),
          size_(shape.size()),
          desc_(descriptor_from_shape(shape, dtype)),
          dtype_(dtype),
          element_size_(tensor_dtype_size(dtype)) { }

    // copy constructor & copy operator

    TensorBase(TensorBase<T> const &tensor) = delete;
    TensorBase<T> const &operator=(TensorBase<T> const &tensor) = delete;

    // move constructor & move operator

    TensorBase(TensorBase<T> &&base) { this->operator=(std::move(base)); }
    TensorBase<T> const &operator=(TensorBase<T> &&base) {
        std::swap(this->data_, base.data_);
        std::swap(this->size_, base.size_);
        std::swap(this->shape_, base.shape_);
        std::swap(this->desc_, base.desc_);
        std::swap(this->dtype_, base.dtype_);
        std::swap(this->element_size_, base.element_size_);
        return *this;
    }

    ~TensorBase() {
        cudnnDestroyTensorDescriptor(this->desc_);
    }

    // data access /////////////////////////////////////////////////////////////

    T const *data() const { return this->data_; }
    T *data() { return this->data_; }
    T data(T *data) { this->data_ = data; }
    dims_t const &dims() const { return this->shape_.dims; }
    dims_t const &strides() const { return this->shape_.strides; }
    int dim(size_t i) const { return this->shape_.dims[i]; }
    int stride(size_t i) const { return this->shape_.strides[i]; }
    TensorShape const &shape() const { return this->shape_; }
    desc_t desc() const { return this->desc_; }
    size_t size() const { return this->size_; }
    dtype_t dtype() const { return this->dtype_; }
    size_t element_size() const { return this->element_size_; }

    // reshape /////////////////////////////////////////////////////////////////

    virtual void reshape(TensorShape const &shape) = 0;

    template <typename... Types> void reshape(Types... args) {
        this->reshape(TensorShape(std::forward<Types>(args)...));
    }

    void reshape_like(TensorBase<T> const &other) { this->reshape(other.shape()); }

    // host data transfer //////////////////////////////////////////////////////

    // assums that the host array has the proper size
    auto from_host(void *host) {
        return memcpy_host_to_gpu(this->data_, host, this->size_ * this->element_size_);
    }

    // assums that the host array has the proper size
    auto to_host(void *host) const {
        return memcpy_gpu_to_host(host, this->data_, this->size_ * this->element_size_);
    }

  protected:
    TensorShape shape_ = {};
    size_t size_ = 0;
    T *data_ = nullptr;
    desc_t desc_ = nullptr;
    dtype_t dtype_ = CUDNN_DATA_TYPE;
    size_t element_size_ = 0;
};

} // end namespace tensor

#endif
