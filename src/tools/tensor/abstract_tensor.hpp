#ifndef TOOLS_TENSOR_ABSTRACT_TENSOR
#define TOOLS_TENSOR_ABSTRACT_TENSOR
#include "../../types.hpp"
#include "../gpu.hpp"
#include "tensor_shape.hpp"
#include <cassert>
#include <cudnn.h>

namespace tensor {

using desc_t = cudnnTensorDescriptor_t;

template <typename T> struct AbstractTensor {
    TensorShape shape = {};
    size_t size = 0;
    T *data = nullptr;
    desc_t desc = nullptr;

    AbstractTensor() = default;
    AbstractTensor(TensorShape const &shape)
        : shape(shape), size(shape.size()) {}
    template <typename... Types>
    AbstractTensor(Types... args)
        : AbstractTensor(TensorShape(std::forward<Types>(args)...)) {}

    AbstractTensor(AbstractTensor const &) = delete;
    AbstractTensor const &operator=(AbstractTensor const &) = delete;

    AbstractTensor(AbstractTensor &&data) { this->operator=(std::move(data)); }
    AbstractTensor const &operator=(AbstractTensor &&data) {
        std::swap(this->data, data.data);
        std::swap(this->size, data.size);
        std::swap(this->shape, data.shape);
        std::swap(this->desc, data.desc);
        return *this;
    }

    virtual ~AbstractTensor() = default;

    virtual void reshape(TensorShape const &shape) = 0;
};

inline desc_t descriptor_from_shape(TensorShape const &shape) {
    desc_t desc = nullptr;;
    if (shape.size() == 0)
        return desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(desc, CUDNN_DATA_TYPE,
                                           shape.dims.size(), shape.dims.data(),
                                           shape.strides.data()));
    return desc;
}

} // end namespace tensor

#endif
