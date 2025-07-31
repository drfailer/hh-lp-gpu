#ifndef TOOLS_TENSOR_TENSOR_VIEW
#define TOOLS_TENSOR_TENSOR_VIEW
#include "abstract_tensor.hpp"
#include <cudnn_ops.h>

namespace tensor {

template <typename T> struct BorrowingTensor : AbstractTensor<T> {
    BorrowingTensor() = default;

    BorrowingTensor(TensorShape const &shape, T *data)
        : AbstractTensor<T>(shape) {
        this->desc = descriptor_from_shape(shape);
        this->data = data;
    }

    BorrowingTensor(BorrowingTensor &&data) { this->operator=(std::move(data)); }
    BorrowingTensor const &operator=(BorrowingTensor &&data) {
        return AbstractTensor<T>::operator=(std::move(data));
    }

    ~BorrowingTensor() override { cudnnDestroyTensorDescriptor(this->desc); }

    void reshape(TensorShape const &shape) override {
        assert(shape.size() <= this->shape.size());
        this->shape = shape;
        this->size = shape.size();
        CUDNN_CHECK(cudnnSetTensorNdDescriptor(
            this->desc, CUDNN_DATA_TYPE, shape.dims.size(), shape.dims.data(),
            shape.strides.data()));
    }
};

} // end namespace tensor

#endif
