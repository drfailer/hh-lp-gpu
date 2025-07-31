#ifndef TOOLS_TENSOR_TENSOR_OWNER
#define TOOLS_TENSOR_TENSOR_OWNER
#include "abstract_tensor.hpp"

namespace tensor {

template <typename T> struct OwningTensor : AbstractTensor<T> {
    OwningTensor() = default;

    template <typename... Types>
    OwningTensor(Types... args)
        : AbstractTensor<T>(std::forward<Types>(args)...) {
        this->desc = descriptor_from_shape(this->shape);
        if (this->size == 0)
            return;
        CUDA_CHECK(alloc_gpu(&this->data, this->size));
    }

    OwningTensor(OwningTensor &&data) { this->operator=(std::move(data)); }
    OwningTensor const &operator=(OwningTensor &&data) {
        return AbstractTensor<T>::operator=(std::move(data));
    }

    ~OwningTensor() override {
        cudaFree(this->data);
        cudnnDestroyTensorDescriptor(this->desc);
    }

    void reshape(TensorShape const &shape) override {
        assert(shape.size() <= this->shape.size());
        this->shape = shape;
        this->size = shape.size();
        CUDNN_CHECK(cudnnSetTensorNdDescriptor(
            this->desc, CUDNN_DATA_TYPE, shape.dims.size(), shape.dims.data(),
            shape.strides.data()));
        cudaFree(this->data);
        if (this->size == 0) {
            this->data = nullptr;
            return;
        }
        CUDA_CHECK(alloc_gpu(&this->data, this->size));
    }
};

} // end namespace tensor

#endif
