#ifndef SRC_TOOLS_TENSOR_TENSOR
#define SRC_TOOLS_TENSOR_TENSOR
#include "tensor_base.hpp"
#include "../../tools/gpu.hpp"
#include "../../types.hpp"
#include <cstdio>
#include <cuda.h>
#include <cudnn_ops.h>
#include <memory>
#include <cassert>

namespace tensor {

class Tensor : public TensorBase<void> {
  public:
    // constructors ////////////////////////////////////////////////////////////

    Tensor() = default;

    Tensor(TensorShape const &shape, data_type_t data_type) : TensorBase<void>(shape, data_type) {
        if (this->size_ == 0)
            return;
        CUDA_CHECK(cudaMalloc(&this->data_, this->size_ * this->element_size_));
    }

    template <typename... Types>
    Tensor(Types... args, data_type_t data_type) : Tensor(TensorShape(std::forward<Types>(args)...), data_type) {}

    Tensor(Tensor &&tensor) : TensorBase<void>(std::move(tensor)) {}
    Tensor const &operator=(Tensor &&tensor) {
        TensorBase<void>::operator=(std::move(tensor));
        return *this;
    }

    ~Tensor() {
        cudaFree(this->data_);
    }

    // reshape /////////////////////////////////////////////////////////////////

    void reshape(TensorShape const &shape) override {
        assert(shape.size() <= this->shape_.size());
        this->shape_ = shape;
        this->size_ = shape.size();
        CUDNN_CHECK(cudnnSetTensorNdDescriptor(
            this->desc_, this->data_type_, shape.dims.size(), shape.dims.data(),
            shape.strides.data()));
        cudaFree(this->data_);
        if (this->size_ == 0) {
            this->data_ = nullptr;
            return;
        }
        CUDA_CHECK(cudaMalloc(&this->data_, this->size_ * this->element_size_));
    }

    // init ////////////////////////////////////////////////////////////////////

    auto random_init(auto lower_bound, auto higher_bound, int seed = 0) {
        switch (this->data_type_) {
        case CUDNN_DATA_FLOAT:
            return memset_random_uniform_gpu<float>((float*)this->data_, this->size_, lower_bound, higher_bound, seed);
            break;
        case CUDNN_DATA_DOUBLE:
            return memset_random_uniform_gpu<double>((double*)this->data_, this->size_, lower_bound, higher_bound, seed);
            break;
        }
        return cudaErrorInvalidValue;
    }

    auto zero() {
        switch (this->data_type_) {
        case CUDNN_DATA_FLOAT:
            return memset_gpu<float>((float*)this->data_, this->size_, 0);
            break;
        case CUDNN_DATA_DOUBLE:
            return memset_gpu<double>((double*)this->data_, this->size_, 0);
            break;
        }
        return cudaErrorInvalidValue;
    }
};

} // end namespace tensor


#endif
