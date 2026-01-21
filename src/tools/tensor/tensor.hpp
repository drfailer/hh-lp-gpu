#ifndef SRC_TOOLS_TENSOR_TENSOR
#define SRC_TOOLS_TENSOR_TENSOR
#include "tensor_base.hpp"
#include "../../tools/gpu.hpp"
#include "../../types.hpp"
#include <cstdio>
#include <cudnn_ops.h>
#include <memory>
#include <cassert>

namespace tensor {

template <typename T>
class Tensor : public TensorBase<T> {
  public:
    // constructors ////////////////////////////////////////////////////////////

    Tensor() = default;

    Tensor(TensorShape const &shape) : TensorBase<T>(shape) {
        if (this->size_ == 0)
            return;
        CUDA_CHECK(alloc_gpu(&this->data_, this->size_));
    }

    template <typename... Types>
    Tensor(Types... args) : Tensor(TensorShape(std::forward<Types>(args)...)) {}

    Tensor(Tensor<T> &&tensor) : TensorBase<T>(std::move(tensor)) {}
    Tensor<T> const &operator=(Tensor<T> &&tensor) {
        TensorBase<T>::operator=(std::move(tensor));
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
            this->desc_, CUDNN_DATA_TYPE, shape.dims.size(), shape.dims.data(),
            shape.strides.data()));
        cudaFree(this->data_);
        if (this->size_ == 0) {
            this->data_ = nullptr;
            return;
        }
        CUDA_CHECK(alloc_gpu(&this->data_, this->size_));
    }

    // init ////////////////////////////////////////////////////////////////////

    auto random_init(T lower_bound, T higher_bound, int seed = 0) {
        return memset_random_uniform_gpu<ftype>(
            this->data_, this->size_, lower_bound, higher_bound, seed);
    }

    auto zero() { return memset_gpu<ftype>(this->data_, this->size_, 0); }
};

} // end namespace tensor


#endif
