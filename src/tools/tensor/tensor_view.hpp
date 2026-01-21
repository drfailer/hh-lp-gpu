#ifndef TOOLS_TENSOR_TENSOR_VIEW
#define TOOLS_TENSOR_TENSOR_VIEW
#include "tensor_base.hpp"
#include <cudnn_ops.h>
#include <cassert>

namespace tensor {

template <typename T>
class TensorView : public TensorBase<const T> {
  public:
    // constructors & destructor ///////////////////////////////////////////////

    TensorView() = default;

    TensorView(TensorShape const &shape, T const *data) : TensorBase<const T>(shape) {
        this->data_ = data;
    }

    TensorView(TensorView<T> &&view) : TensorBase<const T>(std::move(view)) {}

    TensorView<T> const &operator=(TensorView<T> &&view) {
        TensorBase<const T>::operator=(std::move(view));
        return *this;
    }


    // reshape /////////////////////////////////////////////////////////////////

    void reshape(TensorShape const &shape) override {
        assert(shape.size() <= this->shape_.size());
        this->shape_ = shape;
        this->size_ = shape.size();
        CUDNN_CHECK(cudnnSetTensorNdDescriptor(
            this->desc_, CUDNN_DATA_TYPE, shape.dims.size(), shape.dims.data(),
            shape.strides.data()));
    }
};

} // end namespace tensor

#endif
