#ifndef TOOLS_TENSOR_TENSOR_VIEW
#define TOOLS_TENSOR_TENSOR_VIEW
#include "tensor_base.hpp"
#include <cudnn_ops.h>
#include <cassert>

namespace tensor {

class TensorView : public TensorBase<const void> {
  public:
    // constructors & destructor ///////////////////////////////////////////////

    TensorView() = default;

    TensorView(TensorShape const &shape, void const *data, data_type_t data_type) : TensorBase<const void>(shape, data_type) {
        this->data_ = data;
    }

    TensorView(TensorView &&view) : TensorBase<const void>(std::move(view)) {}

    TensorView const &operator=(TensorView &&view) {
        TensorBase<const void>::operator=(std::move(view));
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
