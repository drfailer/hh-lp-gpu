#ifndef TOOLS_TENSOR_TENSOR_SHAPE
#define TOOLS_TENSOR_TENSOR_SHAPE
#include <array>
#include <cstddef>
#include "../../tools/gpu.hpp"
#include "../../types.hpp"

namespace tensor {

// using dims_t = std::array<int, 4>;

struct dims_t {
    int data_[4] = {0};
    int *data() { return data_; }
    int const *data() const { return data_; }
    size_t size() const { return 4; }
    int &operator[](size_t i) { return data_[i]; }
    int operator[](size_t i) const { return data_[i]; }
};

struct TensorShape {
    dims_t dims = {0};
    dims_t strides = {0};

    TensorShape() = default;
    TensorShape(dims_t const &dims, dims_t const &strides)
        : dims(dims), strides(strides) {}
    TensorShape(dims_t const &dims)
        : dims(dims), strides(default_strides(dims)) {}
    TensorShape(int b, int c, int h, int w)
        : TensorShape(dims_t{b, c, h, w}) {}
    TensorShape(int b, int c, int h, int w, int sb, int sc, int sh, int sw)
        : TensorShape(dims_t{b, c, h, w}, dims_t{sb, sc, sh, sw}) {}
    TensorShape(TensorShape const &shape)
        : TensorShape(shape.dims, shape.strides) {}

    TensorShape const &operator=(TensorShape const &shape) {
        if (&shape == this) {
            return *this;
        }
        this->dims = shape.dims;
        this->strides = shape.strides;
        return *this;
    }

    dims_t default_strides(dims_t const &dims) {
        return {dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1};
    }

    size_t size() const {
        return dims[0] * dims[1] * dims[2] * dims[3];
    }
};

template <typename ...Types>
TensorShape shape(Types ...args) {
    return TensorShape(std::forward<Types>(args)...);
}

} // end namespace tensor

#endif
