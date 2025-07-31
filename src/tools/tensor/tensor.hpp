#ifndef TOOLS_TENSOR_TENSOR
#define TOOLS_TENSOR_TENSOR
#include "../../tools/gpu.hpp"
#include "../../types.hpp"
#include "abstract_tensor.hpp"
#include "owning_tensor.hpp"
#include "tensor_view.hpp"
#include <cstdio>
#include <cudnn_ops.h>
#include <memory>

namespace tensor {

// tensor interface ////////////////////////////////////////////////////////////

template <typename T> class TensorInterface {
  public:
    TensorInterface() = default;
    TensorInterface(std::shared_ptr<AbstractTensor<T>> td) : td_(td) {}

    TensorInterface(TensorInterface<T> const &tensor) : td_(tensor.td_) {}
    TensorInterface<T> const &operator=(TensorInterface<T> const &tensor) {
        if (&tensor == this) {
            return *this;
        }
        this->td_ = tensor.td_;
        return *this;
    }

    TensorInterface(TensorInterface<T> &&tensor) : td_(std::move(tensor.td_)) {}
    TensorInterface<T> const &operator=(TensorInterface<T> &&tensor) {
        this->td_ = std::move(tensor.td_);
        return *this;
    }

    // data access /////////////////////////////////////////////////////////////

    T const *data() const { return td_->data; }
    T *data() { return td_->data; }
    void data(T *data) { td_->data = data; }
    dims_t const &dims() const { return td_->shape.dims; }
    dims_t const &strides() const { return td_->shape.strides; }
    int dim(size_t i) const { return td_->shape.dims[i]; }
    int stride(size_t i) const { return td_->shape.strides[i]; }
    TensorShape const &shape() const { return td_->shape; }
    desc_t desc() const { return td_->desc; }
    size_t size() const { return td_->size; }
    bool empty() const { return td_ == nullptr; }

    // reshape /////////////////////////////////////////////////////////////////

    void reshape(TensorShape const &shape) {
        if (empty()) {
            throw std::runtime_error("error: cannot reshape empty tensor.");
        }
        td_->reshape(shape);
    }

    template <typename... Types> void reshape(Types... args) {
        this->reshape(TensorShape(std::forward<Types>(args)...));
    }

    void reshape_like(TensorInterface<T> other) { this->reshape(other.shape()); }

    // view ////////////////////////////////////////////////////////////////////

    TensorInterface<T> view() {
        return tensor_view(this->td_.shape(), this->td_.data());
    }

    TensorInterface<T> view_as(TensorShape const &shape) {
        return tensor_view(shape, this->td_.data());
    }

    // init ////////////////////////////////////////////////////////////////////

    auto random_init(T lower_bound, T higher_bound, int seed = 0) {
        return memset_random_uniform_gpu<ftype>(
            td_->data, td_->size, lower_bound, higher_bound, seed);
    }

    auto zero() { return memset_gpu<ftype>(td_->data, td_->size, 0); }

    // host data transfer //////////////////////////////////////////////////////

    // assums that the host array has the proper size
    auto from_host(T *host) {
        return memcpy_host_to_gpu(td_->data, host, td_->size);
    }

    // assums that the host array has the proper size
    auto to_host(T *host) const {
        return memcpy_gpu_to_host(host, td_->data, td_->size);
    }

  private:
    std::shared_ptr<AbstractTensor<T>> td_ = nullptr;
};

// tensor types ////////////////////////////////////////////////////////////////

// TODO: remove the templates and use void* as a type
// using Tensor = TensorInterface<void>;
// using ConstTensor = TensorInterface<const void>;

template <typename T>
using Tensor = TensorInterface<T>;
template <typename T>
using ConstTensor = TensorInterface<const T>;

// helper functions ////////////////////////////////////////////////////////////

template <typename T, typename... Types> Tensor<T> tensor(Types... args) {
    std::shared_ptr<AbstractTensor<T>> to =
        std::make_shared<OwningTensor<T>>(std::forward<Types>(args)...);
    return Tensor<T>(to);
}

template <typename T> Tensor<T> tensor_like(Tensor<T> const &tensor) {
    std::shared_ptr<AbstractTensor<T>> to =
        std::make_shared<OwningTensor<T>>(tensor.shape());
    return Tensor<T>(to);
}

template <typename T> Tensor<T> tensor_view(TensorShape const &shape, T *data) {
    std::shared_ptr<AbstractTensor<T>> tv =
        std::make_shared<BorrowingTensor<T>>(shape, data);
    return Tensor<T>(tv);
}

template <typename T> Tensor<T> tensor_view_of(Tensor<T> const &tensor) {
    std::shared_ptr<AbstractTensor<T>> tv =
        std::make_shared<BorrowingTensor<T>>(tensor.shape(), tensor.data());
    return Tensor<T>(tv);
}

} // end namespace tensor

#define print_tensor_descriptor(desc)                                          \
    do {                                                                       \
        int n, c, h, w;                                                        \
        int ns, cs, hs, ws;                                                    \
        cudnnDataType_t data_type;                                             \
        cudnnGetTensor4dDescriptor(desc, &data_type, &n, &c, &h, &w, &ns, &cs, \
                                   &hs, &ws);                                  \
        printf(#desc ": [%d, %d, %d, %d]%d : (%d, %d, %d, %d)\n", n, c, h, w,  \
               data_type, ns, cs, hs, ws);                                     \
    } while (0);

#endif
