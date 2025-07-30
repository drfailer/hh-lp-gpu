#ifndef TOOLS_TENSOR_TENSOR
#define TOOLS_TENSOR_TENSOR
#include "../../tools/gpu.hpp"
#include "../../types.hpp"
#include "tensor_data.hpp"
#include <cstdio>
#include <cudnn_ops.h>
#include <memory>

namespace tensor {

template <typename T> class Tensor {
  public:
    Tensor() = default;
    Tensor(dims_t const &dims, dims_t const &strides)
        : td_(std::make_shared<TensorData<T>>(dims, strides)) {}
    Tensor(dims_t const &dims) : td_(std::make_shared<TensorData<T>>(dims)) {}
    Tensor(int b, int c, int h, int w) : Tensor(dims_t{b, c, h, w}) {}

    // TODO: add the possibility to borrow the pointer of an other tensor to
    // view it with a different shape

    T const *data() const { return td_->data; }
    T *data() { return td_->data; }
    dims_t const &dims() const { return td_->dims; }
    dims_t const &strides() const { return td_->strides; }
    int dims(size_t i) const { return td_->dims[i]; }
    int strides(size_t i) const { return td_->strides[i]; }
    desc_t desc() const { return td_->desc; }
    size_t size() const { return td_->size; }

    bool empty() const { return td_ == nullptr; }

    template <typename... Types> void reshape(Types... args) {
        if (empty()) {
            td_ = std::make_shared<TensorData<T>>(args...);
        } else {
            td_->reshape(args...);
        }
    }

    void reshape_like(Tensor<T> other) {
        this->reshape(other.dims(), other.strides());
    }

    auto random_init(T lower_bound, T higher_bound, int seed = 0) {
        return memset_random_uniform_gpu<ftype>(
            td_->data, td_->size, lower_bound, higher_bound, seed);
    }

    auto zero() { return memset_gpu<ftype>(td_->data, td_->size, 0); }

    // assums that the host array has the proper size
    auto from_host(T *host) {
        return memcpy_host_to_gpu(td_->data, host, td_->size);
    }

    // assums that the host array has the proper size
    auto to_host(T *host) const {
        return memcpy_gpu_to_host(host, td_->data, td_->size);
    }

  private:
    std::shared_ptr<TensorData<T>> td_ = nullptr;
};

} // end namespace tensor

#define print_tensor_descriptor(desc)                                          \
    {                                                                          \
        int n, c, h, w;                                                        \
        int ns, cs, hs, ws;                                                    \
        cudnnDataType_t data_type;                                             \
        cudnnGetTensor4dDescriptor(desc, &data_type, &n, &c, &h, &w, &ns, &cs, \
                                   &hs, &ws);                                  \
        printf(#desc ": [%d, %d, %d, %d]%d : (%d, %d, %d, %d)\n", n, c, h, w,  \
               data_type, ns, cs, hs, ws);                                     \
    }

#endif
