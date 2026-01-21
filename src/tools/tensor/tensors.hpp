#ifndef SRC_TOOLS_TENSOR_TENSORS
#define SRC_TOOLS_TENSOR_TENSORS
#include "tensor.hpp"
#include "tensor_view.hpp"

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

namespace tensor {

template <typename T, typename... Types> Tensor<T> tensor(Types... args) {
    return Tensor<T>(std::forward<Types>(args)...);
}

template <typename T> Tensor<T> tensor_like(Tensor<T> const &tensor) {
    return Tensor<T>(tensor.shape());
}

template <typename T> TensorView<T> tensor_view(TensorShape const &shape, T *data) {
    return TensorView<T>(shape, data);
}

template <typename T> TensorView<T> tensor_view_of(Tensor<T> const &tensor) {
    return TensorView<T>(tensor.shape(), tensor.data());
}

} // end namespace tensor

#endif
