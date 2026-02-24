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

template <typename... Types> Tensor tensor(Types... args) {
    return Tensor(TensorShape(std::forward<Types>(args)...), CUDNN_DATA_TYPE);
}

inline Tensor tensor_like(Tensor const &tensor) {
    return Tensor(tensor.shape(), tensor.data_type());
}

inline TensorView tensor_view(TensorShape const &shape, void *data, data_type_t data_type = CUDNN_DATA_TYPE) {
    return TensorView(shape, data, data_type);
}

inline TensorView tensor_view_of(Tensor const &tensor) {
    return TensorView(tensor.shape(), tensor.data(), tensor.data_type());
}

} // end namespace tensor

#endif
