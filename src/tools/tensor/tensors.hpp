#ifndef SRC_TOOLS_TENSOR_TENSORS
#define SRC_TOOLS_TENSOR_TENSORS
#include "tensor.hpp"
#include "tensor_view.hpp"

#define print_tensor_descriptor(desc)                                          \
    do {                                                                       \
        int n, c, h, w;                                                        \
        int ns, cs, hs, ws;                                                    \
        cudnnDataType_t dtype;                                             \
        cudnnGetTensor4dDescriptor(desc, &dtype, &n, &c, &h, &w, &ns, &cs, \
                                   &hs, &ws);                                  \
        printf(#desc ": [%d, %d, %d, %d]%d : (%d, %d, %d, %d)\n", n, c, h, w,  \
               dtype, ns, cs, hs, ws);                                     \
    } while (0);

namespace tensor {

Tensor tensor(TensorShape const &shape, dtype_t dtype = CUDNN_DATA_TYPE) {
    return Tensor(shape, dtype);
}

Tensor tensor(int b, int c, int h, int w, dtype_t dtype = CUDNN_DATA_TYPE) {
    return Tensor(TensorShape(b, c, h, w), dtype);
}

inline Tensor tensor_like(Tensor const &tensor) {
    return Tensor(tensor.shape(), tensor.dtype());
}

inline TensorView tensor_view(TensorShape const &shape, void *data, tensor::dtype_t dtype = CUDNN_DATA_TYPE) {
    return TensorView(shape, data, dtype);
}

inline TensorView tensor_view_of(Tensor const &tensor) {
    return TensorView(tensor.shape(), tensor.data(), tensor.dtype());
}

} // end namespace tensor

#endif
