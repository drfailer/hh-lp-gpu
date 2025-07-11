#ifndef MODEL_DATA_DIMS
#define MODEL_DATA_DIMS

// This a copy of data, for the rest of the graph, dimension is of type shape_t
// which contains vectors for the dimensions and strides of the parameter
// tensors. This struct can be used inside the layers to improve readability.
struct dims_t {
    int inputs = 1;
    int outputs = 1;
    int kernel_width = 1;
    int kernel_height = 1;
    int channels = 1;
    int batch_size = 1;
};

#endif
