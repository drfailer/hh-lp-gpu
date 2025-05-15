#ifndef MODEL_DATA_LAYER_DIMS_H
#define MODEL_DATA_LAYER_DIMS_H
#include <cstdint>

struct LayerDims {
    int64_t inputs = 1;
    int64_t outputs = 1;
    int64_t kernel_width = 1;
    int64_t kernel_height = 1;
    int64_t channels = 1;
    int64_t batch_count = 1;
};


#endif
