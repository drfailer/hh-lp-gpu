#ifndef MODEL_DATA_SHAPE
#define MODEL_DATA_SHAPE
#include <vector>

// IMPORTANT: the first dimension of the parameter is always the batch count.

struct shape_t {
    struct {
        std::vector<int64_t> weights = {};
        std::vector<int64_t> biases = {};
    } dims;
    struct {
        std::vector<int64_t> weights = {};
        std::vector<int64_t> biases = {};
    } strides;
};

#endif
