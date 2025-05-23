#ifndef MODEL_DATA_SHAPE
#define MODEL_DATA_SHAPE
#include "tensor.hpp"

struct shape_t {
    struct {
        vec_t weights = {};
        vec_t biases = {};
    } dims;
    struct {
        vec_t weights = {};
        vec_t biases = {};
    } strides;
};

#endif
