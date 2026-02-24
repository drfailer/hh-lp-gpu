#ifndef MODEL_LAYER_LAYER_H
#define MODEL_LAYER_LAYER_H
#include "../../model/data/cuda_data.hpp"
#include "../../model/data/dims.hpp"
#include "../../model/data/layer_data.hpp"

struct Parameters {
    tensor::Tensor &w;
    tensor::Tensor &b;
};

using InitFwdData = LayerData;
using InitBwdData = LayerData;

struct LayerFwdIn {
    tensor::TensorView &x;
    tensor::Tensor &w;
    tensor::Tensor &b;
};

struct LayerFwdOut {
    tensor::Tensor &y;
};

struct LayerBwdIn {
    tensor::TensorView &dy;
    tensor::TensorView &x;
    tensor::Tensor &y;
    tensor::Tensor &w;
    tensor::Tensor &b;
};

struct LayerBwdOut {
    tensor::Tensor &dx;
    tensor::Tensor &dw;
    tensor::Tensor &db;
};

struct Layer {
    size_t idx = 0;
    dims_t dims;
    tensor::dtype_t dtype;

    Layer(dims_t dims, tensor::dtype_t dtype) : dims(dims), dtype(dtype) {}
    virtual ~Layer() {}

    // get shapes
    virtual LayerParametersShape parameters_shape() const { return {}; }
    virtual LayerIOShape io_shape(tensor::dims_t const &input_dims) const = 0;

    // init functions
    virtual void init_parameters(CUDA cuda, Parameters const &params) {}
    virtual void init_fwd(CUDA cuda, InitFwdData const &data) {}
    virtual void init_bwd(CUDA cuda, InitFwdData const &data) {}

    // fwd and bwd implementation
    virtual void fwd(CUDA cuda, LayerFwdIn const &in, LayerFwdOut const &out) = 0;
    virtual void bwd(CUDA cuda, LayerBwdIn const &in, LayerBwdOut const &out) = 0;
};

#endif
