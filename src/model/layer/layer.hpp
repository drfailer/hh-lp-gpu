#ifndef MODEL_LAYER_LAYER_H
#define MODEL_LAYER_LAYER_H
#include "../../model/data/cuda_data.hpp"
#include "../../model/data/dims.hpp"
#include "../../model/data/layer_data.hpp"

struct Parameters {
    tensor::Tensor<ftype> &w;
    tensor::Tensor<ftype> &b;
};

using InitFwdData = LayerData<ftype>;
using InitBwdData = LayerData<ftype>;

struct LayerFwdIn {
    tensor::Tensor<const ftype> &x;
    tensor::Tensor<ftype> &w;
    tensor::Tensor<ftype> &b;
};

struct LayerFwdOut {
    tensor::Tensor<ftype> &y;
};

struct LayerBwdIn {
    tensor::Tensor<const ftype> &dy;
    tensor::Tensor<const ftype> &x;
    tensor::Tensor<ftype> &y;
    tensor::Tensor<ftype> &w;
    tensor::Tensor<ftype> &b;
};

struct LayerBwdOut {
    tensor::Tensor<ftype> &dx;
    tensor::Tensor<ftype> &dw;
    tensor::Tensor<ftype> &db;
};

template <typename T> struct Layer {
    size_t idx = 0;
    dims_t dims;

    Layer(dims_t dims) : dims(dims) {}
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
