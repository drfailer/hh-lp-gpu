#ifndef MODEL_LAYER_LAYER_H
#define MODEL_LAYER_LAYER_H
#include "../../model/data/cuda_data.hpp"
#include "../../model/data/dims.hpp"
#include "../../model/data/layer_data.hpp"
#include "../../model/data/parameters.hpp"

template <typename T>
struct parameters_t {
    tensor::Tensor<T> &w;
    tensor::Tensor<T> &b;
};

template <typename T>
struct fwd_data_t {
    tensor::Tensor<T> const &w;
    tensor::Tensor<T> const &b;
};

template <typename T>
struct bwd_data_t {
    tensor::Tensor<const T> &x;
    tensor::Tensor<T> &y;
    tensor::Tensor<T> &w;
    tensor::Tensor<T> &b;
    tensor::Tensor<T> &dw;
    tensor::Tensor<T> &db;
};

template <typename T> struct Layer {
    size_t idx = 0;
    dims_t dims;

    Layer(dims_t dims) : dims(dims) {}
    virtual ~Layer() {}

    virtual LayerParametersShape parameters_shape() const {
        return {};
    }
    virtual LayerIOShape io_shape(tensor::dims_t const &input_dims) const = 0;
    virtual void init_parameters(cuda_data_t cuda, parameters_t<T> params) {}

    // override optional
    virtual void init_fwd(cuda_data_t cuda, LayerData<T> const &data) {}
    virtual void init_bwd(cuda_data_t cuda, LayerData<T> const &data) {}

    virtual void fwd(cuda_data_t cuda, fwd_data_t<T> const &data,
                     tensor::Tensor<const T> const &x, tensor::Tensor<T> &y) = 0;
    virtual void bwd(cuda_data_t cuda, bwd_data_t<T> const &data,
                     tensor::Tensor<const T> const &dy, tensor::Tensor<T> &dx) = 0;
};

#endif
