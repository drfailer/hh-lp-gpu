#ifndef MODEL_LAYER_LAYER_H
#define MODEL_LAYER_LAYER_H
#include "../../model/data/cuda_data.hpp"
#include "../../model/data/dims.hpp"
#include "../../model/data/layer_data.hpp"
#include "../../model/data/parameters.hpp"

template <typename T> struct Layer {
    size_t idx = 0;
    dims_t dims;

    Layer(dims_t dims) : dims(dims) {}
    virtual ~Layer() {}

    // TODO: layers should use lists of tensors instead of the current struct
    //       (more flexible in case a layer needs more data, like combined
    //       layers)
    // TODO: create_parameters should be removed and everything should be done
    //       in `init`. However, initialization should not create tensors but
    //       tensor views, and allocation should be done automatically in the
    //       graph (this way, we can skip some allocations, like only use
    //       weights for the inference as well as using only two memory spaces
    //       for input and output). The graph or the network will deallocate
    //       memory if needed.

    virtual Parameters<T> create_parameters() const = 0;
    virtual tensor::dims_t init(cuda_data_t cuda_data, LayerData<T> &state,
                               tensor::dims_t input_dims) = 0;
    virtual tensor::Tensor<T> const &fwd(cuda_data_t cuda_data, LayerData<T> &states,
                                 tensor::Tensor<T> const &input) = 0;
    virtual tensor::Tensor<T> const &bwd(cuda_data_t cuda_data, LayerData<T> &states,
                                 tensor::Tensor<T> const &input,
                                 tensor::Tensor<T> const &output_gradient) = 0;
};

#endif
