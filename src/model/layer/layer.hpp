#ifndef MODEL_LAYER_LAYER_H
#define MODEL_LAYER_LAYER_H
#include "../../model/data/cuda_data.hpp"
#include "../../model/data/dims.hpp"
#include "../../model/data/layer_state.hpp"

template <typename T> struct Layer {
    Layer(dims_t dims) : dims(dims) {}
    size_t idx = 0;
    dims_t dims;

    virtual parameters_t<T> create_parameters() const = 0;
    virtual void init(cuda_data_t cuda_data, LayerState<T> &state,
                      int64_t batch_size) = 0;
    virtual Tensor<T> *fwd(cuda_data_t cuda_data, LayerState<T> &states,
                           Tensor<T> *input) = 0;
    virtual Tensor<T> *bwd(cuda_data_t cuda_data, LayerState<T> &states,
                           Tensor<T> *error) = 0;
};

#endif
