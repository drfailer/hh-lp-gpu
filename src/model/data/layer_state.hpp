#ifndef MODEL_DATA_LAYER_STATE_H
#define MODEL_DATA_LAYER_STATE_H
#include "dims.hpp"
#include <cstddef>

template <typename T> struct LayerState {
    T *input = nullptr;  // input of the forward pass (gpu)
    T *output = nullptr; // output of the forward pass (gpu)
    T *error = nullptr;  // output of the backwards pass (gpu)
    T *weights = nullptr;
    T *biases = nullptr;
    struct {
        T *weights = nullptr;
        T *biases = nullptr;
    } gradients;
};

template <typename T>
LayerState<T> create_layer_state(dims_t dims, bool use_weights,
                                 bool use_biases) {
    LayerState<T> state;
    size_t weights_size = dims.inputs * dims.outputs * dims.kernel_height *
                          dims.kernel_width * dims.channels;

    CUDA_CHECK(alloc_gpu(&state.output, dims.batch_count * dims.outputs));
    CUDA_CHECK(alloc_gpu(&state.error, dims.batch_count * dims.inputs));
    if (use_weights) {
        CUDA_CHECK(alloc_gpu(&state.weights, weights_size));
        CUDA_CHECK(alloc_gpu(&state.gradients.weights,
                             dims.batch_count * weights_size));
    }
    if (use_biases) {
        CUDA_CHECK(alloc_gpu(&state.biases, dims.outputs));
        CUDA_CHECK(alloc_gpu(&state.gradients.biases,
                             dims.batch_count * dims.outputs));
    }
    return state;
}

template <typename T> void destroy_layer_state(LayerState<T> &state) {
    cudaFree(state.output);
    state.output = nullptr;
    cudaFree(state.error);
    state.error = nullptr;
    cudaFree(state.weights);
    state.weights = nullptr;
    cudaFree(state.gradients.weights);
    state.gradients.weights = nullptr;
    cudaFree(state.biases);
    state.biases = nullptr;
    cudaFree(state.gradients.biases);
    state.gradients.biases = nullptr;
}

#endif
