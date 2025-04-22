#ifndef DATA_LAYER_STATE_H
#define DATA_LAYER_STATE_H
#include "../tools/gpu.hpp"
#include <cstdint>

/******************************************************************************/
/*                                   types                                    */
/******************************************************************************/

struct LayerDimentions {
    int64_t nb_nodes = 1;
    int64_t nb_inputs = 1;
    int64_t kernel_size = 1;
};

template <typename T> struct Parameters {
    T *weights = nullptr; // gpu
    T *biases = nullptr;  // gpu
};

template <typename T> struct LayerState {
    T *input = nullptr;   // input of the forward pass (gpu)
    T *output = nullptr;  // output of the forward pass (gpu)
    T *error = nullptr;   // output of the backwards pass (gpu)
    Parameters<T> params; // parameters (weights + baises)
    Parameters<T> grads;  // gradiants
    LayerDimentions dims; // dimentions of the layer
};


template <typename T> struct NetworkState {
    std::vector<LayerState<T>> layer_states;
    // TODO: we should have a configurable type for different loss functions
    // (Adam needs more data)
    T *loss_output;
};

/******************************************************************************/
/*                            parameters functions                            */
/******************************************************************************/

template <typename T>
Parameters<T> parameters_create_gpu(LayerDimentions const &dims) {
    int64_t size = dims.nb_nodes * dims.nb_inputs * dims.kernel_size;
    Parameters<T> parameters;

    CUDA_CHECK(alloc_gpu(&parameters.weights, size));
    CUDA_CHECK(alloc_gpu(&parameters.biases, dims.nb_nodes));
    return parameters;
}

template <typename T> void parameters_destroy_gpu(Parameters<T> &parameters) {
    cudaFree(parameters.weights);
    cudaFree(parameters.biases);
    parameters.weights = nullptr;
    parameters.biases = nullptr;
}

template <typename T>
Parameters<T> parameters_create_host(LayerDimentions const &dims) {
    int64_t size = dims.nb_nodes * dims.nb_inputs * dims.kernel_size;
    Parameters<T> parameters;

    parameters.weights = new T[size];
    parameters.biases = new T[dims.nb_nodes];
    return parameters;
}

template <typename T> void parameters_destroy_host(Parameters<T> &parameters) {
    delete[] parameters.weights;
    delete[] parameters.biases;
    parameters.weights = nullptr;
    parameters.biases = nullptr;
}

template <typename T>
void parameters_host_to_gpu(Parameters<T> &gpu, Parameters<T> const &host,
                            LayerDimentions const &dims) {
    size_t size = dims.kernel_size * dims.nb_nodes * dims.nb_inputs;

    CUDA_CHECK(memcpy_host_to_gpu(gpu.weights, host.weights, size));
    CUDA_CHECK(memcpy_host_to_gpu(gpu.biases, host.biases, dims.nb_nodes));
}

template <typename T>
void parameters_gpu_to_host(Parameters<T> &gpu, Parameters<T> const &host,
                            LayerDimentions const &dims) {
    size_t size = dims.kernel_size * dims.nb_nodes * dims.nb_inputs;

    CUDA_CHECK(memcpy_gpu_to_host(host.weights, gpu.weights, size));
    CUDA_CHECK(memcpy_gpu_to_host(host.biases, gpu.biases, dims.nb_nodes));
}

/******************************************************************************/
/*                           layer state functions                            */
/******************************************************************************/

template <typename T>
LayerState<T> layer_state_create_gpu(LayerDimentions const &dims,
                                     Parameters<T> const &params,
                                     Parameters<T> const &grads) {
    LayerState<T> state;

    // warn: the layer deosn't own the input vector
    CUDA_CHECK(alloc_gpu(&state.output, dims.nb_nodes));
    CUDA_CHECK(alloc_gpu(&state.error, dims.nb_inputs));
    state.params = params;
    state.grads = grads;
    state.dims = dims;
    return state;
}

template <typename T> void layer_state_destroy_gpu(LayerState<T> &state) {
    cudaFree(state.output);
    cudaFree(state.error);
    state.output = nullptr;
    state.error = nullptr;
}

template <typename T>
LayerState<T> layer_state_create_host(LayerDimentions const &dims,
                                      Parameters<T> const &params,
                                      Parameters<T> const &grads) {
    LayerState<T> state;

    // warn: the layer deosn't own the input vector
    state.output = new T[dims.nb_nodes];
    state.error = new T[dims.nb_inputs];
    state.params = params;
    state.grads = grads;
    state.dims = dims;
    return state;
}

template <typename T> void layer_state_destroy_host(LayerState<T> &state) {
    delete[] state.output;
    delete[] state.error;
    state.output = nullptr;
    state.error = nullptr;
}

template <typename T>
void layer_state_host_to_gpu(LayerState<T> &gpu, LayerState<T> &host) {
    CUDA_CHECK(memcpy_host_to_gpu(gpu.output, host.output, gpu.dims.nb_nodes));
    CUDA_CHECK(memcpy_host_to_gpu(gpu.error, host.error, gpu.dims.error));
}

template <typename T>
void layer_state_gpu_to_host(LayerState<T> &host, LayerState<T> &gpu) {
    CUDA_CHECK(memcpy_gpu_to_host(host.output, gpu.output, host.dims.nb_nodes));
    CUDA_CHECK(memcpy_gpu_to_host(host.error, gpu.error, host.dims.error));
}

#endif
