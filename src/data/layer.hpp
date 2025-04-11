#ifndef DATA_LAYER_H
#define DATA_LAYER_H
#include "../tools/gpu.hpp"
#include <cudnn.h>
#include <numeric>
#include <vector>

struct LayerDimentions {
    int64_t nb_nodes = 1;
    int64_t nb_inputs = 1;
    int64_t kernel_size = 1;
};

template <typename T> struct Layer {
    T *weights_host = nullptr;
    T *biases_host = nullptr;
    T *weights_gpu = nullptr;
    T *biases_gpu = nullptr;
    LayerDimentions dims;
};

template <typename T> Layer<T> layer_create(LayerDimentions dims) {
    int64_t size = dims.nb_nodes * dims.nb_inputs * dims.kernel_size;
    T *weights_gpu;
    T *biases_gpu;

    CUDA_CHECK(cudaMalloc((void **)(&weights_gpu), size * sizeof(T)));
    CUDA_CHECK(cudaMalloc((void **)(&biases_gpu), dims.nb_nodes * sizeof(T)));
    return Layer<T>{.weights_host = new T[size],
                    .biases_host = new T[dims.nb_nodes],
                    .weights_gpu = weights_gpu,
                    .biases_gpu = biases_gpu,
                    .dims = dims};
}

template <typename T> void layer_destroy(Layer<T> &layer) {
    CUDA_CHECK(cudaFree(layer.weights_gpu));
    CUDA_CHECK(cudaFree(layer.biases_gpu));
    delete[] layer.weights_host;
    delete[] layer.biases_host;
}

template <typename T> void layer_sync_to_gpu(Layer<T> &layer) {
    int64_t size =
        layer.dims.nb_nodes * layer.dims.nb_inputs * layer.dims.kernel_size;
    CUDA_CHECK(memcpy_host_to_gpu(layer.weights_gpu, layer.weights_host, size));
    CUDA_CHECK(memcpy_host_to_gpu(layer.biases_gpu, layer.biases_host,
                                  layer.dims.nb_nodes));
    cudaDeviceSynchronize();
}

template <typename T> void layer_sync_to_host(Layer<T> &layer) {
    int64_t size =
        layer.dims.nb_nodes * layer.dims.nb_inputs * layer.dims.kernel_size;
    CUDA_CHECK(memcpy_gpu_to_host(layer.weights_host, layer.weights_gpu, size));
    CUDA_CHECK(memcpy_gpu_to_host(layer.biases_host, layer.biases_gpu,
                                  layer.dims.nb_nodes));
    cudaDeviceSynchronize();
}

template <typename T> void layer_init(Layer<T> &layer, T value) {
    int64_t size =
        layer.dims.nb_nodes * layer.dims.nb_inputs * layer.dims.kernel_size;
    for (int64_t i = 0; i < size; ++i) {
        layer.weights_host[i] = value;
    }
    for (int64_t i = 0; i < layer.dims.nb_nodes; ++i) {
        layer.biases_host[i] = value;
    }
    layer_sync_to_gpu(layer);
}

#endif
