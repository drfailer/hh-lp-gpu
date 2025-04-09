#ifndef DATA_LAYER_H
#define DATA_LAYER_H
#include <cudnn.h>
#include <numeric>
#include <vector>

template <typename T> struct Layer {
    T *weights_cpu = nullptr;
    T *biases_cpu = nullptr;
    T *weights_gpu = nullptr;
    T *biases_gpu = nullptr;
    std::vector<int64_t> dims;
};

template <typename T> Layer<T> layer_create(std::vector<int64_t> dims) {
    int64_t size = std::reduce(dims.begin(), dims.end(), 0, std::plus());
    T *weights_gpu;
    T *biases_gpu;

    cudaMalloc((void **)(&weights_gpu), size * sizeof(T));
    cudaMalloc((void **)(&biases_gpu), dims.back() * sizeof(T));
    return Layer<T>{.weights_cpu = new T[size],
                    .biases_cpu = new T[dims.back()],
                    .weights_gpu = weights_gpu,
                    .biases_gpu = biases_gpu,
                    .dims = dims};
}

template <typename T> void layer_destroy(Layer<T> &layer) {
    cudaFree(layer.weights_gpu);
    cudaFree(layer.biases_gpu);
    delete[] layer.weights_cpu;
    delete[] layer.biases_cpu;
}

template <typename T>
void layer_sync_to_gpu(Layer<T> &layer) {
    int64_t size =
        std::reduce(layer.dims.begin(), layer.dims.end(), 0, std::plus());
    cudaMemcpy(layer.weights_gpu, layer.weights_cpu, size * sizeof(*layer.weights_gpu),
               cudaMemcpyHostToDevice);
    cudaMemcpy(layer.biases_gpu, layer.biases_cpu,
               layer.dims.back() * sizeof(*layer.biases_gpu),
               cudaMemcpyHostToDevice);
}

template <typename T>
void layer_sync_to_cpu(Layer<T> &layer) {
    int64_t size =
        std::reduce(layer.dims.begin(), layer.dims.end(), 0, std::plus());
    cudaMemcpy(layer.weights_cpu, layer.weights_gpu, size * sizeof(*layer.weights_gpu),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(layer.biases_cpu, layer.biases_gpu,
               layer.dims.back() * sizeof(*layer.biases_gpu),
               cudaMemcpyDeviceToHost);
}

#endif
