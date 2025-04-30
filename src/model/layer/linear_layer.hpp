#ifndef LAYERS_LINEAR_LAYER_H
#define LAYERS_LINEAR_LAYER_H
#include "../../tools/gpu.hpp"
#include "../../types.hpp"
#include "layer.hpp"
#include "log.h/log.h"
#include <cudnn.h>
#include <cudnn_frontend/graph_interface.h>
#include <cudnn_graph.h>
#include <random>

class LinearLayer : public Layer<ftype> {
  public:
    LinearLayer(cublasHandle_t cublas_handle, int64_t input_dim,
                int64_t output_dim)
        : Layer(LayerDims{.inputs = input_dim, .outputs = output_dim}),
          cublas_handle_(cublas_handle) {}

  public:
    /*
     * Allocates memory for a layer state (output memory for the fwd pass, bwd
     * pass, parameters and gradiants).
     */
    void init(LayerState<ftype> &state) override {
        std::mt19937 mt(0);
        std::uniform_real_distribution<> dist(-0.5, 0.5);
        std::vector<ftype> weights_host(this->dims.inputs * this->dims.outputs,
                                        0);
        std::vector<ftype> biases_host(this->dims.outputs, 0);

        for (size_t i = 0; i < weights_host.size(); ++i) {
            weights_host[i] = dist(mt);
        }
        for (size_t i = 0; i < biases_host.size(); ++i) {
            biases_host[i] = dist(mt);
        }

        INFO_GRP("LinearLayer INIT", INFO_GRP_LAYER_TASK);
        state = create_layer_state<ftype>(this->dims, true, true);
        CUDA_CHECK(memcpy_host_to_gpu(state.weights, weights_host.data(),
                                      weights_host.size()));
        CUDA_CHECK(memcpy_host_to_gpu(state.biases, biases_host.data(),
                                      biases_host.size()));
        cudaDeviceSynchronize();
    }

    ftype *fwd(LayerState<ftype> &state, ftype *input) override {
        INFO_GRP("LinearLayer FWD", INFO_GRP_LAYER_TASK);

        // save input (used for the backwards pass)
        state.input = input;

        CUDA_CHECK(
            memcpy_gpu_to_gpu(state.output, state.biases, state.dims.outputs));
        CUBLAS_CHECK(matvecmul(cublas_handle_, false, state.dims.outputs,
                               state.dims.inputs, 1.f, state.weights,
                               state.input, 1.f, state.output));
        return state.output;
    }

    ftype *bwd(LayerState<ftype> &state, ftype *error) override {
        INFO_GRP("LinearLayer BWD", INFO_GRP_LAYER_TASK);
        // grads_b = biases
        CUDA_CHECK(memcpy_gpu_to_gpu(state.gradiants.biases, error,
                                     state.dims.outputs));
        // w_grad = err * update_inputT
        CUBLAS_CHECK(matmul(cublas_handle_, false, true, state.dims.outputs,
                            state.dims.inputs, 1, 1.f, error, state.input, 0.f,
                            state.gradiants.weights));
        // output_err = errT * weights
        CUBLAS_CHECK(matmul(cublas_handle_, true, false, 1, state.dims.inputs,
                            state.dims.outputs, 1.f, error,
                            state.weights, 0.f, state.error));
        return state.error;
    }

  private:
    cublasHandle_t cublas_handle_ = nullptr;
};

#endif
