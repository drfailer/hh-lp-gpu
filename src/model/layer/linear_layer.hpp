#ifndef LAYERS_LINEAR_LAYER_H
#define LAYERS_LINEAR_LAYER_H
#include "../../tools/gpu.hpp"
#include "../../types.hpp"
#include "layer.hpp"
#include "log.h/log.h"
#include <cudnn.h>
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
     * pass, parameters and gradients).
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

        weights_array.resize(state.dims.batch_count, nullptr);
        inputs_array.resize(state.dims.batch_count, nullptr);
        output_errors_array.resize(state.dims.batch_count, nullptr);
        outputs_array.resize(state.dims.batch_count, nullptr);
        errors_array.resize(state.dims.batch_count, nullptr);
        weights_gradients_array.resize(state.dims.batch_count, nullptr);
    }

    ftype *fwd(LayerState<ftype> &state, ftype *input) override {
        INFO_GRP("LinearLayer FWD", INFO_GRP_LAYER_TASK);

        // save input (used for the backwards pass)
        state.input = input;

        for (int64_t b = 0; b < state.dims.batch_count; ++b) {
            weights_array[b] = state.weights;
            inputs_array[b] = &state.input[b * state.dims.inputs];
            outputs_array[b] = &state.output[b * state.dims.outputs];
        }

        CUDA_CHECK(
            memcpy_gpu_to_gpu(state.output, state.biases,
                              state.dims.outputs * state.dims.batch_count));
        CUBLAS_CHECK(matvecmul(cublas_handle_, false, state.dims.outputs,
                               state.dims.inputs, 1.f, weights_array.data(),
                               inputs_array.data(), 1.f, outputs_array.data(),
                               state.dims.batch_count));
        return state.output;
    }

    ftype *bwd(LayerState<ftype> &state, ftype *error) override {
        INFO_GRP("LinearLayer BWD", INFO_GRP_LAYER_TASK);

        for (int64_t b = 0; b < state.dims.batch_count; ++b) {
            weights_array[b] = state.weights;
            inputs_array[b] = &state.input[b * state.dims.inputs];
            errors_array[b] = &error[b * state.dims.outputs];
            weights_gradients_array[b] =
                &state.gradients
                     .weights[b * state.dims.outputs * state.dims.inputs];
            output_errors_array[b] = &state.error[b * state.dims.outputs];
        }

        // grads_b = biases
        CUDA_CHECK(
            memcpy_gpu_to_gpu(state.gradients.biases, error,
                              state.dims.outputs * state.dims.batch_count));
        // w_grad = err * update_inputT
        CUBLAS_CHECK(matmul(
            cublas_handle_, false, true, state.dims.outputs, state.dims.inputs,
            1, 1.f, errors_array.data(), inputs_array.data(), 0.f,
            weights_gradients_array.data(), state.dims.batch_count));
        // output_err = errT * weights
        CUBLAS_CHECK(matmul(
            cublas_handle_, true, false, 1, state.dims.inputs,
            state.dims.outputs, 1.f, errors_array.data(), weights_array.data(),
            0.f, output_errors_array.data(), state.dims.batch_count));
        return state.error;
    }

  private:
    cublasHandle_t cublas_handle_ = nullptr;

    // arrays of pointers used to call the batch version of sgemv and sgemm
    std::vector<ftype *> weights_array = {};
    std::vector<ftype *> inputs_array = {};
    std::vector<ftype *> outputs_array = {};
    std::vector<ftype *> errors_array = {};
    std::vector<ftype *> output_errors_array = {};
    std::vector<ftype *> weights_gradients_array = {};
};

#endif
