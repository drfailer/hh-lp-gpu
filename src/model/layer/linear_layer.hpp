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
        : Layer(
              dims_t{
                  .inputs = input_dim,
                  .outputs = output_dim,
              },
              shape_t{
                  .dims =
                      {
                          .weights = {1, 1, input_dim, output_dim},
                          .biases = {1, 1, output_dim, 1},
                      },
                  .strides =
                      {
                          .weights{input_dim * output_dim,
                                   input_dim * output_dim, output_dim, 1, 1},
                          .biases = {output_dim, output_dim, 1, 1},
                      }}),
          cublas_handle_(cublas_handle) {}

  public:
    /*
     * Allocates memory for a layer state (output memory for the fwd pass, bwd
     * pass, parameters and gradients).
     */
    LayerState<ftype> create_state() const override {
        INFO_GRP("LinearLayer INIT", INFO_GRP_LAYER_TASK);
        LayerState<ftype> state;

        state = create_layer_state<ftype>(this->dims, true, true);
        CUDA_CHECK(memset_random_uniform_gpu<ftype>(
            state.weights, this->dims.inputs * this->dims.outputs, -0.5, 0.5,
            0));
        CUDA_CHECK(memset_random_uniform_gpu<ftype>(
            state.biases, this->dims.inputs * this->dims.outputs, -0.5, 0.5,
            0));
        cudaDeviceSynchronize();
        return state;
    }

    void init() override {
        weights_array.resize(this->dims.batch_count, nullptr);
        inputs_array.resize(this->dims.batch_count, nullptr);
        output_errors_array.resize(this->dims.batch_count, nullptr);
        outputs_array.resize(this->dims.batch_count, nullptr);
        errors_array.resize(this->dims.batch_count, nullptr);
        weights_gradients_array.resize(this->dims.batch_count, nullptr);
    }

    ftype *fwd(LayerState<ftype> &state, ftype *input) override {
        INFO_GRP("LinearLayer FWD", INFO_GRP_LAYER_TASK);

        // save input (used for the backwards pass)
        state.input = input;

        // TODO: should be done in init
        for (int64_t b = 0; b < this->dims.batch_count; ++b) {
            weights_array[b] = state.weights;
            inputs_array[b] = &state.input[b * this->dims.inputs];
            outputs_array[b] = &state.output[b * this->dims.outputs];
            CUDA_CHECK(memcpy_gpu_to_gpu(outputs_array[b], state.biases,
                                         this->dims.outputs));
        }

        CUBLAS_CHECK(matvecmul(cublas_handle_, false, this->dims.outputs,
                               this->dims.inputs, 1.f, weights_array.data(),
                               inputs_array.data(), 1.f, outputs_array.data(),
                               this->dims.batch_count));
        return state.output;
    }

    ftype *bwd(LayerState<ftype> &state, ftype *error) override {
        INFO_GRP("LinearLayer BWD", INFO_GRP_LAYER_TASK);

        for (int64_t b = 0; b < this->dims.batch_count; ++b) {
            weights_array[b] = state.weights;
            inputs_array[b] = &state.input[b * this->dims.inputs];
            errors_array[b] = &error[b * this->dims.outputs];
            weights_gradients_array[b] =
                &state.gradients
                     .weights[b * this->dims.outputs * this->dims.inputs];
            output_errors_array[b] = &state.error[b * this->dims.outputs];
        }

        // grads_b = biases
        CUDA_CHECK(
            memcpy_gpu_to_gpu(state.gradients.biases, error,
                              this->dims.outputs * this->dims.batch_count));
        // w_grad = err * update_inputT
        CUBLAS_CHECK(matmul(
            cublas_handle_, false, true, this->dims.outputs, this->dims.inputs,
            1, 1.f, errors_array.data(), inputs_array.data(), 0.f,
            weights_gradients_array.data(), this->dims.batch_count));
        // output_err = errT * weights
        CUBLAS_CHECK(matmul(
            cublas_handle_, true, false, 1, this->dims.inputs,
            this->dims.outputs, 1.f, errors_array.data(), weights_array.data(),
            0.f, output_errors_array.data(), this->dims.batch_count));
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
