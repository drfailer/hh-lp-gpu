#ifndef LAYERS_LINEAR_LAYER_H
#define LAYERS_LINEAR_LAYER_H
#include "../../tools/gpu.hpp"
#include "../../types.hpp"
#include "layer.hpp"
#include "log.h/log.h"
#include <cudnn.h>
#include <cudnn_graph.h>
#include <cudnn_ops.h>

class LinearLayer : public Layer<ftype> {
  public:
    LinearLayer(cublasHandle_t cublas_handle, cudnnHandle_t cudnn_handle,
                int64_t input_dim, int64_t output_dim)
        : Layer(dims_t{.inputs = input_dim, .outputs = output_dim}),
          cublas_handle_(cublas_handle), cudnn_handle_(cudnn_handle) {
        CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(&average_tensor));
        CUDNN_CHECK(cudnnSetReduceTensorDescriptor(
            average_tensor, CUDNN_REDUCE_TENSOR_AVG, CUDNN_DATA_FLOAT,
            CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES,
            CUDNN_32BIT_INDICES));
    }

    ~LinearLayer() {
        cudaFree(avg_biases_gradients_ws);
        cudaFree(avg_weights_gradients_ws);
        cudnnDestroyReduceTensorDescriptor(average_tensor);
    }

  public:
    /*
     * Allocates memory for a layer state (output memory for the fwd pass, bwd
     * pass, parameters and gradients).
     */
    parameters_t<ftype> create_parameters() const override {
        INFO_GRP("LinearLayer INIT", INFO_GRP_LAYER_TASK);
        int64_t inputs = this->dims.inputs;
        int64_t outputs = this->dims.outputs;
        parameters_t<ftype> parameters{
            .weights = create_tensor<ftype>({1, 1, outputs, inputs}),
            .biases = create_tensor<ftype>({1, 1, outputs, 1}),
        };

        CUDA_CHECK(memset_random_uniform_gpu<ftype>(
            parameters.weights->data(), outputs * inputs, -0.5, 0.5, 0));
        CUDA_CHECK(memset_random_uniform_gpu<ftype>(parameters.biases->data(),
                                                    outputs, -0.5, 0.5, 0));
        cudaDeviceSynchronize();
        return parameters;
    }

    void init(LayerState<ftype> &state, int64_t batch_size) override {
        int64_t inputs = this->dims.inputs;
        int64_t outputs = this->dims.outputs;

        this->dims.batch_size = batch_size;

        // TODO: use reshape instead
        delete state.output;
        state.output = create_tensor<ftype>({batch_size, 1, outputs, 1});
        delete state.error;
        state.error = create_tensor<ftype>({batch_size, 1, inputs, 1});

        if (batch_size == 1) {
            return;
        }

        weights_array.resize(batch_size, nullptr);
        inputs_array.resize(batch_size, nullptr);
        output_errors_array.resize(batch_size, nullptr);
        outputs_array.resize(batch_size, nullptr);
        errors_array.resize(batch_size, nullptr);
        temp_weights_gradients_array.resize(batch_size, nullptr);

        // create temporary array for the gradients
        temp_weights_gradients.reshape({batch_size, 1, outputs, inputs});
        for (size_t b = 0; b < batch_size; ++b) {
            temp_weights_gradients_array[b] =
                &temp_weights_gradients.data()[b * outputs * inputs];
        }

        // setup tensor descriptors for computing the biases gradients
        // NOTE: the input error tensor has the same dimensions as the output,
        // so it can be used to compute the workspace size
        CUDNN_CHECK(cudnnGetReductionWorkspaceSize(
            cudnn_handle_, average_tensor, state.output->descriptor(),
            state.gradients.biases->descriptor(),
            &avg_biases_gradients_ws_size));
        cudaFree(avg_biases_gradients_ws); // free if needed
        CUDA_CHECK(
            alloc_gpu(&avg_biases_gradients_ws, avg_biases_gradients_ws_size));
        CUDNN_CHECK(cudnnGetReductionWorkspaceSize(
            cudnn_handle_, average_tensor, temp_weights_gradients.descriptor(),
            state.gradients.weights->descriptor(),
            &avg_weights_gradients_ws_size));
        cudaFree(avg_weights_gradients_ws); // free if needed
        CUDA_CHECK(alloc_gpu(&avg_weights_gradients_ws,
                             avg_weights_gradients_ws_size));
    }

    Tensor<ftype> *fwd(LayerState<ftype> &state,
                       Tensor<ftype> *input) override {
        INFO_GRP("LinearLayer FWD", INFO_GRP_LAYER_TASK);

        // save input (used for the backwards pass)
        state.input = input;

        if (this->dims.batch_size > 1) {
            // TODO: should be done in init
            for (int64_t b = 0; b < this->dims.batch_size; ++b) {
                weights_array[b] = state.parameters.weights->data();
                inputs_array[b] = &state.input->data()[b * this->dims.inputs];
                outputs_array[b] =
                    &state.output->data()[b * this->dims.outputs];
                CUDA_CHECK(memcpy_gpu_to_gpu(outputs_array[b],
                                             state.parameters.biases->data(),
                                             this->dims.outputs));
            }

            CUBLAS_CHECK(matvecmul(
                cublas_handle_, false, this->dims.outputs, this->dims.inputs,
                1.f, weights_array.data(), inputs_array.data(), 1.f,
                outputs_array.data(), this->dims.batch_size));
        } else {
            CUDA_CHECK(memcpy_gpu_to_gpu(state.output->data(),
                                         state.parameters.biases->data(),
                                         this->dims.outputs));
            CUBLAS_CHECK(matvecmul(
                cublas_handle_, false, this->dims.outputs, this->dims.inputs,
                1.f, state.parameters.weights->data(), state.input->data(),
                1.f, state.output->data()));
        }

        return state.output;
    }

    Tensor<ftype> *bwd(LayerState<ftype> &state,
                       Tensor<ftype> *error) override {
        INFO_GRP("LinearLayer BWD", INFO_GRP_LAYER_TASK);
        int64_t inputs = this->dims.inputs;
        int64_t outputs = this->dims.outputs;
        int64_t batch_size = this->dims.batch_size;

        if (batch_size > 1) {
            for (int64_t b = 0; b < batch_size; ++b) {
                weights_array[b] = state.parameters.weights->data();
                inputs_array[b] = &state.input->data()[b * inputs];
                errors_array[b] = &error->data()[b * outputs];
                output_errors_array[b] = &state.error->data()[b * inputs];
            }

            // grads_b = error
            ftype alpha = 1, beta = 0;
            CUDNN_CHECK(cudnnReduceTensor(
                cudnn_handle_, average_tensor, nullptr, 0,
                avg_biases_gradients_ws, avg_biases_gradients_ws_size, &alpha,
                error->descriptor(), error->data(), &beta,
                state.gradients.biases->descriptor(),
                state.gradients.biases->data()));
            // w_grad = err * fwd_inputT
            CUBLAS_CHECK(matmul(cublas_handle_, false, true, outputs, inputs, 1,
                                1.f, errors_array.data(), inputs_array.data(),
                                0.f, temp_weights_gradients_array.data(),
                                batch_size));
            // average the gradients
            alpha = 1;
            beta = 0;
            CUDNN_CHECK(cudnnReduceTensor(
                cudnn_handle_, average_tensor, nullptr, 0,
                avg_weights_gradients_ws, avg_weights_gradients_ws_size, &alpha,
                temp_weights_gradients.descriptor(),
                temp_weights_gradients.data(), &beta,
                state.gradients.weights->descriptor(),
                state.gradients.weights->data()));
            // output_err = errT * weights
            CUBLAS_CHECK(matmul(cublas_handle_, true, false, 1, inputs, outputs,
                                1.f, errors_array.data(), weights_array.data(),
                                0.f, output_errors_array.data(), batch_size));
        } else {
            // grads_b = error
            CUDA_CHECK(memcpy_gpu_to_gpu(state.gradients.biases->data(),
                                         error->data(), outputs));

            // w_grad = err * fwd_inputT
            CUBLAS_CHECK(matmul(cublas_handle_, false, true, outputs, inputs, 1,
                                1.f, error->data(), state.input->data(), 0.f,
                                state.gradients.weights->data()));
            // output_err = errT * weights
            CUBLAS_CHECK(matmul(cublas_handle_, true, false, 1, inputs, outputs,
                                1.f, error->data(),
                                state.parameters.weights->data(), 0.f,
                                state.error->data()));
        }

        return state.error;
    }

  private:
    cublasHandle_t cublas_handle_ = nullptr;
    cudnnHandle_t cudnn_handle_ = nullptr;

    cudnnReduceTensorDescriptor_t average_tensor = nullptr;
    ftype *avg_biases_gradients_ws = 0;
    size_t avg_biases_gradients_ws_size = 0;

    Tensor<ftype> temp_weights_gradients;
    ftype *avg_weights_gradients_ws = 0;
    size_t avg_weights_gradients_ws_size = 0;

    // arrays of pointers used to call the batch version of sgemv and sgemm
    std::vector<ftype *> weights_array = {};
    std::vector<ftype *> inputs_array = {};
    std::vector<ftype *> outputs_array = {};
    std::vector<ftype *> errors_array = {};
    std::vector<ftype *> output_errors_array = {};
    std::vector<ftype *> temp_weights_gradients_array = {};
};

#endif
