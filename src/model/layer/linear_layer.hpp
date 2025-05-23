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
          cublas_handle_(cublas_handle), cudnn_handle_(cudnn_handle) {
        CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(&reduce_avg_desc));
        CUDNN_CHECK(cudnnSetReduceTensorDescriptor(
            reduce_avg_desc, CUDNN_REDUCE_TENSOR_AVG, CUDNN_DATA_FLOAT,
            CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES,
            CUDNN_32BIT_INDICES));
        cudnnCreateTensorDescriptor(&biases_gradient_desc);
        cudnnCreateTensorDescriptor(&error_desc);
        cudnnCreateTensorDescriptor(&weights_gradient_desc);
        cudnnCreateTensorDescriptor(&temp_weights_gradients_desc);
    }

    ~LinearLayer() {
        cudaFree(reduce_avg_workspace1);
        cudnnDestroyReduceTensorDescriptor(reduce_avg_desc);
        cudnnDestroyTensorDescriptor(biases_gradient_desc);
        cudnnDestroyTensorDescriptor(weights_gradient_desc);
        cudnnDestroyTensorDescriptor(temp_weights_gradients_desc);
        cudnnDestroyTensorDescriptor(error_desc);

        for (auto &buff : temp_weights_gradients_array) {
            cudaFree(buff);
        }
    }

  public:
    /*
     * Allocates memory for a layer state (output memory for the fwd pass, bwd
     * pass, parameters and gradients).
     */
    layer_state_t<ftype> create_state() const override {
        INFO_GRP("LinearLayer INIT", INFO_GRP_LAYER_TASK);
        layer_state_t<ftype> state;

        state = create_layer_state<ftype>(this->dims, true, true);
        CUDA_CHECK(memset_random_uniform_gpu<ftype>(
            state.parameters.weights, this->dims.inputs * this->dims.outputs,
            -0.5, 0.5, 0));
        CUDA_CHECK(memset_random_uniform_gpu<ftype>(
            state.parameters.biases, this->dims.outputs, -0.5, 0.5, 0));
        cudaDeviceSynchronize();
        return state;
    }

    void init() override {
        if (this->dims.batch_count == 1) {
            return;
        }

        weights_array.resize(this->dims.batch_count, nullptr);
        inputs_array.resize(this->dims.batch_count, nullptr);
        output_errors_array.resize(this->dims.batch_count, nullptr);
        outputs_array.resize(this->dims.batch_count, nullptr);
        errors_array.resize(this->dims.batch_count, nullptr);
        temp_weights_gradients_array.resize(this->dims.batch_count, nullptr);

        // create temporary array for the gradients
        cudaFree(temp_weights_gradients);
        CUDA_CHECK(alloc_gpu(&temp_weights_gradients, this->dims.batch_count *
                                                          this->dims.inputs *
                                                          this->dims.outputs));
        for (size_t b = 0; b < this->dims.batch_count; ++b) {
            temp_weights_gradients_array[b] =
                &temp_weights_gradients[b * this->dims.outputs *
                                        this->dims.inputs];
        }

        // setup tensor descriptors for computing the biases gradients
        CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
            biases_gradient_desc, CUDNN_DATA_FLOAT,
            // this->parameter_shape.dims.biases[0],
            1, this->parameter_shape.dims.biases[1],
            this->parameter_shape.dims.biases[2],
            this->parameter_shape.dims.biases[3],
            this->parameter_shape.strides.biases[0],
            this->parameter_shape.strides.biases[1],
            this->parameter_shape.strides.biases[2],
            this->parameter_shape.strides.biases[3]));
        CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
            error_desc, CUDNN_DATA_FLOAT,
            // this->parameter_shape.dims.biases[0],
            this->dims.batch_count, 1, this->dims.outputs, 1,
            this->dims.outputs, this->dims.outputs, 1, 1));
        CUDNN_CHECK(cudnnGetReductionWorkspaceSize(
            cudnn_handle_, reduce_avg_desc, error_desc, biases_gradient_desc,
            &reduce_avg_workspace1_size));
        cudaFree(reduce_avg_workspace1); // free if needed
        CUDA_CHECK(
            alloc_gpu(&reduce_avg_workspace1, reduce_avg_workspace1_size));

        CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
            weights_gradient_desc, CUDNN_DATA_FLOAT,
            // this->parameter_shape.dims.weights[0],
            1, this->parameter_shape.dims.weights[1],
            this->parameter_shape.dims.weights[2],
            this->parameter_shape.dims.weights[3],
            this->parameter_shape.strides.weights[0],
            this->parameter_shape.strides.weights[1],
            this->parameter_shape.strides.weights[2],
            this->parameter_shape.strides.weights[3]));
        CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
            temp_weights_gradients_desc, CUDNN_DATA_FLOAT,
            this->dims.batch_count, this->parameter_shape.dims.weights[1],
            this->parameter_shape.dims.weights[2],
            this->parameter_shape.dims.weights[3],
            this->parameter_shape.strides.weights[0],
            this->parameter_shape.strides.weights[1],
            this->parameter_shape.strides.weights[2],
            this->parameter_shape.strides.weights[3]));
        CUDNN_CHECK(cudnnGetReductionWorkspaceSize(
            cudnn_handle_, reduce_avg_desc, temp_weights_gradients_desc,
            weights_gradient_desc, &reduce_avg_workspace2_size));
        cudaFree(reduce_avg_workspace2); // free if needed
        CUDA_CHECK(
            alloc_gpu(&reduce_avg_workspace2, reduce_avg_workspace2_size));
    }

    ftype *fwd(layer_state_t<ftype> &state, ftype *input) override {
        INFO_GRP("LinearLayer FWD", INFO_GRP_LAYER_TASK);

        // save input (used for the backwards pass)
        state.input = input;

        if (this->dims.batch_count > 1) {
            // TODO: should be done in init
            for (int64_t b = 0; b < this->dims.batch_count; ++b) {
                weights_array[b] = state.parameters.weights;
                inputs_array[b] = &state.input[b * this->dims.inputs];
                outputs_array[b] = &state.output[b * this->dims.outputs];
                CUDA_CHECK(memcpy_gpu_to_gpu(outputs_array[b],
                                             state.parameters.biases,
                                             this->dims.outputs));
            }

            CUBLAS_CHECK(matvecmul(
                cublas_handle_, false, this->dims.outputs, this->dims.inputs,
                1.f, weights_array.data(), inputs_array.data(), 1.f,
                outputs_array.data(), this->dims.batch_count));
        } else {
            CUDA_CHECK(
                memcpy_gpu_to_gpu(state.output, state.parameters.biases, this->dims.outputs));
            CUBLAS_CHECK(matvecmul(
                cublas_handle_, false, this->dims.outputs, this->dims.inputs,
                1.f, state.parameters.weights, state.input, 1.f, state.output));
        }

        return state.output;
    }

    ftype *bwd(layer_state_t<ftype> &state, ftype *error) override {
        INFO_GRP("LinearLayer BWD", INFO_GRP_LAYER_TASK);

        if (this->dims.batch_count > 1) {
            for (int64_t b = 0; b < this->dims.batch_count; ++b) {
                weights_array[b] = state.parameters.weights;
                inputs_array[b] = &state.input[b * this->dims.inputs];
                errors_array[b] = &error[b * this->dims.outputs];
                output_errors_array[b] = &state.error[b * this->dims.outputs];
            }

            // grads_b = error
            ftype alpha = 1, beta = 0;
            CUDNN_CHECK(cudnnReduceTensor(
                cudnn_handle_, reduce_avg_desc, nullptr, 0,
                reduce_avg_workspace1, reduce_avg_workspace1_size, &alpha,
                error_desc, error, &beta, biases_gradient_desc,
                state.gradients.biases));

            // w_grad = err * update_inputT
            CUBLAS_CHECK(matmul(cublas_handle_, false, true, this->dims.outputs,
                                this->dims.inputs, 1, 1.f, errors_array.data(),
                                inputs_array.data(), 0.f,
                                temp_weights_gradients_array.data(),
                                this->dims.batch_count));
            alpha = 1;
            beta = 0;
            CUDNN_CHECK(cudnnReduceTensor(
                cudnn_handle_, reduce_avg_desc, nullptr, 0,
                reduce_avg_workspace2, reduce_avg_workspace2_size, &alpha,
                temp_weights_gradients_desc, temp_weights_gradients, &beta,
                weights_gradient_desc, state.gradients.weights));
            // output_err = errT * weights
            CUBLAS_CHECK(matmul(cublas_handle_, true, false, 1,
                                this->dims.inputs, this->dims.outputs, 1.f,
                                errors_array.data(), weights_array.data(), 0.f,
                                output_errors_array.data(),
                                this->dims.batch_count));
        } else {
            // grads_b = error
            CUDA_CHECK(memcpy_gpu_to_gpu(state.gradients.biases, error,
                                         this->dims.outputs));

            // w_grad = err * update_inputT
            CUBLAS_CHECK(matmul(cublas_handle_, false, true, this->dims.outputs,
                                this->dims.inputs, 1, 1.f, error, state.input,
                                0.f, state.gradients.weights));
            // output_err = errT * weights
            CUBLAS_CHECK(matmul(cublas_handle_, true, false, 1,
                                this->dims.inputs, this->dims.outputs, 1.f,
                                error, state.parameters.weights, 0.f,
                                state.error));
        }

        return state.error;
    }

  private:
    cublasHandle_t cublas_handle_ = nullptr;
    cudnnHandle_t cudnn_handle_ = nullptr;

    cudnnTensorDescriptor_t biases_gradient_desc = nullptr;
    cudnnTensorDescriptor_t error_desc = nullptr;
    cudnnReduceTensorDescriptor_t reduce_avg_desc = nullptr;
    ftype *reduce_avg_workspace1 = 0;
    size_t reduce_avg_workspace1_size = 0;

    ftype *temp_weights_gradients = nullptr;
    cudnnTensorDescriptor_t weights_gradient_desc = nullptr;
    cudnnTensorDescriptor_t temp_weights_gradients_desc = nullptr;
    ftype *reduce_avg_workspace2 = 0;
    size_t reduce_avg_workspace2_size = 0;

    // arrays of pointers used to call the batch version of sgemv and sgemm
    std::vector<ftype *> weights_array = {};
    std::vector<ftype *> inputs_array = {};
    std::vector<ftype *> outputs_array = {};
    std::vector<ftype *> errors_array = {};
    std::vector<ftype *> output_errors_array = {};
    std::vector<ftype *> temp_weights_gradients_array = {};
};

#endif
