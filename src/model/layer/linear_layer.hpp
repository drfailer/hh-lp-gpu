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
    LinearLayer(int input_dim, int output_dim)
        : Layer(dims_t{.inputs = input_dim, .outputs = output_dim}) {
        CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(&average_tensor));
        CUDNN_CHECK(cudnnSetReduceTensorDescriptor(
            average_tensor, CUDNN_REDUCE_TENSOR_AVG, CUDNN_DATA_TYPE,
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
        int inputs = this->dims.inputs;
        int outputs = this->dims.outputs;
        parameters_t<ftype> parameters{
            .weights = create_tensor<ftype>({1, 1, outputs, inputs}),
            .biases = create_tensor<ftype>({1, 1, outputs, 1}),
        };

        CUDA_CHECK(memset_random_uniform_gpu<ftype>(
            parameters.weights->data(), outputs * inputs, -0.05, 0.05));
        CUDA_CHECK(memset_random_uniform_gpu<ftype>(parameters.biases->data(),
                                                    outputs, -0.05, 0.05));
        return parameters;
    }

    tensor_dims_t init(cuda_data_t cuda_data, LayerState<ftype> &state,
                       tensor_dims_t input_dims) override {
        int inputs = input_dims[1] * input_dims[2] * input_dims[3];
        int outputs = this->dims.outputs;
        auto batch_size = input_dims[0];
        tensor_dims_t output_dims = {batch_size, 1, outputs, 1};

        this->dims.inputs = inputs;
        this->dims.batch_size = batch_size;

        delete state.output;
        state.output = create_tensor<ftype>({batch_size, 1, outputs, 1});
        delete state.error;
        state.error = create_tensor<ftype>({batch_size, 1, inputs, 1});

        if (batch_size == 1) {
            return output_dims;
        }

        weights_array.resize(batch_size, nullptr);
        inputs_array.resize(batch_size, nullptr);
        output_errors_array.resize(batch_size, nullptr);
        outputs_array.resize(batch_size, nullptr);
        errors_array.resize(batch_size, nullptr);
        temp_weights_gradients_array.resize(batch_size, nullptr);

        // create temporary array for the gradients
        temp_weights_gradients.reshape({batch_size, 1, outputs, inputs});

        // preinit array
        for (size_t b = 0; b < batch_size; ++b) {
            temp_weights_gradients_array[b] =
                &temp_weights_gradients.data()[b * outputs * inputs];
            weights_array[b] = state.parameters.weights->data();
            outputs_array[b] = &state.output->data()[b * this->dims.outputs];
            weights_array[b] = state.parameters.weights->data();
            output_errors_array[b] = &state.error->data()[b * inputs];
        }

        // setup tensor descriptors for computing the biases gradients
        // NOTE: the input error tensor has the same dimensions as the output,
        // so it can be used to compute the workspace size
        CUDNN_CHECK(cudnnGetReductionWorkspaceSize(
            cuda_data.cudnn_handle, average_tensor, state.output->descriptor(),
            state.gradients.biases->descriptor(),
            &avg_biases_gradients_ws_size));
        cudaFree(avg_biases_gradients_ws); // free if needed
        CUDA_CHECK(
            alloc_gpu(&avg_biases_gradients_ws, avg_biases_gradients_ws_size));
        CUDNN_CHECK(cudnnGetReductionWorkspaceSize(
            cuda_data.cudnn_handle, average_tensor,
            temp_weights_gradients.descriptor(),
            state.gradients.weights->descriptor(),
            &avg_weights_gradients_ws_size));
        cudaFree(avg_weights_gradients_ws); // free if needed
        CUDA_CHECK(alloc_gpu(&avg_weights_gradients_ws,
                             avg_weights_gradients_ws_size));
        return output_dims;
    }

    Tensor<ftype> *fwd(cuda_data_t cuda_data, LayerState<ftype> &state,
                       Tensor<ftype> *input) override {
        INFO_GRP("LinearLayer FWD", INFO_GRP_LAYER_TASK);

        // save input (used for the backwards pass)
        state.input = input;

        if (this->dims.batch_size > 1) {
            for (int b = 0; b < this->dims.batch_size; ++b) {
                inputs_array[b] = &input->data()[b * this->dims.inputs];
                CUDA_CHECK(memcpy_gpu_to_gpu(outputs_array[b],
                                             state.parameters.biases->data(),
                                             this->dims.outputs));
            }

            CUBLAS_CHECK(matvecmul(cuda_data.cublas_handle, false,
                                   this->dims.outputs, this->dims.inputs, 1.f,
                                   weights_array.data(), inputs_array.data(),
                                   1.f, outputs_array.data(),
                                   this->dims.batch_size));
        } else {
            CUDA_CHECK(memcpy_gpu_to_gpu(state.output->data(),
                                         state.parameters.biases->data(),
                                         this->dims.outputs));
            CUBLAS_CHECK(matvecmul(
                cuda_data.cublas_handle, false, this->dims.outputs,
                this->dims.inputs, 1.f, state.parameters.weights->data(),
                state.input->data(), 1.f, state.output->data()));
        }

        return state.output;
    }

    Tensor<ftype> *bwd(cuda_data_t cuda_data, LayerState<ftype> &state,
                       Tensor<ftype> *error) override {
        INFO_GRP("LinearLayer BWD", INFO_GRP_LAYER_TASK);
        int inputs = this->dims.inputs;
        int outputs = this->dims.outputs;
        int batch_size = this->dims.batch_size;
        auto error_descriptor = state.output->descriptor();
        auto error_data = error->data();

        if (batch_size > 1) {
            for (int b = 0; b < batch_size; ++b) {
                errors_array[b] = &error_data[b * outputs];
            }

            // grads_b = error
            ftype alpha = 1, beta = 0;
            CUDNN_CHECK(cudnnReduceTensor(
                cuda_data.cudnn_handle, average_tensor, nullptr, 0,
                avg_biases_gradients_ws, avg_biases_gradients_ws_size, &alpha,
                error_descriptor, error_data, &beta,
                state.gradients.biases->descriptor(),
                state.gradients.biases->data()));
            // w_grad = err * fwd_inputT
            CUBLAS_CHECK(
                matmul(cuda_data.cublas_handle, false, true, outputs, inputs, 1,
                       1.f, errors_array.data(), inputs_array.data(), 0.f,
                       temp_weights_gradients_array.data(), batch_size));
            // average the gradients
            alpha = 1;
            beta = 0;
            CUDNN_CHECK(cudnnReduceTensor(
                cuda_data.cudnn_handle, average_tensor, nullptr, 0,
                avg_weights_gradients_ws, avg_weights_gradients_ws_size, &alpha,
                temp_weights_gradients.descriptor(),
                temp_weights_gradients.data(), &beta,
                state.gradients.weights->descriptor(),
                state.gradients.weights->data()));
            // output_err = errT * weights
            CUBLAS_CHECK(matmul(cuda_data.cublas_handle, true, false, 1, inputs,
                                outputs, 1.f, errors_array.data(),
                                weights_array.data(), 0.f,
                                output_errors_array.data(), batch_size));
        } else {
            // grads_b = error
            CUDA_CHECK(memcpy_gpu_to_gpu(state.gradients.biases->data(),
                                         error->data(), outputs));

            // w_grad = err * fwd_inputT
            CUBLAS_CHECK(matmul(cuda_data.cublas_handle, false, true, outputs,
                                inputs, 1, 1.f, error->data(),
                                state.input->data(), 0.f,
                                state.gradients.weights->data()));
            // output_err = errT * weights
            CUBLAS_CHECK(matmul(cuda_data.cublas_handle, true, false, 1, inputs,
                                outputs, 1.f, error->data(),
                                state.parameters.weights->data(), 0.f,
                                state.error->data()));
        }

        return state.error;
    }

  private:
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
