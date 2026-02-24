#ifndef LAYERS_LINEAR_LAYER_H
#define LAYERS_LINEAR_LAYER_H
#include "../../tools/gpu.hpp"
#include "../../types.hpp"
#include "layer.hpp"
#include "../../tools/log.h"
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

    ~LinearLayer() override {
        cudaFree(avg_biases_gradients_ws);
        cudaFree(avg_weights_gradients_ws);
        cudnnDestroyReduceTensorDescriptor(average_tensor);
    }

  public:
    /*
     * Allocates memory for a layer state (output memory for the fwd pass, bwd
     * pass, parameters and gradients).
     */
    Parameters<ftype> create_parameters() const override {
        INFO_GRP("LinearLayer INIT", INFO_GRP_LAYER_TASK);
        int inputs = this->dims.inputs;
        int outputs = this->dims.outputs;
        Parameters<ftype> parameters({1, 1, outputs, inputs},
                                     {1, 1, outputs, 1});

        CUDA_CHECK(memset_random_uniform_gpu<ftype>(
            parameters.weights.data(), outputs * inputs, -0.05, 0.05));
        CUDA_CHECK(memset_random_uniform_gpu<ftype>(parameters.biases.data(),
                                                    outputs, -0.05, 0.05));
        return parameters;
    }

    tensor::dims_t init(cuda_data_t cuda_data, LayerData<ftype> &state,
                        tensor::dims_t input_dims) override {
        int inputs = input_dims[1] * input_dims[2] * input_dims[3];
        int outputs = this->dims.outputs;
        auto batch_size = input_dims[0];
        tensor::dims_t output_dims = {batch_size, 1, outputs, 1};

        this->dims.inputs = inputs;
        this->dims.batch_size = batch_size;

        state.y.reshape(batch_size, 1, outputs, 1);
        state.dx.reshape(batch_size, 1, inputs, 1);

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
        temp_weights_gradients.reshape(batch_size, 1, outputs, inputs);

        // preinit array
        for (size_t b = 0; b < batch_size; ++b) {
            temp_weights_gradients_array[b] =
                &temp_weights_gradients.data()[b * outputs * inputs];
            weights_array[b] = state.w.data();
            outputs_array[b] = &state.y.data()[b * this->dims.outputs];
            weights_array[b] = state.w.data();
            output_errors_array[b] = &state.dx.data()[b * inputs];
        }

        // setup tensor descriptors for computing the biases gradients
        // NOTE: the input error tensor has the same dimensions as the output,
        // so it can be used to compute the workspace size
        CUDNN_CHECK(cudnnGetReductionWorkspaceSize(
            cuda_data.cudnn_handle, average_tensor, state.y.desc(),
            state.db.desc(), &avg_biases_gradients_ws_size));
        cudaFree(avg_biases_gradients_ws); // free if needed
        CUDA_CHECK(
            alloc_gpu(&avg_biases_gradients_ws, avg_biases_gradients_ws_size));
        CUDNN_CHECK(cudnnGetReductionWorkspaceSize(
            cuda_data.cudnn_handle, average_tensor,
            temp_weights_gradients.desc(), state.dw.desc(),
            &avg_weights_gradients_ws_size));
        cudaFree(avg_weights_gradients_ws); // free if needed
        CUDA_CHECK(alloc_gpu(&avg_weights_gradients_ws,
                             avg_weights_gradients_ws_size));
        return output_dims;
    }

    tensor::Tensor const &
    fwd(cuda_data_t cuda_data, LayerData<ftype> &state,
        tensor::Tensor const &input) override {
        INFO_GRP("LinearLayer FWD", INFO_GRP_LAYER_TASK);

        if (this->dims.batch_size > 1) {
            for (int b = 0; b < this->dims.batch_size; ++b) {
                inputs_array[b] = &input.data()[b * this->dims.inputs];
                CUDA_CHECK(memcpy_gpu_to_gpu(outputs_array[b], state.b.data(),
                                             this->dims.outputs));
            }

            CUBLAS_CHECK(matvecmul(cuda_data.cublas_handle, false,
                                   this->dims.outputs, this->dims.inputs, 1.f,
                                   weights_array.data(), inputs_array.data(),
                                   1.f, outputs_array.data(),
                                   this->dims.batch_size));
        } else {
            CUDA_CHECK(memcpy_gpu_to_gpu(state.y.data(), state.b.data(),
                                         this->dims.outputs));
            CUBLAS_CHECK(matvecmul(cuda_data.cublas_handle, false,
                                   this->dims.outputs, this->dims.inputs, 1.f,
                                   state.w.data(), input.data(), 1.f,
                                   state.y.data()));
        }

        return state.y;
    }

    tensor::Tensor const &
    bwd(cuda_data_t cuda_data, LayerData<ftype> &state,
        tensor::Tensor const &input,
        tensor::Tensor const &output_gradient) override {
        INFO_GRP("LinearLayer BWD", INFO_GRP_LAYER_TASK);
        int inputs = this->dims.inputs;
        int outputs = this->dims.outputs;
        int batch_size = this->dims.batch_size;
        auto error_descriptor = state.y.desc();
        auto error_data = output_gradient.data();

        if (batch_size > 1) {
            for (int b = 0; b < batch_size; ++b) {
                errors_array[b] = &error_data[b * outputs];
            }

            // grads_b = error
            ftype alpha = 1, beta = 0;
            CUDNN_CHECK(cudnnReduceTensor(
                cuda_data.cudnn_handle, average_tensor, nullptr, 0,
                avg_biases_gradients_ws, avg_biases_gradients_ws_size, &alpha,
                error_descriptor, error_data, &beta, state.db.desc(),
                state.db.data()));
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
                temp_weights_gradients.desc(), temp_weights_gradients.data(),
                &beta, state.dw.desc(), state.dw.data()));
            // output_err = errT * weights
            CUBLAS_CHECK(matmul(cuda_data.cublas_handle, true, false, 1, inputs,
                                outputs, 1.f, errors_array.data(),
                                weights_array.data(), 0.f,
                                output_errors_array.data(), batch_size));
        } else {
            // grads_b = error
            CUDA_CHECK(memcpy_gpu_to_gpu(state.db.data(),
                                         output_gradient.data(), outputs));

            // w_grad = err * fwd_inputT
            CUBLAS_CHECK(matmul(cuda_data.cublas_handle, false, true, outputs,
                                inputs, 1, 1.f, output_gradient.data(),
                                input.data(), 0.f, state.dw.data()));
            // output_err = errT * weights
            CUBLAS_CHECK(matmul(cuda_data.cublas_handle, true, false, 1, inputs,
                                outputs, 1.f, output_gradient.data(),
                                state.w.data(), 0.f, state.dx.data()));
        }

        return state.dx;
    }

  private:
    cudnnReduceTensorDescriptor_t average_tensor = nullptr;
    ftype *avg_biases_gradients_ws = 0;
    size_t avg_biases_gradients_ws_size = 0;

    tensor::Tensor temp_weights_gradients;
    ftype *avg_weights_gradients_ws = 0;
    size_t avg_weights_gradients_ws_size = 0;

    // arrays of pointers used to call the batch version of sgemv and sgemm
    std::vector<ftype *> weights_array = {};
    std::vector<ftype const *> inputs_array = {};
    std::vector<ftype *> outputs_array = {};
    std::vector<ftype const *> errors_array = {};
    std::vector<ftype *> output_errors_array = {};
    std::vector<ftype *> temp_weights_gradients_array = {};
};

#endif
