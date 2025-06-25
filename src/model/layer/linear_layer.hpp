#ifndef LAYERS_LINEAR_LAYER_H
#define LAYERS_LINEAR_LAYER_H
#include "../../kernels/linear_layer_kernel.h"
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
        CUDA_CHECK(cudaFree(avg_biases_gradients_ws));
        CUDNN_CHECK(cudnnDestroyReduceTensorDescriptor(average_tensor));
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
        return output_dims;
    }

    Tensor<ftype> *fwd(cuda_data_t cuda_data, LayerState<ftype> &state,
                       Tensor<ftype> *input) override {
        INFO_GRP("LinearLayer FWD", INFO_GRP_LAYER_TASK);

        // save input (used for the backwards pass)
        state.input = input;

        CUDNN_CHECK(hhlpLinearForward(
            cuda_data.cudnn_handle, state.parameters.weights->data(),
            state.parameters.biases->data(), input->data(),
            state.output->data(), this->dims.inputs, this->dims.outputs,
            this->dims.batch_size, CUDNN_DATA_TYPE));

        return state.output;
    }

    Tensor<ftype> *bwd(cuda_data_t cuda_data, LayerState<ftype> &state,
                       Tensor<ftype> *error) override {
        INFO_GRP("LinearLayer BWD", INFO_GRP_LAYER_TASK);

        // grads_b = error
        CUDNN_CHECK(hhlpLinearBackwardBias(
            cuda_data.cudnn_handle, error->data(),
            state.gradients.biases->data(), this->dims.outputs,
            this->dims.batch_size, CUDNN_DATA_TYPE));
        // w_grad = err * fwd_inputT
        CUDNN_CHECK(hhlpLinearBackwardWeights(
            cuda_data.cudnn_handle, error->data(), state.input->data(),
            state.gradients.weights->data(), this->dims.outputs,
            this->dims.inputs, this->dims.batch_size, CUDNN_DATA_TYPE));
        // output_err = errT * weights
        CUDNN_CHECK(hhlpLinearBackwardData(
            cuda_data.cudnn_handle, error->data(),
            state.parameters.weights->data(), state.error->data(),
            this->dims.outputs, this->dims.inputs, this->dims.batch_size,
            CUDNN_DATA_TYPE));
        return state.error;
    }

  private:
    cudnnReduceTensorDescriptor_t average_tensor = nullptr;
    ftype *avg_biases_gradients_ws = 0;
    size_t avg_biases_gradients_ws_size = 0;
};

#endif
