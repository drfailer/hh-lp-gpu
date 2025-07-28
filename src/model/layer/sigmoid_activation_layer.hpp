#ifndef MODEL_LAYER_SIGMOID_ACTIVATION_LAYER_H
#define MODEL_LAYER_SIGMOID_ACTIVATION_LAYER_H
#include "../../tools/gpu.hpp"
#include "../../types.hpp"
#include "layer.hpp"
#include <cudnn.h>
#include <cudnn_graph.h>
#include <cudnn_ops.h>

struct SigmoidActivationLayer : Layer<ftype> {
    SigmoidActivationLayer() : Layer({}) {
        // sigmoid activation tensor
        CUDNN_CHECK(cudnnCreateActivationDescriptor(&sigmoid_));
        CUDNN_CHECK(cudnnSetActivationDescriptor(
            sigmoid_, CUDNN_ACTIVATION_SIGMOID, CUDNN_NOT_PROPAGATE_NAN, 0));
    }

    ~SigmoidActivationLayer() override {
        cudnnDestroyActivationDescriptor(sigmoid_);
    }

    Parameters<ftype> create_parameters() const override { return {}; }

    tensor_dims_t init(cuda_data_t cuda_data, LayerState<ftype> &state,
                       tensor_dims_t input_dims) override {
        int inputs = input_dims[1] * input_dims[2] * input_dims[3];
        int outputs = inputs;
        int batch_size = input_dims[0];
        this->dims.inputs = inputs;
        this->dims.outputs = outputs;
        this->dims.batch_size = input_dims[0];

        state.output.reshape({batch_size, 1, outputs, 1});
        state.gradients.input.reshape({batch_size, 1, inputs, 1});
        return input_dims;
    }

    Tensor<ftype> const &fwd(cuda_data_t cuda_data, LayerState<ftype> &state,
                             Tensor<ftype> const &input) override {
        INFO_GRP("SigmoidActivationLayer FWD", INFO_GRP_LAYER_TASK);
        ftype alpha = 1, beta = 0;

        CUDNN_CHECK(cudnnActivationForward(
            cuda_data.cudnn_handle, sigmoid_, &alpha, input.descriptor(),
            input.data(), &beta, state.output.descriptor(),
            state.output.data()));
        return state.output;
    }

    Tensor<ftype> const &bwd(cuda_data_t cuda_data, LayerState<ftype> &state,
                             Tensor<ftype> const &input,
                             Tensor<ftype> const &output_gradient) override {
        INFO_GRP("SigmoidActivationLayer BWD", INFO_GRP_LAYER_TASK);
        ftype alpha = 1, beta = 0;

        CUDNN_CHECK(cudnnActivationBackward(
            cuda_data.cudnn_handle, sigmoid_, &alpha, state.output.descriptor(),
            state.output.data(), output_gradient.descriptor(),
            output_gradient.data(), input.descriptor(), input.data(), &beta,
            state.gradients.input.descriptor(), state.gradients.input.data()));
        return state.gradients.input;
    }

  private:
    cudnnActivationDescriptor_t sigmoid_;
};

#endif
