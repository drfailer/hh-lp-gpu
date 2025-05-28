#ifndef MODEL_LAYER_SIGMOID_ACTIVATION_LAYER_H
#define MODEL_LAYER_SIGMOID_ACTIVATION_LAYER_H
#include "../../tools/gpu.hpp"
#include "../../types.hpp"
#include "layer.hpp"
#include <cudnn.h>
#include <cudnn_graph.h>
#include <cudnn_ops.h>
#include <log.h/log.h>

struct SigmoidActivationLayer : Layer<ftype> {
    SigmoidActivationLayer(cudnnHandle_t cudnn_handle, int64_t size)
        : Layer(dims_t{.inputs = size, .outputs = size}),
          cudnn_handle_(cudnn_handle) {
        // sigmoid activation tensor
        CUDNN_CHECK(cudnnCreateActivationDescriptor(&sigmoid_));
        CUDNN_CHECK(cudnnSetActivationDescriptor(
            sigmoid_, CUDNN_ACTIVATION_SIGMOID, CUDNN_NOT_PROPAGATE_NAN, 0));
    }

    ~SigmoidActivationLayer() { cudnnDestroyActivationDescriptor(sigmoid_); }

    parameters_t<ftype> create_parameters() const override {
        return {nullptr, nullptr};
    }

    void init(LayerState<ftype> &state, int64_t batch_size) override {
        this->dims.batch_size = batch_size;

        vec_t output_dims = {this->dims.batch_size, 1, this->dims.outputs, 1};
        vec_t output_strides = {this->dims.outputs, this->dims.outputs, 1, 1};
        vec_t error_dims = {this->dims.batch_size, 1, this->dims.inputs, 1};
        vec_t error_strides = {this->dims.inputs, this->dims.inputs, 1, 1};

        delete state.output;
        state.output = new Tensor<ftype>(output_dims, output_strides);
        delete state.error;
        state.error = new Tensor<ftype>(error_dims, error_strides);
    }

    Tensor<ftype> *fwd(LayerState<ftype> &state,
                       Tensor<ftype> *input) override {
        INFO_GRP("SigmoidActivationLayer FWD", INFO_GRP_LAYER_TASK);
        ftype alpha = 1, beta = 0;

        // save the input for the backwards pass
        state.input = input;

        CUDNN_CHECK(cudnnActivationForward(
            cudnn_handle_, sigmoid_, &alpha, input->descriptor(),
            state.input->data(), &beta, state.output->descriptor(),
            state.output->data()));
        return state.output;
    }

    Tensor<ftype> *bwd(LayerState<ftype> &state,
                       Tensor<ftype> *error) override {
        INFO_GRP("SigmoidActivationLayer BWD", INFO_GRP_LAYER_TASK);
        ftype alpha = 1, beta = 0;

        CUDNN_CHECK(cudnnActivationBackward(
            cudnn_handle_, sigmoid_, &alpha, state.output->descriptor(),
            state.output->data(), error->descriptor(), error->data(),
            state.input->descriptor(), state.input->data(), &beta,
            state.error->descriptor(), state.error->data()));
        return state.error;
    }

  private:
    cudnnActivationDescriptor_t sigmoid_;
    cudnnHandle_t cudnn_handle_ = nullptr;
};

#endif
