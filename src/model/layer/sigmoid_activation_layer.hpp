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

    tensor::dims_t init(cuda_data_t cuda_data, LayerData<ftype> &state,
                        tensor::dims_t input_dims) override {
        int inputs = input_dims[1] * input_dims[2] * input_dims[3];
        int outputs = inputs;
        int batch_size = input_dims[0];
        this->dims.inputs = inputs;
        this->dims.outputs = outputs;
        this->dims.batch_size = input_dims[0];

        state.y.reshape(batch_size, 1, outputs, 1);
        state.dx.reshape(batch_size, 1, inputs, 1);
        return input_dims;
    }

    tensor::Tensor<ftype> const &
    fwd(cuda_data_t cuda_data, LayerData<ftype> &state,
        tensor::Tensor<ftype> const &input) override {
        INFO_GRP("SigmoidActivationLayer FWD", INFO_GRP_LAYER_TASK);
        ftype alpha = 1, beta = 0;

        CUDNN_CHECK(cudnnActivationForward(
            cuda_data.cudnn_handle, sigmoid_, &alpha, input.desc(),
            input.data(), &beta, state.y.desc(), state.y.data()));
        return state.y;
    }

    tensor::Tensor<ftype> const &
    bwd(cuda_data_t cuda_data, LayerData<ftype> &state,
        tensor::Tensor<ftype> const &input,
        tensor::Tensor<ftype> const &output_gradient) override {
        INFO_GRP("SigmoidActivationLayer BWD", INFO_GRP_LAYER_TASK);
        ftype alpha = 1, beta = 0;

        CUDNN_CHECK(cudnnActivationBackward(
            cuda_data.cudnn_handle, sigmoid_, &alpha, state.y.desc(),
            state.y.data(), output_gradient.desc(), output_gradient.data(),
            input.desc(), input.data(), &beta, state.dx.desc(),
            state.dx.data()));
        return state.dx;
    }

  private:
    cudnnActivationDescriptor_t sigmoid_;
};

#endif
