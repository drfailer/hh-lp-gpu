#ifndef MODEL_LAYER_POOLING_LAYER
#define MODEL_LAYER_POOLING_LAYER
#include "../../types.hpp"
#include "layer.hpp"
#include <cudnn_ops.h>

struct PoolingLayer : Layer<ftype> {
    cudnnPoolingDescriptor_t pooling_descriptor;
    cudnnTensorDescriptor_t input_descriptor;

    PoolingLayer(cudnnPoolingMode_t mode, int width, int height,
                 int horizontal_padding = 0, int vertical_padding = 0)
        : Layer<ftype>({.kernel_width = width, .kernel_height = height}) {
        int horizontal_stride = width;
        int vertical_stride = height;

        CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pooling_descriptor));
        // TODO: we might want to use the Nd version at some point
        // cudnnSetPoolingNdDescriptor(
        //     pooling_descriptor, mode, CUDNN_NOT_PROPAGATE_NAN,
        //     window_dims.size(),
        //     window_dims.data(), padding.data(), strides.data());
        CUDNN_CHECK(cudnnSetPooling2dDescriptor(
            pooling_descriptor, mode, CUDNN_NOT_PROPAGATE_NAN, height, width,
            vertical_padding, horizontal_padding, vertical_stride,
            horizontal_stride));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_descriptor));
    }

    ~PoolingLayer() override {
        CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pooling_descriptor));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_descriptor));
    }

    Parameters<ftype> create_parameters() const override { return {}; }

    tensor::dims_t init(cuda_data_t cuda_data, LayerState<ftype> &state,
                        tensor::dims_t input_dims) override {
        tensor::dims_t output_dims;
        cudnnSetTensorNdDescriptorEx(input_descriptor, CUDNN_TENSOR_NCHW,
                                     CUDNN_DATA_TYPE, input_dims.size(),
                                     input_dims.data());
        cudnnGetPoolingNdForwardOutputDim(pooling_descriptor, input_descriptor,
                                          output_dims.size(),
                                          output_dims.data());
        state.gradients.input.reshape(input_dims);
        state.output.reshape(output_dims);
        return output_dims;
    }

    tensor::Tensor<ftype> const &
    fwd(cuda_data_t cuda_data, LayerState<ftype> &state,
        tensor::Tensor<ftype> const &input) override {
        ftype alpha = 1;
        ftype beta = 0;

        CUDNN_CHECK(cudnnPoolingForward(
            cuda_data.cudnn_handle, pooling_descriptor, &alpha, input.desc(),
            input.data(), &beta, state.output.desc(), state.output.data()));
        return state.output;
    }

    tensor::Tensor<ftype> const &
    bwd(cuda_data_t cuda_data, LayerState<ftype> &state,
        tensor::Tensor<ftype> const &input,
        tensor::Tensor<ftype> const &output_gradient) override {
        auto error_descriptor = state.output.desc();
        auto error_data = output_gradient.data();
        ftype alpha = 1;
        ftype beta = 0;

        CUDNN_CHECK(cudnnPoolingBackward(
            cuda_data.cudnn_handle, pooling_descriptor, &alpha,
            state.output.desc(), state.output.data(), error_descriptor,
            error_data, input.desc(), state.input->data(), &beta,
            state.gradients.input.desc(), state.gradients.input.data()));
        return state.gradients.input;
    }
};

#endif
