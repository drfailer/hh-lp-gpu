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

    ~PoolingLayer() {
        CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pooling_descriptor));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_descriptor));
    }

    parameters_t<ftype> create_parameters() const override {
        return {nullptr, nullptr};
    }

    tensor_dims_t init(cuda_data_t cuda_data, LayerState<ftype> &state,
                       tensor_dims_t input_dims) override {
        tensor_dims_t output_dims;
        cudnnSetTensorNdDescriptorEx(input_descriptor, CUDNN_TENSOR_NCHW,
                                     CUDNN_DATA_TYPE, input_dims.size(),
                                     input_dims.data());
        cudnnGetPoolingNdForwardOutputDim(pooling_descriptor, input_descriptor,
                                          output_dims.size(),
                                          output_dims.data());
        state.error = create_tensor<ftype>(input_dims);
        state.output = create_tensor<ftype>(output_dims);
        return output_dims;
    }

    Tensor<ftype> *fwd(cuda_data_t cuda_data, LayerState<ftype> &state,
                       Tensor<ftype> *input) override {
        ftype alpha = 1;
        ftype beta = 0;

        state.input = input;
        CUDNN_CHECK(cudnnPoolingForward(
            cuda_data.cudnn_handle, pooling_descriptor, &alpha,
            input->descriptor(), input->data(), &beta,
            state.output->descriptor(), state.output->data()));
        return state.output;
    }

    Tensor<ftype> *bwd(cuda_data_t cuda_data, LayerState<ftype> &state,
                       Tensor<ftype> *error) override {
        auto error_descriptor = state.output->descriptor();
        auto error_data = error->data();
        ftype alpha = 1;
        ftype beta = 0;

        CUDNN_CHECK(cudnnPoolingBackward(
            cuda_data.cudnn_handle, pooling_descriptor, &alpha,
            state.output->descriptor(), state.output->data(), error_descriptor,
            error_data, state.input->descriptor(), state.input->data(), &beta,
            state.error->descriptor(), state.error->data()));
        return state.error;
    }
};

#endif
