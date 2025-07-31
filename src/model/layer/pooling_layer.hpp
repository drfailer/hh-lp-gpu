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

    virtual LayerIOShape
    io_shape(tensor::dims_t const &input_dims) const override {
        tensor::dims_t output_dims;
        cudnnSetTensorNdDescriptorEx(input_descriptor, CUDNN_TENSOR_NCHW,
                                     CUDNN_DATA_TYPE, input_dims.size(),
                                     input_dims.data());
        cudnnGetPoolingNdForwardOutputDim(pooling_descriptor, input_descriptor,
                                          output_dims.size(),
                                          output_dims.data());
        // here we are stuck and we need the full input
        return {
            .x = tensor::shape(input_dims),
            .y = tensor::shape(output_dims),
        };
    }

    virtual void init_fwd(cuda_data_t cuda,
                          LayerData<ftype> const &data) override {
        tensor::dims_t input_dims = data.x.dims();
        cudnnSetTensorNdDescriptorEx(input_descriptor, CUDNN_TENSOR_NCHW,
                                     CUDNN_DATA_TYPE, input_dims.size(),
                                     input_dims.data());
    }

    void fwd(cuda_data_t cuda, fwd_data_t<ftype> const &data,
             tensor::Tensor<const ftype> const &x,
             tensor::Tensor<ftype> &y) override {
        ftype alpha = 1;
        ftype beta = 0;

        CUDNN_CHECK(cudnnPoolingForward(cuda.cudnn_handle, pooling_descriptor,
                                        &alpha, x.desc(), x.data(), &beta,
                                        y.desc(), y.data()));
    }

    void bwd(cuda_data_t cuda, bwd_data_t<ftype> const &data,
             tensor::Tensor<const ftype> const &dy,
             tensor::Tensor<ftype> &dx) override {
        auto error_descriptor = dy.desc();
        auto error_data = dy.data();
        ftype alpha = 1;
        ftype beta = 0;

        CUDNN_CHECK(cudnnPoolingBackward(
            cuda.cudnn_handle, pooling_descriptor, &alpha, data.y.desc(),
            data.y.data(), error_descriptor, error_data, data.x.desc(),
            data.x.data(), &beta, dx.desc(), dx.data()));
    }
};

#endif
