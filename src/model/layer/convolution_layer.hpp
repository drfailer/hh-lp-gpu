#ifndef MODEL_LAYER_CONVOLUTION_LAYER
#define MODEL_LAYER_CONVOLUTION_LAYER
#include "../../tools/gpu.hpp"
#include "../../types.hpp"
#include "layer.hpp"
#include "log.h/log.h"
#include <cudnn.h>
#include <cudnn_cnn.h>
#include <cudnn_graph.h>
#include <cudnn_ops.h>

struct ConvolutionLayer : Layer<ftype> {
    bool use_biases;
    cudnnConvolutionFwdAlgo_t fwd_algo;
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
    cudnnConvolutionDescriptor_t convolution_descriptor;
    cudnnFilterDescriptor_t filter_descriptor;
    int input_height;
    int input_width;

    ftype *convolution_fw_ws;
    size_t convolution_fw_ws_size;

    ftype *convolution_bw_data_ws;
    size_t convolution_bw_data_ws_size;

    ftype *convolution_bw_filter_ws;
    size_t convolution_bw_filter_ws_size;

    // n = c = 1;
    // h     = IMAGE_H;
    // w     = IMAGE_W;

    ConvolutionLayer(
        int inputs, int outputs, int input_width, int input_height,
        int kernel_width, int kernel_height,
        cudnnConvolutionFwdAlgo_t fwd_algo = CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        cudnnConvolutionBwdDataAlgo_t bwd_data_algo =
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
        cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo =
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
        bool use_biases = true)
        : Layer({.inputs = inputs,
                 .outputs = outputs,
                 .kernel_width = kernel_width,
                 .kernel_height = kernel_height}),
          use_biases(use_biases), fwd_algo(fwd_algo),
          bwd_data_algo(bwd_data_algo), bwd_filter_algo(bwd_filter_algo),
          input_height(input_width), input_width(input_width) {
        constexpr size_t convDims = 2;
        constexpr int padA[convDims] = {0, 0};
        constexpr int filterStrideA[convDims] = {1, 1};
        constexpr int upscaleA[convDims] = {1, 1};
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
        CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(
            convolution_descriptor, convDims, padA, filterStrideA, upscaleA,
            CUDNN_CROSS_CORRELATION, CUDNN_DATA_TYPE));
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_descriptor));

        const int tensorDims = 4;
        const int filterDimA[tensorDims] = {outputs, inputs, (int)kernel_height,
                                            (int)kernel_width};

        CUDNN_CHECK(cudnnSetFilterNdDescriptor(
            filter_descriptor, CUDNN_DATA_TYPE, CUDNN_TENSOR_NCHW, tensorDims,
            filterDimA));
    }

    ~ConvolutionLayer() {
        CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(convolution_descriptor));
        CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_descriptor));
    }

    parameters_t<ftype> create_parameters() const {
        parameters_t<ftype> parameters;

        parameters.weights = create_tensor<ftype>(
            {dims.outputs, dims.inputs, dims.kernel_height, dims.kernel_width});

        if (use_biases) {
            // todo: the baises should be created here
            // note that having an Input layer would be great
            parameters.biases = nullptr;
        }

        return parameters;
    }

    tensor_dims_t init(cuda_data_t cuda_data, LayerState<ftype> &state,
                       tensor_dims_t input_dims) {
        constexpr int tensor_dims = 4;
        int tensorOuputDimA[tensor_dims] = {input_dims.n, input_dims.c,
                                            input_dims.h, input_dims.w};

        delete state.error;
        state.error = create_tensor_from_dims<ftype>(input_dims);

        tensor_dims_t output_dims;
        cudnnTensorDescriptor_t srcTensorDesc;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&srcTensorDesc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_TYPE, input_dims.n,
            input_dims.c, input_dims.h, input_dims.w));
        CUDNN_CHECK(cudnnGetConvolutionNdForwardOutputDim(
            convolution_descriptor, srcTensorDesc, filter_descriptor,
            tensor_dims, tensorOuputDimA));
        output_dims.n = tensorOuputDimA[0];
        output_dims.c = tensorOuputDimA[1];
        output_dims.h = tensorOuputDimA[2];
        output_dims.w = tensorOuputDimA[3];
        state.parameters.biases = create_tensor<ftype>(
            {1, output_dims.c, output_dims.h, output_dims.w});
        state.gradients.biases = create_tensor<ftype>(
            {1, output_dims.c, output_dims.h, output_dims.w});

        delete state.output;
        state.output = create_tensor_from_dims<ftype>(output_dims);

        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
            cuda_data.cudnn_handle, srcTensorDesc, filter_descriptor,
            convolution_descriptor, state.output->descriptor(), fwd_algo,
            &convolution_fw_ws_size));
        delete convolution_fw_ws;
        CUDA_CHECK(alloc_gpu(&convolution_fw_ws, convolution_fw_ws_size));

        int filter_dims[4] = {this->dims.outputs, this->dims.inputs,
            this->dims.kernel_height, this->dims.kernel_height};
        cudnnSetFilterNdDescriptor(filter_descriptor, CUDNN_DATA_TYPE,
                                   CUDNN_TENSOR_NCHW, 4, filter_dims);

        CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
            cuda_data.cudnn_handle, filter_descriptor,
            state.output->descriptor(), convolution_descriptor, srcTensorDesc,
            bwd_data_algo, &convolution_bw_data_ws_size));
        delete convolution_bw_data_ws;
        CUDA_CHECK(
            alloc_gpu(&convolution_bw_data_ws, convolution_bw_data_ws_size));

        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            cuda_data.cudnn_handle, srcTensorDesc, state.output->descriptor(),
            convolution_descriptor, filter_descriptor, bwd_filter_algo,
            &convolution_bw_filter_ws_size));
        delete convolution_bw_filter_ws;
        CUDA_CHECK(alloc_gpu(&convolution_bw_filter_ws,
                             convolution_bw_filter_ws_size));

        CUDNN_CHECK(cudnnDestroyTensorDescriptor(srcTensorDesc));
        return output_dims;
    }

    Tensor<ftype> *fwd(cuda_data_t cuda_data, LayerState<ftype> &state,
                       Tensor<ftype> *input) {
        ftype alpha = 1, beta = 0;

        state.input = input; // save the input

        CUDNN_CHECK(cudnnConvolutionForward(
            cuda_data.cudnn_handle, &alpha, input->descriptor(), input->data(),
            filter_descriptor, state.parameters.weights->data(),
            convolution_descriptor, fwd_algo, convolution_fw_ws,
            convolution_fw_ws_size, &beta, state.output->descriptor(),
            state.output->data()));
        beta = 1;
        // TODO: this doesn't work when batch_size > 1
        CUDNN_CHECK(cudnnAddTensor(cuda_data.cudnn_handle, &alpha,
                                   state.parameters.biases->descriptor(),
                                   state.parameters.biases->data(), &beta,
                                   state.output->descriptor(),
                                   state.output->data()));
        return state.output;
    }

    Tensor<ftype> *bwd(cuda_data_t cuda_data, LayerState<ftype> &state,
                       Tensor<ftype> *error) {
        ftype alpha = 1, beta = 0;
        // compute biases gradient (gradient / biases)
        cudnnConvolutionBackwardBias(cuda_data.cudnn_handle, &alpha,
                                     error->descriptor(), error->data(), &beta,
                                     state.gradients.biases->descriptor(),
                                     state.gradients.biases->data());
        // compute weights gradient (gradient / weights)
        cudnnConvolutionBackwardFilter(
            cuda_data.cudnn_handle, &alpha, state.input->descriptor(),
            state.input->data(), error->descriptor(), error->data(),
            convolution_descriptor, bwd_filter_algo, convolution_bw_filter_ws,
            convolution_bw_filter_ws_size, &beta, filter_descriptor,
            state.gradients.weights->data()); // TODO: we want to use a larger
                                              // temporary array here, and
                                              // average the resulting tensors
                                              // to get the final values for the
                                              // gradients
        // compute the output error (gradient / data)
        cudnnConvolutionBackwardData(
            cuda_data.cudnn_handle, &alpha, filter_descriptor,
            state.parameters.weights->data(), error->descriptor(),
            error->data(), convolution_descriptor, bwd_data_algo,
            convolution_bw_data_ws, convolution_bw_data_ws_size, &beta,
            state.error->descriptor(), state.error->data());

        return state.error;
    }
};

#endif
