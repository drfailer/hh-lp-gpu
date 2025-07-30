#ifndef MODEL_LAYER_CONVOLUTION_LAYER
#define MODEL_LAYER_CONVOLUTION_LAYER
#include "../../tools/gpu.hpp"
#include "../../types.hpp"
#include "layer.hpp"
#include <cassert>
#include <cudnn.h>
#include <cudnn_cnn.h>
#include <cudnn_graph.h>
#include <cudnn_ops.h>

struct ConvolutionLayer : Layer<ftype> {
    bool use_biases = false;
    cudnnConvolutionFwdAlgo_t fwd_algo;
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
    cudnnConvolutionDescriptor_t convolution_descriptor = nullptr;
    cudnnFilterDescriptor_t filter_descriptor = nullptr;
    int input_height = 1;
    int input_width = 1;

    ftype *convolution_fw_ws = nullptr;
    size_t convolution_fw_ws_size = 0;

    ftype *convolution_bw_data_ws = nullptr;
    size_t convolution_bw_data_ws_size = 0;

    ftype *convolution_bw_filter_ws = nullptr;
    size_t convolution_bw_filter_ws_size = 0;

    // We need to define the input of the layer in case the output of the
    // previous layer dosn't have the right shape.
    cudnnTensorDescriptor_t input_descriptor;

    // filter data
    static constexpr size_t convolution_dims = 2;
    int padding[convolution_dims] = {0, 0};
    int filter_strides[convolution_dims] = {1, 1};
    int filter_upscale[convolution_dims] = {1, 1};

    // n = c = 1;
    // h     = IMAGE_H;
    // w     = IMAGE_W;

    ConvolutionLayer(
        int inputs, int outputs, int input_width, int input_height,
        int kernel_width, int kernel_height, bool use_biases = true,
        cudnnConvolutionFwdAlgo_t fwd_algo = CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        cudnnConvolutionBwdDataAlgo_t bwd_data_algo =
            CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
        cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo =
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0)
        : Layer({.inputs = inputs,
                 .outputs = outputs,
                 .kernel_width = kernel_width,
                 .kernel_height = kernel_height}),
          use_biases(use_biases), fwd_algo(fwd_algo),
          bwd_data_algo(bwd_data_algo), bwd_filter_algo(bwd_filter_algo),
          input_height(input_width), input_width(input_width) {
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
        CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(
            convolution_descriptor, convolution_dims, padding, filter_strides,
            filter_upscale, CUDNN_CROSS_CORRELATION, CUDNN_DATA_TYPE));
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_descriptor));

        const tensor::dims_t filter_dim = {outputs, inputs, (int)kernel_height,
                                           (int)kernel_width};

        CUDNN_CHECK(cudnnSetFilterNdDescriptor(
            filter_descriptor, CUDNN_DATA_TYPE, CUDNN_TENSOR_NCHW,
            filter_dim.size(), filter_dim.data()));

        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_descriptor));
    }

    ~ConvolutionLayer() override {
        CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(convolution_descriptor));
        CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_descriptor));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_descriptor));
        cudaFree(convolution_fw_ws);
        cudaFree(convolution_bw_filter_ws);
        cudaFree(convolution_bw_data_ws);
    }

    Parameters<ftype> create_parameters() const override {
        Parameters<ftype> parameters;

        parameters.weights.reshape(dims.outputs, dims.inputs,
                                   dims.kernel_height, dims.kernel_width);
        CUDA_CHECK(parameters.weights.random_init(-0.05, 0.05));

        if (use_biases) {
            parameters.biases.reshape(1, dims.outputs, 1, 1);
            CUDA_CHECK(parameters.biases.random_init(-0.05, 0.05));
        }

        int filter_dims[4] = {dims.outputs, dims.inputs, dims.kernel_height,
                              dims.kernel_height};
        CUDNN_CHECK(
            cudnnSetFilterNdDescriptor(filter_descriptor, CUDNN_DATA_TYPE,
                                       CUDNN_TENSOR_NCHW, 4, filter_dims));

        return parameters;
    }

    tensor::dims_t init(cuda_data_t cuda_data, LayerData<ftype> &state,
                        tensor::dims_t input_dims) override {
        tensor::dims_t output_dims;

        dims.batch_size = input_dims[0];

        // properly set the input descriptor
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_TYPE, input_dims[0],
            input_dims[1], input_height, input_width));
        assert(input_height == input_dims[2]);
        assert(input_width == input_dims[3]);

        state.dx.reshape(input_dims);

        CUDNN_CHECK(cudnnGetConvolutionNdForwardOutputDim(
            convolution_descriptor, input_descriptor, filter_descriptor,
            output_dims.size(), output_dims.data()));

        state.y.reshape(output_dims);

        if (use_biases) {
            assert(output_dims[1] == dims.outputs);
        }

        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
            cuda_data.cudnn_handle, input_descriptor, filter_descriptor,
            convolution_descriptor, state.y.desc(), fwd_algo,
            &convolution_fw_ws_size));
        cudaFree(convolution_fw_ws);
        CUDA_CHECK(alloc_gpu(&convolution_fw_ws, convolution_fw_ws_size));

        CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
            cuda_data.cudnn_handle, filter_descriptor, state.y.desc(),
            convolution_descriptor, input_descriptor, bwd_data_algo,
            &convolution_bw_data_ws_size));
        cudaFree(convolution_bw_data_ws);
        CUDA_CHECK(
            alloc_gpu(&convolution_bw_data_ws, convolution_bw_data_ws_size));

        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            cuda_data.cudnn_handle, input_descriptor, state.y.desc(),
            convolution_descriptor, filter_descriptor, bwd_filter_algo,
            &convolution_bw_filter_ws_size));
        cudaFree(convolution_bw_filter_ws);
        CUDA_CHECK(alloc_gpu(&convolution_bw_filter_ws,
                             convolution_bw_filter_ws_size));
        return output_dims;
    }

    tensor::Tensor<ftype> const &
    fwd(cuda_data_t cuda_data, LayerData<ftype> &state,
        tensor::Tensor<ftype> const &input) override {
        ftype alpha = 1, beta = 0;

        CUDNN_CHECK(cudnnConvolutionForward(
            cuda_data.cudnn_handle, &alpha, input.desc(), input.data(),
            filter_descriptor, state.w.data(), convolution_descriptor, fwd_algo,
            convolution_fw_ws, convolution_fw_ws_size, &beta, state.y.desc(),
            state.y.data()));

        alpha = 1;
        beta = 1;
        CUDNN_CHECK(cudnnAddTensor(cuda_data.cudnn_handle, &alpha,
                                   state.b.desc(), state.b.data(), &beta,
                                   state.y.desc(), state.y.data()));
        return state.y;
    }

    tensor::Tensor<ftype> const &
    bwd(cuda_data_t cuda_data, LayerData<ftype> &state,
        tensor::Tensor<ftype> const &input,
        tensor::Tensor<ftype> const &output_gradient) override {
        ftype alpha = 1.0 / dims.batch_size, beta = 0;
        // The shape of the input error might be wrong if the next layer is
        // linear, so we need to use the shape of the output.
        auto output_gradient_descriptor = state.y.desc();
        auto output_gradient_data = output_gradient.data();

        // compute biases gradient (gradient / biases)
        CUDNN_CHECK(cudnnConvolutionBackwardBias(
            cuda_data.cudnn_handle, &alpha, output_gradient_descriptor,
            output_gradient_data, &beta, state.db.desc(), state.db.data()));

        // compute weights gradient (gradient / weights)
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(
            cuda_data.cudnn_handle, &alpha, state.x.desc(), state.x.data(),
            output_gradient_descriptor, output_gradient_data,
            convolution_descriptor, bwd_filter_algo, convolution_bw_filter_ws,
            convolution_bw_filter_ws_size, &beta, filter_descriptor,
            state.dw.data()));

        // compute the output error (gradient / data)
        alpha = 1;
        CUDNN_CHECK(cudnnConvolutionBackwardData(
            cuda_data.cudnn_handle, &alpha, filter_descriptor, state.w.data(),
            output_gradient_descriptor, output_gradient_data,
            convolution_descriptor, bwd_data_algo, convolution_bw_data_ws,
            convolution_bw_data_ws_size, &beta, state.dx.desc(),
            state.dx.data()));

        return state.dx;
    }
};

#endif
