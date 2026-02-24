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

struct ConvolutionLayer : Layer {
    bool use_biases = false;
    cudnnConvolutionFwdAlgo_t fwd_algo;
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
    cudnnConvolutionDescriptor_t convolution_descriptor = nullptr;
    cudnnFilterDescriptor_t filter_descriptor = nullptr;
    int input_height = 1;
    int input_width = 1;
    int nb_inputs = 1;
    int nb_outputs = 1;
    int kernel_width = 0;
    int kernel_height = 0;

    void *convolution_fw_ws = nullptr;
    size_t convolution_fw_ws_size = 0;

    void *convolution_bw_data_ws = nullptr;
    size_t convolution_bw_data_ws_size = 0;

    void *convolution_bw_filter_ws = nullptr;
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
            CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
        tensor::dtype_t dtype = CUDNN_DATA_TYPE)
        : Layer({.inputs = inputs,
                 .outputs = outputs,
                 .kernel_width = kernel_width,
                 .kernel_height = kernel_height}, dtype),
          use_biases(use_biases), fwd_algo(fwd_algo),
          bwd_data_algo(bwd_data_algo), bwd_filter_algo(bwd_filter_algo),
          input_height(input_width), input_width(input_width),
          nb_inputs(inputs), nb_outputs(outputs), kernel_width(kernel_width),
          kernel_height(kernel_height) {
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
        CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(
            convolution_descriptor, convolution_dims, padding, filter_strides,
            filter_upscale, CUDNN_CROSS_CORRELATION, CUDNN_DATA_TYPE));
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_descriptor));
        int filter_dims[4] = {outputs, inputs, kernel_height, kernel_width};
        CUDNN_CHECK(
            cudnnSetFilterNdDescriptor(filter_descriptor, CUDNN_DATA_TYPE,
                                       CUDNN_TENSOR_NCHW, 4, filter_dims));
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

    LayerParametersShape parameters_shape() const override {
        tensor::TensorShape bias_shape = {{0}, {0}};

        if (use_biases) {
            bias_shape = tensor::shape(1, dims.outputs, 1, 1);
        }
        return LayerParametersShape{
            .w = tensor::shape(dims.outputs, dims.inputs, dims.kernel_height,
                               dims.kernel_width),
            .b = bias_shape,
        };
    }

    LayerIOShape io_shape(tensor::dims_t const &input_dims) const override {
        int batch_size = input_dims[0];
        tensor::dims_t output_dims;

        // TODO: we should be able to remove this at some point
        // properly set the input descriptor
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_TYPE, input_dims[0],
            input_dims[1], input_dims[2], input_dims[3]));
        assert(input_height == input_dims[2]);
        assert(input_width == input_dims[3]);

        assert(input_dims[2] == input_height);
        assert(input_dims[3] == input_width);
        CUDNN_CHECK(cudnnGetConvolutionNdForwardOutputDim(
            convolution_descriptor, input_descriptor, filter_descriptor,
            output_dims.size(), output_dims.data()));
        assert(output_dims[1] == nb_outputs);
        return LayerIOShape{
            .x = tensor::shape(batch_size, 1, input_height, input_width),
            .y = tensor::shape(output_dims),
        };
    }

    void init_parameters(CUDA cuda, Parameters const &params) override {
        CUDA_CHECK(params.w.random_init(-0.05, 0.05));
        if (use_biases) {
            CUDA_CHECK(params.b.random_init(-0.05, 0.05));
        }
    }

    void init_fwd(CUDA cuda, InitFwdData const &data) override {
        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
            cuda.cudnn_handle, input_descriptor, filter_descriptor,
            convolution_descriptor, data.y.desc(), fwd_algo,
            &convolution_fw_ws_size));
        cudaFree(convolution_fw_ws);
        CUDA_CHECK(alloc_gpu(&convolution_fw_ws, convolution_fw_ws_size));
    }

    void init_bwd(CUDA cuda, LayerData const &data) override {
        CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
            cuda.cudnn_handle, filter_descriptor, data.y.desc(),
            convolution_descriptor, input_descriptor, bwd_data_algo,
            &convolution_bw_data_ws_size));
        cudaFree(convolution_bw_data_ws);
        CUDA_CHECK(alloc_gpu(&convolution_bw_data_ws, convolution_bw_data_ws_size));

        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            cuda.cudnn_handle, input_descriptor, data.y.desc(),
            convolution_descriptor, filter_descriptor, bwd_filter_algo,
            &convolution_bw_filter_ws_size));
        cudaFree(convolution_bw_filter_ws);
        CUDA_CHECK(alloc_gpu(&convolution_bw_filter_ws, convolution_bw_filter_ws_size));
    }

    void fwd(CUDA cuda, LayerFwdIn const &in, LayerFwdOut const &out) override {
        ftype alpha = 1, beta = 0;

        CUDNN_CHECK(cudnnConvolutionForward(
            cuda.cudnn_handle, &alpha, in.x.desc(), in.x.data(),
            filter_descriptor, in.w.data(), convolution_descriptor, fwd_algo,
            convolution_fw_ws, convolution_fw_ws_size, &beta, out.y.desc(),
            out.y.data()));

        if (!use_biases) {
            return;
        }

        alpha = 1;
        beta = 1;
        CUDNN_CHECK(cudnnAddTensor(cuda.cudnn_handle, &alpha, in.b.desc(),
                                   in.b.data(), &beta, out.y.desc(),
                                   out.y.data()));
    }

    void bwd(CUDA cuda, LayerBwdIn const &in, LayerBwdOut const &out) override {
        ftype alpha = 1.0 / dims.batch_size, beta = 0;
        // The shape of the input error might be wrong if the next layer is
        // linear, so we need to use the shape of the output.
        auto output_gradient_descriptor = in.dy.desc();
        auto output_gradient_data = in.dy.data();

        // compute biases gradient (gradient / biases)
        CUDNN_CHECK(cudnnConvolutionBackwardBias(
            cuda.cudnn_handle, &alpha, output_gradient_descriptor,
            output_gradient_data, &beta, out.db.desc(), out.db.data()));

        // compute weights gradient (gradient / weights)
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(
            cuda.cudnn_handle, &alpha, in.x.desc(), in.x.data(),
            output_gradient_descriptor, output_gradient_data,
            convolution_descriptor, bwd_filter_algo, convolution_bw_filter_ws,
            convolution_bw_filter_ws_size, &beta, filter_descriptor,
            out.dw.data()));

        // compute the output error (gradient / data)
        alpha = 1;
        CUDNN_CHECK(cudnnConvolutionBackwardData(
            cuda.cudnn_handle, &alpha, filter_descriptor, in.w.data(),
            output_gradient_descriptor, output_gradient_data,
            convolution_descriptor, bwd_data_algo, convolution_bw_data_ws,
            convolution_bw_data_ws_size, &beta, out.dx.desc(), out.dx.data()));
    }
};

#endif
