#ifndef LAYERS_LINEAR_LAYER_H
#define LAYERS_LINEAR_LAYER_H
#include "../../kernels/linear_layer_kernel.h"
#include "../../tools/gpu.hpp"
#include "../../types.hpp"
#include "layer.hpp"
#include <cassert>
#include <cudnn.h>
#include <cudnn_graph.h>
#include <cudnn_ops.h>

class LinearLayer : public Layer {
  public:
    int nb_inputs;
    int nb_outputs;

    LinearLayer(int nb_inputs, int nb_outputs, tensor::dtype_t dtype = CUDNN_DATA_TYPE)
        : Layer(dims_t{.inputs = nb_inputs, .outputs = nb_outputs}, dtype),
          nb_inputs(nb_inputs), nb_outputs(nb_outputs) {
        CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(&average_tensor));
        CUDNN_CHECK(cudnnSetReduceTensorDescriptor(
            average_tensor, CUDNN_REDUCE_TENSOR_AVG, CUDNN_DATA_TYPE,
            CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES,
            CUDNN_32BIT_INDICES));
    }

    ~LinearLayer() override {
        CUDNN_CHECK(cudnnDestroyReduceTensorDescriptor(average_tensor));
    }

  public:
    LayerParametersShape parameters_shape() const override {
        return LayerParametersShape{
            .w = tensor::shape(1, 1, nb_outputs, nb_inputs),
            .b = tensor::shape(1, 1, nb_outputs, 1),
        };
    }

    LayerIOShape io_shape(tensor::dims_t const &input_dims) const override {
        int batch_size = input_dims[0];
        return LayerIOShape{
            .x = tensor::shape(batch_size, 1, nb_inputs, 1),
            .y = tensor::shape(batch_size, 1, nb_outputs, 1),
        };
    }

    void init_parameters(CUDA cuda, Parameters const &params) override {
        params.w.random_init(-0.05, 0.05);
        params.b.random_init(-0.05, 0.05);
    }

    void init_fwd(CUDA cuda, InitFwdData const &data) override {
        this->dims.batch_size = data.x.dim(0);
    }

    void fwd(CUDA cuda, LayerFwdIn const &in, LayerFwdOut const &out) override {
        INFO_GRP("LinearLayer FWD", INFO_GRP_LAYER_TASK);

        CUDNN_CHECK(hhlpLinearForward(cuda.cudnn_handle, in.w.data(),
                                      in.b.data(), in.x.data(), out.y.data(),
                                      nb_inputs, nb_outputs,
                                      this->dims.batch_size, CUDNN_DATA_TYPE));
    }

    void bwd(CUDA cuda, LayerBwdIn const &in, LayerBwdOut const &out) override {
        INFO_GRP("LinearLayer BWD", INFO_GRP_LAYER_TASK);

        // grads_b = error
        CUDNN_CHECK(hhlpLinearBackwardBias(
            cuda.cudnn_handle, in.dy.data(), out.db.data(), this->dims.outputs,
            this->dims.batch_size, CUDNN_DATA_TYPE));
        // w_grad = err * fwd_inputT
        CUDNN_CHECK(hhlpLinearBackwardWeights(
            cuda.cudnn_handle, in.dy.data(), in.x.data(), out.dw.data(),
            this->dims.outputs, this->dims.inputs, this->dims.batch_size,
            CUDNN_DATA_TYPE));
        // output_err = errT * weights
        CUDNN_CHECK(hhlpLinearBackwardData(
            cuda.cudnn_handle, in.dy.data(), in.w.data(), out.dx.data(),
            nb_outputs, nb_inputs, this->dims.batch_size, CUDNN_DATA_TYPE));
    }

  private:
    cudnnReduceTensorDescriptor_t average_tensor = nullptr;
};

#endif
