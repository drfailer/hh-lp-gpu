#ifndef MODEL_OPTIMIZER_SGD_OPTIMIZER_H
#define MODEL_OPTIMIZER_SGD_OPTIMIZER_H
#include "../../tools/gpu.hpp"
#include "../../types.hpp"
#include "optimizer.hpp"
#include <cudnn.h>
#include <cudnn_graph.h>
#include <cudnn_ops.h>
#include <log.h/log.h>
#include <vector>

class SGDOptimizer : public Optimizer<ftype> {
  public:
    struct UpdateGraph {
        cudnnTensorDescriptor_t parameter_tensor = nullptr;
        cudnnTensorDescriptor_t gradient_tensor = nullptr;
        ftype *workspace = nullptr;
        size_t workspace_size = 0;

        UpdateGraph() {
            cudnnCreateTensorDescriptor(&parameter_tensor);
            cudnnCreateTensorDescriptor(&gradient_tensor);
        }

        ~UpdateGraph() {
            CUDA_CHECK(cudaFree(workspace));
            cudnnDestroyTensorDescriptor(parameter_tensor);
            cudnnDestroyTensorDescriptor(gradient_tensor);
        }
    };

    struct LayerUpdateData {
        std::shared_ptr<UpdateGraph> update_weights = nullptr;
        std::shared_ptr<UpdateGraph> update_biases = nullptr;
    };

  public:
    SGDOptimizer(cudnnHandle_t cudnn_handle) : cudnn_handle_(cudnn_handle) {
        CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(&reduce_tensor));
        CUDNN_CHECK(cudnnSetReduceTensorDescriptor(
            reduce_tensor, CUDNN_REDUCE_TENSOR_AVG, CUDNN_DATA_FLOAT,
            CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES,
            CUDNN_32BIT_INDICES));
    }

    ~SGDOptimizer() { cudnnDestroyReduceTensorDescriptor(reduce_tensor); }

    void init(LayerState<ftype> const &state) override {
        if (!has_params(state)) {
            return;
        }

        auto dims = state.dims;
        update_data_.update_weights = create_update_graph(
            {dims.batch_count, 1, dims.outputs, dims.inputs},
            {dims.outputs * dims.inputs, dims.outputs * dims.inputs,
             dims.inputs, 1});

        if (has_biases(state)) {
            update_data_.update_biases =
                create_update_graph({dims.batch_count, 1, dims.outputs, 1},
                                    {dims.outputs, dims.outputs, 1, 1});
        }
    }

    std::shared_ptr<UpdateGraph>
    create_update_graph(std::vector<int64_t> const &dims,
                        std::vector<int64_t> const &strides) {
        auto data = std::make_shared<UpdateGraph>();

        // param = param - learning_rate * gradient
        // NOTE: the parameter is not batched so the first dimension is always 1
        CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
            data->parameter_tensor, CUDNN_DATA_FLOAT, 1, dims[1], dims[2],
            dims[3], strides[0], strides[1], strides[2], strides[3]));
        CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
            data->gradient_tensor, CUDNN_DATA_FLOAT, dims[0], dims[1], dims[2],
            dims[3], strides[0], strides[1], strides[2], strides[3]));
        CUDNN_CHECK(cudnnGetReductionWorkspaceSize(
            cudnn_handle_, reduce_tensor, data->gradient_tensor,
            data->parameter_tensor, &data->workspace_size));
        CUDA_CHECK(alloc_gpu(&data->workspace, data->workspace_size));
        return data;
    }

    void optimize(LayerState<ftype> const &state,
                  ftype learning_rate) override {
        INFO_GRP("Optimizer", INFO_GRP_LAYER_TASK);
        if (!has_params(state)) {
            return;
        }

        if (update_data_.update_weights) {
            optimize_params(update_data_.update_weights, state.weights,
                            state.gradients.weights, learning_rate,
                            state.dims.batch_count);
        }

        if (update_data_.update_biases) {
            optimize_params(update_data_.update_biases, state.biases,
                            state.gradients.biases, learning_rate,
                            state.dims.batch_count);
        }
    }

    void optimize_params(auto opt, ftype *parameter, ftype *gradient,
                         ftype learning_rate, size_t batch_count) {
        ftype alpha = -learning_rate, beta = 1;

        /* Tensor operation : C = reduce op( alpha * A ) + beta * C */
        // cudnnReduceTensor(cudnnHandle_t handle,
        //                   const cudnnReduceTensorDescriptor_t
        //                   reduceTensorDesc, void *indices, size_t
        //                   indicesSizeInBytes, void *workspace, size_t
        //                   workspaceSizeInBytes, const void *alpha, const
        //                   cudnnTensorDescriptor_t aDesc, const void *A, const
        //                   void *beta, const cudnnTensorDescriptor_t cDesc,
        //                   void *C);
        if (batch_count > 1) {
            CUDNN_CHECK(cudnnReduceTensor(
                cudnn_handle_, reduce_tensor, nullptr, 0, opt->workspace,
                opt->workspace_size, &alpha, opt->gradient_tensor, gradient,
                &beta, opt->parameter_tensor, parameter));
        } else {
            CUDNN_CHECK(cudnnAddTensor(cudnn_handle_, &alpha,
                                       opt->gradient_tensor, gradient, &beta,
                                       opt->parameter_tensor, parameter));
        }
    }

    std::shared_ptr<Optimizer<ftype>> copy() const override {
        return std::make_shared<SGDOptimizer>(cudnn_handle_);
    }

  public:
    bool has_params(LayerState<ftype> const &state) {
        return state.weights != nullptr && state.biases != nullptr;
    }

    bool has_biases(LayerState<ftype> const &state) {
        return state.biases != nullptr && state.gradients.biases != nullptr;
    }

  private:
    LayerUpdateData update_data_;
    cudnnReduceTensorDescriptor_t reduce_tensor = nullptr;
    cudnnHandle_t cudnn_handle_ = nullptr;
};

#endif
