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
    };

    struct LayerUpdateData {
        UpdateGraph update_weights;
        UpdateGraph update_biases;
    };

    UpdateGraph create_update_graph(cudnnHandle_t handle,
                                    cudnnReduceTensorDescriptor_t reduce_tensor,
                                    std::vector<int64_t> const &dims,
                                    std::vector<int64_t> const &strides) const {
        UpdateGraph graph;

        cudnnCreateTensorDescriptor(&graph.parameter_tensor);
        cudnnCreateTensorDescriptor(&graph.gradient_tensor);
        CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
            graph.parameter_tensor, CUDNN_DATA_FLOAT, 1, dims[1], dims[2],
            dims[3], strides[0], strides[1], strides[2], strides[3]));
        CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
            graph.gradient_tensor, CUDNN_DATA_FLOAT, dims[0], dims[1], dims[2],
            dims[3], strides[0], strides[1], strides[2], strides[3]));
        CUDNN_CHECK(cudnnGetReductionWorkspaceSize(
            handle, reduce_tensor, graph.gradient_tensor,
            graph.parameter_tensor, &graph.workspace_size));
        CUDA_CHECK(alloc_gpu(&graph.workspace, graph.workspace_size));
        return graph;
    }

    void destroy_update_graph(UpdateGraph &graph) {
        CUDA_CHECK(cudaFree(graph.workspace));
        cudnnDestroyTensorDescriptor(graph.parameter_tensor);
        cudnnDestroyTensorDescriptor(graph.gradient_tensor);
        graph = {0};
    }

  public:
    SGDOptimizer(cudnnHandle_t cudnn_handle) : cudnn_handle_(cudnn_handle) {
        CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(&reduce_tensor));
        CUDNN_CHECK(cudnnSetReduceTensorDescriptor(
            reduce_tensor, CUDNN_REDUCE_TENSOR_AVG, CUDNN_DATA_FLOAT,
            CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES,
            CUDNN_32BIT_INDICES));
    }

    ~SGDOptimizer() {
        cudnnDestroyReduceTensorDescriptor(reduce_tensor);
        destroy_update_graph(update_data_.update_weights);
        destroy_update_graph(update_data_.update_biases);
    }

    void optimize(LayerState<ftype> const &state,
                  ftype learning_rate) override {
        INFO_GRP("Optimizer", INFO_GRP_LAYER_TASK);
        if (update_data_.update_weights.parameter_tensor) {
            optimize_params(update_data_.update_weights, state.weights,
                            state.gradients.weights, learning_rate);
        }

        if (update_data_.update_biases.parameter_tensor) {
            optimize_params(update_data_.update_biases, state.biases,
                            state.gradients.biases, learning_rate);
        }
    }

    void optimize_params(auto &opt, ftype *parameter, ftype *gradient,
                         ftype learning_rate) {
        ftype alpha = -learning_rate, beta = 1;

        // param = param - learning_rate * gradient

        if (has_batch_) {
            /* Tensor operation : C = reduce op( alpha * A ) + beta * C */
            CUDNN_CHECK(cudnnReduceTensor(
                cudnn_handle_, reduce_tensor, nullptr, 0, opt.workspace,
                opt.workspace_size, &alpha, opt.gradient_tensor, gradient,
                &beta, opt.parameter_tensor, parameter));
        } else {
            CUDNN_CHECK(cudnnAddTensor(cudnn_handle_, &alpha,
                                       opt.gradient_tensor, gradient, &beta,
                                       opt.parameter_tensor, parameter));
        }
    }

    std::shared_ptr<Optimizer<ftype>>
    create(shape_t const &shape) const override {
        auto result = std::make_shared<SGDOptimizer>(cudnn_handle_);

        if (shape.dims.weights.size() > 0) {
            result->has_batch_ = shape.dims.weights[0] > 1;
            result->update_data_.update_weights =
                create_update_graph(cudnn_handle_, reduce_tensor,
                                    shape.dims.weights, shape.strides.weights);
        }

        if (shape.dims.biases.size() > 0) {
            result->has_batch_ = shape.dims.biases[0] > 1;
            result->update_data_.update_biases =
                create_update_graph(cudnn_handle_, reduce_tensor,
                                    shape.dims.biases, shape.strides.biases);
        }
        return result;
    }

  private:
    cudnnHandle_t cudnn_handle_ = nullptr;
    cudnnReduceTensorDescriptor_t reduce_tensor = nullptr;
    LayerUpdateData update_data_;
    bool has_batch_ = false;
};

#endif
