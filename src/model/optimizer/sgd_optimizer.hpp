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
        ftype *workspace = nullptr;
        cudnnTensorDescriptor_t parameter_tensor;
        cudnnTensorDescriptor_t gradiant_tensor;

        UpdateGraph() {
            cudnnCreateTensorDescriptor(&parameter_tensor);
            cudnnCreateTensorDescriptor(&gradiant_tensor);
        }

        ~UpdateGraph() {
            cudnnDestroyTensorDescriptor(parameter_tensor);
            cudnnDestroyTensorDescriptor(gradiant_tensor);
        }
    };

    struct LayerUpdateData {
        std::shared_ptr<UpdateGraph> update_weights = nullptr;
        std::shared_ptr<UpdateGraph> update_biases = nullptr;
    };

  public:
    SGDOptimizer(cudnnHandle_t cudnn_handle) : cudnn_handle_(cudnn_handle) {}

    void init(LayerState<ftype> const &state) override {
        if (!has_params(state)) {
            return;
        }
        auto dims = state.dims;
        update_data_.update_weights = create_update_graph(
            {1, 1, dims.outputs, dims.inputs},
            {1, dims.outputs * dims.inputs, dims.inputs, 1});

        if (has_biases(state)) {
            update_data_.update_biases = create_update_graph(
                {1, 1, dims.outputs, 1}, {1, dims.outputs, 1, 1});
        }
    }

    std::shared_ptr<UpdateGraph>
    create_update_graph(std::vector<int64_t> const &dims,
                        std::vector<int64_t> const &strides) {
        auto data = std::make_shared<UpdateGraph>();

        // param = param - learning_rate * gradiant
        // TODO: add the batch size
        cudnnSetTensor4dDescriptorEx(
            data->parameter_tensor, CUDNN_DATA_FLOAT, dims[0], dims[1], dims[2],
            dims[3], strides[0], strides[1], strides[2], strides[3]);
        cudnnSetTensor4dDescriptorEx(
            data->gradiant_tensor, CUDNN_DATA_FLOAT, dims[0], dims[1], dims[2],
            dims[3], strides[0], strides[1], strides[2], strides[3]);
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
                            state.gradiants.weights, learning_rate);
        }

        if (update_data_.update_biases) {
            optimize_params(update_data_.update_biases, state.biases,
                            state.gradiants.biases, learning_rate);
        }
    }

    void optimize_params(auto opt, ftype *parameter, ftype *gradiant,
                         ftype learning_rate) {
        ftype beta = 1;
        learning_rate *= -1;
        cudnnAddTensor(cudnn_handle_, &learning_rate, opt->gradiant_tensor,
                       gradiant, &beta, opt->parameter_tensor, parameter);
    }

    std::shared_ptr<Optimizer<ftype>> copy() const override {
        return std::make_shared<SGDOptimizer>(cudnn_handle_);
    }

  public:
    bool has_params(LayerState<ftype> const &state) {
        return state.weights != nullptr && state.biases != nullptr;
    }

    bool has_biases(LayerState<ftype> const &state) {
        return state.biases != nullptr && state.gradiants.biases != nullptr;
    }

  private:
    LayerUpdateData update_data_;
    cudnnHandle_t cudnn_handle_ = nullptr;
};

#endif
