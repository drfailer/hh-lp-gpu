#ifndef OPTIMIZERS_SGD_OPTIMIZER_H
#define OPTIMIZERS_SGD_OPTIMIZER_H
#include "../data/opt_layer_data.hpp"
#include "../types.hpp"
#include "optimizer.hpp"

class SGDOptimizer : public Optimizer<ftype> {
  public:
    struct UpdateGraph {
        cudnn_frontend::graph::Graph graph;
        ftype *workspace = nullptr;
        tensor_attr_t tensor;
        tensor_attr_t gradiant_tensor;
        tensor_attr_t scaled_gradiant_tensor;
        tensor_attr_t result_tensor;
        tensor_attr_t scale_tensor;
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
        update_graphs_.update_weights = create_update_graph(
            {1, dims.nb_nodes, dims.nb_inputs},
            {dims.nb_nodes * dims.nb_inputs, dims.nb_inputs, 1});

        if (has_biases(state)) {
            update_graphs_.update_biases = create_update_graph(
                {1, dims.nb_nodes, 1}, {dims.nb_nodes, 1, 1});
        }
    }

    void optimize(LayerState<ftype> const &state,
                  ftype learning_rate) override {
        INFO_GRP("Optimizer", INFO_GRP_LAYER_TASK);
        if (!has_params(state)) {
            return;
        }

        if (update_graphs_.update_weights) {
            optimize_params(update_graphs_.update_weights, state.params.weights,
                            state.grads.weights, learning_rate);
        }

        if (update_graphs_.update_biases) {
            optimize_params(update_graphs_.update_biases, state.params.biases,
                            state.grads.biases, learning_rate);
        }
    }

    void optimize_params(auto opt, ftype *parameter, ftype *gradiant,
                         ftype learning_rate) {
        MemoryMap mem = {
            {opt->scale_tensor, &learning_rate},
            {opt->tensor, parameter},
            {opt->gradiant_tensor, gradiant},
            {opt->scaled_gradiant_tensor, gradiant},
            {opt->result_tensor, parameter},
        };

        CUDNN_CHECK(opt->graph.execute(cudnn_handle_, mem, opt->workspace));
    }

    std::shared_ptr<UpdateGraph>
    create_update_graph(std::vector<int64_t> const &dims,
                        std::vector<int64_t> const &strides) {
        namespace fe = cudnn_frontend;
        auto data = std::make_shared<UpdateGraph>();
        data->graph.set_io_data_type(fe::DataType_t::FLOAT)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);

        // tenors attributes
        auto learning_rate_attributes = fe::graph::Tensor_attributes()
                                            .set_name("learning rate")
                                            .set_dim({1, 1, 1})
                                            .set_stride({1, 1, 1});
        auto param_attributes = fe::graph::Tensor_attributes()
                                    .set_name("parameter")
                                    .set_dim(dims)
                                    .set_stride(strides);
        auto grad_attributes = fe::graph::Tensor_attributes()
                                   .set_name("parameter gradiant")
                                   .set_dim(dims)
                                   .set_stride(strides);
        auto scale_attributes = fe::graph::Pointwise_attributes()
                                    .set_name("learning_rate * gradiant")
                                    .set_mode(fe::PointwiseMode_t::MUL);
        auto substract_attributes =
            fe::graph::Pointwise_attributes()
                .set_name("parameters - learning_rate * gradiant")
                .set_mode(fe::PointwiseMode_t::SUB);

        // graph inputs:

        data->tensor = data->graph.tensor(param_attributes);
        data->gradiant_tensor = data->graph.tensor(grad_attributes);
        data->scale_tensor = data->graph.tensor(learning_rate_attributes);

        // operations:

        // scaled_gradiant = gradiants * learning_rate
        data->scaled_gradiant_tensor = data->graph.pointwise(
            data->scale_tensor, data->gradiant_tensor, scale_attributes);
        // parameters -= scaled_gradiant
        data->result_tensor = data->graph.pointwise(
            data->tensor, data->scaled_gradiant_tensor, substract_attributes);

        // result:

        data->result_tensor->set_output(true);

        CUDNN_CHECK(data->graph.validate());
        CUDNN_CHECK(data->graph.build(cudnn_handle_, {fe::HeurMode_t::A}));

        int64_t workspace_size;
        CUDNN_CHECK(data->graph.get_workspace_size(workspace_size));
        CUDA_CHECK(alloc_gpu(&data->workspace, workspace_size));

        return data;
    }

    std::shared_ptr<Optimizer<ftype>> copy() const override {
        return std::make_shared<SGDOptimizer>(cudnn_handle_);
    }

  public:
    bool has_params(LayerState<ftype> const &state) {
        return state.params.weights != nullptr &&
               state.params.biases != nullptr;
    }

    bool has_biases(LayerState<ftype> const &state) {
        return state.params.biases != nullptr && state.grads.biases != nullptr;
    }

  private:
    LayerUpdateData update_graphs_;
    cudnnHandle_t cudnn_handle_ = nullptr;
};

#endif
