#ifndef TASK_OPTIMIZER_SGD_OPTIMIZER_TASK_H
#define TASK_OPTIMIZER_SGD_OPTIMIZER_TASK_H
#include "../../tools/timer.hpp"
#include "optimizer_task.hpp"

class SGDOptimizerTask : public OptimizerTask {
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
    SGDOptimizerTask(size_t nb_threads, cudnnHandle_t cudnn_handle,
                     cublasHandle_t cublas_handle)
        : OptimizerTask("SGD Optimizer", nb_threads, cudnn_handle,
                        cublas_handle) {}

#warning                                                                       \
    "Generalize dims and strides: changing the Parameters type and adding the dims and the strides here would be a good idea"

    void init(NetworkState<ftype> const &state) override {
        update_graphs_ =
            std::vector<LayerUpdateData>(state.layer_states.size());

        for (size_t idx = 0; idx < update_graphs_.size(); ++idx) {
            if (!has_params(state.layer_states[idx])) {
                WARN("layer " << idx << " has no parameters");
                continue;
            }
            auto dims = state.layer_states[idx].dims;
            update_graphs_[idx].update_weights = create_update_graph(
                {1, dims.nb_nodes, dims.nb_inputs},
                {dims.nb_nodes * dims.nb_inputs, dims.nb_inputs, 1});

            if (has_biases(state.layer_states[idx])) {
                update_graphs_[idx].update_biases = create_update_graph(
                    {1, dims.nb_nodes, 1}, {dims.nb_nodes, 1, 1});
            }
        }
    }

    void execute(std::shared_ptr<OptLayerData<ftype>> data) override {
        INFO_GRP("OptimizerTask", INFO_GRP_LAYER_TASK);
        if (!has_params(data->state)) {
            this->addResult(data);
            return;
        }

        if (update_graphs_[data->idx].update_weights) {
            optimize(update_graphs_[data->idx].update_weights,
                     data->state.params.weights, data->state.grads.weights,
                     data->learning_rate);
        }

        if (update_graphs_[data->idx].update_biases) {
            optimize(update_graphs_[data->idx].update_biases,
                     data->state.params.biases, data->state.grads.biases,
                     data->learning_rate);
        }
        this->addResult(data);
    }

    std::shared_ptr<hh::AbstractTask<OptimizerTaskIO>> copy() override {
        timer_start(sgd_copy);
        auto copy = std::make_shared<SGDOptimizerTask>(this->numberThreads(),
                                                       cudnn(), cublas());
        timer_end(sgd_copy);
        timer_report_prec(sgd_copy, milliseconds);
        return copy;
    }

    void optimize(auto opt, ftype *parameter, ftype *gradiant,
                  ftype learning_rate) {
        MemoryMap mem = {
            {opt->scale_tensor, &learning_rate},
            {opt->tensor, parameter},
            {opt->gradiant_tensor, gradiant},
            {opt->scaled_gradiant_tensor, gradiant},
            {opt->result_tensor, parameter},
        };

        CUDNN_CHECK(opt->graph.execute(cudnn(), mem, opt->workspace));
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
        CUDNN_CHECK(data->graph.build(cudnn(), {fe::HeurMode_t::A}));

        int64_t workspace_size;
        CUDNN_CHECK(data->graph.get_workspace_size(workspace_size));
        CUDA_CHECK(alloc_gpu(&data->workspace, workspace_size));

        return data;
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
    SGDOptimizerTask(size_t nb_threads, cudnnHandle_t cudnn_handle,
                     cublasHandle_t cublas_handle,
                     std::vector<LayerUpdateData> const &update_graphs)
        : SGDOptimizerTask(nb_threads, cudnn_handle, cublas_handle) {
        this->update_graphs_ = update_graphs;
    }

  private:
    std::vector<LayerUpdateData> update_graphs_ = {};
};

#endif
