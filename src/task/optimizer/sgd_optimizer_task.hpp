#ifndef TASK_OPTIMIZER_SGD_OPTIMIZER_TASK_H
#define TASK_OPTIMIZER_SGD_OPTIMIZER_TASK_H
#include "optimizer_task.hpp"

/* We need an optimizer state
 *
 * EXTERIOR -[TrainingData]->  PipelineState
 * EXTERIOR -[InferenceData]-> PipelineState
 *
 * PipelineState -[FwdData]->     LayerTask
 * LayerTask     -[FwdData]->     PipelineState
 * PipelineState -[LossBwdData]-> LossTask
 * LossTask      -[LossBwdData]-> PipelieState
 * PipelineState -[BwdData]->     LayerTask
 * LayerTask     -[BwdData]->     PipelineState
 *
 * PipelineState  -[OptData]->           OptimizerState
 * OptimizerState -[OptLayerData(idx)]-> OptLayerTask<NB_LAYER>
 * OptLayerTask   -[OptLayerData(idx)]-> OptmizerSTate
 * OptimizerState -[OptData]->           PipelineState
 *
 * The optimizer task should be parallelized on the CPU so the gradiants are
 * changed in parallel and the update is done in parallel as well.
 *
 * The UpdateData and the corresponding funcitons in the layers should be
 * removed as everything will be done in the optmizer task.
 *
 */

class SGDOptimizerTask : public OptimizerTask {
  public:
    struct LayerUpdateData {
        cudnn_frontend::graph::Graph graph;
        ftype *workspace = nullptr;
        tensor_attr_t tensor;
        tensor_attr_t gradiant_tensor;
        tensor_attr_t scaled_gradiant_tensor;
        tensor_attr_t result_tensor;
        tensor_attr_t scale_tensor;
    };

  public:
    SGDOptimizerTask(size_t nb_threads, cudnnHandle_t cudnn_handle,
                     cublasHandle_t cublas_handle)
        : OptimizerTask("SGD Optimizer", nb_threads, cudnn_handle,
                        cublas_handle),
          kernel_cache_(std::make_shared<cudnn_frontend::KernelCache>()) {}

#warning "we need an init funciton that create NB_LAYERS graphs so we don't have to rebuild the graphs every iteration"

    void execute(std::shared_ptr<OptLayerData<ftype>> data) override {
        INFO_GRP("OptimizerTask", INFO_GRP_LAYER_TASK);
        if (!has_params(data->state)) {
            this->addResult(data);
            return;
        }
        auto dims = data->state.dims;

#warning "Generalize dims and strides: changing the Parameters type and adding the dims and the strides here would be a good idea"

        // TODO: this dims and strides are valid only for the linear layer, this
        // code should be generatlized. Note: changing the Parameters type and
        // adding the dims and the strides here would be a good idea.
        optimize(data->state.params.biases, data->state.grads.biases,
                 data->learning_rate, {1, dims.nb_nodes, dims.nb_inputs},
                 {dims.nb_nodes * dims.nb_inputs, dims.nb_inputs, 1});

        // update biases if requried
        if (has_biases(data->state)) {
            optimize(data->state.params.biases, data->state.grads.biases,
                     data->learning_rate, {1, dims.nb_nodes, 1},
                     {dims.nb_nodes, 1, 1});
        }
        this->addResult(data);
    }

    std::shared_ptr<hh::AbstractTask<OptimizerTaskIO>> copy() override {
        return std::make_shared<SGDOptimizerTask>(this->numberThreads(),
                                                  cudnn(), cublas());
    }

    void optimize(ftype *parameter, ftype *gradiant, ftype learning_rate,
                  std::vector<int64_t> const &dims,
                  std::vector<int64_t> const &strides) {
        auto opt = create_update_graph(dims, strides);
        MemoryMap mem = {
            {opt.scale_tensor, &learning_rate},
            {opt.tensor, parameter},
            {opt.gradiant_tensor, gradiant},
            {opt.scaled_gradiant_tensor, gradiant},
            {opt.result_tensor, parameter},
        };

        CUDNN_CHECK(opt.graph.execute(cudnn(), mem, opt.workspace));
    }

    LayerUpdateData create_update_graph(std::vector<int64_t> const &dims,
                                        std::vector<int64_t> const &strides) {
        namespace fe = cudnn_frontend;
        LayerUpdateData data;
        // do not work :( v
        // data.graph.set_dynamic_shape_enabled(true).set_kernel_cache(
        //     kernel_cache_);
        data.graph.set_io_data_type(fe::DataType_t::FLOAT)
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

        data.tensor = data.graph.tensor(param_attributes);
        data.gradiant_tensor = data.graph.tensor(grad_attributes);
        data.scale_tensor = data.graph.tensor(learning_rate_attributes);

        // operations:

        // scaled_gradiant = gradiants * learning_rate
        data.scaled_gradiant_tensor = data.graph.pointwise(
            data.scale_tensor, data.gradiant_tensor, scale_attributes);
        // parameters -= scaled_gradiant
        data.result_tensor = data.graph.pointwise(
            data.tensor, data.scaled_gradiant_tensor, substract_attributes);

        // result:

        data.result_tensor->set_output(true);

        CUDNN_CHECK(data.graph.validate());
        CUDNN_CHECK(data.graph.build(cudnn(), {fe::HeurMode_t::A}));

        int64_t workspace_size;
        CUDNN_CHECK(data.graph.get_workspace_size(workspace_size));
        CUDA_CHECK(alloc_gpu(&data.workspace, workspace_size));

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
    std::shared_ptr<cudnn_frontend::KernelCache> kernel_cache_ = nullptr;
};

#endif
