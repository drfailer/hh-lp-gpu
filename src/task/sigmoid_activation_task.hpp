#ifndef TASK_SIGMOID_ACTIVATION_TASK_H
#define TASK_SIGMOID_ACTIVATION_TASK_H
#include "../cudnn_operations.hpp"
#include "../tools/gpu.hpp"
#include "layer_task.hpp"
#include <log.h/log.h>

struct SigmoidActivationTask : LayerTask {
    SigmoidActivationTask(std::string const &name, cudnnHandle_t cudnn_handle,
                          cublasHandle_t cublas_handle, size_t layer_idx,
                          LayerDimentions const &dims)
        : LayerTask(name, cudnn_handle, cublas_handle, layer_idx) {
        build_fwd_graph(dims);
        build_bwd_graph(dims);
        CUDA_CHECK(alloc_gpu(&output_gpu_, dims.nb_inputs));
    }
    SigmoidActivationTask(cudnnHandle_t cudnn_handle,
                          cublasHandle_t cublas_handle, size_t layer_idx,
                          LayerDimentions const &dims)
        : SigmoidActivationTask("SigmoidActivationTask", cudnn_handle,
                                cublas_handle, layer_idx, dims) {}

    ~SigmoidActivationTask() {
        cudaFree(fwd_.workspace);
        cudaFree(output_gpu_);
    }

    void execute(std::shared_ptr<FwdData<ftype>> fwd_data) override {
        namespace fe = cudnn_frontend;
        INFO_GRP("SigmoidActivationTask FWD", INFO_GRP_LAYER_TASK);

        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>,
                           void *>
            memory_map = {{fwd_.input_tensor, fwd_data->input_gpu},
                          {fwd_.output_tensor, output_gpu_}};
        CUDNN_CHECK(fwd_.graph.execute(cudnn(), memory_map, fwd_.workspace));
        fwd_data->input_gpu = output_gpu_;
        this->addResult(fwd_data);
    }

    void execute(std::shared_ptr<BwdData<ftype>> bwd_data) override {
        // TODO: execute the bwd graph
    }

    void build_fwd_graph(LayerDimentions const &dims) {
        namespace fe = cudnn_frontend;
        auto &graph = fwd_.graph;

        fwd_.input_tensor =
            fwd_.graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("input")
                                  .set_dim({1, dims.nb_inputs, 1})
                                  .set_stride({dims.nb_inputs, 1, 1})
                                  .set_data_type(fe::DataType_t::FLOAT));
        fwd_.output_tensor = fwd_.graph.pointwise(
            fwd_.input_tensor,
            fe::graph::Pointwise_attributes()
                .set_name("sigmoid")
                .set_mode(fe::PointwiseMode_t::SIGMOID_FWD)
                .set_compute_data_type(fe::DataType_t::FLOAT));
        fwd_.output_tensor->set_output(true).set_data_type(
            fe::DataType_t::FLOAT);

        CUDNN_CHECK(graph.validate());
        CUDNN_CHECK(graph.build_operation_graph(cudnn()));
        CUDNN_CHECK(graph.create_execution_plans({fe::HeurMode_t::A}));
        CUDNN_CHECK(graph.check_support(cudnn()));
        CUDNN_CHECK(graph.build_plans(
            cudnn(), fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));

        int64_t workspace_size;
        CUDNN_CHECK(graph.get_workspace_size(workspace_size));
        CUDA_CHECK(alloc_gpu(&fwd_.workspace, workspace_size));
    }

    void build_bwd_graph(LayerDimentions const &dims) {
        // TODO
    }

  private:
    ftype *output_gpu_ = nullptr;
    struct {
        cudnn_frontend::graph::Graph graph;
        tensor_attr_t input_tensor;
        tensor_attr_t output_tensor;
        float *workspace;
    } fwd_;
    cudnn_frontend::graph::Graph bwd_graph_;
};

#endif
