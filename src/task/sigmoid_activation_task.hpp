#ifndef TASK_SIGMOID_ACTIVATION_TASK_H
#define TASK_SIGMOID_ACTIVATION_TASK_H
#include "../tools/gpu.hpp"
#include "layer_task.hpp"
#include <log.h/log.h>

struct SigmoidActivationTask : LayerTask {
    SigmoidActivationTask(std::string const &name, cudnnHandle_t cudnn_handle,
                          cublasHandle_t cublas_handle,
                          LayerDimentions const &dims)
        : LayerTask(name, cudnn_handle, cublas_handle, -1) {
        build_fwd_graph(dims);
        build_bwd_graph(dims);
        CUDA_CHECK(alloc_gpu(&fwd_output_gpu_, dims.nb_inputs));
        CUDA_CHECK(alloc_gpu(&bwd_output_gpu_, dims.nb_inputs));
    }
    SigmoidActivationTask(cudnnHandle_t cudnn_handle,
                          cublasHandle_t cublas_handle,
                          LayerDimentions const &dims)
        : SigmoidActivationTask("SigmoidActivationTask", cudnn_handle,
                                cublas_handle, dims) {}

    ~SigmoidActivationTask() {
        cudaFree(fwd_.workspace);
        cudaFree(fwd_output_gpu_);
        cudaFree(bwd_output_gpu_);
    }

    void execute(std::shared_ptr<FwdData<ftype>> fwd_data) override {
        namespace fe = cudnn_frontend;
        INFO_GRP("SigmoidActivationTask FWD", INFO_GRP_LAYER_TASK);

        // save the input for the backwards pass
        fwd_input_gpu_ = fwd_data->input_gpu;

        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>,
                           void *>
            memory_map = {{fwd_.input_tensor, fwd_data->input_gpu},
                          {fwd_.output_tensor, fwd_output_gpu_}};
        CUDNN_CHECK(fwd_.graph.execute(cudnn(), memory_map, fwd_.workspace));
        fwd_data->input_gpu = fwd_output_gpu_;
        this->addResult(fwd_data);
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

    void execute(std::shared_ptr<BwdData<ftype>> bwd_data) override {
        namespace fe = cudnn_frontend;
        INFO_GRP("SigmoidActivationTask BWD", INFO_GRP_LAYER_TASK);

        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>,
                           void *>
            memory_map = {
                {bwd_.err_tensor, bwd_data->err_gpu},
                {bwd_.fwd_input_tensor, fwd_input_gpu_},
                {bwd_.output_tensor, bwd_output_gpu_},
            };
        CUDNN_CHECK(bwd_.graph.execute(cudnn(), memory_map, bwd_.workspace));
        bwd_data->err_gpu = bwd_output_gpu_;
        this->addResult(bwd_data);
    }

    void build_bwd_graph(LayerDimentions const &dims) {
        namespace fe = cudnn_frontend;
        auto &graph = bwd_.graph;

        bwd_.err_tensor =
            bwd_.graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("input error")
                                  .set_dim({1, dims.nb_inputs, 1})
                                  .set_stride({dims.nb_inputs, 1, 1})
                                  .set_data_type(fe::DataType_t::FLOAT));
        bwd_.fwd_input_tensor =
            bwd_.graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("fwd input")
                                  .set_dim({1, dims.nb_inputs, 1})
                                  .set_stride({dims.nb_inputs, 1, 1})
                                  .set_data_type(fe::DataType_t::FLOAT));
        bwd_.output_tensor = bwd_.graph.pointwise(
            bwd_.err_tensor, bwd_.fwd_input_tensor,
            fe::graph::Pointwise_attributes()
                .set_name("sigmoid prime")
                .set_mode(fe::PointwiseMode_t::SIGMOID_BWD)
                .set_compute_data_type(fe::DataType_t::FLOAT));
        bwd_.output_tensor->set_output(true).set_data_type(
            fe::DataType_t::FLOAT);

        CUDNN_CHECK(graph.validate());
        CUDNN_CHECK(graph.build_operation_graph(cudnn()));
        CUDNN_CHECK(graph.create_execution_plans({fe::HeurMode_t::A}));
        CUDNN_CHECK(graph.check_support(cudnn()));
        CUDNN_CHECK(graph.build_plans(
            cudnn(), fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));

        int64_t workspace_size;
        CUDNN_CHECK(graph.get_workspace_size(workspace_size));
        CUDA_CHECK(alloc_gpu(&bwd_.workspace, workspace_size));
    }

  private:
    ftype *fwd_input_gpu_ = nullptr;
    ftype *fwd_output_gpu_ = nullptr;
    ftype *bwd_output_gpu_ = nullptr;
    struct {
        cudnn_frontend::graph::Graph graph;
        tensor_attr_t input_tensor;
        tensor_attr_t output_tensor;
        float *workspace;
    } fwd_;
    struct {
        cudnn_frontend::graph::Graph graph;
        tensor_attr_t fwd_input_tensor;
        tensor_attr_t err_tensor;
        tensor_attr_t output_tensor;
        float *workspace;
    } bwd_;
};

#endif
