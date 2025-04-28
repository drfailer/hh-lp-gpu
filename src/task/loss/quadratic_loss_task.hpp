#ifndef TASK_LOSS_QUADRATIC_LOSS_TASK_H
#define TASK_LOSS_QUADRATIC_LOSS_TASK_H
#include "../../data/loss_bwd_data.hpp"
#include "../../data/loss_fwd_data.hpp"
#include "../../tools/gpu.hpp"
#include "../../types.hpp"
#include "loss_task.hpp"
#include <hedgehog/hedgehog.h>

class QuadraticLossTask : public LossTask {
  public:
    QuadraticLossTask(int64_t size, cudnnHandle_t cudnn_handle,
                      cublasHandle_t cublas_handle)
        : LossTask("QuadraticLossTask", cudnn_handle, cublas_handle) {
        // TODO: make batch size configurable
        create_bwd_graph(size);
    }

    void init(NetworkState<ftype> &state) override {
        CUDA_CHECK(alloc_gpu(&state.loss_output,
                             (size_t)state.layer_states.back().dims.nb_nodes));
    }

    void execute(std::shared_ptr<LossFwdData<ftype>>) override {
        INFO_GRP("QuadraticLossTask FWD", INFO_GRP_LAYER_TASK);
        ERROR("unimplemented");
        exit(1);
        // ftype diff = ground_truth - output;
        // return 0.5 * diff * diff;
    }

    void execute(std::shared_ptr<LossBwdData<ftype>> data) override {
        INFO_GRP("QuadraticLossTask BWD", INFO_GRP_LAYER_TASK);
        // return output - ground_truth;

        MemoryMap mem_map = {
            {bwd.error_tensor, data->states.loss_output},
            {bwd.model_output_tensor, data->input},
            {bwd.ground_truth_tensor, data->ground_truth},
        };
        CUDNN_CHECK(bwd.graph.execute(cudnn(), mem_map, bwd.workspace));
        this->addResult(std::make_shared<BwdData<ftype>>(data->states,
                    data->states.loss_output, data->learning_rate));
    }

    void create_bwd_graph(int64_t size) {
        namespace fe = cudnn_frontend;
        std::vector<int64_t> dims = {1, size, 1}, strides = {size, 1, 1};

        bwd.graph.set_io_data_type(fe::DataType_t::FLOAT)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);

        bwd.model_output_tensor =
            bwd.graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("mode output")
                                 .set_dim(dims)
                                 .set_stride(strides));
        bwd.ground_truth_tensor =
            bwd.graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("ground truth")
                                 .set_dim(dims)
                                 .set_stride(strides));
        bwd.error_tensor = bwd.graph.pointwise(
            bwd.model_output_tensor, bwd.ground_truth_tensor,
            fe::graph::Pointwise_attributes().set_mode(
                fe::PointwiseMode_t::SUB));

        bwd.error_tensor->set_output(true);

        CUDNN_CHECK(bwd.graph.validate());
        CUDNN_CHECK(bwd.graph.build(cudnn(), {fe::HeurMode_t::A}));

        int64_t workspace_size;
        CUDNN_CHECK(bwd.graph.get_workspace_size(workspace_size));
        CUDA_CHECK(alloc_gpu(&bwd.workspace, workspace_size));
    }

  private:
    struct {
        cudnn_frontend::graph::Graph graph;
        tensor_attr_t model_output_tensor;
        tensor_attr_t ground_truth_tensor;
        tensor_attr_t error_tensor;
        ftype *workspace = nullptr;
    } bwd;
};

#endif
