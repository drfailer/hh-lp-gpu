#ifndef MODEL_LOSS_QUADRATIC_LOSS_H
#define MODEL_LOSS_QUADRATIC_LOSS_H
#include "../../types.hpp"
#include "loss.hpp"
#include <log.h/log.h>
#include "../../tools/gpu.hpp"

class QuadraticLoss : public Loss<ftype> {
  public:
    QuadraticLoss(cudnnHandle_t cudnn_handle) : cudnn_handle_(cudnn_handle) {}

  public:
    void init(int64_t size) override { create_bwd_graph_graph(size); }

    void fwd(ftype *model_output, ftype *ground_truth, ftype *result) override {
        INFO_GRP("QuadraticLossTask FWD", INFO_GRP_LAYER_TASK);
        ERROR("unimplemented");
        exit(1);
        // ftype diff = ground_truth - output;
        // return 0.5 * diff * diff;
    }

    void bwd(ftype *model_output, ftype *ground_truth, ftype *result) override {
        INFO_GRP("QuadraticLossTask BWD", INFO_GRP_LAYER_TASK);
        // return output - ground_truth;

        MemoryMap mem_map = {
            {bwd_graph.error_tensor, result},
            {bwd_graph.model_output_tensor, model_output},
            {bwd_graph.ground_truth_tensor, ground_truth},
        };
        CUDNN_CHECK(bwd_graph.graph.execute(cudnn_handle_, mem_map,
                                            bwd_graph.workspace));
    }

    void create_bwd_graph_graph(int64_t size) {
        namespace fe = cudnn_frontend;
        std::vector<int64_t> dims = {1, size, 1}, strides = {size, 1, 1};

        bwd_graph.graph.set_io_data_type(fe::DataType_t::FLOAT)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);

        bwd_graph.model_output_tensor =
            bwd_graph.graph.tensor(fe::graph::Tensor_attributes()
                                       .set_name("mode output")
                                       .set_dim(dims)
                                       .set_stride(strides));
        bwd_graph.ground_truth_tensor =
            bwd_graph.graph.tensor(fe::graph::Tensor_attributes()
                                       .set_name("ground truth")
                                       .set_dim(dims)
                                       .set_stride(strides));
        bwd_graph.error_tensor = bwd_graph.graph.pointwise(
            bwd_graph.model_output_tensor, bwd_graph.ground_truth_tensor,
            fe::graph::Pointwise_attributes().set_mode(
                fe::PointwiseMode_t::SUB));

        bwd_graph.error_tensor->set_output(true);

        CUDNN_CHECK(bwd_graph.graph.validate());
        CUDNN_CHECK(bwd_graph.graph.build(cudnn_handle_, {fe::HeurMode_t::A}));

        int64_t workspace_size;
        CUDNN_CHECK(bwd_graph.graph.get_workspace_size(workspace_size));
        CUDA_CHECK(alloc_gpu(&bwd_graph.workspace, workspace_size));
    }

  private:
    struct {
        cudnn_frontend::graph::Graph graph;
        tensor_attr_t model_output_tensor;
        tensor_attr_t ground_truth_tensor;
        tensor_attr_t error_tensor;
        ftype *workspace = nullptr;
    } bwd_graph;
    cudnnHandle_t cudnn_handle_ = nullptr;
};

#endif
