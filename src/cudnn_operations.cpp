#include "cudnn_operations.hpp"
#include "tools/defer.hpp"
#include "tools/gpu.hpp"
#include "tools/timer.hpp"
#include <cudnn.h>
#include <cudnn_frontend.h>
#include <cudnn_graph.h>

/*
 * C[m, n] = A[m, n] 0 B[m, n]
 */
void hadamard(ftype *A_host, ftype *B_host, int64_t m, int64_t n,
              ftype *C_host) {
    namespace fe = cudnn_frontend;
    // gpu pointers
    ftype *A_gpu = nullptr, *B_gpu = nullptr;
    ftype *C_gpu = nullptr;

    // allocate gpu memory
    cudaMalloc((void **)(&A_gpu), m * n * sizeof(*A_gpu));
    cudaMalloc((void **)(&B_gpu), m * n * sizeof(*B_gpu));
    cudaMalloc((void **)(&C_gpu), m * n * sizeof(*C_gpu));

    // copy cpu memory to gpu
    cudaMemcpy(A_gpu, A_host, m * n * sizeof(*A_gpu), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B_host, m * n * sizeof(*B_gpu), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    timer_start(hadamard_graph_creation);
    fe::graph::Graph graph;

    // input tensors (should be at least 3d)
    auto A = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("A")
                              .set_dim({1, m, n})
                              .set_stride({m * n, n, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto B = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("B")
                              .set_dim({1, m, n})
                              .set_stride({m * n, n, 1})
                              .set_data_type(fe::DataType_t::FLOAT));

    // add pointwise operation to the graph
    auto C = graph.pointwise(A, B,
                             fe::graph::Pointwise_attributes()
                                 .set_name("hadamard")
                                 .set_mode(fe::PointwiseMode_t::MUL)
                                 .set_compute_data_type(fe::DataType_t::FLOAT));

    // C should be set as output
    C->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    CUDNN_CHECK(graph.validate());

    // map the tensors attributes to the gpu memory
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *>
        memory_map = {{A, A_gpu}, {B, B_gpu}, {C, C_gpu}};

    // graph configuration
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    CUDNN_CHECK(graph.build_operation_graph(handle));
    CUDNN_CHECK(graph.create_execution_plans({fe::HeurMode_t::A}));
    CUDNN_CHECK(graph.check_support(handle));
    CUDNN_CHECK(
        graph.build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));

    int64_t workspace_size;
    CUDNN_CHECK(graph.get_workspace_size(workspace_size));
    float *workspace = nullptr;
    cudaMalloc((void **)(&workspace), workspace_size * sizeof(*workspace));
    timer_end(hadamard_graph_creation);

    // execute
    timer_start(hadamard_graph_execution);
    CUDNN_CHECK(graph.execute(handle, memory_map, workspace));
    timer_end(hadamard_graph_execution);

    timer_report(hadamard_graph_creation);
    timer_report(hadamard_graph_execution);

    // copy result matrix to cpu
    cudaMemcpy(C_host, C_gpu, m * n * sizeof(*C_gpu), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // free
    cudaFree(workspace);
    cudnnDestroy(handle);
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);
}

/*
 * C[m, k] = A[m, n] * B[n, k]
 */
void matmul(ftype *A_host, ftype *B_host, int64_t m, int64_t n, int64_t k,
            ftype *C_host) {
    namespace fe = cudnn_frontend;

    ftype *A_gpu = nullptr, *B_gpu = nullptr;
    ftype *C_gpu = nullptr;

    cudaMalloc((void **)(&A_gpu), m * n * sizeof(*A_gpu));
    cudaMalloc((void **)(&B_gpu), n * k * sizeof(*B_gpu));
    cudaMalloc((void **)(&C_gpu), m * k * sizeof(*C_gpu));

    cudaMemcpy(A_gpu, A_host, m * n * sizeof(*A_gpu), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B_host, n * k * sizeof(*B_gpu), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    timer_start(matmul_graph_creation);
    fe::graph::Graph graph;

    auto A = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("A")
                              .set_dim({1, m, n})
                              .set_stride({m * n, n, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto B = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("B")
                              .set_dim({1, n, k})
                              .set_stride({n * k, k, 1})
                              .set_data_type(fe::DataType_t::FLOAT));

    auto C = graph.matmul(
        A, B,
        fe::graph::Matmul_attributes().set_name("matmul").set_compute_data_type(
            fe::DataType_t::FLOAT));
    C->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    CUDNN_CHECK(graph.validate());

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *>
        memory_map = {{A, A_gpu}, {B, B_gpu}, {C, C_gpu}};

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    CUDNN_CHECK(graph.build_operation_graph(handle));
    CUDNN_CHECK(graph.create_execution_plans({fe::HeurMode_t::A}));
    CUDNN_CHECK(graph.check_support(handle));
    CUDNN_CHECK(
        graph.build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));

    int64_t workspace_size;
    CUDNN_CHECK(graph.get_workspace_size(workspace_size));
    float *workspace = nullptr;
    cudaMalloc((void **)(&workspace), workspace_size * sizeof(*workspace));
    timer_end(matmul_graph_creation);

    timer_start(matmul_graph_execution);
    CUDNN_CHECK(graph.execute(handle, memory_map, workspace));
    timer_end(matmul_graph_execution);

    timer_report(matmul_graph_creation);
    timer_report(matmul_graph_execution);

    cudaMemcpy(C_host, C_gpu, m * k * sizeof(*C_gpu), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(workspace);
    cudnnDestroy(handle);
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);
}

void sigmoid(ftype *A_host, int64_t size) {
    namespace fe = cudnn_frontend;
    ftype *A_gpu = nullptr;
    cudnnHandle_t handle;

    timer_start(sigmoid_graph_creation);
    fe::graph::Graph graph;

    CUDA_CHECK(alloc_gpu(&A_gpu, size));
    defer(cudaFree(A_gpu));
    CUDA_CHECK(memcpy_host_to_gpu(A_gpu, A_host, size));
    cudaDeviceSynchronize();

    cudnnCreate(&handle);
    defer(cudnnDestroy(handle));

    auto input_tensor = graph.tensor(fe::graph::Tensor_attributes()
                                         .set_name("input")
                                         .set_dim({1, size, 1})
                                         .set_stride({size, 1, 1})
                                         .set_data_type(fe::DataType_t::FLOAT));
    auto output_tensor = graph.pointwise(
        input_tensor, fe::graph::Pointwise_attributes()
                          .set_name("sigmoid")
                          .set_mode(fe::PointwiseMode_t::SIGMOID_FWD)
                          .set_compute_data_type(fe::DataType_t::FLOAT));
    output_tensor->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    CUDNN_CHECK(graph.validate());
    CUDNN_CHECK(graph.build_operation_graph(handle));
    CUDNN_CHECK(graph.create_execution_plans({fe::HeurMode_t::A}));
    CUDNN_CHECK(graph.check_support(handle));
    CUDNN_CHECK(
        graph.build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));

    int64_t workspace_size;
    ftype *workspace = nullptr;
    CUDNN_CHECK(graph.get_workspace_size(workspace_size));
    CUDA_CHECK(alloc_gpu(&workspace, workspace_size));
    defer(cudaFree(workspace))
    timer_end(sigmoid_graph_creation);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *>
        memory_map = {{input_tensor, A_gpu}, {output_tensor, A_gpu}};

    timer_start(sigmoid_graph_execution);
    CUDNN_CHECK(graph.execute(handle, memory_map, workspace));
    timer_end(sigmoid_graph_execution);

    timer_report(sigmoid_graph_creation);
    timer_report(sigmoid_graph_execution);

    CUDA_CHECK(memcpy_gpu_to_host(A_host, A_gpu, size));
    cudaDeviceSynchronize();
}
