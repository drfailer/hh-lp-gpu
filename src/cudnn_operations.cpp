#include "cudnn_operations.hpp"
#include <cudnn.h>
#include <cudnn_frontend.h>
#include <cudnn_graph.h>
#include "utils.hpp"

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

  // execute
  CUDNN_CHECK(graph.execute(handle, memory_map, workspace));

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

  CUDNN_CHECK(graph.execute(handle, memory_map, workspace));

  cudaMemcpy(C_host, C_gpu, m * k * sizeof(*C_gpu), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  cudaFree(workspace);
  cudnnDestroy(handle);
  cudaFree(A_gpu);
  cudaFree(B_gpu);
  cudaFree(C_gpu);
}
