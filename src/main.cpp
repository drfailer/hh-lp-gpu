#include <cstdint>
#include <cstring>
#include <cudnn.h>
#include <cudnn_frontend.h>
#include <cudnn_graph.h>
#include <iostream>

#define CUDNN_CHECK(expr)                                                      \
  {                                                                            \
    auto result = expr;                                                        \
    if (result.is_bad()) {                                                     \
      std::cerr << "[ERROR]: " __FILE__ ":" << __LINE__ << ": "                \
                << result.get_message() << std::endl;                          \
    }                                                                          \
  }

using ftype = float;

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

  assert(cudaSuccess == cudaMemcpy(A_gpu, A_host, m * n * sizeof(*A_gpu), cudaMemcpyHostToDevice));
  assert(cudaSuccess == cudaMemcpy(B_gpu, B_host, n * k * sizeof(*B_gpu), cudaMemcpyHostToDevice));
  cudaDeviceSynchronize();

  fe::graph::Graph graph;

  auto A = graph.tensor(fe::graph::Tensor_attributes()
                            .set_name("A")
                            .set_dim({m, n, 1})
                            .set_stride({n, 1, 1})
                            .set_data_type(fe::DataType_t::FLOAT));
  auto B = graph.tensor(fe::graph::Tensor_attributes()
                            .set_name("B")
                            .set_dim({n, k, 1})
                            .set_stride({k, 1, 1})
                            .set_data_type(fe::DataType_t::FLOAT));

  /* auto C = graph.matmul( */
  /*     A, B, */
  /*     fe::graph::Matmul_attributes().set_name("matmul").set_compute_data_type( */
  /*         fe::DataType_t::FLOAT)); */
  auto C = graph.pointwise(A, B,
                           fe::graph::Pointwise_attributes()
                               .set_name("hadamard")
                               .set_mode(fe::PointwiseMode_t::MUL)
                               .set_compute_data_type(fe::DataType_t::FLOAT));
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

  assert(cudaSuccess == cudaMemcpy(C_host, C_gpu, m * k * sizeof(*C_gpu), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  cudaFree(workspace);
  cudnnDestroy(handle);
  cudaFree(A_gpu);
  cudaFree(B_gpu);
  cudaFree(C_gpu);
}

void init(ftype *A, ftype *B, ftype *C, int64_t m, int64_t n, int64_t k) {
  // init A
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      A[i * n + j] = i + j;
    }
  }

  // init B
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t j = 0; j < k; ++j) {
      B[i * k + j] = i * j;
    }
  }

  // init C to zero
  memset(C, 0, m * k * sizeof(*C));
}

void display_matrix(std::string const &name, ftype *matrix, int64_t rows,
                    int64_t cols) {
  std::cout << name << " =" << std::endl;
  for (int64_t i = 0; i < rows; ++i) {
    std::cout << "\t";
    for (int64_t j = 0; j < cols; ++j) {
      std::cout << (float)matrix[i * cols + j] << " ";
    }
    std::cout << std::endl;
  }
}

int main(int, char **) {
  std::cout << "testing cudnn and cublas" << std::endl;
  constexpr int64_t m = 10;
  constexpr int64_t n = 10;
  constexpr int64_t k = 10;
  ftype *A = new ftype[m * n];
  ftype *B = new ftype[n * k];
  ftype *C = new ftype[m * k];

  init(A, B, C, m, n, k);
  display_matrix("A", A, m, n);
  display_matrix("B", B, n, k);
  display_matrix("C", C, m, k);

  matmul(A, B, m, n, k, C);

  display_matrix("C", C, m, k);

  delete[] A;
  delete[] B;
  delete[] C;
  return 0;
}
