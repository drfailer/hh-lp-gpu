#include <cstdint>
#include <cudnn.h>
#include <cudnn_frontend.h>
#include <iostream>
#include <cstring>

using f8 = int8_t;
using ftype = f8;

/*
 * C[m, k] = A[m, n] * B[n, k]
 */
void matmul(ftype *A, ftype *B, size_t n, size_t m, size_t k, ftype *C) {
  namespace fe = cudnn_frontend;

  f8 *A_gpu = nullptr, *B_gpu = nullptr, *C_gpu = nullptr;

  cudaMalloc((void **)(&A_gpu), m * n * sizeof(*A_gpu));
  cudaMalloc((void **)(&B_gpu), n * k * sizeof(*B_gpu));
  cudaMalloc((void **)(&C_gpu), m * k * sizeof(*B_gpu));

  cudaMemcpy(A_gpu, A, m * n * sizeof(*A_gpu), cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu, B, m * n * sizeof(*B_gpu), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  fe::graph::Graph graph;

  // TODO

  cudaMemcpy(C, C_gpu, m * k * sizeof(*C_gpu), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

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

void display_matrix(std::string const &name, ftype *matrix, size_t rows,
                    size_t cols) {
  std::cout << name << " =" << std::endl;
  for (size_t i = 0; i < rows; ++i) {
    std::cout << "\t";
    for (size_t j = 0; j < cols; ++j) {
      std::cout << (float)matrix[i * cols + j] << " ";
    }
    std::cout << std::endl;
  }
}

int main(int, char **) {
  std::cout << "testing cudnn and cublas" << std::endl;
  constexpr size_t m = 3;
  constexpr size_t n = 3;
  constexpr size_t k = 2;
  ftype *A = new ftype[m * n];
  ftype *B = new ftype[n * k];
  ftype *C = new ftype[m * k];

  init(A, B, C, m, n, k);
  display_matrix("A", A, m, n);
  display_matrix("B", B, n, k);
  display_matrix("C", C, m, k);

  delete[] A;
  delete[] B;
  delete[] C;
  return 0;
}
