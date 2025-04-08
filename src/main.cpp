#include <cstdint>
#include <cstring>
#include <iostream>
#include "cudnn_operations.hpp"

// design:
//
// REQUIREMENTS:
// - dynamically load a model from a file.
// - functions should be defined in the task and not in the layers.
// - only gpu layers.
//
// inheritance of layers VS type attribute VS ~type template attribute~
// ~methods~ VS external functions
//
// type template attribute forces to use a tuple to store the layers in the
// model.
//
// External functions in tasks is better because easier to change. A lambda task
// will be useful here.
//
// inheritance would work better with HH, allowing to override methods of the
// fwd and bwd tasks
//
// ~~~BAD
// template <typename ...Layers>
// struct Model {
//   std::tuple<Layers...> layers;
// };
// struct MyLayerType : Layer { custom layer };
// struct MyLayerTypeTask : LayerTask<LayerType> {
//   void execute(LayerTaskData<Forward>);
//   void execute(LayerTaskData<Backward>);
// };
// ~~~BAD
//
// Defining a model entierly at compile time disable the possibility to load an
// unknown model from a file. Therefore, we should use a dynamic type.
//
// DATA:
//
// layer:
// - weights_gpu, weights_cpu
// - biases_gpu, biases_cpu
//
// model:
// - layers
//
// TASK:
//
// - Activation, ActivationDerivative
// - Loss, LossDerivative
// - Optimization
//
// - ForwardPassState
// - BackwardPassState

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
  constexpr int64_t m = 3;
  constexpr int64_t n = 3;
  constexpr int64_t k = 3;
  ftype *A = new ftype[m * n];
  ftype *B = new ftype[n * k];
  ftype *C = new ftype[m * k];

  init(A, B, C, m, n, k);
  display_matrix("A", A, m, n);
  display_matrix("B", B, n, k);
  display_matrix("C", C, m, k);

  if (m == n && n == k) hadamard(A, B, m, n, C);
  display_matrix("C", C, m, k);

  matmul(A, B, m, n, k, C);
  display_matrix("C", C, m, k);

  delete[] A;
  delete[] B;
  delete[] C;
  return 0;
}
