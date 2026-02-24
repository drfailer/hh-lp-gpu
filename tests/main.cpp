#include "layers.hpp"
#include "cublas_function.hpp"
#include "../src/tools/defer.hpp"
#include "../src/tools/communicator.hpp"

// globals handls for the tests
cudnnHandle_t CUDNN_HANDLE;
cublasHandle_t CUBLAS_HANDLE;


int main(int argc, char **argv) {
    cudnnCreate(&CUDNN_HANDLE);
    defer(cudnnDestroy(CUDNN_HANDLE));
    cublasCreate_v2(&CUBLAS_HANDLE);
    defer(cublasDestroy_v2(CUBLAS_HANDLE));

    MPIService service(&argc, &argv, false);

    utest_start();

    // urun_test(matvecmul_n);
    // urun_test(matvecmul_t);
    // urun_test(matvecmul_batch_n);
    // urun_test(matmul_n_n);
    // urun_test(matmul_t_n);
    // urun_test(matmul_n_t);
    // urun_test(matmul_t_t);
    // urun_test(matmul_batch_n_n);

    urun_test(linear_layer_fwd);
    urun_test(linear_layer_bwd);
    urun_test(linear_layer_fwd_batched);
    urun_test(linear_layer_bwd_batched);
    urun_test(sigmoid_activation_fwd);
    urun_test(sigmoid_activation_bwd);
    urun_test(sgd_optimizer);
    // online inference and training
    urun_test(inference);
    urun_test(training);
    // mnist
    urun_test(mnist);
    urun_test(mnist_batched);

    // urun_test_args(mnist_multi_node, &service);

    utest_end();
    return UTEST_STATUS;
}
