#include "../tools/mnist/minist_loader.hpp"
#include "cudnn_operations.hpp"
#include "data/layer_state.hpp"
#include "graph/network_graph.hpp"
#include "layers/linear_layer.hpp"
#include "layers/sigmoid_activation_layer.hpp"
#include "loss/quadratic_loss.hpp"
#include "optimizers/sgd_optimizer.hpp"
#include "tools/defer.hpp"
#include "tools/gpu.hpp"
#include "tools/utest.hpp"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <ostream>
#include <random>
#include <unistd.h>

// globals handls for the tests
cudnnHandle_t CUDNN_HANDLE;
cublasHandle_t CUBLAS_HANDLE;

ftype sigmoid(ftype x) { return 1.0 / (1.0 + std::exp(-x)); }

ftype sigmoid_derivative(ftype x) { return sigmoid(x) * (1.0 - sigmoid(x)); }

std::pair<LayerState<ftype>, LayerState<ftype>>
create_test_layer(LayerDimentions const &dims) {
    Parameters<ftype> params_gpu = parameters_create_gpu<ftype>(dims);
    Parameters<ftype> grads_gpu = parameters_create_gpu<ftype>(dims);
    LayerState<ftype> gpu =
        layer_state_create_gpu<ftype>(dims, params_gpu, grads_gpu);
    Parameters<ftype> params_host = parameters_create_host<ftype>(dims);
    Parameters<ftype> grads_host = parameters_create_host<ftype>(dims);
    LayerState<ftype> host =
        layer_state_create_host<ftype>(dims, params_host, grads_host);
    return {host, gpu};
}

Parameters<ftype> create_test_parameters_gpu(LayerDimentions const &dims,
                                             ftype value) {
    Parameters<ftype> gpu = parameters_create_gpu<ftype>(dims);
    Parameters<ftype> host = parameters_create_host<ftype>(dims);
    size_t size = dims.kernel_size * dims.nb_nodes * dims.nb_nodes;
    for (size_t i = 0; i < size; ++i) {
        host.weights[i] = value;
    }
    for (size_t i = 0; i < (size_t)dims.nb_inputs; ++i) {
        host.biases[i] = value;
    }
    parameters_host_to_gpu(gpu, host, dims);
    parameters_destroy_host(host);
    return gpu;
}

Parameters<ftype> init_test_parameters_gpu(LayerDimentions const &dims,
                                           Parameters<ftype> &gpu) {
    Parameters<ftype> host = parameters_create_host<ftype>(dims);
    for (int64_t i = 0; i < dims.nb_nodes; ++i) {
        for (int64_t j = 0; j < dims.nb_inputs; ++j) {
            host.weights[i * dims.nb_inputs + j] = i + j + 1;
            std::cout << host.weights[i * dims.nb_inputs + j] << " ";
        }
        std::cout << std::endl;
    }
    for (int64_t i = 0; i < dims.nb_nodes; ++i) {
        host.biases[i] = i + 1;
    }
    parameters_host_to_gpu(gpu, host, dims);
    parameters_destroy_host(host);
    return gpu;
}

Parameters<ftype> create_test_parameters_gpu(LayerDimentions const &dims) {
    Parameters<ftype> gpu = parameters_create_gpu<ftype>(dims);
    init_test_parameters_gpu(dims, gpu);
    return gpu;
}

ftype *create_test_gpu_array(ftype *host, size_t size) {
    ftype *gpu = nullptr;
    CUDA_CHECK(alloc_gpu(&gpu, size));
    CUDA_CHECK(memcpy_host_to_gpu(gpu, host, size));
    cudaDeviceSynchronize();
    return gpu;
}

void init(ftype *A, ftype *B, ftype *C, ftype *V, int64_t m, int64_t n,
          int64_t k) {
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

    // init V
    for (int64_t i = 0; i < m; ++i) {
        V[i] = i;
    }
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

int mnist_get_label(ftype *arr) {
    size_t imax = 0;

    for (size_t i = 1; i < 10; ++i) {
        if (arr[i] > arr[imax]) {
            imax = i;
        }
    }
    return imax;
}

UTest(cdnn_operations) {
    constexpr int64_t m = 3;
    constexpr int64_t n = 3;
    constexpr int64_t k = 3;
    ftype *A = new ftype[m * n];
    ftype *B = new ftype[n * k];
    ftype *C = new ftype[m * k];
    ftype *V = new ftype[m];

    std::cout << "perform matrix multiplication: C[m, k] = A[m, n] * B[n, k]"
              << std::endl;
    std::cout << "m = " << m << std::endl;
    std::cout << "n = " << n << std::endl;
    std::cout << "k = " << k << std::endl;

    init(A, B, C, V, m, n, k);
    display_matrix("A", A, m, n);
    display_matrix("B", B, n, k);
    display_matrix("C", C, m, k);

    if (m == n && n == k)
        hadamard(A, B, m, n, C);
    display_matrix("C", C, m, k);

    matmul(A, B, m, n, k, C);
    display_matrix("C", C, m, k);

    sigmoid(V, m);
    display_matrix("V", V, m, 1);

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] V;
}

UTest(matvecmul_n) {
    constexpr size_t m = 10;
    constexpr size_t n = 10'000;
    ftype *mat_host = nullptr, *vec_host = nullptr, *gt_host = nullptr,
          *out_host = nullptr;
    ftype *mat_gpu = nullptr, *vec_gpu = nullptr, *out_gpu = nullptr;
    std::mt19937 rand(0);
    std::uniform_real_distribution<ftype> dist(0, 1);

    // allocate on host
    mat_host = new ftype[m * n];
    defer(delete[] mat_host);
    vec_host = new ftype[n];
    defer(delete[] vec_host);
    gt_host = new ftype[m];
    defer(delete[] gt_host);
    out_host = new ftype[m];
    defer(delete[] out_host);

    // allocate on gpu
    CUDA_CHECK(alloc_gpu(&mat_gpu, m * n));
    defer(cudaFree(mat_gpu));
    CUDA_CHECK(alloc_gpu(&vec_gpu, n));
    defer(cudaFree(vec_gpu));
    CUDA_CHECK(alloc_gpu(&out_gpu, m));
    defer(cudaFree(out_gpu));

    // init
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            mat_host[i * n + j] = dist(rand);
        }
    }
    for (size_t i = 0; i < n; ++i) {
        vec_host[i] = dist(rand);
    }
    CUDA_CHECK(memcpy_host_to_gpu(mat_gpu, mat_host, m * n));
    CUDA_CHECK(memcpy_host_to_gpu(vec_gpu, vec_host, n));
    cudaDeviceSynchronize();

    // matrix vector multiplication on host
    for (size_t i = 0; i < m; ++i) {
        ftype sum = 0;
        for (size_t j = 0; j < n; ++j) {
            sum += mat_host[i * n + j] * vec_host[j];
        }
        gt_host[i] = sum;
    }

    // matrix vector multiplication on the gpu
    CUBLAS_CHECK(matvecmul(CUBLAS_HANDLE, false, m, n, 1.f, mat_gpu, vec_gpu,
                           0.f, out_gpu));
    CUDA_CHECK(memcpy_gpu_to_host(out_host, out_gpu, m));
    cudaDeviceSynchronize();

    // verify the result
    for (size_t i = 0; i < m; ++i) {
        uassert_float_equal(gt_host[i], out_host[i], 1e-2);
    }
}

UTest(matvecmul_t) {
    constexpr size_t m = 10;
    constexpr size_t n = 10'000;
    ftype *mat_host = nullptr, *vec_host = nullptr, *gt_host = nullptr,
          *out_host = nullptr;
    ftype *mat_gpu = nullptr, *vec_gpu = nullptr, *out_gpu = nullptr;
    std::mt19937 rand(0);
    std::uniform_real_distribution<ftype> dist(0, 1);

    // allocate on host
    mat_host = new ftype[m * n];
    defer(delete[] mat_host);
    vec_host = new ftype[m];
    defer(delete[] vec_host);
    gt_host = new ftype[n];
    defer(delete[] gt_host);
    out_host = new ftype[n];
    defer(delete[] out_host);

    // allocate on gpu
    CUDA_CHECK(alloc_gpu(&mat_gpu, m * n));
    defer(cudaFree(mat_gpu));
    CUDA_CHECK(alloc_gpu(&vec_gpu, m));
    defer(cudaFree(vec_gpu));
    CUDA_CHECK(alloc_gpu(&out_gpu, n));
    defer(cudaFree(out_gpu));

    // init
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            mat_host[i * n + j] = dist(rand);
        }
    }
    for (size_t i = 0; i < m; ++i) {
        vec_host[i] = dist(rand);
    }
    CUDA_CHECK(memcpy_host_to_gpu(mat_gpu, mat_host, m * n));
    CUDA_CHECK(memcpy_host_to_gpu(vec_gpu, vec_host, m));
    cudaDeviceSynchronize();

    // matrix vector multiplication on host
    for (size_t i = 0; i < n; ++i) {
        ftype sum = 0;
        for (size_t j = 0; j < m; ++j) {
            sum += mat_host[j * n + i] * vec_host[j];
        }
        gt_host[i] = sum;
    }

    // matrix vector multiplication on the gpu
    CUBLAS_CHECK(matvecmul(CUBLAS_HANDLE, true, m, n, 1.f, mat_gpu, vec_gpu,
                           0.f, out_gpu));
    CUDA_CHECK(memcpy_gpu_to_host(out_host, out_gpu, n));
    cudaDeviceSynchronize();

    // verify the result
    for (size_t i = 0; i < n; ++i) {
        uassert_float_equal(gt_host[i], out_host[i], 1e-2);
    }
}

UTest(matmul_n_n) {
    INFO("matmul n n")
    constexpr size_t m = 10;
    constexpr size_t n = 10'000;
    constexpr size_t k = 100;
    ftype *A_host = nullptr, *B_host = nullptr, *GT_host = nullptr,
          *C_host = nullptr;
    ftype *A_gpu = nullptr, *B_gpu = nullptr, *C_gpu = nullptr;
    std::mt19937 rand(0);
    std::uniform_real_distribution<ftype> dist(0, 1);
    auto init_matrix = [&dist, &rand](ftype *mat, size_t rows, size_t cols) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                mat[i * cols + j] = dist(rand);
            }
        }
    };

    // allocate on host
    A_host = new ftype[m * k];
    defer(delete[] A_host);
    B_host = new ftype[k * n];
    defer(delete[] B_host);
    C_host = new ftype[m * n];
    defer(delete[] C_host);
    GT_host = new ftype[m * n];
    defer(delete[] GT_host);

    // allocate on gpu
    CUDA_CHECK(alloc_gpu(&A_gpu, m * k));
    defer(cudaFree(A_gpu));
    CUDA_CHECK(alloc_gpu(&B_gpu, k * n));
    defer(cudaFree(B_gpu));
    CUDA_CHECK(alloc_gpu(&C_gpu, m * n));
    defer(cudaFree(C_gpu));

    // init
    init_matrix(A_host, m, k);
    init_matrix(B_host, k, n);
    CUDA_CHECK(memcpy_host_to_gpu(A_gpu, A_host, m * k));
    CUDA_CHECK(memcpy_host_to_gpu(B_gpu, B_host, k * n));
    cudaDeviceSynchronize();

    // matrix multiplication on host
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            GT_host[i * n + j] = 0;
            for (size_t e = 0; e < k; ++e) {
                GT_host[i * n + j] += A_host[i * k + e] * B_host[e * n + j];
            }
        }
    }

    // matrix multiplication on the gpu
    matmul(CUBLAS_HANDLE, false, false, m, n, k, (ftype)1, A_gpu, B_gpu,
           (ftype)0, C_gpu);
    CUDA_CHECK(memcpy_gpu_to_host(C_host, C_gpu, m * n));

    // verify the results
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            uassert_float_equal(GT_host[i * n + j], C_host[i * n + j], 1e-1);
        }
    }
}

UTest(matmul_t_n) {
    INFO("matmul t n")
    constexpr size_t m = 10;
    constexpr size_t n = 10'000;
    constexpr size_t k = 100;
    ftype *A_host = nullptr, *B_host = nullptr, *GT_host = nullptr,
          *C_host = nullptr;
    ftype *A_gpu = nullptr, *B_gpu = nullptr, *C_gpu = nullptr;
    std::mt19937 rand(0);
    std::uniform_real_distribution<ftype> dist(0, 1);
    auto init_matrix = [&dist, &rand](ftype *mat, size_t rows, size_t cols) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                mat[i * cols + j] = dist(rand);
            }
        }
    };

    // allocate on host
    A_host = new ftype[k * m];
    defer(delete[] A_host);
    B_host = new ftype[k * n];
    defer(delete[] B_host);
    C_host = new ftype[m * n];
    defer(delete[] C_host);
    GT_host = new ftype[m * n];
    defer(delete[] GT_host);

    // allocate on gpu
    CUDA_CHECK(alloc_gpu(&A_gpu, m * k));
    defer(cudaFree(A_gpu));
    CUDA_CHECK(alloc_gpu(&B_gpu, k * n));
    defer(cudaFree(B_gpu));
    CUDA_CHECK(alloc_gpu(&C_gpu, m * n));
    defer(cudaFree(C_gpu));

    // init
    init_matrix(A_host, k, m);
    init_matrix(B_host, k, n);
    CUDA_CHECK(memcpy_host_to_gpu(A_gpu, A_host, k * m));
    CUDA_CHECK(memcpy_host_to_gpu(B_gpu, B_host, k * n));
    cudaDeviceSynchronize();

    // matrix multiplication on host
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            GT_host[i * n + j] = 0;
            for (size_t e = 0; e < k; ++e) {
                GT_host[i * n + j] += A_host[e * m + i] * B_host[e * n + j];
            }
        }
    }

    // matrix multiplication on the gpu
    matmul(CUBLAS_HANDLE, true, false, m, n, k, (ftype)1, A_gpu, B_gpu,
           (ftype)0, C_gpu);
    CUDA_CHECK(memcpy_gpu_to_host(C_host, C_gpu, m * n));

    // verify the results
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            uassert_float_equal(GT_host[i * n + j], C_host[i * n + j], 1e-1);
        }
    }
}

UTest(matmul_n_t) {
    INFO("matmul n t")
    constexpr size_t m = 10;
    constexpr size_t n = 10'000;
    constexpr size_t k = 100;
    ftype *A_host = nullptr, *B_host = nullptr, *GT_host = nullptr,
          *C_host = nullptr;
    ftype *A_gpu = nullptr, *B_gpu = nullptr, *C_gpu = nullptr;
    std::mt19937 rand(0);
    std::uniform_real_distribution<ftype> dist(0, 1);
    auto init_matrix = [&dist, &rand](ftype *mat, size_t rows, size_t cols) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                mat[i * cols + j] = dist(rand);
            }
        }
    };

    // allocate on host
    A_host = new ftype[m * k];
    defer(delete[] A_host);
    B_host = new ftype[n * k];
    defer(delete[] B_host);
    C_host = new ftype[m * n];
    defer(delete[] C_host);
    GT_host = new ftype[m * n];
    defer(delete[] GT_host);

    // allocate on gpu
    CUDA_CHECK(alloc_gpu(&A_gpu, m * k));
    defer(cudaFree(A_gpu));
    CUDA_CHECK(alloc_gpu(&B_gpu, n * k));
    defer(cudaFree(B_gpu));
    CUDA_CHECK(alloc_gpu(&C_gpu, m * n));
    defer(cudaFree(C_gpu));

    // init
    init_matrix(A_host, m, k);
    init_matrix(B_host, n, k);
    CUDA_CHECK(memcpy_host_to_gpu(A_gpu, A_host, m * k));
    CUDA_CHECK(memcpy_host_to_gpu(B_gpu, B_host, n * k));
    cudaDeviceSynchronize();

    // matrix multiplication on host
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            GT_host[i * n + j] = 0;
            for (size_t e = 0; e < k; ++e) {
                GT_host[i * n + j] += A_host[i * k + e] * B_host[j * k + e];
            }
        }
    }

    // matrix multiplication on the gpu
    matmul(CUBLAS_HANDLE, false, true, m, n, k, (ftype)1, A_gpu, B_gpu,
           (ftype)0, C_gpu);
    CUDA_CHECK(memcpy_gpu_to_host(C_host, C_gpu, m * n));

    // verify the results
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            uassert_float_equal(GT_host[i * n + j], C_host[i * n + j], 1e-1);
        }
    }
}

UTest(matmul_t_t) {
    INFO("matmul t t")
    constexpr size_t m = 10;
    constexpr size_t n = 10'000;
    constexpr size_t k = 100;
    ftype *A_host = nullptr, *B_host = nullptr, *GT_host = nullptr,
          *C_host = nullptr;
    ftype *A_gpu = nullptr, *B_gpu = nullptr, *C_gpu = nullptr;
    std::mt19937 rand(0);
    std::uniform_real_distribution<ftype> dist(0, 1);
    auto init_matrix = [&dist, &rand](ftype *mat, size_t rows, size_t cols) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                mat[i * cols + j] = dist(rand);
            }
        }
    };

    // allocate on host
    A_host = new ftype[k * m];
    defer(delete[] A_host);
    B_host = new ftype[n * k];
    defer(delete[] B_host);
    C_host = new ftype[m * n];
    defer(delete[] C_host);
    GT_host = new ftype[m * n];
    defer(delete[] GT_host);

    // allocate on gpu
    CUDA_CHECK(alloc_gpu(&A_gpu, k * m));
    defer(cudaFree(A_gpu));
    CUDA_CHECK(alloc_gpu(&B_gpu, n * k));
    defer(cudaFree(B_gpu));
    CUDA_CHECK(alloc_gpu(&C_gpu, m * n));
    defer(cudaFree(C_gpu));

    // init
    init_matrix(A_host, k, m);
    init_matrix(B_host, n, k);
    CUDA_CHECK(memcpy_host_to_gpu(A_gpu, A_host, k * m));
    CUDA_CHECK(memcpy_host_to_gpu(B_gpu, B_host, n * k));
    cudaDeviceSynchronize();

    // matrix multiplication on host
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            GT_host[i * n + j] = 0;
            for (size_t e = 0; e < k; ++e) {
                GT_host[i * n + j] += A_host[e * m + i] * B_host[j * k + e];
            }
        }
    }

    // matrix multiplication on the gpu
    matmul(CUBLAS_HANDLE, true, true, m, n, k, (ftype)1, A_gpu, B_gpu, (ftype)0,
           C_gpu);
    CUDA_CHECK(memcpy_gpu_to_host(C_host, C_gpu, m * n));

    // verify the results
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            uassert_float_equal(GT_host[i * n + j], C_host[i * n + j], 1e-1);
        }
    }
}

UTest(linear_layer_fwd) {
    constexpr int64_t nb_nodes = 3;
    constexpr int64_t nb_inputs = 3;
    LayerDimentions dims = {
        .nb_nodes = nb_nodes, .nb_inputs = nb_inputs, .kernel_size = 1};
    ftype input_host[nb_inputs] = {1, 2, 3}, output_host[nb_nodes] = {0};
    ftype *input_gpu = nullptr;

    auto params = create_test_parameters_gpu(dims, 1);
    defer(parameters_destroy_gpu(params));
    LayerState<ftype> layer_state = layer_state_create_gpu(dims, params, {});

    CUDA_CHECK(alloc_gpu(&input_gpu, nb_inputs));
    defer(cudaFree(input_gpu));
    CUDA_CHECK(memcpy_host_to_gpu(input_gpu, input_host, nb_inputs));
    cudaDeviceSynchronize();

    LinearLayer linear_layer(CUBLAS_HANDLE, nb_inputs, nb_nodes);

    ftype *output_gpu = linear_layer.fwd(layer_state, input_gpu);

    urequire(output_gpu == layer_state.output);
    CUDA_CHECK(memcpy_gpu_to_host(output_host, output_gpu, nb_nodes));
    cudaDeviceSynchronize();

    for (size_t i = 0; i < nb_nodes; ++i) {
        uassert_equal(output_host[i], 7);
    }
}

UTest(linear_layer_bwd) {
    constexpr int64_t nb_nodes = 3;
    constexpr int64_t nb_inputs = 4;
    LayerDimentions dims = {
        .nb_nodes = nb_nodes, .nb_inputs = nb_inputs, .kernel_size = 1};
    ftype input_host[nb_inputs] = {1, 2, 3, 4},
          input_err_host[nb_nodes] = {100, 10, 1},
          output_err_host[nb_inputs] = {0};
    ftype *input_gpu = nullptr, *err_gpu = nullptr;

    // init input and output gpu buffers
    input_gpu = create_test_gpu_array(input_host, nb_inputs);
    defer(cudaFree(input_gpu));
    err_gpu = create_test_gpu_array(input_err_host, nb_inputs);
    defer(cudaFree(err_gpu));

    // init the parameters
    Parameters<ftype> params = create_test_parameters_gpu(dims);
    defer(parameters_destroy_gpu(params));
    Parameters<ftype> grads = parameters_create_gpu<ftype>(dims);
    defer(parameters_destroy_gpu(grads));

    LayerState<ftype> layer_state = layer_state_create_gpu(dims, params, grads);
    defer(layer_state_destroy_gpu(layer_state));

    LinearLayer linear_layer(CUBLAS_HANDLE, nb_inputs, nb_nodes);
    linear_layer.fwd(layer_state, input_gpu);
    ftype *output_err_gpu = linear_layer.bwd(layer_state, err_gpu);

    urequire(output_err_gpu == layer_state.error);
    CUDA_CHECK(memcpy_gpu_to_host(output_err_host, output_err_gpu, nb_inputs));
    cudaDeviceSynchronize();

    uassert_equal(output_err_host[0], 123);
    uassert_equal(output_err_host[1], 234);
    uassert_equal(output_err_host[2], 345);
    uassert_equal(output_err_host[3], 456);
}

UTest(sigmoid_activation_fwd) {
    constexpr int64_t nb_nodes = 3;
    constexpr int64_t nb_inputs = 3;
    LayerDimentions dims = {
        .nb_nodes = nb_nodes, .nb_inputs = nb_inputs, .kernel_size = 1};
    ftype input_host[nb_inputs] = {1, 2, 3}, output_host[nb_nodes] = {0};
    ftype *input_gpu = nullptr;
    LayerState<ftype> layer_state = layer_state_create_gpu<ftype>(dims, {}, {});
    defer(layer_state_destroy_gpu(layer_state));
    NetworkState<ftype> network_state({layer_state}, nullptr);

    CUDA_CHECK(alloc_gpu(&input_gpu, nb_inputs));
    defer(cudaFree(input_gpu));
    CUDA_CHECK(memcpy_host_to_gpu(input_gpu, input_host, nb_inputs));
    cudaDeviceSynchronize();

    SigmoidActivationLayer sigmoid_layer(CUDNN_HANDLE, nb_inputs);
    ftype *output_gpu = sigmoid_layer.fwd(layer_state, input_gpu);

    urequire(output_gpu != input_gpu);
    CUDA_CHECK(memcpy_gpu_to_host(output_host, output_gpu, nb_inputs));
    cudaDeviceSynchronize();

    for (size_t i = 0; i < nb_nodes; ++i) {
        uassert_float_equal(output_host[i], sigmoid(input_host[i]), 1e-6);
    }
}

UTest(sigmoid_activation_bwd) {
    constexpr int64_t nb_nodes = 6;
    constexpr int64_t nb_inputs = 6;
    LayerDimentions dims = {
        .nb_nodes = nb_nodes, .nb_inputs = nb_inputs, .kernel_size = 1};
    ftype input_host[nb_inputs] = {1, 2, 3, 4, 5, 6},
          err[nb_inputs] = {10, 10, 10, 10, 10, 10},
          output_host[nb_nodes] = {0};
    ftype *input_gpu = nullptr, *err_gpu = nullptr;
    LayerState<ftype> layer_state = layer_state_create_gpu<ftype>(dims, {}, {});
    defer(layer_state_destroy_gpu(layer_state));
    NetworkState<ftype> network_state({layer_state}, nullptr);

    CUDA_CHECK(alloc_gpu(&input_gpu, nb_inputs));
    CUDA_CHECK(alloc_gpu(&err_gpu, nb_inputs));
    defer(cudaFree(input_gpu));
    defer(cudaFree(err_gpu));
    CUDA_CHECK(memcpy_host_to_gpu(input_gpu, input_host, nb_inputs));
    CUDA_CHECK(memcpy_host_to_gpu(err_gpu, err, nb_inputs));
    cudaDeviceSynchronize();

    SigmoidActivationLayer sigmoid_layer(CUDNN_HANDLE, nb_inputs);
    sigmoid_layer.fwd(layer_state, input_gpu);
    ftype *output_gpu = sigmoid_layer.bwd(layer_state, err_gpu);

    CUDA_CHECK(memcpy_gpu_to_host(output_host, output_gpu, nb_inputs));
    cudaDeviceSynchronize();

    for (size_t i = 0; i < nb_nodes; ++i) {
        uassert_float_equal(output_host[i],
                            err[i] * sigmoid_derivative(input_host[i]), 1e-6);
    }
}

UTest(inference) {
    constexpr size_t nb_nodes = 3;
    constexpr size_t nb_inputs = 3;
    ftype input_host[nb_inputs] = {1, 1, 1}, output_host[nb_nodes] = {0};
    ftype *input_gpu = nullptr;
    NetworkGraph graph;
    NetworkState<ftype> state;

    CUDA_CHECK(alloc_gpu(&input_gpu, nb_inputs));
    defer(cudaFree(input_gpu));
    CUDA_CHECK(memcpy_host_to_gpu(input_gpu, input_host, nb_inputs));
    cudaDeviceSynchronize();

    graph.set_loss<QuadraticLoss>(CUDNN_HANDLE);
    graph.set_optimizer<SGDOptimizer>(1, CUDNN_HANDLE);

    graph.add_layer<LinearLayer>(CUBLAS_HANDLE, nb_inputs, nb_nodes);
    graph.add_layer<SigmoidActivationLayer>(CUDNN_HANDLE, nb_nodes);
    graph.build();

    graph.init_network_state(state);
    defer(graph.destroy_network_state(state));

    init_test_parameters_gpu(state.layer_states[0].dims,
                             state.layer_states[0].params);

    graph.executeGraph(true);
    graph.pushData(std::make_shared<InferenceData<ftype>>(state, input_gpu));
    ftype *output_gpu = graph.get<InferenceData<ftype>>()->input;
    graph.terminate();

    CUDA_CHECK(memcpy_gpu_to_host(output_host, output_gpu, nb_nodes));
    cudaDeviceSynchronize();

    for (size_t i = 0; i < nb_nodes; ++i) {
        ftype weights_input_bias = input_host[0] * (i + 1) +
                                   output_host[1] * (i + 2) +
                                   output_host[2] * (i + 3) + i + 1;
        ftype expected_value = sigmoid(weights_input_bias);
        uassert_float_equal(output_host[i], expected_value, 1e-6);
    }
}

UTest(training) {
    constexpr size_t nb_nodes = 10;
    constexpr size_t nb_inputs = 28 * 28;
    constexpr ftype learning_rate = 0.1;
    constexpr ftype epochs = 1;
    NetworkGraph graph;
    NetworkState<ftype> state;
    MNISTLoader loader;

    DataSet<ftype> data_set =
        loader.load_ds("../data/mnist/train-labels-idx1-ubyte",
                       "../data/mnist/train-images-idx3-ubyte");
    defer(destroy_data_set(data_set));

    urequire(data_set.datas.size() == 60'000);

    graph.set_loss<QuadraticLoss>(CUDNN_HANDLE);
    graph.set_optimizer<SGDOptimizer>(2, CUDNN_HANDLE);

    graph.add_layer<LinearLayer>(CUBLAS_HANDLE, nb_inputs, 32);
    graph.add_layer<SigmoidActivationLayer>(CUDNN_HANDLE, 32);
    graph.cut_layer();
    graph.add_layer<LinearLayer>(CUBLAS_HANDLE, 32, 32);
    graph.add_layer<SigmoidActivationLayer>(CUDNN_HANDLE, 32);
    graph.cut_layer();
    graph.add_layer<LinearLayer>(CUBLAS_HANDLE, 32, 10);
    graph.add_layer<SigmoidActivationLayer>(CUDNN_HANDLE, nb_nodes);
    graph.build();

    graph.init_network_state(state);
    defer(graph.destroy_network_state(state));

    graph.executeGraph(true);
    graph.pushData(std::make_shared<TrainingData<ftype>>(
        state, data_set, learning_rate, epochs));
    graph.terminate();

    graph.createDotFile("train.dot", hh::ColorScheme::EXECUTION,
                        hh::StructureOptions::QUEUE);
}

UTest(evaluate_mnist) {
    constexpr ftype learning_rate = 0.01;
    constexpr size_t epochs = 2;
    MNISTLoader loader;

    DataSet<ftype> training_set =
        loader.load_ds("../data/mnist/train-labels-idx1-ubyte",
                       "../data/mnist/train-images-idx3-ubyte");
    defer(destroy_data_set(training_set));
    DataSet<ftype> testing_set =
        loader.load_ds("../data/mnist/t10k-labels-idx1-ubyte",
                       "../data/mnist/t10k-images-idx3-ubyte");
    defer(destroy_data_set(testing_set));

    NetworkGraph graph;

    graph.set_loss<QuadraticLoss>(CUDNN_HANDLE);
    graph.set_optimizer<SGDOptimizer>(1, CUDNN_HANDLE);

    graph.add_layer<LinearLayer>(CUBLAS_HANDLE, 28 * 28, 10);
    graph.add_layer<SigmoidActivationLayer>(CUDNN_HANDLE, 10);

    graph.build();

    NetworkState<ftype> network;
    graph.init_network_state(network);
    defer(graph.destroy_network_state(network));

    graph.executeGraph(true);

    INFO("Inference before training...");

    std::vector<ftype> expected(10), found(10);
    size_t success = 0, errors = 0;
    for (auto data : testing_set.datas) {
        graph.pushData(
            std::make_shared<InferenceData<ftype>>(network, data.input));
        auto *output = graph.get<InferenceData<ftype>>()->input;
        graph.cleanGraph();
        CUDA_CHECK(memcpy_gpu_to_host(expected.data(), data.ground_truth, 10));
        CUDA_CHECK(memcpy_gpu_to_host(found.data(), output, 10));
        int expected_label = mnist_get_label(expected.data());
        int found_label = mnist_get_label(found.data());

        if (found_label == expected_label) {
            ++success;
        } else {
            ++errors;
        }
    }
    ftype accuracy_start = (ftype)success / (ftype)testing_set.datas.size();
    std::cout << "accuracy: " << accuracy_start << std::endl;
    std::cout << "success: " << success << ", errors: " << errors << std::endl;

    INFO("start training (learning_rate = " << learning_rate
                                            << ", epochs = " << epochs << ")");
    graph.pushData(std::make_shared<TrainingData<ftype>>(
        network, training_set, learning_rate, epochs));
    (void)graph.get<TrainingData<ftype>>();
    graph.cleanGraph();

    success = 0;
    errors = 0;
    for (auto data : testing_set.datas) {
        graph.pushData(
            std::make_shared<InferenceData<ftype>>(network, data.input));
        auto *output = graph.get<InferenceData<ftype>>()->input;
        graph.cleanGraph();
        CUDA_CHECK(memcpy_gpu_to_host(expected.data(), data.ground_truth, 10));
        CUDA_CHECK(memcpy_gpu_to_host(found.data(), output, 10));
        int expected_label = mnist_get_label(expected.data());
        int found_label = mnist_get_label(found.data());

        if (found_label == expected_label) {
            ++success;
        } else {
            ++errors;
        }
    }
    graph.terminate();

    ftype accuracy_end = (ftype)success / (ftype)testing_set.datas.size();
    std::cout << "accuracy: " << accuracy_end << std::endl;
    std::cout << "success: " << success << ", errors: " << errors << std::endl;

    uassert(accuracy_end > 10 * accuracy_start);

    graph.createDotFile("train_mnist.dot", hh::ColorScheme::EXECUTION,
                        hh::StructureOptions::QUEUE);
}

int main(int, char **) {
    cudnnCreate(&CUDNN_HANDLE);
    defer(cudnnDestroy(CUDNN_HANDLE));
    cublasCreate_v2(&CUBLAS_HANDLE);
    defer(cublasDestroy_v2(CUBLAS_HANDLE));

    run_test(cdnn_operations);
    run_test(matvecmul_n);
    run_test(matvecmul_t);
    run_test(matmul_n_n);
    run_test(matmul_t_n);
    run_test(matmul_n_t);
    run_test(matmul_t_t);

    run_test(linear_layer_fwd);
    run_test(linear_layer_bwd);
    run_test(sigmoid_activation_fwd);
    run_test(sigmoid_activation_bwd);

    run_test(inference);
    run_test(training);

    run_test(evaluate_mnist);
    return 0;
}
