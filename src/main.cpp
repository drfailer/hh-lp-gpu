#include "../tools/mnist/minist_loader.hpp"
// #include "cudnn_operations.hpp"
#include "../tools/batch_generator.hpp"
#include "graph/network_graph.hpp"
#include "model/data/layer_state.hpp"
#include "model/layer/linear_layer.hpp"
#include "model/layer/sigmoid_activation_layer.hpp"
#include "model/loss/quadratic_loss.hpp"
#include "model/optimizer/sgd_optimizer.hpp"
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

void init_test_parameters(layer_state_t<ftype> &state, dims_t dims,
                          ftype value) {
    int64_t weights_size = dims.inputs * dims.outputs;
    ftype *weights = new ftype[weights_size];
    defer(delete[] weights);
    ftype *biases = new ftype[dims.outputs];
    defer(delete[] biases);

    for (int64_t i = 0; i < weights_size; ++i) {
        weights[i] = value;
    }
    for (int64_t i = 0; i < dims.outputs; ++i) {
        biases[i] = value;
    }
    CUDA_CHECK(memcpy_host_to_gpu(state.parameters->weights->data(), weights,
                                  weights_size));
    CUDA_CHECK(memcpy_host_to_gpu(state.parameters->biases->data(), biases,
                                  dims.outputs));
}

void init_test_parameters(layer_state_t<ftype> &state, dims_t dims) {
    ftype *weights = new ftype[dims.inputs * dims.outputs];
    defer(delete[] weights);
    ftype *biases = new ftype[dims.outputs];
    defer(delete[] biases);

    for (int64_t i = 0; i < dims.outputs; ++i) {
        for (int64_t j = 0; j < dims.inputs; ++j) {
            weights[i * dims.inputs + j] = i + j + 1;
            std::cout << weights[i * dims.inputs + j] << " ";
        }
        std::cout << std::endl;
    }
    for (int64_t i = 0; i < dims.outputs; ++i) {
        biases[i] = i + 1;
    }
    CUDA_CHECK(memcpy_host_to_gpu(state.parameters->weights->data(), weights,
                                  dims.inputs * dims.outputs));
    CUDA_CHECK(memcpy_host_to_gpu(state.parameters->biases->data(), biases,
                                  dims.outputs));
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

UTest(matvecmul_batch_n) {
    constexpr size_t m = 10;
    constexpr size_t n = 10'000;
    constexpr size_t batch_count = 2;
    ftype *mat_host[batch_count] = {0}, *vec_host[batch_count] = {0},
          *gt_host[batch_count] = {0}, *out_host[batch_count] = {0};
    ftype *mat_gpu[batch_count] = {0}, *vec_gpu[batch_count] = {0},
          *out_gpu[batch_count] = {0};
    std::mt19937 rand(0);
    std::uniform_real_distribution<ftype> dist(0, 1);

    // allocate on host
    for (size_t b = 0; b < batch_count; ++b) {
        mat_host[b] = new ftype[m * n];
        vec_host[b] = new ftype[n];
        gt_host[b] = new ftype[m];
        out_host[b] = new ftype[m];
    }
    defer({
        for (size_t b = 0; b < batch_count; ++b) {
            delete[] mat_host[b];
            delete[] vec_host[b];
            delete[] gt_host[b];
            delete[] out_host[b];
        }
    });

    // allocate on gpu
    for (size_t b = 0; b < batch_count; ++b) {
        CUDA_CHECK(alloc_gpu(&mat_gpu[b], m * n));
        CUDA_CHECK(alloc_gpu(&vec_gpu[b], n));
        CUDA_CHECK(alloc_gpu(&out_gpu[b], m));
    }
    defer({
        for (size_t b = 0; b < batch_count; ++b) {
            defer(cudaFree(mat_gpu[b]));
            defer(cudaFree(vec_gpu[b]));
            defer(cudaFree(out_gpu[b]));
        }
    });

    // init
    for (size_t b = 0; b < batch_count; ++b) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                mat_host[b][i * n + j] = dist(rand);
            }
        }
        for (size_t i = 0; i < n; ++i) {
            vec_host[b][i] = dist(rand);
        }
        CUDA_CHECK(memcpy_host_to_gpu(mat_gpu[b], mat_host[b], m * n));
        CUDA_CHECK(memcpy_host_to_gpu(vec_gpu[b], vec_host[b], n));
    }
    cudaDeviceSynchronize();

    // matrix vector multiplication on host
    for (size_t b = 0; b < batch_count; ++b) {
        for (size_t i = 0; i < m; ++i) {
            ftype sum = 0;
            for (size_t j = 0; j < n; ++j) {
                sum += mat_host[b][i * n + j] * vec_host[b][j];
            }
            gt_host[b][i] = sum;
        }
    }

    // matrix vector multiplication on the gpu
    CUBLAS_CHECK(matvecmul(CUBLAS_HANDLE, false, m, n, 1.f, mat_gpu, vec_gpu,
                           0.f, out_gpu, batch_count));
    for (size_t b = 0; b < batch_count; ++b) {
        CUDA_CHECK(memcpy_gpu_to_host(out_host[b], out_gpu[b], m));
    }
    cudaDeviceSynchronize();

    // verify the result
    for (size_t b = 0; b < batch_count; ++b) {
        for (size_t i = 0; i < m; ++i) {
            uassert_float_equal(gt_host[b][i], out_host[b][i], 1e-2);
        }
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

UTest(matmul_batch_n_n) {
    constexpr size_t m = 10;
    constexpr size_t n = 10'000;
    constexpr size_t k = 100;
    constexpr size_t batch_count = 2;
    ftype *A_host[batch_count] = {0}, *B_host[batch_count] = {0},
          *GT_host[batch_count] = {0}, *C_host[batch_count] = {0};
    ftype *A_gpu[batch_count] = {0}, *B_gpu[batch_count] = {0},
          *C_gpu[batch_count] = {0};
    std::mt19937 rand(0);
    std::uniform_real_distribution<ftype> dist(0, 1);
    auto init_matrix = [&dist, &rand](ftype **mat, size_t rows, size_t cols) {
        for (size_t b = 0; b < batch_count; ++b) {
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    mat[b][i * cols + j] = dist(rand);
                }
            }
        }
    };

    // allocate on host
    for (size_t b = 0; b < batch_count; ++b) {
        A_host[b] = new ftype[m * k];
        B_host[b] = new ftype[k * n];
        C_host[b] = new ftype[m * n];
        GT_host[b] = new ftype[m * n];
    }
    defer({
        for (size_t b = 0; b < batch_count; ++b) {
            defer(delete[] A_host[b]);
            defer(delete[] B_host[b]);
            defer(delete[] C_host[b]);
            defer(delete[] GT_host[b]);
        }
    });

    // allocate on gpu
    for (size_t b = 0; b < batch_count; ++b) {
        CUDA_CHECK(alloc_gpu(&A_gpu[b], m * k));
        CUDA_CHECK(alloc_gpu(&B_gpu[b], k * n));
        CUDA_CHECK(alloc_gpu(&C_gpu[b], m * n));
    }
    defer({
        for (size_t b = 0; b < batch_count; ++b) {
            defer(cudaFree(A_gpu[b]));
            defer(cudaFree(B_gpu[b]));
            defer(cudaFree(C_gpu[b]));
        }
    });

    // init
    init_matrix(A_host, m, k);
    init_matrix(B_host, k, n);
    for (size_t b = 0; b < batch_count; ++b) {
        CUDA_CHECK(memcpy_host_to_gpu(A_gpu[b], A_host[b], m * k));
        CUDA_CHECK(memcpy_host_to_gpu(B_gpu[b], B_host[b], k * n));
    }
    cudaDeviceSynchronize();

    // matrix multiplication on host
    for (size_t b = 0; b < batch_count; ++b) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                GT_host[b][i * n + j] = 0;
                for (size_t e = 0; e < k; ++e) {
                    GT_host[b][i * n + j] +=
                        A_host[b][i * k + e] * B_host[b][e * n + j];
                }
            }
        }
    }

    // matrix multiplication on the gpu
    matmul(CUBLAS_HANDLE, false, false, m, n, k, (ftype)1, A_gpu, B_gpu,
           (ftype)0, C_gpu, batch_count);
    for (size_t b = 0; b < batch_count; ++b) {
        CUDA_CHECK(memcpy_gpu_to_host(C_host[b], C_gpu[b], m * n));
    }

    // verify the results
    for (size_t b = 0; b < batch_count; ++b) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                uassert_float_equal(GT_host[b][i * n + j], C_host[b][i * n + j],
                                    1e-1);
            }
        }
    }
}

UTest(linear_layer_fwd) {
    constexpr int64_t inputs = 3;
    constexpr int64_t outputs = 3;
    dims_t dims = {.inputs = inputs, .outputs = outputs};
    ftype input_host[inputs] = {1, 2, 3}, output_host[outputs] = {0};
    Tensor<ftype> input_gpu({1, 1, inputs, 1}, {inputs, inputs, 1, 1});

    CUDA_CHECK(memcpy_host_to_gpu(input_gpu.data(), input_host, inputs));
    cudaDeviceSynchronize();

    LinearLayer linear_layer(CUBLAS_HANDLE, CUDNN_HANDLE, inputs, outputs);
    layer_state_t<ftype> state = linear_layer.create_state();
    defer(destroy_layer_state(state));
    init_test_parameters(state, dims, 1);
    linear_layer.init(state, 1);

    Tensor<ftype> *output_gpu = linear_layer.fwd(state, &input_gpu);

    urequire(output_gpu == state.output);
    CUDA_CHECK(memcpy_gpu_to_host(output_host, output_gpu->data(), outputs));
    cudaDeviceSynchronize();

    for (size_t i = 0; i < outputs; ++i) {
        uassert_equal(output_host[i], 7);
    }
}

UTest(linear_layer_bwd) {
    constexpr int64_t inputs = 4;
    constexpr int64_t outputs = 3;
    dims_t dims = {.inputs = inputs, .outputs = outputs};
    ftype input_host[inputs] = {1, 2, 3, 4},
          input_err_host[outputs] = {100, 10, 1}, output_err_host[inputs] = {0};
    Tensor<ftype> input_gpu({1, 1, inputs, 1}, {inputs, inputs, 1, 1});
    Tensor<ftype> err_gpu({1, 1, inputs, 1}, {inputs, inputs, 1, 1});

    // init input and output gpu buffers
    CUDA_CHECK(memcpy_host_to_gpu(input_gpu.data(), input_host, inputs));
    CUDA_CHECK(memcpy_host_to_gpu(err_gpu.data(), input_err_host, inputs));

    LinearLayer linear_layer(CUBLAS_HANDLE, CUDNN_HANDLE, inputs, outputs);
    layer_state_t<ftype> state = linear_layer.create_state();
    init_test_parameters(state, dims);
    defer(destroy_layer_state(state));
    linear_layer.init(state, 1);

    linear_layer.fwd(state, &input_gpu);
    Tensor<ftype> *output_err_gpu = linear_layer.bwd(state, &err_gpu);

    urequire(output_err_gpu == state.error);
    CUDA_CHECK(
        memcpy_gpu_to_host(output_err_host, output_err_gpu->data(), inputs));
    cudaDeviceSynchronize();

    uassert_equal(output_err_host[0], 123);
    uassert_equal(output_err_host[1], 234);
    uassert_equal(output_err_host[2], 345);
    uassert_equal(output_err_host[3], 456);
}

UTest(sigmoid_activation_fwd) {
    constexpr int64_t outputs = 3;
    constexpr int64_t inputs = 3;
    ftype input_host[inputs] = {1, 2, 3}, output_host[outputs] = {0};
    Tensor<ftype> input_gpu({1, 1, inputs, 1}, {inputs, inputs, 1, 1});

    CUDA_CHECK(memcpy_host_to_gpu(input_gpu.data(), input_host, inputs));
    cudaDeviceSynchronize();

    SigmoidActivationLayer sigmoid_layer(CUDNN_HANDLE, inputs);
    layer_state_t<ftype> state = sigmoid_layer.create_state();
    defer(destroy_layer_state(state));
    sigmoid_layer.init(state, 1);
    Tensor<ftype> *output_gpu = sigmoid_layer.fwd(state, &input_gpu);

    urequire(output_gpu != &input_gpu);
    CUDA_CHECK(memcpy_gpu_to_host(output_host, output_gpu->data(), inputs));
    cudaDeviceSynchronize();

    for (size_t i = 0; i < outputs; ++i) {
        uassert_float_equal(output_host[i], sigmoid(input_host[i]), 1e-6);
    }
}

UTest(sigmoid_activation_bwd) {
    constexpr int64_t outputs = 6;
    constexpr int64_t inputs = 6;
    ftype input_host[inputs] = {1, 2, 3, 4, 5, 6},
          err[inputs] = {10, 10, 10, 10, 10, 10}, output_host[outputs] = {0};
    Tensor<ftype> input_gpu({1, 1, inputs, 1}, {inputs, inputs, 1, 1});
    Tensor<ftype> err_gpu({1, 1, inputs, 1}, {inputs, inputs, 1, 1});

    CUDA_CHECK(memcpy_host_to_gpu(input_gpu.data(), input_host, inputs));
    CUDA_CHECK(memcpy_host_to_gpu(err_gpu.data(), err, inputs));
    cudaDeviceSynchronize();

    SigmoidActivationLayer sigmoid_layer(CUDNN_HANDLE, inputs);
    layer_state_t<ftype> state = sigmoid_layer.create_state();
    defer(destroy_layer_state(state));
    sigmoid_layer.init(state, 1);
    sigmoid_layer.fwd(state, &input_gpu);
    Tensor<ftype> *output_gpu = sigmoid_layer.bwd(state, &err_gpu);

    CUDA_CHECK(memcpy_gpu_to_host(output_host, output_gpu->data(), inputs));
    cudaDeviceSynchronize();

    for (size_t i = 0; i < outputs; ++i) {
        uassert_float_equal(output_host[i],
                            err[i] * sigmoid_derivative(input_host[i]), 1e-6);
    }
}

UTest(sgd_optimizer) {
    constexpr int64_t inputs = 3;
    constexpr int64_t outputs = 2;
    constexpr ftype learning_rate = 0.01;
    ftype weights[inputs * outputs] = {1, 2, 3, 4, 4, 6};
    ftype weights_gradients[inputs * outputs] = {1, 1, 1, 1, 1, 1};
    ftype biases[outputs] = {1, 2};
    ftype biases_gradients[outputs] = {1, 1};
    vec_t weights_dims = {1, 1, inputs, outputs},
          weights_strides = {inputs * outputs, inputs * outputs, outputs, 1},
          biases_dims = {1, 1, outputs, 1},
          biases_strides = {outputs, outputs, 1, 1};
    Tensor<ftype> weights_gpu(weights_dims, weights_strides),
        biases_gpu(biases_dims, biases_strides),
        gradient_weights_gpu(weights_dims, weights_strides),
        gradient_biases_gpu(biases_dims, biases_strides);
    Parameter<ftype> parameters(&weights_gpu, &biases_gpu);
    Parameter<ftype> gradients(&gradient_weights_gpu, &gradient_biases_gpu);
    layer_state_t<ftype> state = {
        .parameters = &parameters,
        .gradients = &gradients,
    };
    SGDOptimizer optimizer_factory(CUDNN_HANDLE);

    CUDA_CHECK(
        memcpy_host_to_gpu(weights_gpu.data(), weights, inputs * outputs));
    CUDA_CHECK(memcpy_host_to_gpu(gradient_weights_gpu.data(), weights_gradients,
                                  inputs * outputs));
    CUDA_CHECK(memcpy_host_to_gpu(biases_gpu.data(), biases, outputs));
    CUDA_CHECK(
        memcpy_host_to_gpu(gradient_biases_gpu.data(), biases_gradients, outputs));

    auto sgd = optimizer_factory.create();
    sgd->optimize(state, learning_rate);

    ftype result_weights[inputs * outputs] = {0}, result_biases[outputs] = {0};
    CUDA_CHECK(memcpy_gpu_to_host(result_weights, weights_gpu.data(),
                                  outputs * inputs));
    CUDA_CHECK(memcpy_gpu_to_host(result_biases, biases_gpu.data(), outputs));

    for (size_t i = 0; i < inputs * outputs; ++i) {
        uassert_float_equal(result_weights[i],
                            weights[i] - learning_rate * weights_gradients[i],
                            1e-6);
    }

    for (size_t i = 0; i < outputs; ++i) {
        uassert_float_equal(result_biases[i],
                            biases[i] - learning_rate * biases_gradients[i],
                            1e-6);
    }
}

UTest(inference) {
    constexpr size_t outputs = 3;
    constexpr size_t inputs = 3;
    ftype input_host[inputs] = {1, 1, 1}, output_host[outputs] = {0};
    Tensor<ftype> input_gpu({1, 1, inputs, 1}, {inputs, inputs, 1, 1});
    NetworkGraph graph;
    NetworkState<ftype> state;

    CUDA_CHECK(memcpy_host_to_gpu(input_gpu.data(), input_host, inputs));
    cudaDeviceSynchronize();

    graph.set_loss<QuadraticLoss>(CUDNN_HANDLE);
    graph.set_optimizer<SGDOptimizer>(1, CUDNN_HANDLE);

    graph.add_layer<LinearLayer>(CUBLAS_HANDLE, CUDNN_HANDLE, inputs, outputs);
    graph.add_layer<SigmoidActivationLayer>(CUDNN_HANDLE, outputs);

    graph.build();

    state = graph.create_state();
    defer(graph.destroy_state(state));
    graph.init(state, 1);

    init_test_parameters(state.layers[0],
                         dims_t{.inputs = inputs, .outputs = outputs});

    graph.executeGraph(true);
    graph.pushData(std::make_shared<InferenceData<ftype>>(state, &input_gpu));
    Tensor<ftype> *output_gpu = graph.get<InferenceData<ftype>>()->input;
    graph.terminate();

    CUDA_CHECK(memcpy_gpu_to_host(output_host, output_gpu->data(), outputs));
    cudaDeviceSynchronize();

    for (size_t i = 0; i < outputs; ++i) {
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

    graph.add_layer<LinearLayer>(CUBLAS_HANDLE, CUDNN_HANDLE, nb_inputs, 32);
    graph.add_layer<SigmoidActivationLayer>(CUDNN_HANDLE, 32);
    graph.cut_layer();
    graph.add_layer<LinearLayer>(CUBLAS_HANDLE, CUDNN_HANDLE, 32, 32);
    graph.add_layer<SigmoidActivationLayer>(CUDNN_HANDLE, 32);
    graph.cut_layer();
    graph.add_layer<LinearLayer>(CUBLAS_HANDLE, CUDNN_HANDLE, 32, 10);
    graph.add_layer<SigmoidActivationLayer>(CUDNN_HANDLE, nb_nodes);

    graph.build();

    state = graph.create_state();
    defer(graph.destroy_state(state));

    graph.init(state, 1);

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

    graph.add_layer<LinearLayer>(CUBLAS_HANDLE, CUDNN_HANDLE, 28 * 28, 10);
    graph.add_layer<SigmoidActivationLayer>(CUDNN_HANDLE, 10);

    graph.build();

    NetworkState<ftype> state = graph.create_state();
    defer(graph.destroy_state(state));

    graph.init(state, 1);

    graph.executeGraph(true);

    INFO("Inference before training...");

    std::vector<ftype> expected(10), found(10);
    size_t success = 0, errors = 0;
    for (auto data : testing_set.datas) {
        graph.pushData(
            std::make_shared<InferenceData<ftype>>(state, data.input));
        Tensor<ftype> *output = graph.get<InferenceData<ftype>>()->input;
        graph.cleanGraph();
        CUDA_CHECK(
            memcpy_gpu_to_host(expected.data(), data.ground_truth->data(), 10));
        CUDA_CHECK(memcpy_gpu_to_host(found.data(), output->data(), 10));
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
        state, training_set, learning_rate, epochs));
    (void)graph.get<TrainingData<ftype>>();
    graph.cleanGraph();

    INFO("Evaluate the model...");

    success = 0;
    errors = 0;
    for (auto data : testing_set.datas) {
        graph.pushData(
            std::make_shared<InferenceData<ftype>>(state, data.input));
        Tensor<ftype> *output = graph.get<InferenceData<ftype>>()->input;
        graph.cleanGraph();
        CUDA_CHECK(
            memcpy_gpu_to_host(expected.data(), data.ground_truth->data(), 10));
        CUDA_CHECK(memcpy_gpu_to_host(found.data(), output->data(), 10));
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

UTest(evaluate_mnist_batched) {
    constexpr ftype learning_rate = 0.1;
    constexpr size_t epochs = 3;
    constexpr size_t batch_count = 16;
    MNISTLoader loader;
    BatchGenerator<ftype> batch_generator(0);

    DataSet<ftype> training_data =
        loader.load_ds("../data/mnist/train-labels-idx1-ubyte",
                       "../data/mnist/train-images-idx3-ubyte");
    defer(destroy_data_set(training_data));
    DataSet<ftype> training_set =
        batch_generator.generate(training_data, 28 * 28, 10, batch_count);
    DataSet<ftype> testing_set =
        loader.load_ds("../data/mnist/t10k-labels-idx1-ubyte",
                       "../data/mnist/t10k-images-idx3-ubyte");
    defer(destroy_data_set(testing_set));

    NetworkGraph graph;

    graph.set_loss<QuadraticLoss>(CUDNN_HANDLE);
    graph.set_optimizer<SGDOptimizer>(1, CUDNN_HANDLE);

    graph.add_layer<LinearLayer>(CUBLAS_HANDLE, CUDNN_HANDLE, 28 * 28, 10);
    graph.add_layer<SigmoidActivationLayer>(CUDNN_HANDLE, 10);

    graph.build();

    NetworkState<ftype> network = graph.create_state();
    defer(graph.destroy_state(network));

    graph.executeGraph(true);

    INFO("Inference before training...");

    graph.init(network, 1);

    std::vector<ftype> expected(10), found(10);
    size_t success = 0, errors = 0;
    for (auto data : testing_set.datas) {
        graph.pushData(
            std::make_shared<InferenceData<ftype>>(network, data.input));
        Tensor<ftype> *output = graph.get<InferenceData<ftype>>()->input;
        graph.cleanGraph();
        CUDA_CHECK(
            memcpy_gpu_to_host(expected.data(), data.ground_truth->data(), 10));
        CUDA_CHECK(memcpy_gpu_to_host(found.data(), output->data(), 10));
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

    graph.init(network, batch_count);

    INFO("start training (learning_rate = " << learning_rate
                                            << ", epochs = " << epochs << ")");
    graph.pushData(std::make_shared<TrainingData<ftype>>(
        network, training_set, learning_rate, epochs));
    (void)graph.get<TrainingData<ftype>>();
    graph.cleanGraph();

    INFO("Evaluate the model...");

    graph.init(network, 1);

    success = 0;
    errors = 0;
    for (auto data : testing_set.datas) {
        graph.pushData(
            std::make_shared<InferenceData<ftype>>(network, data.input));
        Tensor<ftype> *output = graph.get<InferenceData<ftype>>()->input;
        graph.cleanGraph();
        CUDA_CHECK(
            memcpy_gpu_to_host(expected.data(), data.ground_truth->data(), 10));
        CUDA_CHECK(memcpy_gpu_to_host(found.data(), output->data(), 10));
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

    uassert(accuracy_end > accuracy_start);

    graph.createDotFile("train_mnist_batch.dot", hh::ColorScheme::EXECUTION,
                        hh::StructureOptions::QUEUE);
}

int main(int, char **) {
    cudnnCreate(&CUDNN_HANDLE);
    defer(cudnnDestroy(CUDNN_HANDLE));
    cublasCreate_v2(&CUBLAS_HANDLE);
    defer(cublasDestroy_v2(CUBLAS_HANDLE));

    // run_test(matvecmul_n);
    // run_test(matvecmul_t);
    // run_test(matvecmul_batch_n);
    // run_test(matmul_n_n);
    // run_test(matmul_t_n);
    // run_test(matmul_n_t);
    // run_test(matmul_t_t);
    // run_test(matmul_batch_n_n);

    // run_test(linear_layer_fwd);
    // run_test(linear_layer_bwd);
    // run_test(sigmoid_activation_fwd);
    // run_test(sigmoid_activation_bwd);
    // run_test(sgd_optimizer);

    // run_test(inference);
    // run_test(training);

    // run_test(evaluate_mnist);
    run_test(evaluate_mnist_batched);
    return 0;
}
