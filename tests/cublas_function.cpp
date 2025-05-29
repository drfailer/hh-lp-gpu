#include "cublas_function.hpp"
#include "../src/types.hpp"
#include "../src/tools/defer.hpp"
#include <random>

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

    // verify the result
    for (size_t i = 0; i < n; ++i) {
        uassert_float_equal(gt_host[i], out_host[i], 1e-2);
    }
}

UTest(matvecmul_batch_n) {
    constexpr size_t m = 10;
    constexpr size_t n = 10'000;
    constexpr size_t batch_size = 2;
    ftype *mat_host[batch_size] = {0}, *vec_host[batch_size] = {0},
          *gt_host[batch_size] = {0}, *out_host[batch_size] = {0};
    ftype *mat_gpu[batch_size] = {0}, *vec_gpu[batch_size] = {0},
          *out_gpu[batch_size] = {0};
    std::mt19937 rand(0);
    std::uniform_real_distribution<ftype> dist(0, 1);

    // allocate on host
    for (size_t b = 0; b < batch_size; ++b) {
        mat_host[b] = new ftype[m * n];
        vec_host[b] = new ftype[n];
        gt_host[b] = new ftype[m];
        out_host[b] = new ftype[m];
    }
    defer({
        for (size_t b = 0; b < batch_size; ++b) {
            delete[] mat_host[b];
            delete[] vec_host[b];
            delete[] gt_host[b];
            delete[] out_host[b];
        }
    });

    // allocate on gpu
    for (size_t b = 0; b < batch_size; ++b) {
        CUDA_CHECK(alloc_gpu(&mat_gpu[b], m * n));
        CUDA_CHECK(alloc_gpu(&vec_gpu[b], n));
        CUDA_CHECK(alloc_gpu(&out_gpu[b], m));
    }
    defer({
        for (size_t b = 0; b < batch_size; ++b) {
            defer(cudaFree(mat_gpu[b]));
            defer(cudaFree(vec_gpu[b]));
            defer(cudaFree(out_gpu[b]));
        }
    });

    // init
    for (size_t b = 0; b < batch_size; ++b) {
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

    // matrix vector multiplication on host
    for (size_t b = 0; b < batch_size; ++b) {
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
                           0.f, out_gpu, batch_size));
    for (size_t b = 0; b < batch_size; ++b) {
        CUDA_CHECK(memcpy_gpu_to_host(out_host[b], out_gpu[b], m));
    }

    // verify the result
    for (size_t b = 0; b < batch_size; ++b) {
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
    constexpr size_t batch_size = 2;
    ftype *A_host[batch_size] = {0}, *B_host[batch_size] = {0},
          *GT_host[batch_size] = {0}, *C_host[batch_size] = {0};
    ftype *A_gpu[batch_size] = {0}, *B_gpu[batch_size] = {0},
          *C_gpu[batch_size] = {0};
    std::mt19937 rand(0);
    std::uniform_real_distribution<ftype> dist(0, 1);
    auto init_matrix = [&dist, &rand](ftype **mat, size_t rows, size_t cols) {
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    mat[b][i * cols + j] = dist(rand);
                }
            }
        }
    };

    // allocate on host
    for (size_t b = 0; b < batch_size; ++b) {
        A_host[b] = new ftype[m * k];
        B_host[b] = new ftype[k * n];
        C_host[b] = new ftype[m * n];
        GT_host[b] = new ftype[m * n];
    }
    defer({
        for (size_t b = 0; b < batch_size; ++b) {
            defer(delete[] A_host[b]);
            defer(delete[] B_host[b]);
            defer(delete[] C_host[b]);
            defer(delete[] GT_host[b]);
        }
    });

    // allocate on gpu
    for (size_t b = 0; b < batch_size; ++b) {
        CUDA_CHECK(alloc_gpu(&A_gpu[b], m * k));
        CUDA_CHECK(alloc_gpu(&B_gpu[b], k * n));
        CUDA_CHECK(alloc_gpu(&C_gpu[b], m * n));
    }
    defer({
        for (size_t b = 0; b < batch_size; ++b) {
            defer(cudaFree(A_gpu[b]));
            defer(cudaFree(B_gpu[b]));
            defer(cudaFree(C_gpu[b]));
        }
    });

    // init
    init_matrix(A_host, m, k);
    init_matrix(B_host, k, n);
    for (size_t b = 0; b < batch_size; ++b) {
        CUDA_CHECK(memcpy_host_to_gpu(A_gpu[b], A_host[b], m * k));
        CUDA_CHECK(memcpy_host_to_gpu(B_gpu[b], B_host[b], k * n));
    }

    // matrix multiplication on host
    for (size_t b = 0; b < batch_size; ++b) {
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
           (ftype)0, C_gpu, batch_size);
    for (size_t b = 0; b < batch_size; ++b) {
        CUDA_CHECK(memcpy_gpu_to_host(C_host[b], C_gpu[b], m * n));
    }

    // verify the results
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                uassert_float_equal(GT_host[b][i * n + j], C_host[b][i * n + j],
                                    1e-1);
            }
        }
    }
}
