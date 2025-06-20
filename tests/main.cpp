#include "layers.hpp"
#include "cublas_function.hpp"
#include "../src/tools/defer.hpp"

// globals handls for the tests
cudnnHandle_t CUDNN_HANDLE;
cublasHandle_t CUBLAS_HANDLE;


int main(int, char **) {
    cudnnCreate(&CUDNN_HANDLE);
    defer(cudnnDestroy(CUDNN_HANDLE));
    cublasCreate_v2(&CUBLAS_HANDLE);
    defer(cublasDestroy_v2(CUBLAS_HANDLE));

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
    // urun_test(linear_layer_bwd);
    urun_test(linear_layer_fwd_batched);
    // urun_test(linear_layer_bwd_batched);
    // urun_test(sigmoid_activation_fwd);
    // urun_test(sigmoid_activation_bwd);
    // urun_test(sgd_optimizer);

    // urun_test(inference);
    // urun_test(training);

    urun_test(mnist);
    // urun_test(mnist_batched);

    utest_end();

    return UTEST_STATUS;
}

// TILE_[dims]
// WARP_TILE_[dims]
// THREAD_TILE_[dims]

// for (int cta_n = 0; cta_n < GemmN; cta_n += CtaTileN) {                     // for each threadblock_y           } threadblock-level concurrency
//   for (int cta_m = 0; cta_m < GemmM; cta_m += CtaTileM) {                   //    for each threadblock_x        }
//
//     for (int cta_k = 0; cta_k < GemmK; cta_k += CtaTileK) {                 //       "GEMM mainloop" - no unrolling
//                                                                             //                       - one iteration of this loop is one "stage"
//                                                                             //
//       for (int warp_n = 0; warp_n < CtaTileN; warp_n += WarpTileN) {        // for each warp_y                  } warp-level parallelism
//         for (int warp_m = 0; warp_m < CtaTileM; warp_m += WarpTileM) {      //    for each warp_x               }
//                                                                             //
//           for (int warp_k = 0; warp_k < CtaTileK; warp_k += WarpTileK) {         //       fully unroll across CtaTileK
//                                                                             //         - one iteration of this loop is one "k Group"
//                                                                             //
//             for (int mma_k = 0; mma_k < WarpTileK; mma_k += MmaK) {         // for each mma instruction         } instruction-level parallelism
//               for (int mma_n = 0; mma_n < WarpTileN; mma_n += MmaN) {       //    for each mma instruction      }
//                 for (int mma_m = 0; mma_m < WarpTileM; mma_m += MmaM) {


// constexpr unsigned int TILE_X{128U};
// constexpr unsigned int TILE_Y{128U};
// constexpr unsigned int TILE_K{16U};
//
// // The skew size is used to avoid bank conflicts in shared memory.
// constexpr size_t BLOCK_TILE_SKEW_SIZE_X{16U};
// constexpr size_t BLOCK_TILE_SKEW_SIZE_Y{16U};
//
// constexpr unsigned int WARP_TILE_X{32U};
// constexpr unsigned int WARP_TILE_Y{64U};
// constexpr unsigned int NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
// constexpr unsigned int NUM_WARPS_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};
// static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0U);
// static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0U);
//
// constexpr unsigned int THREAD_TILE_X{16U};
// constexpr unsigned int THREAD_TILE_Y{16U};
// constexpr unsigned int THREAD_TILE_K{16U};
//
// constexpr unsigned int NUM_THREADS_PER_BLOCK{NUM_WARPS_X * NUM_WARPS_Y *
//                                              32U};
//
// dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
// dim3 const grid_dim{
//     (n + TILE_X - 1U) / TILE_X,
//     (m + TILE_Y - 1U) / TILE_Y,
//     1U};
