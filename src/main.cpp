#include "cudnn_operations.hpp"
#include "data/layer.hpp"
#include "task/fully_connected_layer_task.hpp"
#include "tools/defer.hpp"
#include "tools/gpu.hpp"
#include <cstdint>
#include <cstring>
#include <iostream>

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
// - weights_gpu, weights_host
// - biases_gpu, biases_host
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
//
// Idea: add the a ans z tensors in the layer

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

void test_cdnn_operations() {
    constexpr int64_t m = 3;
    constexpr int64_t n = 3;
    constexpr int64_t k = 1;
    ftype *A = new ftype[m * n];
    ftype *B = new ftype[n * k];
    ftype *C = new ftype[m * k];

    std::cout << "perform matrix multiplication: C[m, k] = A[m, n] * B[n, k]"
              << std::endl;
    std::cout << "m = " << m << std::endl;
    std::cout << "n = " << n << std::endl;
    std::cout << "k = " << k << std::endl;

    init(A, B, C, m, n, k);
    display_matrix("A", A, m, n);
    display_matrix("B", B, n, k);
    display_matrix("C", C, m, k);

    if (m == n && n == k)
        hadamard(A, B, m, n, C);
    display_matrix("C", C, m, k);

    matmul(A, B, m, n, k, C);
    display_matrix("C", C, m, k);

    delete[] A;
    delete[] B;
    delete[] C;
}

void test_fully_connected_layer_fwd() {
    constexpr int64_t nb_nodes = 3;
    constexpr int64_t nb_inputs = 3;
    LayerDimentions dims = {
        .nb_nodes = nb_nodes, .nb_inputs = nb_inputs, .kernel_size = 1};
    ftype input_host[nb_inputs] = {1, 2, 3}, output_host[nb_nodes] = {0};
    ftype *input_gpu = nullptr, *output_gpu = nullptr;
    cudnnHandle_t cudnn_handle;
    cublasHandle_t cublas_handle;

    Layer<ftype> layer = layer_create<ftype>(dims);
    defer(layer_destroy(layer));

    std::vector<Layer<ftype>> layers = {layer};

    cudnnCreate(&cudnn_handle);
    defer(cudnnDestroy(cudnn_handle));

    cublasCreate_v2(&cublas_handle);
    defer(cublasDestroy_v2(cublas_handle));

    layer_init(layer, (ftype)1);

    CUDA_CHECK(alloc_gpu(&input_gpu, nb_inputs));
    defer(cudaFree(input_gpu));

    CUDA_CHECK(alloc_gpu(&output_gpu, nb_nodes));
    defer(cudaFree(output_gpu));

    CUDA_CHECK(memcpy_host_to_gpu(input_gpu, input_host, nb_inputs));
    cudaDeviceSynchronize();

    hh::Graph<LayerTaskType> graph;
    auto fc_layer_task = std::make_shared<FullyConnectedLayerTask>(
        cudnn_handle, cublas_handle,
        std::vector<int64_t>({1, nb_nodes, nb_inputs}));

    graph.inputs(fc_layer_task);
    graph.outputs(fc_layer_task);

    graph.executeGraph(true);
    graph.pushData(std::make_shared<FwdData<ftype>>(layers.begin(), input_gpu,
                                                    output_gpu));
    graph.finishPushingData();

    auto result = graph.getBlockingResult();

    ftype *result_output_gpu = std::get<0>(*result)->output_gpu;

    assert(output_gpu == result_output_gpu);
    CUDA_CHECK(memcpy_gpu_to_host(output_host, output_gpu, nb_nodes));
    cudaDeviceSynchronize();

    graph.waitForTermination();

    for (size_t i = 0; i < nb_nodes; ++i) {
        std::cout << output_host[i] << std::endl;
    }
}

int main(int, char **) {
    /* test_cdnn_operations(); */
    test_fully_connected_layer_fwd();
    return 0;
}
