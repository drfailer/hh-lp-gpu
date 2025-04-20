#include "cudnn_operations.hpp"
#include "data/layer_state.hpp"
#include "data/update_data.hpp"
#include "graph/network_graph.hpp"
#include "task/linear_layer_task.hpp"
#include "task/loss/quadratic_loss_task.hpp"
#include "task/sigmoid_activation_task.hpp"
#include "tools/defer.hpp"
#include "tools/gpu.hpp"
#include "tools/utest.hpp"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <ostream>

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

Parameters<ftype> init_test_parameters_gpu(LayerDimentions const &dims, Parameters<ftype> &gpu) {
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

template <typename T> std::shared_ptr<T> hh_get_result(auto &graph) {
    return std::get<std::shared_ptr<T>>(*graph.getBlockingResult());
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

// globals handls for the tests
cudnnHandle_t CUDNN_HANDLE;
cublasHandle_t CUBLAS_HANDLE;

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

// UTest(linear_layer_init) {
//     constexpr int64_t nb_nodes = 3;
//     constexpr int64_t nb_inputs = 3;
//     LayerDimentions dims = {
//         .nb_nodes = nb_nodes, .nb_inputs = nb_inputs, .kernel_size = 1};
//     NetworkState<ftype> network_state;
//
//     urequire(network_state.layer_states.size() == 1);
//
//     hh::Graph<LayerTaskIO> graph;
//     auto fc_layer_task =
//         std::make_shared<LinearLayerTask>(CUDNN_HANDLE, CUBLAS_HANDLE, 0,
//         dims);
//
//     graph.inputs(fc_layer_task);
//     graph.outputs(fc_layer_task);
//
//     graph.init(network_state)
//
//     graph.executeGraph(true);
//     graph.pushData(std::make_shared<InitData<ftype>>(network_state));
//     graph.finishPushingData();
//     auto init_data = hh_get_result<InitData<ftype>>(graph);
//     graph.waitForTermination();
//
//     urequire(&init_data->nn == &network_state);
//     auto state = init_data->nn[0];
//
//     uassert(state.dims.nb_nodes == nb_nodes);
//     uassert(state.dims.nb_inputs == nb_inputs);
//     uassert(state.dims.kernel_size == 1);
//     uassert(state.input == nullptr);
//     uassert(state.error != nullptr);
//     uassert(state.output != nullptr);
//     uassert(state.params.weights != nullptr);
//     uassert(state.params.biases != nullptr);
//     uassert(state.grads.weights != nullptr);
//     uassert(state.grads.biases != nullptr);
//
//     layer_state_destroy_gpu(state);
//     parameters_destroy_gpu(state.params);
//     parameters_destroy_gpu(state.grads);
//     uassert(state.input == nullptr);
//     uassert(state.error == nullptr);
//     uassert(state.output == nullptr);
//     uassert(state.params.weights == nullptr);
//     uassert(state.params.biases == nullptr);
//     uassert(state.grads.weights == nullptr);
//     uassert(state.grads.biases == nullptr);
// }

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
    defer(layer_state_destroy_gpu(layer_state));
    NetworkState<ftype> network_state({layer_state}, nullptr);

    CUDA_CHECK(alloc_gpu(&input_gpu, nb_inputs));
    defer(cudaFree(input_gpu));
    CUDA_CHECK(memcpy_host_to_gpu(input_gpu, input_host, nb_inputs));
    cudaDeviceSynchronize();

    hh::Graph<LayerTaskIO> graph;
    auto fc_layer_task =
        std::make_shared<LinearLayerTask>(CUDNN_HANDLE, CUBLAS_HANDLE, 0, dims);

    graph.inputs(fc_layer_task);
    graph.outputs(fc_layer_task);

    graph.executeGraph(true);
    graph.pushData(std::make_shared<FwdData<ftype>>(network_state, input_gpu));
    graph.finishPushingData();
    ftype *output_gpu = hh_get_result<FwdData<ftype>>(graph)->input;
    graph.waitForTermination();

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
    NetworkState<ftype> network_state({layer_state}, nullptr);

    hh::Graph<LayerTaskIO> graph;
    auto fc_layer_task =
        std::make_shared<LinearLayerTask>(CUDNN_HANDLE, CUBLAS_HANDLE, 0, dims);
    fc_layer_task->execute(
        std::make_shared<FwdData<ftype>>(network_state, input_gpu));

    graph.inputs(fc_layer_task);
    graph.outputs(fc_layer_task);

    graph.executeGraph(true);
    graph.pushData(std::make_shared<BwdData<ftype>>(network_state, err_gpu));
    graph.finishPushingData();
    ftype *output_err_gpu = hh_get_result<BwdData<ftype>>(graph)->error;
    graph.waitForTermination();

    urequire(output_err_gpu == layer_state.error);
    CUDA_CHECK(memcpy_gpu_to_host(output_err_host, output_err_gpu, nb_inputs));
    cudaDeviceSynchronize();

    uassert_equal(output_err_host[0], 123);
    uassert_equal(output_err_host[1], 234);
    uassert_equal(output_err_host[2], 345);
    uassert_equal(output_err_host[3], 456);
}

UTest(linear_layer_update) {
    constexpr int64_t nb_nodes = 3;
    constexpr int64_t nb_inputs = 3;
    LayerDimentions dims = {
        .nb_nodes = nb_nodes, .nb_inputs = nb_inputs, .kernel_size = 1};
    ftype learning_rate = 0.5;
    ftype weights_host[nb_nodes * nb_inputs] = {0}, biases_host[nb_nodes] = {0};
    ftype grads_value = 10;

    // init the parameters
    Parameters<ftype> params = create_test_parameters_gpu(dims);
    defer(parameters_destroy_gpu(params));
    Parameters<ftype> grads = create_test_parameters_gpu(dims, grads_value);
    defer(parameters_destroy_gpu(grads));

    LayerState<ftype> layer_state = layer_state_create_gpu(dims, params, grads);
    defer(layer_state_destroy_gpu(layer_state));
    NetworkState<ftype> network_state({layer_state}, nullptr);

    hh::Graph<LayerTaskIO> graph;
    auto fc_layer_task =
        std::make_shared<LinearLayerTask>(CUDNN_HANDLE, CUBLAS_HANDLE, 0, dims);

    graph.inputs(fc_layer_task);
    graph.outputs(fc_layer_task);

    graph.executeGraph(true);
    graph.pushData(
        std::make_shared<UpdateData<ftype>>(network_state, learning_rate));
    graph.finishPushingData();
    auto state =
        hh_get_result<UpdateData<ftype>>(graph)->states.layer_states[0];
    graph.waitForTermination();

    // make sure that the weights gradiants were not modified
    size_t size = nb_inputs * nb_nodes;
    CUDA_CHECK(memcpy_gpu_to_host(weights_host, state.grads.weights, size));
    CUDA_CHECK(memcpy_gpu_to_host(biases_host, state.grads.biases, nb_nodes));
    cudaDeviceSynchronize();
    for (size_t i = 0; i < size; ++i) {
        uassert_equal(weights_host[i], grads_value);
    }
    for (size_t i = 0; i < nb_inputs; ++i) {
        uassert_equal(biases_host[i], grads_value);
    }

    // testing the values of the weights
    CUDA_CHECK(memcpy_gpu_to_host(weights_host, state.params.weights, size));
    CUDA_CHECK(memcpy_gpu_to_host(biases_host, state.params.biases, nb_nodes));
    cudaDeviceSynchronize();
    for (int64_t i = 0; i < nb_nodes; ++i) {
        for (int64_t j = 0; j < nb_inputs; ++j) {
            ftype expected_value = i + j + 1 - learning_rate * grads_value;
            uassert_equal(weights_host[i * nb_inputs + j], expected_value);
        }
    }
    for (int64_t i = 0; i < nb_nodes; ++i) {
        ftype expected_value = i + 1 - learning_rate * grads_value;
        uassert_equal(biases_host[i], expected_value);
    }
}

// UTest(sigmoid_activation_init) {
//     constexpr int64_t nb_nodes = 3;
//     constexpr int64_t nb_inputs = 3;
//     LayerDimentions dims = {
//         .nb_nodes = nb_nodes, .nb_inputs = nb_inputs, .kernel_size = 1};
//     NetworkState<ftype> network_state;
//
//     hh::Graph<LayerTaskIO> graph;
//     auto sig_task = std::make_shared<SigmoidActivationTask>(
//         CUDNN_HANDLE, CUBLAS_HANDLE, 0, dims);
//
//     graph.inputs(sig_task);
//     graph.outputs(sig_task);
//
//     graph.executeGraph(true);
//     graph.pushData(std::make_shared<InitData<ftype>>(network_state));
//     graph.finishPushingData();
//     auto state = hh_get_result<InitData<ftype>>(graph)->nn[0];
//     graph.waitForTermination();
//
//     uassert(state.dims.nb_nodes == nb_nodes);
//     uassert(state.dims.nb_inputs == nb_inputs);
//     uassert(state.dims.kernel_size == 1);
//     uassert(state.input == nullptr);
//     uassert(state.error != nullptr);
//     uassert(state.output != nullptr);
//     uassert(state.params.weights == nullptr);
//     uassert(state.params.biases == nullptr);
//     uassert(state.grads.weights == nullptr);
//     uassert(state.grads.biases == nullptr);
//
//     layer_state_destroy_gpu(state);
//     uassert(state.input == nullptr);
//     uassert(state.error == nullptr);
//     uassert(state.output == nullptr);
//     uassert(state.params.weights == nullptr);
//     uassert(state.params.biases == nullptr);
//     uassert(state.grads.weights == nullptr);
//     uassert(state.grads.biases == nullptr);
// }

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

    hh::Graph<LayerTaskIO> graph;
    auto sig_task = std::make_shared<SigmoidActivationTask>(
        CUDNN_HANDLE, CUBLAS_HANDLE, 0, dims);

    graph.inputs(sig_task);
    graph.outputs(sig_task);

    graph.executeGraph(true);
    graph.pushData(std::make_shared<FwdData<ftype>>(network_state, input_gpu));
    graph.finishPushingData();
    ftype *output_gpu = hh_get_result<FwdData<ftype>>(graph)->input;
    graph.waitForTermination();

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

    hh::Graph<LayerTaskIO> graph;
    auto sig_task = std::make_shared<SigmoidActivationTask>(
        CUDNN_HANDLE, CUBLAS_HANDLE, 0, dims);
    // init sig_task (we expect the forward pass to be done at this point)
    sig_task->execute(
        std::make_shared<FwdData<ftype>>(network_state, input_gpu));

    graph.inputs(sig_task);
    graph.outputs(sig_task);

    graph.executeGraph(true);
    graph.pushData(std::make_shared<BwdData<ftype>>(network_state, err_gpu));
    graph.finishPushingData();
    ftype *output_gpu = hh_get_result<BwdData<ftype>>(graph)->error;
    graph.waitForTermination();

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

    graph.set_loss(
        std::make_shared<QuadraticLossTask>(10, CUDNN_HANDLE, CUBLAS_HANDLE));

    graph.add_layer(std::make_shared<LinearLayerTask>(
        CUDNN_HANDLE, CUBLAS_HANDLE, 0,
        LayerDimentions{
            .nb_nodes = nb_nodes, .nb_inputs = nb_inputs, .kernel_size = 1}));
    graph.add_layer(std::make_shared<SigmoidActivationTask>(
        CUDNN_HANDLE, CUBLAS_HANDLE, 1,
        LayerDimentions{
            .nb_nodes = nb_nodes, .nb_inputs = nb_nodes, .kernel_size = 1}));
    graph.build();

    graph.init_network_state(state);
    defer(graph.destroy_network_state(state));

    init_test_parameters_gpu(state.layer_states[0].dims,
            state.layer_states[0].params);

    graph.executeGraph(true);
    graph.pushData(std::make_shared<InferenceData<ftype>>(state, input_gpu));
    ftype *output_gpu = hh_get_result<InferenceData<ftype>>(graph)->input;
    graph.waitForTermination();

    CUDA_CHECK(memcpy_gpu_to_host(output_host, output_gpu, nb_nodes));
    cudaDeviceSynchronize();

    for (size_t i = 0; i < nb_nodes; ++i) {
        ftype weights_input_bias =
            input_host[0] * (i + 1) + output_host[1] * (i + 2) +
            output_host[2] * (i + 3) + i + 1;
        ftype expected_value = sigmoid(weights_input_bias);
        uassert_float_equal(output_host[i], expected_value, 1e-6);
    }
}

int main(int, char **) {
    cudnnCreate(&CUDNN_HANDLE);
    defer(cudnnDestroy(CUDNN_HANDLE));
    cublasCreate_v2(&CUBLAS_HANDLE);
    defer(cublasDestroy_v2(CUBLAS_HANDLE));

    run_test(cdnn_operations);
    run_test(linear_layer_fwd);
    run_test(linear_layer_bwd);
    run_test(linear_layer_update);
    run_test(sigmoid_activation_fwd);
    run_test(sigmoid_activation_bwd);
    run_test(inference);
    return 0;
}
