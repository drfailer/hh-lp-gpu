#include "layers.hpp"
#include "../src/graph/network_graph.hpp"
#include "../src/model/data/layer_state.hpp"
#include "../src/model/layer/convolution_layer.hpp"
#include "../src/model/layer/pooling_layer.hpp"
#include "../src/model/layer/linear_layer.hpp"
#include "../src/model/layer/sigmoid_activation_layer.hpp"
#include "../src/model/loss/quadratic_loss.hpp"
#include "../src/model/optimizer/sgd_optimizer.hpp"
#include "../src/tools/defer.hpp"
#include "../src/tools/gpu.hpp"
#include "../src/tools/timer.hpp"
#include "../tools/batch_generator.hpp"
#include "../tools/mnist/mnist_loader.hpp"
#include "utest.hpp"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <ostream>
#include <unistd.h>

ftype sigmoid(ftype x) { return 1.0 / (1.0 + std::exp(-x)); }

ftype sigmoid_derivative(ftype x) { return sigmoid(x) * (1.0 - sigmoid(x)); }

void init_test_parameters(LayerState<ftype> &state, dims_t dims, ftype value) {
    int weights_size = dims.inputs * dims.outputs;
    ftype *weights = new ftype[weights_size];
    defer(delete[] weights);
    ftype *biases = new ftype[dims.outputs];
    defer(delete[] biases);

    for (int i = 0; i < weights_size; ++i) {
        weights[i] = value;
    }
    for (int i = 0; i < dims.outputs; ++i) {
        biases[i] = value;
    }
    state.parameters.weights->from_host(weights);
    state.parameters.biases->from_host(biases);
}

void init_test_parameters(LayerState<ftype> &state, dims_t dims) {
    ftype *weights = new ftype[dims.inputs * dims.outputs];
    defer(delete[] weights);
    ftype *biases = new ftype[dims.outputs];
    defer(delete[] biases);

    for (int i = 0; i < dims.outputs; ++i) {
        for (int j = 0; j < dims.inputs; ++j) {
            weights[i * dims.inputs + j] = i + j + 1;
            std::cout << weights[i * dims.inputs + j] << " ";
        }
        std::cout << std::endl;
    }
    for (int i = 0; i < dims.outputs; ++i) {
        biases[i] = i + 1;
    }
    state.parameters.weights->from_host(weights);
    state.parameters.biases->from_host(biases);
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

float evaluate_mnist(NetworkGraph &graph, DataSet<ftype> &testing_set,
                     std::shared_ptr<NNState<ftype>> &state,
                     int batch_size = 1) {
    int success = 0;
    int errors = 0;
    std::vector<ftype> expected(batch_size * 10, 0), found(batch_size * 10, 0);

    timer_start(evaluate_mnist);
    for (auto data : testing_set.datas) {
        Tensor<ftype> *output = graph.predict(state, data.input);
        CUDA_CHECK(data.ground_truth->to_host(expected.data()));
        CUDA_CHECK(output->to_host(found.data()));

        for (size_t i = 0; i < batch_size; ++i) {
            int expected_label = mnist_get_label(&expected.data()[i * 10]);
            int found_label = mnist_get_label(&found.data()[i * 10]);

            if (found_label == expected_label) {
                ++success;
            } else {
                ++errors;
            }
        }
    }
    timer_end(evaluate_mnist);

    float accuracy =
        (ftype)success / (ftype)(batch_size * testing_set.datas.size());
    std::cout << "accuracy: " << accuracy << std::endl;
    std::cout << "success: " << success << ", errors: " << errors << std::endl;
    timer_report_prec(evaluate_mnist, milliseconds);

    return accuracy;
}

UTest(linear_layer_fwd) {
    constexpr int inputs = 3;
    constexpr int outputs = 3;
    dims_t dims = {.inputs = inputs, .outputs = outputs};
    ftype input_host[inputs] = {1, 2, 3}, output_host[outputs] = {0};
    Tensor<ftype> input_gpu({1, 1, inputs, 1}, {inputs, inputs, 1, 1});

    input_gpu.from_host(input_host);

    LinearLayer linear_layer(inputs, outputs);
    LayerState<ftype> state;
    state.set_parameters(linear_layer.create_parameters());
    init_test_parameters(state, dims, 1);
    linear_layer.init({CUDNN_HANDLE, CUBLAS_HANDLE}, state, {1, 1, inputs, 1});

    Tensor<ftype> *output_gpu =
        linear_layer.fwd({CUDNN_HANDLE, CUBLAS_HANDLE}, state, &input_gpu);

    urequire(output_gpu == state.output);
    output_gpu->to_host(output_host);

    for (size_t i = 0; i < outputs; ++i) {
        uassert_equal(output_host[i], 7);
    }
}

UTest(linear_layer_bwd) {
    constexpr int inputs = 4;
    constexpr int outputs = 3;
    dims_t dims = {.inputs = inputs, .outputs = outputs};
    ftype input_host[inputs] = {1, 2, 3, 4},
          input_err_host[outputs] = {100, 10, 1}, output_err_host[inputs] = {0};
    Tensor<ftype> input_gpu({1, 1, inputs, 1}, {inputs, inputs, 1, 1});
    Tensor<ftype> err_gpu({1, 1, inputs, 1}, {inputs, inputs, 1, 1});

    // init input and output gpu buffers
    input_gpu.from_host(input_host);
    err_gpu.from_host(input_err_host);

    LinearLayer linear_layer(inputs, outputs);
    LayerState<ftype> state;
    state.set_parameters(linear_layer.create_parameters());
    init_test_parameters(state, dims);
    linear_layer.init({CUDNN_HANDLE, CUBLAS_HANDLE}, state, {1, 1, inputs, 1});

    linear_layer.fwd({CUDNN_HANDLE, CUBLAS_HANDLE}, state, &input_gpu);
    Tensor<ftype> *output_err_gpu =
        linear_layer.bwd({CUDNN_HANDLE, CUBLAS_HANDLE}, state, &err_gpu);

    urequire(output_err_gpu == state.error);
    output_err_gpu->to_host(output_err_host);

    uassert_equal(output_err_host[0], 123);
    uassert_equal(output_err_host[1], 234);
    uassert_equal(output_err_host[2], 345);
    uassert_equal(output_err_host[3], 456);
}

UTest(linear_layer_fwd_batched) {
    constexpr int inputs = 3;
    constexpr int outputs = 3;
    constexpr int batch_size = 4;
    dims_t dims = {.inputs = inputs, .outputs = outputs};
    ftype input_host[batch_size * inputs] = {0},
                                  output_host[batch_size * outputs] = {0};
    Tensor<ftype> input_gpu({batch_size, 1, inputs, 1}, {inputs, inputs, 1, 1});

    for (size_t i = 0; i < batch_size * inputs; ++i) {
        input_host[i] = i + 1;
    }

    input_gpu.from_host(input_host);

    LinearLayer linear_layer(inputs, outputs);
    LayerState<ftype> state;
    state.set_parameters(linear_layer.create_parameters());
    init_test_parameters(state, dims, 1);
    linear_layer.init({CUDNN_HANDLE, CUBLAS_HANDLE}, state,
                      {batch_size, 1, inputs, 1});

    Tensor<ftype> *output_gpu =
        linear_layer.fwd({CUDNN_HANDLE, CUBLAS_HANDLE}, state, &input_gpu);

    urequire(output_gpu == state.output);
    output_gpu->to_host(output_host);

    for (size_t i = 0; i < outputs; ++i) {
        uassert_equal(output_host[i], 7);
    }
    for (size_t i = 0; i < outputs; ++i) {
        uassert_equal(output_host[i + outputs], 16);
    }
    for (size_t i = 0; i < outputs; ++i) {
        uassert_equal(output_host[i + 2 * outputs], 25);
    }
    for (size_t i = 0; i < outputs; ++i) {
        uassert_equal(output_host[i + 3 * outputs], 34);
    }
}

UTest(linear_layer_bwd_batched) {
    constexpr int inputs = 4;
    constexpr int outputs = 3;
    constexpr int batch_size = 2;
    dims_t dims = {
        .inputs = inputs, .outputs = outputs, .batch_size = batch_size};
    ftype input_host[batch_size * inputs] = {1, 2, 3, 4, 5, 6, 7, 8};
    ftype input_err_host[batch_size * outputs] = {1, 10, 100, 100, 10, 1};
    ftype output_err_host[batch_size * inputs] = {0};
    ftype biases_gradient_host[outputs] = {0};
    ftype weights_gradient_host[outputs * outputs] = {0};
    Tensor<ftype> input_gpu({batch_size, 1, inputs, 1}, {inputs, inputs, 1, 1});
    Tensor<ftype> input_err_gpu({batch_size, 1, outputs, 1},
                                {outputs, outputs, 1, 1});

    // init input and output gpu buffers
    input_gpu.from_host(input_host);
    input_err_gpu.from_host(input_err_host);

    LinearLayer linear_layer(inputs, outputs);
    LayerState<ftype> state;
    state.set_parameters(linear_layer.create_parameters());
    init_test_parameters(state, dims);
    linear_layer.init({CUDNN_HANDLE, CUBLAS_HANDLE}, state,
                      {batch_size, 1, inputs, 1});

    linear_layer.fwd({CUDNN_HANDLE, CUBLAS_HANDLE}, state, &input_gpu);
    Tensor<ftype> *output_err_gpu =
        linear_layer.bwd({CUDNN_HANDLE, CUBLAS_HANDLE}, state, &input_err_gpu);

    urequire(output_err_gpu == state.error);
    output_err_gpu->to_host(output_err_host);

    uassert_equal(output_err_host[0], 321);
    uassert_equal(output_err_host[1], 432);
    uassert_equal(output_err_host[2], 543);
    uassert_equal(output_err_host[3], 654);

    uassert_equal(output_err_host[4], 123);
    uassert_equal(output_err_host[5], 234);
    uassert_equal(output_err_host[6], 345);
    uassert_equal(output_err_host[7], 456);

    state.gradients.biases->to_host(biases_gradient_host);
    for (size_t i = 0; i < outputs; ++i) {
        ftype sum = 0;
        for (size_t b = 0; b < batch_size; ++b) {
            sum += input_err_host[b * outputs + i];
        }
        ftype expected = sum / batch_size;
        uassert_float_equal(biases_gradient_host[i], expected, 1e-6);
    }
    state.gradients.weights->to_host(weights_gradient_host);
    for (size_t i = 0; i < outputs; ++i) {
        for (size_t j = 0; j < inputs; ++j) {
            ftype sum = 0;
            for (size_t b = 0; b < batch_size; ++b) {
                sum += input_err_host[b * outputs + i] *
                       input_host[b * inputs + j];
            }
            ftype expected = sum / batch_size;
            uassert_float_equal(weights_gradient_host[i * inputs + j], expected,
                                1e-6);
        }
    }
}

UTest(sigmoid_activation_fwd) {
    constexpr int outputs = 3;
    constexpr int inputs = 3;
    ftype input_host[inputs] = {1, 2, 3}, output_host[outputs] = {0};
    Tensor<ftype> input_gpu({1, 1, inputs, 1}, {inputs, inputs, 1, 1});

    input_gpu.from_host(input_host);

    SigmoidActivationLayer sigmoid_layer;
    LayerState<ftype> state;
    sigmoid_layer.init({CUDNN_HANDLE, CUBLAS_HANDLE}, state, {1, 1, inputs, 1});
    Tensor<ftype> *output_gpu =
        sigmoid_layer.fwd({CUDNN_HANDLE, CUBLAS_HANDLE}, state, &input_gpu);

    urequire(output_gpu != &input_gpu);
    output_gpu->to_host(output_host);

    for (size_t i = 0; i < outputs; ++i) {
        uassert_float_equal(output_host[i], sigmoid(input_host[i]), 1e-6);
    }
}

UTest(sigmoid_activation_bwd) {
    constexpr int outputs = 6;
    constexpr int inputs = 6;
    ftype input_host[inputs] = {1, 2, 3, 4, 5, 6},
          err_host[inputs] = {10, 10, 10, 10, 10, 10},
          output_host[outputs] = {0};
    Tensor<ftype> input_gpu({1, 1, inputs, 1}, {inputs, inputs, 1, 1});
    Tensor<ftype> err_gpu({1, 1, inputs, 1}, {inputs, inputs, 1, 1});

    input_gpu.from_host(input_host);
    err_gpu.from_host(err_host);

    SigmoidActivationLayer sigmoid_layer;
    LayerState<ftype> state;
    sigmoid_layer.init({CUDNN_HANDLE, CUBLAS_HANDLE}, state, {1, 1, inputs, 1});
    sigmoid_layer.fwd({CUDNN_HANDLE, CUBLAS_HANDLE}, state, &input_gpu);
    Tensor<ftype> *output_gpu =
        sigmoid_layer.bwd({CUDNN_HANDLE, CUBLAS_HANDLE}, state, &err_gpu);

    output_gpu->to_host(output_host);

    for (size_t i = 0; i < outputs; ++i) {
        uassert_float_equal(output_host[i],
                            err_host[i] * sigmoid_derivative(input_host[i]),
                            1e-6);
    }
}

UTest(sgd_optimizer) {
    constexpr int inputs = 3;
    constexpr int outputs = 2;
    constexpr ftype learning_rate = 0.001;
    ftype weights[inputs * outputs] = {1, 2, 3, 4, 4, 6};
    ftype weights_gradients[inputs * outputs] = {1, 1, 1, 1, 1, 1};
    ftype biases[outputs] = {1, 2};
    ftype biases_gradients[outputs] = {1, 1};
    tensor_dims_t weights_dims = {1, 1, inputs, outputs},
          biases_dims = {1, 1, outputs, 1};
    LayerState<ftype> state;
    SGDOptimizer optimizer_factory(learning_rate);

    state.parameters.weights = create_tensor<ftype>(weights_dims);
    state.parameters.biases = create_tensor<ftype>(biases_dims);
    state.gradients.weights = create_tensor<ftype>(weights_dims);
    state.gradients.biases = create_tensor<ftype>(biases_dims);

    CUDA_CHECK(memcpy_host_to_gpu(state.parameters.weights->data(), weights,
                                  inputs * outputs));
    CUDA_CHECK(memcpy_host_to_gpu(state.gradients.weights->data(),
                                  weights_gradients, inputs * outputs));
    CUDA_CHECK(
        memcpy_host_to_gpu(state.parameters.biases->data(), biases, outputs));
    CUDA_CHECK(memcpy_host_to_gpu(state.gradients.biases->data(),
                                  biases_gradients, outputs));

    auto sgd = optimizer_factory.create();
    sgd->optimize({CUDNN_HANDLE, CUBLAS_HANDLE}, state);

    ftype result_weights[inputs * outputs] = {0}, result_biases[outputs] = {0};
    CUDA_CHECK(memcpy_gpu_to_host(
        result_weights, state.parameters.weights->data(), outputs * inputs));
    CUDA_CHECK(memcpy_gpu_to_host(result_biases,
                                  state.parameters.biases->data(), outputs));

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

    CUDA_CHECK(memcpy_host_to_gpu(input_gpu.data(), input_host, inputs));

    graph.add_layer<LinearLayer>(inputs, outputs);
    graph.add_layer<SigmoidActivationLayer>();

    graph.build();

    auto state = graph.create_state();
    graph.init_state(state, {1, 1, inputs, 1});

    init_test_parameters(state->layers[0],
                         dims_t{.inputs = inputs, .outputs = outputs});

    graph.executeGraph(true);
    graph.pushData(std::make_shared<PredictionData<ftype>>(state, &input_gpu));
    Tensor<ftype> *output_gpu = graph.get<PredictionData<ftype>>()->input;
    graph.terminate();

    CUDA_CHECK(memcpy_gpu_to_host(output_host, output_gpu->data(), outputs));

    for (size_t i = 0; i < outputs; ++i) {
        ftype weights_input_bias = input_host[0] * (i + 1) +
                                   output_host[1] * (i + 2) +
                                   output_host[2] * (i + 3) + i + 1;
        ftype expected_value = sigmoid(weights_input_bias);
        uassert_float_equal(output_host[i], expected_value, 1e-6);
    }
}

UTest(training) {
    constexpr size_t nb_inputs = 28 * 28;
    constexpr ftype learning_rate = 0.01;
    constexpr ftype epochs = 1;
    NetworkGraph graph;
    MNISTLoader loader;

    DataSet<ftype> data_set =
        loader.load_ds("../data/mnist/train-labels-idx1-ubyte",
                       "../data/mnist/train-images-idx3-ubyte");
    defer(destroy_data_set(data_set));

    urequire(data_set.datas.size() == 60'000);

    graph.set_loss<QuadraticLoss>();
    graph.set_optimizer<SGDOptimizer>(3, learning_rate);

    graph.add_layer<LinearLayer>(nb_inputs, 32);
    graph.add_layer<SigmoidActivationLayer>();
    graph.cut_layer();
    graph.add_layer<LinearLayer>(32, 32);
    graph.add_layer<SigmoidActivationLayer>();
    graph.cut_layer();
    graph.add_layer<LinearLayer>(32, 10);
    graph.add_layer<SigmoidActivationLayer>();

    graph.build();

    auto state = graph.create_state();

    graph.init_state(state, {1, 1, nb_inputs, 1});

    graph.executeGraph(true);
    timer_start(training);
    graph.pushData(
        std::make_shared<TrainingData<ftype>>(state, data_set, epochs));
    (void)graph.get<TrainingData<ftype>>();
    timer_end(training);
    graph.terminate();

    timer_report_prec(training, milliseconds);

    graph.createDotFile("train.dot", hh::ColorScheme::EXECUTION,
                        hh::StructureOptions::QUEUE);
}

UTest(mnist) {
    constexpr ftype learning_rate = 0.001;
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

    graph.set_loss<QuadraticLoss>();
    graph.set_optimizer<SGDOptimizer>(1, learning_rate);

    graph.add_layer<ConvolutionLayer>(1, 20, 28, 28, 5, 5);
    graph.add_layer<PoolingLayer>(CUDNN_POOLING_MAX, 2, 2);
    graph.add_layer<LinearLayer>(12 * 12 * 20, 10);

    // graph.add_layer<LinearLayer>(28 * 28, 10);

    graph.add_layer<SigmoidActivationLayer>();

    graph.build();

    auto state = graph.create_state();
    graph.init_state(state, {1, 1, 28, 28});

    graph.executeGraph(true);

    INFO("Inference before training...");
    ftype accuracy_start = evaluate_mnist(graph, testing_set, state);

    INFO("start training (learning_rate = " << learning_rate
                                            << ", epochs = " << epochs << ")");
    timer_start(online_training);
    graph.pushData(
        std::make_shared<TrainingData<ftype>>(state, training_set, epochs));
    (void)graph.get<TrainingData<ftype>>();
    timer_end(online_training);
    graph.cleanGraph();

    timer_report_prec(online_training, milliseconds);

    INFO("Evaluate the model...");
    ftype accuracy_end = evaluate_mnist(graph, testing_set, state);

    graph.terminate();

    uassert(accuracy_end > accuracy_start);

    graph.createDotFile("train_mnist.dot", hh::ColorScheme::EXECUTION,
                        hh::StructureOptions::QUEUE);
}

UTest(mnist_batched) {
    constexpr ftype learning_rate = 0.001;
    constexpr size_t epochs = 10;
    constexpr size_t batch_size = 64;
    constexpr size_t test_batch_size = 1'000;
    MNISTLoader loader;
    BatchGenerator<ftype> batch_generator(0);

    DataSet<ftype> training_data =
        loader.load_ds("../data/mnist/train-labels-idx1-ubyte",
                       "../data/mnist/train-images-idx3-ubyte");
    DataSet<ftype> training_set =
        batch_generator.generate(training_data, batch_size);
    defer(destroy_data_set(training_set));
    destroy_data_set(training_data);
    DataSet<ftype> testing_set =
        loader.load_ds("../data/mnist/t10k-labels-idx1-ubyte",
                       "../data/mnist/t10k-images-idx3-ubyte", test_batch_size);
    defer(destroy_data_set(testing_set));

    NetworkGraph graph;

    graph.set_loss<QuadraticLoss>();
    graph.set_optimizer<SGDOptimizer>(1, learning_rate);

    graph.add_layer<ConvolutionLayer>(1, 20, 28, 28, 5, 5);
    graph.add_layer<PoolingLayer>(CUDNN_POOLING_MAX, 2, 2);
    graph.add_layer<LinearLayer>(12 * 12 * 20, 10);


    // graph.add_layer<LinearLayer>(28 * 28, 10);

    graph.add_layer<SigmoidActivationLayer>();

    graph.build();

    auto state = graph.create_state();

    graph.executeGraph(true);

    INFO("Inference before training...");
    graph.init_state(state, {test_batch_size, 1, 28, 28});
    ftype accuracy_start =
        evaluate_mnist(graph, testing_set, state, test_batch_size);

    graph.init_state(state, {batch_size, 1, 28, 28});

    INFO("start training (learning_rate = " << learning_rate
                                            << ", epochs = " << epochs << ")");
    timer_start(batch_training);
    graph.train(state, training_set, epochs);
    timer_end(batch_training);

    timer_report_prec(batch_training, milliseconds);

    INFO("Evaluate the model...");
    graph.init_state(state, {test_batch_size, 1, 28, 28});
    ftype accuracy_end =
        evaluate_mnist(graph, testing_set, state, test_batch_size);

    graph.terminate();

    uassert(accuracy_end > accuracy_start);

    graph.createDotFile("train_mnist_batch.dot", hh::ColorScheme::EXECUTION,
                        hh::StructureOptions::QUEUE);
}
