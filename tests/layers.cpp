#include "layers.hpp"
#include "../src/graph/network_graph.hpp"
#include "../src/graph/distributed_network_graph.hpp"
#include "../src/model/layer/convolution_layer.hpp"
#include "../src/model/layer/linear_layer.hpp"
#include "../src/model/layer/pooling_layer.hpp"
#include "../src/model/layer/sigmoid_activation_layer.hpp"
#include "../src/model/loss/quadratic_loss.hpp"
#include "../src/model/optimizer/sgd_optimizer.hpp"
#include "../src/tools/gpu.hpp"
#include "../src/tools/timer.hpp"
#include "../tools/batch_generator.hpp"
#include "../tools/mnist/mnist_loader.hpp"
#include "utest.hpp"
#include <cstring>
#include <iostream>
#include <ostream>
#include <unistd.h>

template <typename T>
LayerData<T> init_layer_and_get_layer_data(auto layer,
                                           tensor::dims_t const &input_dims,
                                           bool bwd) {
    CUDA cuda{CUDNN_HANDLE, CUBLAS_HANDLE};
    LayerData<ftype> data;

    // create and init parameters
    auto param_shape = layer->parameters_shape();
    data.w = tensor::tensor<ftype>(param_shape.w);
    data.b = tensor::tensor<ftype>(param_shape.b);
    layer->init_parameters(cuda, {data.w, data.b});

    // data initialization
    auto io_shape = layer->io_shape(input_dims);

    // fwd init
    data.x = tensor::tensor_view<ftype>(io_shape.x, nullptr);
    data.y = tensor::tensor<ftype>(io_shape.y);
    layer->init_fwd(cuda, data);

    if (!bwd) {
        return data;
    }

    // bwd init
    data.dx = tensor::tensor<ftype>(io_shape.x);
    data.dy = tensor::tensor_view<ftype>(io_shape.y, nullptr);
    data.dw = tensor::tensor_like<ftype>(data.w);
    data.db = tensor::tensor_like<ftype>(data.b);
    layer->init_bwd(cuda, data);
    return data;
}

ftype sigmoid(ftype x) { return 1.0 / (1.0 + std::exp(-x)); }

ftype sigmoid_derivative(ftype x) { return sigmoid(x) * (1.0 - sigmoid(x)); }

void init_test_parameters(LayerData<ftype> &data, dims_t dims, ftype value) {
    int weights_size = dims.inputs * dims.outputs;
    std::vector<ftype> weights(weights_size, value);
    std::vector<ftype> biases(dims.outputs, value);
    data.w.from_host(weights.data());
    data.b.from_host(biases.data());
}

void init_test_parameters(LayerData<ftype> &data, dims_t dims) {
    std::vector<ftype> weights(dims.inputs * dims.outputs);
    std::vector<ftype> biases(dims.outputs);

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
    data.w.from_host(weights.data());
    data.b.from_host(biases.data());
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
                     std::shared_ptr<NetworkData<ftype>> &data,
                     int batch_size = 1) {
    int success = 0;
    int errors = 0;
    std::vector<ftype> expected(batch_size * 10, 0), found(batch_size * 10, 0);

    timer_start(evaluate_mnist);
    for (auto &test_data : testing_set.datas) {
        auto &output = graph.predict(data, test_data.input);
        CUDA_CHECK(test_data.ground_truth.to_host(expected.data()));
        CUDA_CHECK(output.to_host(found.data()));

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
    auto input_gpu = tensor::tensor<ftype>(1, 1, inputs, 1);

    input_gpu.from_host(input_host);

    LinearLayer layer(inputs, outputs);
    LayerData<ftype> ld =
        init_layer_and_get_layer_data<ftype>(&layer, input_gpu.dims(), false);
    init_test_parameters(ld, dims, 1);

    ld.x.data(input_gpu.data());
    layer.fwd({CUDNN_HANDLE, CUBLAS_HANDLE}, {ld.x, ld.w, ld.b}, {ld.y});
    ld.y.to_host(output_host);

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
    auto input_gpu = tensor::tensor<ftype>(1, 1, inputs, 1);
    auto err_gpu = tensor::tensor<ftype>(1, 1, inputs, 1);

    // init input and output gpu buffers
    input_gpu.from_host(input_host);
    err_gpu.from_host(input_err_host);

    LinearLayer layer(inputs, outputs);
    LayerData<ftype> ld =
        init_layer_and_get_layer_data<ftype>(&layer, input_gpu.dims(), true);
    init_test_parameters(ld, dims);

    ld.x.data(input_gpu.data());
    layer.fwd({CUDNN_HANDLE, CUBLAS_HANDLE}, {ld.x, ld.w, ld.b}, {ld.y});
    ld.dy.data(err_gpu.data());
    layer.bwd({CUDNN_HANDLE, CUBLAS_HANDLE}, {ld.dy, ld.x, ld.y, ld.w, ld.b},
              {ld.dx, ld.dw, ld.db});
    ld.dx.to_host(output_err_host);

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
    auto input_gpu = tensor::tensor<ftype>(batch_size, 1, inputs, 1);

    for (size_t i = 0; i < batch_size * inputs; ++i) {
        input_host[i] = i + 1;
    }

    input_gpu.from_host(input_host);

    LinearLayer layer(inputs, outputs);
    LayerData<ftype> ld =
        init_layer_and_get_layer_data<ftype>(&layer, input_gpu.dims(), false);
    init_test_parameters(ld, dims, 1);

    ld.x.data(input_gpu.data());
    layer.fwd({CUDNN_HANDLE, CUBLAS_HANDLE}, {ld.x, ld.w, ld.b}, {ld.y});
    ld.y.to_host(output_host);

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
    ftype weights_gradient_host[inputs * outputs] = {0};
    auto input_gpu = tensor::tensor<ftype>(batch_size, 1, inputs, 1);
    auto input_err_gpu = tensor::tensor<ftype>(batch_size, 1, outputs, 1);

    // init input and output gpu buffers
    input_gpu.from_host(input_host);
    input_err_gpu.from_host(input_err_host);

    LinearLayer layer(inputs, outputs);
    LayerData<ftype> ld =
        init_layer_and_get_layer_data<ftype>(&layer, input_gpu.dims(), true);
    init_test_parameters(ld, dims);

    ld.x.data(input_gpu.data());
    layer.fwd({CUDNN_HANDLE, CUBLAS_HANDLE}, {ld.x, ld.w, ld.b}, {ld.y});
    ld.dy.data(input_err_gpu.data());
    layer.bwd({CUDNN_HANDLE, CUBLAS_HANDLE}, {ld.dy, ld.x, ld.y, ld.w, ld.b},
              {ld.dx, ld.dw, ld.db});
    ld.dx.to_host(output_err_host);

    uassert_equal(output_err_host[0], 321);
    uassert_equal(output_err_host[1], 432);
    uassert_equal(output_err_host[2], 543);
    uassert_equal(output_err_host[3], 654);

    uassert_equal(output_err_host[4], 123);
    uassert_equal(output_err_host[5], 234);
    uassert_equal(output_err_host[6], 345);
    uassert_equal(output_err_host[7], 456);

    ld.db.to_host(biases_gradient_host);
    for (size_t i = 0; i < outputs; ++i) {
        ftype sum = 0;
        for (size_t b = 0; b < batch_size; ++b) {
            sum += input_err_host[b * outputs + i];
        }
        ftype expected = sum / batch_size;
        uassert_float_equal(biases_gradient_host[i], expected, 1e-6);
    }
    ld.dw.to_host(weights_gradient_host);
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
    auto input_gpu = tensor::tensor<ftype>(1, 1, inputs, 1);

    input_gpu.from_host(input_host);

    SigmoidActivationLayer layer;
    LayerData<ftype> ld =
        init_layer_and_get_layer_data<ftype>(&layer, input_gpu.dims(), true);

    ld.x.data(input_gpu.data());
    layer.fwd({CUDNN_HANDLE, CUBLAS_HANDLE}, {ld.x, ld.w, ld.b}, {ld.y});
    ld.y.to_host(output_host);

    for (size_t i = 0; i < outputs; ++i) {
        uassert_float_equal(output_host[i], sigmoid(input_host[i]), 1e-6);
    }
}

UTest(sigmoid_activation_bwd) {
    constexpr int outputs = 6;
    constexpr int inputs = 6;
    ftype input_host[inputs] = {1, 2, 3, 4, 5, 6},
          input_err_host[inputs] = {10, 10, 10, 10, 10, 10},
          output_host[outputs] = {0};
    auto input_gpu = tensor::tensor<ftype>(1, 1, inputs, 1);
    auto input_err_gpu = tensor::tensor<ftype>(1, 1, inputs, 1);

    input_gpu.from_host(input_host);
    input_err_gpu.from_host(input_err_host);

    SigmoidActivationLayer layer;
    LayerData<ftype> ld =
        init_layer_and_get_layer_data<ftype>(&layer, input_gpu.dims(), true);

    ld.x.data(input_gpu.data());
    layer.fwd({CUDNN_HANDLE, CUBLAS_HANDLE}, {ld.x, ld.w, ld.b}, {ld.y});
    ld.dy.data(input_err_gpu.data());
    layer.bwd({CUDNN_HANDLE, CUBLAS_HANDLE}, {ld.dy, ld.x, ld.y, ld.w, ld.b},
              {ld.dx, ld.dw, ld.db});
    ld.dx.to_host(output_host);

    for (size_t i = 0; i < outputs; ++i) {
        uassert_float_equal(
            output_host[i],
            input_err_host[i] * sigmoid_derivative(input_host[i]), 1e-6);
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
    tensor::dims_t weights_dims = {1, 1, inputs, outputs},
                   biases_dims = {1, 1, outputs, 1};
    LayerData<ftype> ld;
    SGDOptimizer optimizer_factory(learning_rate);

    ld.w = tensor::tensor<ftype>(weights_dims);
    ld.b = tensor::tensor<ftype>(biases_dims);
    ld.dw = tensor::tensor<ftype>(weights_dims);
    ld.db = tensor::tensor<ftype>(biases_dims);

    CUDA_CHECK(ld.w.from_host(weights));
    CUDA_CHECK(ld.dw.from_host(weights_gradients));
    CUDA_CHECK(ld.b.from_host(biases));
    CUDA_CHECK(ld.db.from_host(biases_gradients));

    auto sgd = optimizer_factory.copy();
    sgd->optimize({CUDNN_HANDLE, CUBLAS_HANDLE}, {ld.dw, ld.db}, {ld.w, ld.b});

    ftype result_weights[inputs * outputs] = {0}, result_biases[outputs] = {0};
    CUDA_CHECK(ld.w.to_host(result_weights));
    CUDA_CHECK(ld.b.to_host(result_biases));

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
    auto input_gpu = tensor::tensor<ftype>(1, 1, inputs, 1);
    NetworkGraph graph;

    CUDA_CHECK(memcpy_host_to_gpu(input_gpu.data(), input_host, inputs));

    graph.add_layer<LinearLayer>(inputs, outputs);
    graph.add_layer<SigmoidActivationLayer>();

    graph.build();
    graph.executeGraph(true);

    auto data = graph.init_parameters();
    graph.init(data, {1, 1, inputs, 1});

    init_test_parameters(data->layers_datas[0],
                         dims_t{.inputs = inputs, .outputs = outputs});

    graph.pushData(std::make_shared<PredictionData<ftype>>(data, &input_gpu));
    auto output_gpu = graph.get<PredictionData<ftype>>()->input;
    graph.terminate();

    output_gpu->to_host(output_host);

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

    urequire(data_set.datas.size() == 60'000);

    graph.set_loss<QuadraticLoss>();
    graph.set_optimizer<SGDOptimizer>(3, learning_rate);

    graph.add_layer<LinearLayer>(nb_inputs, 32);
    graph.add_layer<SigmoidActivationLayer>();
    graph.add_layer<LinearLayer>(32, 32);
    graph.add_layer<SigmoidActivationLayer>();
    graph.add_layer<LinearLayer>(32, 10);
    graph.add_layer<SigmoidActivationLayer>();

    graph.build();
    graph.executeGraph(true);

    auto data = graph.init_parameters();
    graph.init(data, {1, 1, nb_inputs, 1});

    timer_start(training);
    graph.pushData(
        std::make_shared<TrainingData<ftype>>(data, data_set, epochs));
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
    DataSet<ftype> testing_set =
        loader.load_ds("../data/mnist/t10k-labels-idx1-ubyte",
                       "../data/mnist/t10k-images-idx3-ubyte");

    NetworkGraph graph;

    graph.set_loss<QuadraticLoss>();
    graph.set_optimizer<SGDOptimizer>(1, learning_rate);

    graph.add_layer<ConvolutionLayer>(1, 20, 28, 28, 5, 5);
    graph.add_layer<PoolingLayer>(CUDNN_POOLING_MAX, 2, 2);
    graph.add_layer<LinearLayer>(12 * 12 * 20, 10);
    // graph.add_layer<LinearLayer>(28 * 28, 10);

    graph.add_layer<SigmoidActivationLayer>();

    graph.build();
    graph.executeGraph(true);

    auto data = graph.init_parameters();
    graph.init(data, {1, 1, 28, 28});

    INFO("Inference before training...");
    ftype accuracy_start = evaluate_mnist(graph, testing_set, data);

    INFO("start training (learning_rate = " << learning_rate
                                            << ", epochs = " << epochs << ")");
    timer_start(online_training);
    graph.pushData(
        std::make_shared<TrainingData<ftype>>(data, training_set, epochs));
    (void)graph.get<TrainingData<ftype>>();
    timer_end(online_training);
    graph.cleanGraph();

    timer_report_prec(online_training, milliseconds);

    INFO("Evaluate the model...");
    ftype accuracy_end = evaluate_mnist(graph, testing_set, data);

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
        batch_generator.generate(std::move(training_data), batch_size);
    DataSet<ftype> testing_set =
        loader.load_ds("../data/mnist/t10k-labels-idx1-ubyte",
                       "../data/mnist/t10k-images-idx3-ubyte", test_batch_size);

    NetworkGraph graph;

    graph.set_loss<QuadraticLoss>();
    graph.set_optimizer<SGDOptimizer>(1, learning_rate);

    graph.add_layer<ConvolutionLayer>(1, 20, 28, 28, 5, 5);
    graph.add_layer<PoolingLayer>(CUDNN_POOLING_MAX, 2, 2);
    graph.add_layer<LinearLayer>(12 * 12 * 20, 10);
    // graph.add_layer<LinearLayer>(28 * 28, 10);

    graph.add_layer<SigmoidActivationLayer>();

    graph.build();
    graph.executeGraph(true);

    auto data = graph.init_parameters();

    INFO("Inference before training...");
    graph.init(data, {test_batch_size, 1, 28, 28});
    ftype accuracy_start =
        evaluate_mnist(graph, testing_set, data, test_batch_size);

    graph.init(data, {batch_size, 1, 28, 28});

    INFO("start training (learning_rate = " << learning_rate
                                            << ", epochs = " << epochs << ")");
    timer_start(batch_training);
    graph.train(data, training_set, epochs);
    timer_end(batch_training);

    timer_report_prec(batch_training, milliseconds);

    INFO("Evaluate the model...");
    graph.init(data, {test_batch_size, 1, 28, 28});
    ftype accuracy_end =
        evaluate_mnist(graph, testing_set, data, test_batch_size);

    graph.terminate();

    uassert(accuracy_end > accuracy_start);

    graph.createDotFile("train_mnist_batch_single_node.dot", hh::ColorScheme::EXECUTION,
                        hh::StructureOptions::QUEUE);
}

UTestArgs(mnist_multi_node, CommService *service) {
    constexpr ftype learning_rate = 0.001;
    constexpr size_t epochs = 10;
    constexpr size_t batch_size = 64;
    constexpr size_t test_batch_size = 1'000;
    MNISTLoader loader;
    BatchGenerator<ftype> batch_generator(0);
    DataSet<ftype> training_data, training_set, testing_set;

    // if (service->rank() == 0) {
    //     training_data = loader.load_ds("../data/mnist/train-labels-idx1-ubyte",
    //                                    "../data/mnist/train-images-idx3-ubyte");
    //     training_set = batch_generator.generate(std::move(training_data), batch_size);
    //     testing_set = loader.load_ds("../data/mnist/t10k-labels-idx1-ubyte",
    //                                  "../data/mnist/t10k-images-idx3-ubyte",
    //                                  test_batch_size);
    // }

    DistributedNetworkGraph graph(service);

    graph.set_loss<QuadraticLoss>();
    graph.set_optimizer<SGDOptimizer>(1, learning_rate);

    graph.add_layer<ConvolutionLayer>(1, 20, 28, 28, 5, 5);
    graph.add_layer<PoolingLayer>(CUDNN_POOLING_MAX, 2, 2);
    graph.cut_layer();
    graph.add_layer<LinearLayer>(12 * 12 * 20, 64);
    graph.add_layer<SigmoidActivationLayer>();
    graph.cut_layer();
    graph.add_layer<LinearLayer>(64, 10);
    graph.add_layer<SigmoidActivationLayer>();

    graph.build();
    graph.executeGraph(true);

    std::cout << "initalizing parameters" << std::endl;
    auto data = graph.init_parameters();
    std::cout << "parameters initialized" << std::endl;

    INFO("Inference before training...");
    graph.init(data, {test_batch_size, 1, 28, 28});
    // ftype accuracy_start =
    //     evaluate_mnist(graph, testing_set, data, test_batch_size);
    //
    //  graph.init(data, {batch_size, 1, 28, 28});

    // INFO("start training (learning_rate = " << learning_rate
    //                                         << ", epochs = " << epochs << ")");
    // timer_start(batch_training);
    // graph.train(data, training_set, epochs);
    // timer_end(batch_training);
    //
    // timer_report_prec(batch_training, milliseconds);
    //
    // INFO("Evaluate the model...");
    // graph.init(data, {test_batch_size, 1, 28, 28});
    // ftype accuracy_end =
    //     evaluate_mnist(graph, testing_set, data, test_batch_size);

    service->barrier();
    graph.terminate();
    std::cout << "graph terminated" << std::endl;

    // uassert(accuracy_end > accuracy_start);

    std::ostringstream oss;
    oss << "train_mnist_batch_multinode_" << service->rank() << ".dot";
    graph.createDotFile(oss.str(), hh::ColorScheme::EXECUTION,
            hh::StructureOptions::QUEUE);
}
