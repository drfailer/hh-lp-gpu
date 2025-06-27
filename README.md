# Hedgehog Learning Pipeline

## Requirements

- `cmake` (>= 3.16)
- `cuda-12.9`
- `cublas-12.9`
- `cudnn-9`

## Example

```cpp
NetworkGraph graph;

// configure the optimizer
graph.set_loss<QuadraticLoss>();
graph.set_optimizer<SGDOptimizer>(1, learning_rate);

// layers
graph.add_layer<ConvolutionLayer>(1, 20, 28, 28, 5, 5);
graph.add_layer<PoolingLayer>(CUDNN_POOLING_MAX, 2, 2);
graph.add_layer<LinearLayer>(12 * 12 * 20, 10);
graph.add_layer<SigmoidActivationLayer>();

// build the hedgehog graph
graph.build();

// create the state (allocate and initalize parameters)
auto state = graph.create_state();

// start Hedgehog
graph.executeGraph(true);

//  train the model
graph.init_state(state, {batch_size, 1, 28, 28});
graph.train(state, training_set, epochs);

// use the model
graph.init_state(state, {test_batch_size, 1, 28, 28});
Tensor<float> *result1 = graph.predict(state, test_data1);
Tensor<float> *result2 = graph.predict(state, test_data2);
// ...

graph.terminate();
```
