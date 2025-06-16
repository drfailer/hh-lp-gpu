#include <cuda_fp16.h>
#include <cudnn_graph.h>
#include <stdio.h>

/******************************************************************************/
/*                                cuda kernels                                */
/******************************************************************************/

/*
 * output = weights * input + biases
 * TODO: use the x axis to compute over the batch
 */
template <int BLOCK_SIZE, typename DataType>
__global__ void
_hhlpLinearForward(DataType const *weights, DataType const *biases,
                   DataType const *inputs, DataType *outputs, int nb_inputs,
                   int nb_outputs, int batch_size) {
    int batch_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int block_idx = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int output_idx = batch_idx * nb_outputs + block_idx;

    if (batch_idx >= batch_size || block_idx >= nb_outputs)
        return;

    DataType result = 0;
    DataType const *weights_row = &weights[block_idx * nb_inputs];
    DataType const *input = &inputs[batch_idx * nb_inputs];
    for (int i = 0; i < nb_inputs; ++i) {
        result += weights_row[i] * input[i];
    }
    outputs[output_idx] = result + biases[block_idx];
}

/*
 * biases_gradient = output_gradient
 */
template <int BLOCK_SIZE, typename DataType>
__global__ void _hhlpLinearBackwardBias(DataType const *output_gradient,
                                        DataType *biases_gradient,
                                        int nb_outputs, int batch_size) {
    int block_idx = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (block_idx >= nb_outputs)
        return;

    DataType result = 0;
    for (int b = 0; b < batch_size; ++b) {
        result += output_gradient[b * nb_outputs + block_idx];
    }
    biases_gradient[block_idx] = result / (DataType)batch_size;
}

/*
 * weights_gradient = output_gradient * inputT
 */
template <int BLOCK_SIZE, typename DataType>
__global__ void
_hhlpLinearBackwardWeights(DataType const *output_gradient,
                           DataType const *input, DataType *weights_gradient,
                           int nb_outputs, int nb_inputs, int batch_size) {
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (row >= nb_outputs || col >= nb_inputs)
        return;

    DataType result = 0;
    for (int b = 0; b < batch_size; ++b) {
        result +=
            output_gradient[b * nb_outputs + row] * input[b * nb_inputs + col];
    }
    weights_gradient[row * nb_inputs + col] = result / (DataType)batch_size;
}

/*
 * input_gradientT = output_gradientT * weights
 */
template <int BLOCK_SIZE, typename DataType>
__global__ void
_hhlpLinearBackwardData(DataType const *output_gradients,
                        DataType const *weights, DataType *input_gradient,
                        int nb_outputs, int nb_inputs, int batch_size) {
    int batch_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int block_idx = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int output_idx = batch_idx * nb_inputs + block_idx;

    if (batch_idx >= batch_size || block_idx >= nb_inputs)
        return;

    DataType result = 0;
    DataType const *weights_col = &weights[block_idx];
    DataType const *output_gradient = &output_gradients[batch_idx * nb_outputs];
    for (int i = 0; i < nb_outputs; ++i) {
        result += output_gradient[i] * weights_col[i * nb_inputs];
    }

    input_gradient[output_idx] = result;
}

/******************************************************************************/
/*                             external functions                             */
/******************************************************************************/

#define EXTERNAL_FUNCTION_IMPL(KERNEL, ...)                                    \
    cudnnStatus_t KERNEL(__VA_ARGS__) {                                        \
        cudaStream_t stream;                                                   \
        dim3 threads(1, std::min(nb_outputs, 32));                             \
        dim3 grid(1, std::max(1, nb_outputs / 32));                            \
                                                                               \
        cudnnGetStream(cudnn_handle, &stream);                                 \
        switch (data_type) {                                                   \
        case CUDNN_DATA_FLOAT:                                                 \
            KERNEL<32><<<grid, threads, 0, stream>>>(KERNEL_PARAMS(float));    \
            break;                                                             \
        case CUDNN_DATA_DOUBLE:                                                \
            KERNEL<32><<<grid, threads, 0, stream>>>(KERNEL_PARAMS(double));   \
            break;                                                             \
        case CUDNN_DATA_HALF:                                                  \
            KERNEL<32><<<grid, threads, 0, stream>>>(KERNEL_PARAMS(int16_t));  \
            break;                                                             \
        default:                                                               \
            return cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE;        \
            break;                                                             \
        }                                                                      \
        return cudnnStatus_t::CUDNN_STATUS_SUCCESS;                            \
    }

#define LAUNCH_KERNEL(target_type, kernel)                                     \
    {                                                                          \
        using type = target_type;                                              \
        kernel;                                                                \
    }
#define SWITCH_CUDNN_TYPE(data_type, kernel)                                   \
    switch (data_type) {                                                       \
    case CUDNN_DATA_FLOAT:                                                     \
        LAUNCH_KERNEL(float, kernel)                                           \
        break;                                                                 \
    case CUDNN_DATA_DOUBLE:                                                    \
        LAUNCH_KERNEL(double, kernel)                                          \
        break;                                                                 \
    case CUDNN_DATA_HALF:                                                      \
        LAUNCH_KERNEL(__half, kernel)                                          \
        break;                                                                 \
    default:                                                                   \
        return cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE;            \
        break;                                                                 \
    }

// TODO: it would be better to take tensor descriptor as argument instead
cudnnStatus_t hhlpLinearForward(cudnnHandle_t cudnn_handle, void const *weights,
                                void const *biases, void const *input,
                                void *output, int nb_inputs, int nb_outputs,
                                int batch_size, cudnnDataType_t data_type) {
    cudaStream_t stream;
    cudnnGetStream(cudnn_handle, &stream);

    dim3 threads(32, 32);
    dim3 grid(std::max<int>(1, batch_size / threads.x),
              std::max<int>(1, nb_outputs / threads.y));

    SWITCH_CUDNN_TYPE(
        data_type,
        (_hhlpLinearForward<32><<<grid, threads, 0, stream>>>(
            (type const *)weights, (type const *)biases, (type const *)input,
            (type *)output, nb_inputs, nb_outputs, batch_size)));
    return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t hhlpLinearBackwardBias(cudnnHandle_t cudnn_handle,
                                     void const *error, void *biases_gradient,
                                     int nb_outputs, int batch_size,
                                     cudnnDataType_t data_type) {
    cudaStream_t stream;
    cudnnGetStream(cudnn_handle, &stream);

    dim3 threads(1, 32);
    dim3 grid(1, (nb_outputs + threads.y - 1) / threads.y);

    SWITCH_CUDNN_TYPE(
        data_type, (_hhlpLinearBackwardBias<32><<<grid, threads, 0, stream>>>(
                       (type const *)error, (type *)biases_gradient, nb_outputs,
                       batch_size)));
    return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t hhlpLinearBackwardWeights(cudnnHandle_t cudnn_handle,
                                        void const *output_gradient,
                                        void const *input,
                                        void *weights_gradient, int nb_outputs,
                                        int nb_inputs, int batch_size,
                                        cudnnDataType_t data_type) {
    cudaStream_t stream;
    cudnnGetStream(cudnn_handle, &stream);

    dim3 threads(32, 32);
    dim3 grid((nb_inputs + threads.x - 1) / threads.x,
              (nb_outputs + threads.y - 1) / threads.y);

    SWITCH_CUDNN_TYPE(
        data_type,
        (_hhlpLinearBackwardWeights<4><<<grid, threads, 0, stream>>>(
            (type *)output_gradient, (type *)input, (type *)weights_gradient,
            nb_outputs, nb_inputs, batch_size)));
    return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t hhlpLinearBackwardData(cudnnHandle_t cudnn_handle,
                                     void const *output_gradient,
                                     void const *weights, void *input_gradient,
                                     int nb_outputs, int nb_inputs,
                                     int batch_size,
                                     cudnnDataType_t data_type) {
    cudaStream_t stream;
    cudnnGetStream(cudnn_handle, &stream);

    dim3 threads(32, 32);
    dim3 grid(std::max<int>(1, batch_size / threads.x),
              std::max<int>(1, nb_inputs / threads.y));

    SWITCH_CUDNN_TYPE(
        data_type,
        (_hhlpLinearBackwardData<32><<<grid, threads, 0, stream>>>(
            (type *)output_gradient, (type *)weights, (type *)input_gradient,
            nb_outputs, nb_inputs, batch_size)));
    return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
}
