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
__global__ void _hhlpLinearForward(DataType const *weights,
                                   DataType const *biases,
                                   DataType const *input, DataType *output,
                                   int nb_inputs, int nb_outputs) {
    int output_idx = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (output_idx >= nb_outputs)
        return;

    DataType result = 0;
    DataType const *weights_row = &weights[output_idx * nb_inputs];
    for (int i = 0; i < nb_inputs; ++i) {
        result += weights_row[i] * input[i];
    }
    output[output_idx] = result + biases[output_idx];
}

/*
 * biases_gradient = error
 */
template <int BLOCK_SIZE, typename DataType>
__global__ void _hhlpLinearBackwardBias(DataType const *error,
                                        DataType *biases_gradient,
                                        int nb_outputs) {
    int idx = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (idx >= nb_outputs)
        return;

    biases_gradient[idx] = error[idx];
}

/*
 * weights_gradient = output_gradient * inputT
 */
template <int BLOCK_SIZE, typename DataType>
__global__ void _hhlpLinearBackwardWeights(DataType const *output_gradient,
                                           DataType const *input,
                                           DataType *weights_gradient,
                                           int nb_outputs, int nb_inputs) {
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (row >= nb_outputs || col >= nb_inputs)
        return;

    weights_gradient[row * nb_inputs + col] = output_gradient[row] * input[col];
}

/*
 * input_gradientT = output_gradientT * weights
 */
template <int BLOCK_SIZE, typename DataType>
__global__ void _hhlpLinearBackwardData(DataType const *output_gradient,
                                        DataType const *weights,
                                        DataType *input_gradient,
                                        int nb_outputs, int nb_inputs) {
    int output_idx = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (output_idx >= nb_inputs)
        return;

    DataType result = 0;
    DataType const *weights_col = &weights[output_idx];
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
        LAUNCH_KERNEL(int16_t, kernel)                                         \
        break;                                                                 \
    default:                                                                   \
        return cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE;            \
        break;                                                                 \
    }

// TODO: it would be better to take tensor descriptor as argument instead
cudnnStatus_t hhlpLinearForward(cudnnHandle_t cudnn_handle, void const *weights,
                                void const *biases, void const *input,
                                void *output, int nb_inputs, int nb_outputs,
                                cudnnDataType_t data_type) {
    cudaStream_t stream;
    cudnnGetStream(cudnn_handle, &stream);

    dim3 threads(1, std::min(nb_outputs, 32));
    dim3 grid(1, std::max(1, nb_outputs / 32));

    SWITCH_CUDNN_TYPE(
        data_type,
        (_hhlpLinearForward<32><<<grid, threads, 0, stream>>>(
            (type const *)weights, (type const *)biases, (type const *)input,
            (type *)output, nb_inputs, nb_outputs)));
    return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t hhlpLinearBackwardBias(cudnnHandle_t cudnn_handle,
                                     void const *error, void *biases_gradient,
                                     int nb_outputs,
                                     cudnnDataType_t data_type) {
    cudaStream_t stream;
    cudnnGetStream(cudnn_handle, &stream);

    dim3 threads(1, std::min(nb_outputs, 32));
    dim3 grid(1, std::max(1, nb_outputs / 32));

    SWITCH_CUDNN_TYPE(
        data_type, (_hhlpLinearBackwardBias<32><<<grid, threads, 0, stream>>>(
                       (type *)error, (type *)biases_gradient, nb_outputs)));
    return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t hhlpLinearBackwardWeights(cudnnHandle_t cudnn_handle,
                                        void const *output_gradient,
                                        void const *input,
                                        void *weights_gradient, int nb_outputs,
                                        int nb_inputs,
                                        cudnnDataType_t data_type) {
    cudaStream_t stream;
    cudnnGetStream(cudnn_handle, &stream);

    dim3 threads(std::min(nb_inputs, 32), std::min(nb_outputs, 32));
    dim3 grid(std::max(1, nb_inputs / 32), std::max(1, nb_outputs / 32));

    SWITCH_CUDNN_TYPE(
        data_type,
        (_hhlpLinearBackwardWeights<32><<<grid, threads, 0, stream>>>(
            (type *)output_gradient, (type *)input, (type *)weights_gradient,
            nb_outputs, nb_inputs)));
    return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t hhlpLinearBackwardData(cudnnHandle_t cudnn_handle,
                                     void const *output_gradient,
                                     void const *weights, void *input_gradient,
                                     int nb_outputs, int nb_inputs,
                                     cudnnDataType_t data_type) {
    cudaStream_t stream;
    cudnnGetStream(cudnn_handle, &stream);

    dim3 threads(1, std::min(nb_inputs, 32));
    dim3 grid(1, std::max(1, nb_inputs / 32));

    SWITCH_CUDNN_TYPE(
        data_type, (_hhlpLinearBackwardData<32><<<grid, threads, 0, stream>>>(
                       (type *)output_gradient, (type *)weights,
                       (type *)input_gradient, nb_outputs, nb_inputs)));
    return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
}
