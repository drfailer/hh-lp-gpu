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
__global__ void hhlpLinearForwardKernel(DataType const *weights,
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
__global__ void hhlpLinearBackwardBiasKernel(DataType const *error,
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
__global__ void hhlpLinearBackwardWeightsKernel(DataType const *output_gradient,
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
__global__ void hhlpLinearBackwardDataKernel(DataType const *output_gradient,
                                             DataType const *weights,
                                             DataType *input_gradient,
                                             int nb_outputs, int nb_inputs) {
    int block_idx = blockIdx.y * BLOCK_SIZE;

    if (block_idx + threadIdx.y >= nb_inputs)
        return;

    __shared__ DataType output_gradient_block[BLOCK_SIZE];
    DataType result = 0;

    // TODO: simplify this
    for (int i = 0; i < nb_outputs; i += BLOCK_SIZE) {
        if (i + threadIdx.y < nb_outputs) {
            output_gradient_block[threadIdx.y] =
                output_gradient[i + threadIdx.y];
        }
        __syncthreads();

        for (int j = 0; j < BLOCK_SIZE && i + j < nb_outputs; ++j) {
            result += output_gradient_block[j] *
                      weights[(i + j) * nb_inputs + threadIdx.y];
        }
        __syncthreads();
    }

    input_gradient[block_idx + threadIdx.y] = result;
}

/******************************************************************************/
/*                             external functions                             */
/******************************************************************************/

#define EXTERNAL_FUNCTION_IMPL(KERNEL, ...)                                    \
    cudnnStatus_t KERNEL(__VA_ARGS__) {                                        \
        cudaStream_t stream;                                                   \
        dim3 threads(1, 32);                                                   \
        dim3 grid(1, nb_outputs / 32);                                         \
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

// TODO: it would be better to take tensor descriptor as argument instead
cudnnStatus_t hhlpLinearForward(cudnnHandle_t cudnn_handle, void const *weights,
                                void const *biases, void const *input,
                                void *output, int nb_inputs, int nb_outputs,
                                cudnnDataType_t data_type) {
    cudaStream_t stream;
    cudnnGetStream(cudnn_handle, &stream);

    dim3 threads(1, std::min(nb_outputs, 32));
    dim3 grid(1, std::max(1, nb_outputs / 32));

    switch (data_type) {
    case CUDNN_DATA_FLOAT:
        hhlpLinearForwardKernel<32><<<grid, threads, 0, stream>>>(
            (float const *)weights, (float const *)biases, (float const *)input,
            (float *)output, nb_inputs, nb_outputs);
        break;
    case CUDNN_DATA_DOUBLE:
        hhlpLinearForwardKernel<32><<<grid, threads, 0, stream>>>(
            (double *)weights, (double *)biases, (double *)input,
            (double *)output, nb_inputs, nb_outputs);
        break;
    case CUDNN_DATA_HALF:
        hhlpLinearForwardKernel<32><<<grid, threads, 0, stream>>>(
            (int16_t *)weights, (int16_t *)biases, (int16_t *)input,
            (int16_t *)output, nb_inputs, nb_outputs);
        break;
    default:
        return cudnnStatus_t ::CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE;
        break;
    }
    return cudnnStatus_t ::CUDNN_STATUS_SUCCESS;
}

#define KERNEL_PARAMS(type) (type *)error, (type *)biases_gradient, nb_outputs
EXTERNAL_FUNCTION_IMPL(hhlpLinearBackwardBiasKernel, cudnnHandle_t cudnn_handle,
                       void const *error, void *biases_gradient, int nb_outputs,
                       cudnnDataType_t data_type)
#undef KERNEL_PARAMS

#define KERNEL_PARAMS(type)                                                    \
    (type *)output_gradient, (type *)input, (type *)weights_gradient,          \
        nb_outputs, nb_inputs
EXTERNAL_FUNCTION_IMPL(hhlpLinearBackwardWeightsKernel,
                       cudnnHandle_t cudnn_handle, void const *output_gradient,
                       void const *input, void *weights_gradient,
                       int nb_outputs, int nb_inputs, cudnnDataType_t data_type)
#undef KERNEL_PARAMS

#define KERNEL_PARAMS(type)                                                    \
    (type *)output_gradient, (type *)weights, (type *)input_gradient,          \
        nb_outputs, nb_inputs
EXTERNAL_FUNCTION_IMPL(hhlpLinearBackwardDataKernel, cudnnHandle_t cudnn_handle,
                       void const *output_gradient, void const *weights,
                       void *input_gradient, int nb_outputs, int nb_inputs,
                       cudnnDataType_t data_type)
#undef KERNEL_PARAMS
