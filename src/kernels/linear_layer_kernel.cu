#include <cuda_fp16.h>
#include <cudnn_graph.h>
#include <stdio.h>

// clang-format off
#define REDUCE(BS, T, SMEM, tid)\
    if (BS >= 512) { if (tid < 256) { SMEM[tid] += SMEM[tid + 256]; } __syncthreads(); }\
    if (BS >= 256) { if (tid < 128) { SMEM[tid] += SMEM[tid + 128]; } __syncthreads(); }\
    if (BS >= 128) { if (tid < 64) { SMEM[tid] += SMEM[tid + 64]; } __syncthreads(); }\
    warp_reduce<BS, T>(SMEM, tid);
// clang-format on

/******************************************************************************/
/*                                cuda kernels                                */
/******************************************************************************/

__device__ __half operator+=(__half volatile &lhs, __half volatile const &rhs) {
    return (__half &)lhs += (const __half &)rhs;
}

template <unsigned int BLOCK_SIZE, typename DataType>
__device__ void warp_reduce(volatile DataType *shared_data, unsigned int tid) {
    // clang-format off
    if (BLOCK_SIZE >= 64) { if (tid < 32) shared_data[tid] += shared_data[tid + 32]; }
    if (BLOCK_SIZE >= 32) { if (tid < 16) shared_data[tid] += shared_data[tid + 16]; }
    if (BLOCK_SIZE >= 16) { if (tid < 8)  shared_data[tid] += shared_data[tid + 8]; }
    if (BLOCK_SIZE >= 8) { if (tid < 4)  shared_data[tid] += shared_data[tid + 4]; }
    if (BLOCK_SIZE >= 4) { if (tid < 2)  shared_data[tid] += shared_data[tid + 2]; }
    if (BLOCK_SIZE >= 2) { if (tid < 1)  shared_data[tid] += shared_data[tid + 1]; }
    // clang-format on
}

/*
 * output = weights * input + biases
 *
 * blockIdx.x -> output row
 * threadIdx.y -> batch
 * threadIdx.x -> partial sum over the row
 *
 * source: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 */
template <unsigned int REDUCE_BLOCK_SIZE, unsigned int BATCH_BLOCK_SIZE,
          typename DataType>
__global__ void
_hhlpLinearForward(DataType const *weights, DataType const *biases,
                   DataType const *inputs, DataType *outputs, int nb_inputs,
                   int nb_outputs, int batch_size) {
    int batch_idx = blockIdx.y * BATCH_BLOCK_SIZE + threadIdx.y;
    unsigned int tid = threadIdx.x;

    if (batch_idx >= batch_size)
        return;

    __shared__ DataType results[BATCH_BLOCK_SIZE * REDUCE_BLOCK_SIZE];

    DataType *result = &results[threadIdx.y * REDUCE_BLOCK_SIZE];
    DataType const *weights_row = &weights[blockIdx.x * nb_inputs];
    DataType const *input = &inputs[batch_idx * nb_inputs];

    result[tid] = 0;
    for (int idx = tid; idx < nb_inputs; idx += REDUCE_BLOCK_SIZE) {
        DataType w = weights_row[idx];
        DataType i = input[idx];
        result[tid] += w * i;
    }
    __syncthreads();

    REDUCE(REDUCE_BLOCK_SIZE, DataType, result, tid);

    if (tid == 0) {
        outputs[batch_idx * nb_outputs + blockIdx.x] =
            result[0] + biases[blockIdx.x];
    }
}

/*
 * biases_gradient = output_gradient
 * TODO:
 * - is it better to load 2 elements from the same column?
 */
template <unsigned int BATCH_BLOCK_SIZE, unsigned int OUTPUT_BLOCK_SIZE,
          unsigned int THREAD_BLOCK_SIZE, typename DataType>
__global__ void _hhlpLinearBackwardBias(DataType const *output_gradient,
                                        DataType *biases_gradient,
                                        int nb_outputs, int batch_size) {
    __shared__ DataType
        results[OUTPUT_BLOCK_SIZE][THREAD_BLOCK_SIZE][BATCH_BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned output_idx =
        blockIdx.y * OUTPUT_BLOCK_SIZE + THREAD_BLOCK_SIZE * threadIdx.y;

    auto result = results[output_idx];
#pragma unroll
    for (int t = 0; t < THREAD_BLOCK_SIZE; ++t) {
        result[t][tid] = 0;
    }
    for (int idx = tid; idx < batch_size; idx += BATCH_BLOCK_SIZE) {
#pragma unroll
        for (int t = 0; t < THREAD_BLOCK_SIZE; ++t) {
            if (output_idx + t < nb_outputs)
                result[t][tid] +=
                    output_gradient[idx * nb_outputs + output_idx + t];
        }
    }
    __syncthreads();

#pragma unroll
    for (int t = 0; t < THREAD_BLOCK_SIZE; ++t) {
        REDUCE(BATCH_BLOCK_SIZE, DataType, result[t], tid);
    }

    if (tid == 0) {
#pragma unroll
        for (int t = 0; t < THREAD_BLOCK_SIZE; ++t) {
            if (output_idx + t < nb_outputs)
                biases_gradient[output_idx + t] =
                    result[t][0] / (DataType)batch_size;
        }
    }
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
    int batch_idx = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int block_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
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

#define CEIL_DIV(ttl_size, block_size) (ttl_size + block_size - 1) / block_size

// TODO: it would be better to take tensor descriptor as argument instead
cudnnStatus_t hhlpLinearForward(cudnnHandle_t cudnn_handle, void const *weights,
                                void const *biases, void const *input,
                                void *output, int nb_inputs, int nb_outputs,
                                int batch_size, cudnnDataType_t data_type) {
    cudaStream_t stream;
    cudnnGetStream(cudnn_handle, &stream);

    constexpr unsigned int batch_threads = 32;
    constexpr unsigned int reduce_threads = 32;

    // TODO: compute the number of threads using the warp tile size
    dim3 threads(reduce_threads, batch_threads, 1);
    dim3 grid(nb_outputs, CEIL_DIV(batch_size, threads.y), 1);

    SWITCH_CUDNN_TYPE(
        data_type,
        (_hhlpLinearForward<reduce_threads, batch_threads>
         <<<grid, threads, 0, stream>>>(
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

    if (batch_size == 1) {
        SWITCH_CUDNN_TYPE(data_type, cudaMemcpy(biases_gradient, error,
                                                nb_outputs * sizeof(type),
                                                cudaMemcpyDeviceToDevice));
    } else {
        constexpr unsigned int batch_block_size = 32;
        constexpr unsigned int output_block_size = 64;
        constexpr unsigned int thread_block_size = 2;
        dim3 threads(batch_block_size, output_block_size / thread_block_size);
        dim3 grid(1, CEIL_DIV(nb_outputs, threads.y));

        SWITCH_CUDNN_TYPE(
            data_type,
            (_hhlpLinearBackwardBias<batch_block_size, output_block_size,
                                     thread_block_size>
             <<<grid, threads, 0, stream>>>((type const *)error,
                                            (type *)biases_gradient, nb_outputs,
                                            batch_size)));
    }
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
    dim3 grid(CEIL_DIV(nb_inputs, threads.x), CEIL_DIV(nb_outputs, threads.y));

    SWITCH_CUDNN_TYPE(
        data_type,
        (_hhlpLinearBackwardWeights<32><<<grid, threads, 0, stream>>>(
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
    dim3 grid(CEIL_DIV(nb_inputs, threads.x), CEIL_DIV(batch_size, threads.y));

    SWITCH_CUDNN_TYPE(
        data_type,
        (_hhlpLinearBackwardData<32><<<grid, threads, 0, stream>>>(
            (type *)output_gradient, (type *)weights, (type *)input_gradient,
            nb_outputs, nb_inputs, batch_size)));
    return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
}
