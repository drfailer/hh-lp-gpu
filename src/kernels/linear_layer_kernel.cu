#include "linear_layer_kernel.h"
#include <cuda_fp16.h>
#include <cudnn_graph.h>

#define BREAK_IF(block_cond, cond)                                             \
    if constexpr (block_cond) {                                                \
        if (cond) {                                                            \
            break;                                                             \
        }                                                                      \
    }
#define BREAK_IF_BLOCK(block_size, cond) BREAK_IF(block_size > 1, cond)

// clang-format off
#define REDUCE(BS, T, SMEM, tid)                                                                   \
    if constexpr (BS >= 512) { if (tid < 256) { SMEM[tid] += SMEM[tid + 256]; } __syncthreads(); } \
    if constexpr (BS >= 256) { if (tid < 128) { SMEM[tid] += SMEM[tid + 128]; } __syncthreads(); } \
    if constexpr (BS >= 128) { if (tid < 64) { SMEM[tid] += SMEM[tid + 64]; } __syncthreads(); }   \
    warp_reduce<BS, T>(SMEM, tid);
// clang-format on

__device__ __half operator+=(__half volatile &lhs, __half volatile const &rhs) {
    return (__half &)lhs += (const __half &)rhs;
}

template <u32 BLOCK_SIZE, typename DataType>
__device__ void warp_reduce(volatile DataType *shared_data, u32 tid) {
    // clang-format off
    if constexpr (BLOCK_SIZE >= 64) { if (tid < 32) shared_data[tid] += shared_data[tid + 32]; }
    if constexpr (BLOCK_SIZE >= 32) { if (tid < 16) shared_data[tid] += shared_data[tid + 16]; }
    if constexpr (BLOCK_SIZE >= 16) { if (tid < 8)  shared_data[tid] += shared_data[tid + 8]; }
    if constexpr (BLOCK_SIZE >= 8) { if (tid < 4)  shared_data[tid] += shared_data[tid + 4]; }
    if constexpr (BLOCK_SIZE >= 4) { if (tid < 2)  shared_data[tid] += shared_data[tid + 2]; }
    if constexpr (BLOCK_SIZE >= 2) { if (tid < 1)  shared_data[tid] += shared_data[tid + 1]; }
    // clang-format on
}

/******************************************************************************/
/*                                cuda kernels                                */
/******************************************************************************/

/*
 * output = weights * input + biases
 *
 * blockIdx.x -> output row
 * threadIdx.y -> batch
 * threadIdx.x -> partial sum over the row
 *
 * source: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 */
template <u32 BATCH_BLOCK_SIZE, u32 REDUCE_BLOCK_SIZE, typename DataType>
__global__ void
_hhlpLinearForward(DataType const *weights, DataType const *biases,
                   DataType const *inputs, DataType *outputs, u32 nb_inputs,
                   u32 nb_outputs, u32 batch_size) {
    u32 batch_idx = blockIdx.y * BATCH_BLOCK_SIZE + threadIdx.y;
    u32 tid = threadIdx.x;

    if (batch_idx >= batch_size)
        return;

    __shared__ DataType results[BATCH_BLOCK_SIZE * REDUCE_BLOCK_SIZE];

    DataType *result = &results[threadIdx.y * REDUCE_BLOCK_SIZE];
    DataType const *weights_row = &weights[blockIdx.x * nb_inputs];
    DataType const *input = &inputs[batch_idx * nb_inputs];

    result[tid] = 0;
    for (u32 idx = tid; idx < nb_inputs; idx += REDUCE_BLOCK_SIZE) {
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
 */
template <u32 BATCH_BLOCK_SIZE, u32 OUTPUT_BLOCK_SIZE, u32 THREAD_BLOCK_SIZE,
          typename DataType>
__global__ void _hhlpLinearBackwardBias(DataType const *output_gradient,
                                        DataType *biases_gradient,
                                        u32 nb_outputs, u32 batch_size) {
    __shared__ DataType
        results[OUTPUT_BLOCK_SIZE][THREAD_BLOCK_SIZE][BATCH_BLOCK_SIZE];
    u32 tid = threadIdx.x;
    unsigned output_idx = blockIdx.y * OUTPUT_BLOCK_SIZE * THREAD_BLOCK_SIZE +
                          threadIdx.y * THREAD_BLOCK_SIZE;

    auto result = results[threadIdx.y];

    if (output_idx >= nb_outputs)
        return;

    // clang-format off
    // zero result
    #pragma unroll
    for (u32 t = 0; t < THREAD_BLOCK_SIZE; ++t) {
        result[t][tid] = 0;
    }
    // clang-format on

    // clang-format off
    for (u32 idx = tid; idx < batch_size; idx += BATCH_BLOCK_SIZE) {
        #pragma unroll
        for (u32 t = 0; t < THREAD_BLOCK_SIZE; ++t) {
            BREAK_IF_BLOCK(THREAD_BLOCK_SIZE, (output_idx + t) >= nb_outputs);
            result[t][tid] += output_gradient[idx * nb_outputs + output_idx + t];
        }
    }
    // clang-format on
    __syncthreads();

// clang-format on
#pragma unroll
    for (u32 t = 0; t < THREAD_BLOCK_SIZE; ++t) {
        BREAK_IF_BLOCK(THREAD_BLOCK_SIZE, (output_idx + t) >= nb_outputs);
        REDUCE(BATCH_BLOCK_SIZE, DataType, result[t], tid);
    }
    // clang-format on

    // clang-format off
    if (tid == 0) {
        #pragma unroll
        for (u32 t = 0; t < THREAD_BLOCK_SIZE; ++t) {
            BREAK_IF_BLOCK(THREAD_BLOCK_SIZE, (output_idx + t) >= nb_outputs);
            biases_gradient[output_idx + t] = result[t][0] / (DataType)batch_size;
        }
    }
    // clang-format on
}

/*
 * weights_gradient = output_gradient * inputT
 */
template <u32 BATCH_BLOCK_SIZE, u32 INPUT_BLOCK_SIZE, u32 OUTPUT_BLOCK_SIZE,
          u32 THREAD_INPUT_BLOCK_SIZE, u32 THREAD_OUTPUT_BLOCK_SIZE,
          typename DataType>
__global__ void
_hhlpLinearBackwardWeights(DataType const *output_gradient,
                           DataType const *input, DataType *weights_gradient,
                           u32 nb_outputs, u32 nb_inputs, u32 batch_size) {
    u32 row = blockIdx.y * OUTPUT_BLOCK_SIZE * THREAD_OUTPUT_BLOCK_SIZE +
              threadIdx.y * THREAD_OUTPUT_BLOCK_SIZE;
    u32 col = blockIdx.x * INPUT_BLOCK_SIZE * THREAD_INPUT_BLOCK_SIZE +
              threadIdx.x * THREAD_INPUT_BLOCK_SIZE;
    u32 tid = threadIdx.z;

    if (row >= nb_outputs || col >= nb_inputs)
        return;

    __shared__ DataType
        results[OUTPUT_BLOCK_SIZE][THREAD_OUTPUT_BLOCK_SIZE][INPUT_BLOCK_SIZE]
               [THREAD_INPUT_BLOCK_SIZE][BATCH_BLOCK_SIZE];

    // zero result
    // clang-format off
    #pragma unroll
    for (u32 to = 0; to < THREAD_OUTPUT_BLOCK_SIZE; ++to) {
        #pragma unroll
        for (u32 ti = 0; ti < THREAD_INPUT_BLOCK_SIZE; ++ti) {
            results[threadIdx.y][to][threadIdx.x][ti][tid] = 0;
        }
    }
    // clang-format on

    // compute over batches
    // clang-format off
    for (u32 b = tid; b < batch_size; b += BATCH_BLOCK_SIZE) {
        #pragma unroll
        for (u32 to = 0; to < THREAD_OUTPUT_BLOCK_SIZE; ++to) {
            BREAK_IF_BLOCK(THREAD_OUTPUT_BLOCK_SIZE, (row + to) >= nb_outputs);
            DataType o = output_gradient[b * nb_outputs + (row + to)];

            #pragma unroll
            for (u32 ti = 0; ti < THREAD_INPUT_BLOCK_SIZE; ++ti) {
                BREAK_IF_BLOCK(THREAD_INPUT_BLOCK_SIZE, (col + ti) >= nb_inputs);
                DataType i = input[b * nb_inputs + (col + ti)];
                results[threadIdx.y][to][threadIdx.x][ti][tid] += o * i;
            }
        }
    }
    // clang-format on
    __syncthreads();

    // reduction
    // clang-format off
    #pragma unroll
    for (u32 to = 0; to < THREAD_OUTPUT_BLOCK_SIZE; ++to) {
        BREAK_IF_BLOCK(THREAD_OUTPUT_BLOCK_SIZE, (row + to) >= nb_outputs);

        #pragma unroll
        for (u32 ti = 0; ti < THREAD_INPUT_BLOCK_SIZE; ++ti) {
            BREAK_IF_BLOCK(THREAD_INPUT_BLOCK_SIZE, (col + ti) >= nb_inputs);
            REDUCE(BATCH_BLOCK_SIZE, DataType,
                   results[threadIdx.y][to][threadIdx.x][ti], tid);
        }
    }
    // clang-format on

    // clang-format off
    if (tid == 0) {
        #pragma unroll
        for (u32 to = 0; to < THREAD_OUTPUT_BLOCK_SIZE; ++to) {
            BREAK_IF_BLOCK(THREAD_OUTPUT_BLOCK_SIZE, (row + to) >= nb_outputs);

            #pragma unroll
            for (u32 ti = 0; ti < THREAD_INPUT_BLOCK_SIZE; ++ti) {
                BREAK_IF_BLOCK(THREAD_INPUT_BLOCK_SIZE,
                               (col + ti) >= nb_inputs);
                weights_gradient[(row + to) * nb_inputs + (col + ti)] =
                    results[threadIdx.y][to][threadIdx.x][ti][0] /
                    (DataType)batch_size;
            }
        }
    }
    // clang-format on
}

/*
 * input_gradientT = output_gradientT * weights
 * TODO: when THREAD_BLOCK_SIZE == 4, we can use 128B load with floats
 */
template <u32 BATCH_BLOCK_SIZE, u32 REDUCE_BLOCK_SIZE, u32 THREAD_BLOCK_SIZE,
          typename DataType>
__global__ void
_hhlpLinearBackwardData(DataType const *output_gradients,
                        DataType const *weights, DataType *input_gradient,
                        u32 nb_outputs, u32 nb_inputs, u32 batch_size) {
    u32 batch_idx = blockIdx.y * BATCH_BLOCK_SIZE + threadIdx.y;
    u32 tid = threadIdx.x;

    if (batch_idx >= batch_size)
        return;

    __shared__ DataType
        results[BATCH_BLOCK_SIZE][THREAD_BLOCK_SIZE][REDUCE_BLOCK_SIZE];

    auto result = results[threadIdx.y];
    u32 col_block_idx = blockIdx.x * THREAD_BLOCK_SIZE;
    DataType const *weights_col = &weights[col_block_idx];
    DataType const *output_gradient = &output_gradients[batch_idx * nb_outputs];

    // clang-format off
    // zero result
    #pragma unroll
    for (u32 t = 0; t < THREAD_BLOCK_SIZE; ++t) {
        BREAK_IF_BLOCK(THREAD_BLOCK_SIZE, col_block_idx + t >= nb_inputs);
        result[t][tid] = 0;
    }
    // clang-format on

    // clang-format off
    for (u32 idx = tid; idx < nb_outputs; idx += REDUCE_BLOCK_SIZE) {
        DataType o = output_gradient[idx];

        #pragma unroll
        for (u32 t = 0; t < THREAD_BLOCK_SIZE; ++t) {
            BREAK_IF_BLOCK(THREAD_BLOCK_SIZE, (col_block_idx + t) >= nb_inputs);
            DataType w = weights_col[idx * nb_inputs + t];
            result[t][tid] += o * w;
        }
    }
    // clang-format on
    __syncthreads();

    // clang-format off
    #pragma unroll
    for (u32 t = 0; t < THREAD_BLOCK_SIZE; ++t) {
        BREAK_IF_BLOCK(THREAD_BLOCK_SIZE, (col_block_idx + t) >= nb_inputs);
        REDUCE(REDUCE_BLOCK_SIZE, DataType, result[t], tid);
    }
    // clang-format on

    // clang-format off
    if (tid == 0) {
        #pragma unroll
        for (u32 t = 0; t < THREAD_BLOCK_SIZE; ++t) {
            BREAK_IF_BLOCK(THREAD_BLOCK_SIZE, (col_block_idx + t) >= nb_inputs);
            input_gradient[batch_idx * nb_inputs +
                           blockIdx.x * THREAD_BLOCK_SIZE + t] = result[t][0];
        }
    }
    // clang-format on
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

cudnnStatus_t hhlpLinearForward(cudnnHandle_t cudnn_handle, void const *weights,
                                void const *biases, void const *input,
                                void *output, u32 nb_inputs, u32 nb_outputs,
                                u32 batch_size, cudnnDataType_t data_type) {
    cudaStream_t stream;
    cudnnGetStream(cudnn_handle, &stream);

#define IMPL(batch_block_size, reduce_block_size)                              \
    {                                                                          \
        dim3 threads(reduce_block_size, batch_block_size, 1);                  \
        dim3 grid(nb_outputs, CEIL_DIV(batch_size, threads.y), 1);             \
                                                                               \
        static_assert(batch_block_size * reduce_block_size <= 1024);           \
        SWITCH_CUDNN_TYPE(                                                     \
            data_type,                                                         \
            (_hhlpLinearForward<batch_block_size, reduce_block_size>           \
             <<<grid, threads, 0, stream>>>(                                   \
                 (type const *)weights, (type const *)biases,                  \
                 (type const *)input, (type *)output, nb_inputs, nb_outputs,   \
                 batch_size)));                                                \
    }
    // select implementation
    // clang-format off
    switch (batch_size) {
    case 1: IMPL(1, 512); break;
    case 32: IMPL(8, 128); break;
    case 64: IMPL(16, 64); break;
    default: IMPL(32, 32); break;
    }
    // clang-format on
#undef IMPL
    return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t hhlpLinearBackwardBias(cudnnHandle_t cudnn_handle,
                                     void const *error, void *biases_gradient,
                                     u32 nb_outputs, u32 batch_size,
                                     cudnnDataType_t data_type) {
    cudaStream_t stream;
    cudnnGetStream(cudnn_handle, &stream);

#define IMPL(batch_block_size, output_block_size, thread_block_size)           \
    {                                                                          \
        static_assert(batch_block_size * output_block_size <= 1024);           \
        dim3 threads(batch_block_size, output_block_size);                     \
        dim3 grid(1, CEIL_DIV(nb_outputs, threads.y));                         \
                                                                               \
        SWITCH_CUDNN_TYPE(                                                     \
            data_type,                                                         \
            (_hhlpLinearBackwardBias<batch_block_size, output_block_size,      \
                                     thread_block_size>                        \
             <<<grid, threads, 0, stream>>>((type const *)error,               \
                                            (type *)biases_gradient,           \
                                            nb_outputs, batch_size)));         \
    }

    // select implementation
    // clang-format off
    switch (batch_size) {
    case 1:
        SWITCH_CUDNN_TYPE(data_type, cudaMemcpy(biases_gradient, error,
                                                nb_outputs * sizeof(type),
                                                cudaMemcpyDeviceToDevice));
        break;
    case 32: IMPL(1, 1024, 1); break;
    // TODO: we nee more benchmarks for this
    case 64: IMPL(2, 512, 1); break;
    default: IMPL(32, 32, 1); break;
    }
    // clang-format on
#undef IMPL
    return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t hhlpLinearBackwardWeights(cudnnHandle_t cudnn_handle,
                                        void const *output_gradient,
                                        void const *input,
                                        void *weights_gradient, u32 nb_outputs,
                                        u32 nb_inputs, u32 batch_size,
                                        cudnnDataType_t data_type) {
    cudaStream_t stream;
    cudnnGetStream(cudnn_handle, &stream);

#define IMPL(batch_block_size, input_block_size, output_block_size,            \
             thread_input_block_size, thread_output_block_size)                \
    {                                                                          \
        static_assert(                                                         \
            input_block_size * output_block_size * batch_block_size <= 1024);  \
        dim3 threads(input_block_size, output_block_size, batch_block_size);   \
        dim3 grid(                                                             \
            CEIL_DIV(nb_inputs, (threads.x * thread_input_block_size)),        \
            CEIL_DIV(nb_outputs, (threads.y * thread_output_block_size)), 1);  \
                                                                               \
        SWITCH_CUDNN_TYPE(                                                     \
            data_type,                                                         \
            (_hhlpLinearBackwardWeights<                                       \
                batch_block_size, input_block_size, output_block_size,         \
                thread_input_block_size, thread_output_block_size>             \
             <<<grid, threads, 0, stream>>>(                                   \
                 (type *)output_gradient, (type *)input,                       \
                 (type *)weights_gradient, nb_outputs, nb_inputs,              \
                 batch_size)));                                                \
    }
    // select implementation
    // clang-format off
    switch (batch_size) {
    // BUG: the result is wrong when the batch_block_size is greater than 1
    // case 64: IMPL(1, 32, 32, 4, 1); break;
    default: IMPL(1, 32, 32, 4, 1); break;
    }
    // clang-format on
#undef IMPL
    return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t hhlpLinearBackwardData(cudnnHandle_t cudnn_handle,
                                     void const *output_gradient,
                                     void const *weights, void *input_gradient,
                                     u32 nb_outputs, u32 nb_inputs,
                                     u32 batch_size,
                                     cudnnDataType_t data_type) {
    cudaStream_t stream;
    cudnnGetStream(cudnn_handle, &stream);

#define IMPL(batch_block_size, reduce_block_size, thread_block_size)           \
    {                                                                          \
        static_assert(batch_block_size * reduce_block_size <= 1024);           \
        dim3 threads(reduce_block_size, batch_block_size);                     \
        dim3 grid(CEIL_DIV(nb_inputs, thread_block_size),                      \
                  CEIL_DIV(batch_size, threads.y), 1);                         \
                                                                               \
        SWITCH_CUDNN_TYPE(                                                     \
            data_type,                                                         \
            (_hhlpLinearBackwardData<batch_block_size, reduce_block_size,      \
                                     thread_block_size>                        \
             <<<grid, threads, 0, stream>>>(                                   \
                 (type *)output_gradient, (type *)weights,                     \
                 (type *)input_gradient, nb_outputs, nb_inputs, batch_size))); \
    }
    // select implementation
    // clang-format off
    switch (batch_size) {
    case 1: IMPL(32, 32, 1); break;
    case 64: IMPL(64, 16, 1); break;
    default: IMPL(32, 32, 1); break;
    }
    // clang-format on
#undef IMPL
    return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
}
