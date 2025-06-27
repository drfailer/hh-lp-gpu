#ifndef KERNELS_LINEAR_LAYER_KERNEL
#define KERNELS_LINEAR_LAYER_KERNEL
#include <cudnn_graph.h>

using u32 = unsigned int;

cudnnStatus_t hhlpLinearForward(cudnnHandle_t cudnn_handle, void const *weights,
                                void const *biases, void const *input,
                                void *output, u32 nb_inputs, u32 nb_outputs,
                                u32 batch_size, cudnnDataType_t data_type);
cudnnStatus_t hhlpLinearBackwardBias(cudnnHandle_t cudnn_handle,
                                     void const *error, void *biases_gradient,
                                     u32 nb_outputs, u32 batch_size,
                                     cudnnDataType_t data_type);
cudnnStatus_t hhlpLinearBackwardWeights(cudnnHandle_t cudnn_handle,
                                        void const *output_gradient,
                                        void const *input,
                                        void *weights_gradient, u32 nb_outputs,
                                        u32 nb_inputs, u32 batch_size,
                                        cudnnDataType_t data_type);
cudnnStatus_t hhlpLinearBackwardData(cudnnHandle_t cudnn_handle,
                                     void const *output_gradient,
                                     void const *weights, void *input_gradient,
                                     u32 nb_outputs, u32 nb_inputs,
                                     u32 batch_size, cudnnDataType_t data_type);

#endif
