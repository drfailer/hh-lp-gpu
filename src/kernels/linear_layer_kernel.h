#ifndef KERNELS_LINEAR_LAYER_KERNEL
#define KERNELS_LINEAR_LAYER_KERNEL
#include <cudnn_graph.h>

cudnnStatus_t hhlpLinearForward(cudnnHandle_t cudnn_handle,
                                      void const *weights, void const *biases,
                                      void const *input, void *output,
                                      int nb_inputs, int nb_outputs,
                                      cudnnDataType_t data_type);
cudnnStatus_t hhlpLinearBackwardBiasKernel(cudnnHandle_t cudnn_handle,
                                           void const *error,
                                           void *biases_gradient,
                                           int nb_outputs,
                                           cudnnDataType_t data_type);
cudnnStatus_t hhlpLinearBackwardWeightsKernel(cudnnHandle_t cudnn_handle,
                                              void const *output_gradient,
                                              void const *input,
                                              void *weights_gradient,
                                              int nb_outputs, int nb_inputs,
                                              cudnnDataType_t data_type);
cudnnStatus_t hhlpLinearBackwardDataKernel(cudnnHandle_t cudnn_handle,
                                           void const *output_gradient,
                                           void const *weights,
                                           void *input_gradient, int nb_outputs,
                                           int nb_inputs,
                                           cudnnDataType_t data_type);

#endif
