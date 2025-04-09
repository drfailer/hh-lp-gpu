#ifndef TYPES_H
#define TYPES_H
#include <cudnn.h>
#include <cudnn_frontend/graph_interface.h>
#include <cudnn_graph.h>

using ftype = float;
using tensor_attr_t = std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>;
constexpr cudnn_frontend::DataType_t COMPUTE_DATA_TYPE =
    cudnn_frontend::DataType_t::FLOAT;

#endif
