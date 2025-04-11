#ifndef CUDNN_OPERATIONS_H
#define CUDNN_OPERATIONS_H
#include <cstdint>
#include "types.hpp"

void hadamard(ftype *A_host, ftype *B_host, int64_t m, int64_t n, ftype *C_host);
void matmul(ftype *A_host, ftype *B_host, int64_t m, int64_t n, int64_t k,
            ftype *C_host);
void sigmoid(ftype *A_host, int64_t size);

#endif
