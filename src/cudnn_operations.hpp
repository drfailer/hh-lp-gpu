#ifndef CUDNN_OPERATIONS_H
#define CUDNN_OPERATIONS_H
#include <cstdint>

#define CUDNN_CHECK(expr)                                                      \
  {                                                                            \
    auto result = expr;                                                        \
    if (result.is_bad()) {                                                     \
      std::cerr << "[ERROR]: " __FILE__ ":" << __LINE__ << ": "                \
                << result.get_message() << std::endl;                          \
    }                                                                          \
  }

using ftype = float;

void hadamard(ftype *A_host, ftype *B_host, int64_t m, int64_t n, ftype *C_host);
void matmul(ftype *A_host, ftype *B_host, int64_t m, int64_t n, int64_t k,
            ftype *C_host);

#endif
