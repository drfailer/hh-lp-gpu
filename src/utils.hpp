#ifndef UTILS_H
#define UTILS_H

#define CUDNN_CHECK(expr)                                                      \
  {                                                                            \
    auto result = expr;                                                        \
    if (result.is_bad()) {                                                     \
      std::cerr << "[ERROR]: " __FILE__ ":" << __LINE__ << ": "                \
                << result.get_message() << std::endl;                          \
    }                                                                          \
  }

#endif
