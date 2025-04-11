#ifndef CPP_UTILS_UTEST_H
#define CPP_UTILS_UTEST_H
#include <iostream>
#include <string>
#include <type_traits>

#define Test(name)                                                             \
    void test_function_##name(                                                 \
        [[maybe_unused]] utest::test_status_t &__test_status__)

#define run_test(name)                                                         \
    {                                                                          \
        utest::test_status_t status{false, 0};                                 \
        test_function_##name(status);                                          \
        utest::report_status(#name, status);                                   \
    }

#define uassert(expr)                                                          \
    if (utest::assert_impl("ASSERT", expr, #expr, __FILE__, __LINE__) != 0) {  \
        ++__test_status__.nb_assert_failed;                                    \
    }
#define uassert_equal(lhs, rhs)                                                \
    if (utest::assert_equal_impl("ASSERT", lhs, rhs, #lhs, #rhs, __FILE__,     \
                                 __LINE__) != 0) {                             \
        ++__test_status__.nb_assert_failed;                                    \
    }
#define urequire(expr)                                                         \
    if (utest::assert_impl("REQUIRE", expr, #expr, __FILE__, __LINE__) != 0) { \
        __test_status__.require_failed = true;                                 \
        return;                                                                \
    }

namespace utest {

struct test_status_t {
    bool require_failed = false;
    size_t nb_assert_failed = 0;
};

inline void report_status(std::string const &test_name,
                          test_status_t const &status) {
    if (!status.require_failed && status.nb_assert_failed == 0) {
        std::cerr << "\033[0;32m[TEST PASSED] " << test_name << "\033[0m"
                  << std::endl;
    } else {
        std::cerr << "\033[1;31m[TEST FAILED] " << test_name << ":\033[0m ";

        if (status.require_failed) {
            std::cerr << "failed on require." << std::endl;
        } else {
            std::cerr << status.nb_assert_failed << " assertion failed."
                      << std::endl;
        }
    }
}

inline void report_error(std::string const &group, std::string const filename,
                         size_t line) {
    std::cerr << "\033[1;31m[" << group << " ERROR]:\033[0m " << filename << ":"
              << line << ":";
}

inline int assert_impl(std::string const &group, bool expr,
                       std::string const &expr_str, std::string const &filename,
                       size_t line) {
    if (!expr) {
        report_error(group, filename, line);
        std::cerr << "`" << expr_str << "` evaluated to false." << std::endl;
        return 1;
    }
    return 0;
}

template <typename T1, typename T2>
inline int assert_equal_impl(std::string const &group, T1 const &lhs,
                             T2 const &rhs, std::string const &lhs_str,
                             std::string const &rhs_str,
                             std::string const &filename, size_t line) {
    if constexpr (!std::is_convertible_v<T1, T2>) {
        report_error(group, filename, line);
        std::cerr << " type `" << typeid(lhs).name() << "` not convertible to `"
                  << typeid(rhs).name() << "`." << std::endl;
        return 1;
    }
    if (lhs != ((T1)rhs)) {
        report_error(group, filename, line);
        std::cerr << "`" << lhs_str << "` not equal to `" << rhs_str << "`."
                  << std::endl;
        return 1;
    }
    return 0;
}

} // namespace utest

#endif
