#ifndef CPP_UTILS_UTEST_H
#define CPP_UTILS_UTEST_H
#define UTEST
#include <iostream>
#include <string>

#define UTest(name) void test_function_##name()

#define uassert(expr)                                                          \
    utest::Test::assert_true("ASSERT", expr, #expr, __FILE__, __LINE__)
#define uassert_equal(lhs, rhs)                                                \
    utest::Test::assert_equal("ASSERT", lhs, rhs, #lhs, #rhs, __FILE__,        \
                              __LINE__)
#define urequire(expr)                                                         \
    if (utest::Test::assert_true("REQUIRE", expr, #expr, __FILE__,             \
                                 __LINE__) != 0) {                             \
        utest::Test::require_fail();                                           \
        return;                                                                \
    }

#ifdef UTEST
#define run_test(name)                                                         \
    try {                                                                      \
        utest::Test::reset();                                                  \
        test_function_##name();                                                \
        utest::Test::report(#name);                                            \
    } catch (utest::PanicException const &e) {                                 \
        utest::Test::report_panic(#name, e);                                   \
    } catch (std::exception const &e) {                                        \
        utest::Test::report_exception(#name, e);                               \
    }

#define upanic(msg) utest::Test::panic(msg, __FILE__, __LINE__);
#else
#define run_test(name)
#define upanic(msg)
#endif

namespace utest {

struct PanicException : std::runtime_error {
    PanicException(std::string const &msg) : std::runtime_error(msg) {}
};

struct Test {
    static inline bool require_failed = false;
    static inline size_t nb_assert_failed = 0;

    static inline void require_fail() { require_failed = true; }

    static inline void reset() {
        require_failed = false;
        nb_assert_failed = 0;
    }

    static inline void error(std::string const &group, std::string const file,
                             size_t line, std::string const &msg) {
        std::cerr << "\033[1;31m[" << group << " ERROR]:\033[0m " << file << ":"
                  << line << ": " << msg << std::endl;
    }
    static inline void panic(std::string const &msg, std::string const file,
                             size_t line) {
        error("PANIC", file, line, msg);
        throw PanicException(msg);
    }

    static inline void report(std::string const &test_name) {
        if (!require_failed && nb_assert_failed == 0) {
            std::cerr << "\033[0;32m[TEST PASSED] " << test_name << "\033[0m"
                      << std::endl;
        } else {
            std::cerr << "\033[1;31m[TEST FAILED] " << test_name << ":\033[0m ";

            if (require_failed) {
                std::cerr << "failed on require." << std::endl;
            } else {
                std::cerr << nb_assert_failed << " assertion failed."
                          << std::endl;
            }
        }
    }

    static inline void report_exception(std::string const &test_name,
                                        std::exception const &e) {
        std::cerr << "\033[1;31m[TEST FAILED] " << test_name
                  << ":\033[0m unhandled exception: " << e.what() << std::endl;
    }

    static inline void report_panic(std::string const &test_name,
                                    PanicException const &e) {
        std::cerr << "\033[1;31m[TEST FAILED] " << test_name
                  << ":\033[0m program panic error (" << e.what() << ")."
                  << std::endl;
    }

    static inline int assert_true(std::string const &group, bool expr,
                                  std::string const &expr_str,
                                  std::string const &file, size_t line) {
        if (!expr) {
            error(group, file, line, "`" + expr_str + "` evaluated to false.");
            ++nb_assert_failed;
            return 1;
        }
        return 0;
    }

    static inline int assert_equal(std::string const &group, auto const &lhs,
                                   auto const &rhs, std::string const &lhs_str,
                                   std::string const &rhs_str,
                                   std::string const &file, size_t line) {
        if (lhs != rhs) {
            error(group, file, line, lhs_str + " != " + rhs_str + ".");
            ++nb_assert_failed;
            return 1;
        }
        return 0;
    }
};

} // namespace utest

#endif
