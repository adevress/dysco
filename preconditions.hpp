#ifndef DYSCO_PRECONDITIONS_HPP
#define DYSCO_PRECONDITIONS_HPP

/**
 * @file
 * Self-contained contract helpers, modelled on `arcane::enforce_precondition`
 * (from arcane-lang's `utils/contracts.hpp`) but with no external dependency:
 * the diagnostic is built with the standard library instead of {fmt}.
 *
 * The call syntax is unchanged:
 *
 * @code
 *   void scale(double factor) {
 *     enforce_precondition(factor > 0.0);   // throws at this line
 *     ...
 *   }
 * @endcode
 *
 * The source location defaults to `source_location::current()` evaluated at
 * the call site, so the reported line/function are those of the *parent*
 * function making the check.  Where `std::source_location` is available
 * (C++20) it is used directly; otherwise a small builtin-backed stand-in with
 * the same interface is provided, keeping the project buildable as C++17.
 */

#include <cstdlib>
#include <stdexcept>
#include <string>

#if defined(__has_include)
#  if __has_include(<source_location>) && __cplusplus >= 202002L
#    include <source_location>
#    define DYSCO_HAS_STD_SOURCE_LOCATION 1
#  endif
#endif
#ifndef DYSCO_HAS_STD_SOURCE_LOCATION
#  define DYSCO_HAS_STD_SOURCE_LOCATION 0
#endif

namespace dyscostman {

#if DYSCO_HAS_STD_SOURCE_LOCATION
using source_location = std::source_location;
#else
/// Minimal C++17 stand-in for `std::source_location`.  `current()` takes the
/// compiler builtins as default arguments, so they are expanded at the call
/// site (this is how libstdc++/libc++ implement the standard type).
struct source_location {
  [[nodiscard]] static constexpr source_location current(
      const char* const file = __builtin_FILE(),
      const char* const function = __builtin_FUNCTION(),
      const unsigned line = __builtin_LINE(),
      const unsigned column = 0) noexcept {
    return source_location(file, function, line, column);
  }

  constexpr source_location() noexcept = default;

  [[nodiscard]] constexpr const char* file_name() const noexcept {
    return _file;
  }
  [[nodiscard]] constexpr const char* function_name() const noexcept {
    return _function;
  }
  [[nodiscard]] constexpr unsigned line() const noexcept { return _line; }
  [[nodiscard]] constexpr unsigned column() const noexcept { return _column; }

 private:
  constexpr source_location(const char* file, const char* function,
                            unsigned line, unsigned column) noexcept
      : _file(file), _function(function), _line(line), _column(column) {}

  const char* _file = "";
  const char* _function = "";
  unsigned _line = 0;
  unsigned _column = 0;
};
#endif  // DYSCO_HAS_STD_SOURCE_LOCATION

namespace intern {

/// @brief Build the diagnostic string for a failed precondition.
[[nodiscard]] inline std::string preconditionFailureMessage(
    const source_location& location) {
  std::string message = "precondition failed at ";
  message += location.file_name();
  message += ':';
  message += std::to_string(location.line());
  message += ':';
  message += std::to_string(location.column());
  message += " in function '";
  message += location.function_name();
  message += '\'';
  return message;
}

/// @brief Report a precondition failure at the given location and throw a
/// `std::logic_error`.
[[noreturn]] inline void log_and_throw(const source_location& location) {
  throw std::logic_error(preconditionFailureMessage(location));
}

}  // namespace intern

/// @brief Enforce a precondition, throwing `std::logic_error` on failure.
/// @param condition the condition that must hold
/// @param location source location of the precondition, captured at the call
/// site (the line inside the parent function by default)
inline void enforce_precondition(
    bool condition,
    const source_location location = source_location::current()) {
  if (!condition) {
    intern::log_and_throw(location);
  }
}

/// @brief Mark a code path as unreachable, throwing `std::logic_error` if it
/// is executed.
/// @param location source location of the unreachable code, captured at the
/// call site
[[noreturn]] inline void unreachable(
    const source_location location = source_location::current()) {
  intern::log_and_throw(location);
}

}  // namespace dyscostman

#endif  // DYSCO_PRECONDITIONS_HPP
