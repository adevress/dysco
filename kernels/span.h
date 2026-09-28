#ifndef DYSCO_SPAN_H
#define DYSCO_SPAN_H

/**
 * @file
 * `dysco::span`: a single name that resolves to `std::span` when the toolchain
 * provides C++20, and to a small, API-compatible C++17 fallback otherwise.
 *
 * Dysco is deliberately kept at C++17 (it is also built as part of older
 * Casacore releases), so `std::span` cannot be assumed.  Using this alias
 * keeps the accelerated backend written in terms of views instead of raw
 * pointer/size pairs, and the fallback simply disappears once the C++20
 * standard is available.
 *
 * Note: only the host-side API uses this type.  Device kernels still receive
 * plain pointers (a `span` is not device-annotated), and the launchers forward
 * `data()`/`size()`.
 */

#include <cstddef>
#include <type_traits>
#include <utility>

// `__cpp_lib_span` only exists after a standard library header (e.g. <span>
// or <version>) has been included, so probe the header directly instead.
#if defined(__has_include)
#if __has_include(<span>) && __cplusplus >= 202002L
#include <span>
#define DYSCO_USE_STD_SPAN 1
#endif
#endif

#ifdef DYSCO_USE_STD_SPAN
namespace dysco {
template <class T>
using span = std::span<T>;
}  // namespace dysco
#else
namespace dysco {

inline constexpr std::size_t dynamic_extent = static_cast<std::size_t>(-1);

template <class T>
class span;

template <class T>
class span {
 public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using iterator = pointer;
  using const_iterator = const_pointer;

  constexpr span() noexcept = default;
  constexpr span(pointer data, size_type count) noexcept
      : _data(data), _size(count) {}

  // Converting constructor: span<T> -> span<const T> is allowed, the reverse
  // is not.
  template <class U, std::enable_if_t<std::is_convertible_v<U (*)[], T (*)[]>,
                                      int> = 0>
  constexpr span(const span<U>& other) noexcept
      : _data(other.data()), _size(other.size()) {}

  // From a C array.
  template <class U, std::size_t N,
            std::enable_if_t<std::is_convertible_v<U (*)[], T (*)[]>, int> = 0>
  constexpr span(U (&array)[N]) noexcept : _data(array), _size(N) {}

  // From any contiguous container with data()/size() (std::vector,
  // std::array, ...), excluding span itself.
  template <class Container,
            class = decltype(static_cast<pointer>(
                std::declval<Container&>().data())),
            class = decltype(std::declval<Container&>().size()),
            class = std::enable_if_t<
                !std::is_same_v<std::remove_cv_t<Container>, span>>>
  constexpr span(Container& container) noexcept
      : _data(container.data()), _size(container.size()) {}

  [[nodiscard]] constexpr pointer data() const noexcept { return _data; }
  [[nodiscard]] constexpr size_type size() const noexcept { return _size; }
  [[nodiscard]] constexpr bool empty() const noexcept { return _size == 0; }
  constexpr reference operator[](size_type i) const noexcept { return _data[i]; }
  constexpr reference front() const noexcept { return _data[0]; }
  constexpr reference back() const noexcept { return _data[_size - 1]; }
  constexpr iterator begin() const noexcept { return _data; }
  constexpr iterator end() const noexcept { return _data + _size; }

  [[nodiscard]] constexpr span subspan(
      size_type offset, size_type count = dynamic_extent) const noexcept {
    return span(_data + offset,
                count == dynamic_extent ? _size - offset : count);
  }
  [[nodiscard]] constexpr span first(size_type count) const noexcept {
    return span(_data, count);
  }
  [[nodiscard]] constexpr span last(size_type count) const noexcept {
    return span(_data + _size - count, count);
  }

 private:
  pointer _data = nullptr;
  size_type _size = 0;
};

template <class T>
span(T*, std::size_t) -> span<T>;
template <class Container>
span(Container&)
    -> span<std::remove_const_t<
        std::remove_pointer_t<decltype(std::declval<Container&>().data())>>>;

}  // namespace dysco
#endif  // DYSCO_USE_STD_SPAN

#endif  // DYSCO_SPAN_H
