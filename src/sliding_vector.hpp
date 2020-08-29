/**
 * @file sliding_vector.hpp.
 * @brief Bounded sliding vector.
 * @author Aiz (c).
 * @date 2020.
 */
#pragma once

#include <vector>

namespace aiz {

template <typename T>
class SlidingVector {
 public:
  explicit SlidingVector(size_t capacity) : kCapacity(capacity) {
    mVec.reserve(capacity);
  };

  void pushBack(const T &v) noexcept(
      std::is_nothrow_copy_constructible<T>::value) {
    static_assert(std::is_copy_constructible<T>::value,
                  "T must be copy constructible");
    return emplaceBack(v);
  }

  template <typename P, typename = typename std::enable_if<
                            std::is_constructible<T, P &&>::value>::type>
  void pushBack(P &&v) noexcept(std::is_nothrow_constructible<T, P &&>::value) {
    return emplaceBack(std::forward<P>(v));
  }

  const std::vector<T> &vecCRef() const { return mVec; }

  void clear() const { mVec.clear(); }

 private:
  template <typename... Args>
  void emplaceBack(Args &&... args) noexcept(
      std::is_nothrow_constructible<T, Args &&...>::value) {
    static_assert(std::is_constructible<T, Args &&...>::value,
                  "T must be constructible with Args&&...");

    if (mVec.size() >= kCapacity) mVec.erase(mVec.begin());
    mVec.push_back(T(std::forward<Args>(args)...));
  }

  const size_t kCapacity;
  std::vector<T> mVec;
};

}  // namespace aiz
