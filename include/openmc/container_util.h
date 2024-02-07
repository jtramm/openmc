#ifndef OPENMC_CONTAINER_UTIL_H
#define OPENMC_CONTAINER_UTIL_H

#include <algorithm> // for find
#include <iterator>  // for begin, end
#include <unordered_set>

namespace openmc {

template<class C, class T>
inline bool contains(const C& v, const T& x)
{
  return std::end(v) != std::find(std::begin(v), std::end(v), x);
}

template<class T>
inline bool has_matching_element(const std::unordered_set<T>& elements, const vector<T>& vec)
{
  for (T num : vec) {
    if (elements.find(num) != elements.end()) {
      return true;
    }
  }
  return false;
}

} // namespace openmc

#endif // OPENMC_CONTAINER_UTIL_H
