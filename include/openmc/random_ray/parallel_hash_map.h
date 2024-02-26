#ifndef OPENMC_RANDOM_RAY_PARALLEL_HASH_MAP_H
#define OPENMC_RANDOM_RAY_PARALLEL_HASH_MAP_H

#include "openmc/openmp_interface.h"

#include <memory>

// If I make this entirely abstract, with types for both key and value,
// then I need to provide a comparator and a hash for the key. These
// are annoying functors and clutter the other class definitions.
// The need goes away if I limit it to int types etc, but I would like it to
// handle 64 bits I think.

// The better way I think is to just specialize it for a specific key use case,
// so that everything can live in this header. Thus, it will assume the key is
// a std::pair<uint64_t, uint64_t>.

// This version of the class is by far the simplest and would work great
// for most cases. However, the weakness is that the number of buckets
// needs to be fixed in advance of the first iteration, and in general
// we have no halfway decent estimate of how many FSRS there are going
// to be. So, if we pick a reasonable number like 10k, we are in big
// trouble if we have even just like 10M FSRS in the problem. In this
// case, we are going to hit the wall in terms of insertion performance.
// The issue is each bucket is a linked list, so this is trash if we
// have more than just a few items in each bucket. It's not a problem
// after the first iteration, but then things fall apart. It is much
// better to have it be a 2D hash idea, which would allow for the buckets
// to dynamically resize themselves to allow for efficient access no matter
// if the problem is large or small. TH
template<typename KeyType, typename ValueType, typename HashFunctor,
  typename EqualFunctor>
class ParallelHashMapSimple {

public:
  ParallelHashMapSimple(int n_buckets = 1000)
  {
    // Set the maximum load factor to a very high value.
    my_map.max_load_factor(std::numeric_limits<float>::max());

    // Rehash the map to have a specific number of buckets.
    // This will not change due to the high load factor being set.
    my_map.rehash(n_buckets);

    locks_.resize(map.bucket_count());
  }

  // Accesses element for a given sr and bin combo
  ValueType* operator[](KeyType key)
  {
    int bucket = map_.bucket(key);
    locks_[bucket].lock();
    ValueType* value = map_[key].get();
    locks_[bucket].unlock();
    return value;
  }

  // Copies everything into a vector, clears all buckets
  void empty_to_vector(vector<T>& vec) {}

private:
  vector<OpenMPMutex> locks_;
  std::unordered_map<KeyType, std::unique_ptr<ValueType>, HashFunctor,
    EqualFunctor>
    map_;
};

// This would be great. EXCEPT it makes no sense that I'm going to
// get out a different map type than I declared with the definition.
// If I had accessors, then it'd be fine to do whatever I want inside,
// but this is no good on the outside.

// Maybe it's OK though to just declare it as

// We HAVE to use pointers, as
template<typename KeyType, typename ValueType, typename HashFunctor,
  typename EqualFunctor>
class ParallelHashMap {

  struct Bucket {
    OpenMPMutex lock_;
    std::unordered_map<KeyType, std::unique_ptr<ValueType>, HashFunctor,
      EqualFunctor>
      map_;
  };

public:
  ParallelHashMap(int n_buckets = 1000) : buckets_(n_buckets) {}

  // Copies everything into a vector, clears all buckets
  vector<ValueType> copy_into_vector()
  {
    vector<ValueType> v;
    for (auto& bucket : buckets_) {
      for (auto& pair : bucket.map_) {
        v.push_back(*pair.second.get());
      }
    }
  }

  lock(const KeyType& key)
  {
    Bucket& bucket = get_bucket(key);
    bucket.lock();
  }

  unlock(const KeyType& key)
  {
    Bucket& bucket = get_bucket(key);
    bucket.unlock();
  }

  void clear()
  {
    for (auto& bucket : buckets_) {
      bucket.map_.clear();
    }
  }

  bool contains(const KeyType& key)
  {
    Bucket& bucket = get_bucket(key);
    return bucket.map_.contains(key);
  }

  ValueType& operator[](const KeyType& key)
  {
    Bucket& bucket = get_bucket(key);
    auto& uptr = bucket.map_[key];

    return *uptr->get();
  }

private:
  Bucket& get_bucket(const KeyType& key)
  {
    buckets_[hash(key) % buckets_.size()];
  }

  HashFunctor hash;
  vector<Bucket> buckets_;
};

#endif // OPENMC_RANDOM_RAY_PARALLEL_HASH_MAP_H