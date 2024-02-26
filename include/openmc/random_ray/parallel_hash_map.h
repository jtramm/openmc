#ifndef OPENMC_RANDOM_RAY_PARALLEL_HASH_MAP_H
#define OPENMC_RANDOM_RAY_PARALLEL_HASH_MAP_H

#include "openmc/openmp_interface.h"

#include <memory>

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

  // Copies everything into a vector, clears all buckets
  int64_t move_contents_into_vector(vector<ValueType>& v)
  {
    int64_t n_new = 0;
    for (auto& bucket : buckets_) {
      for (auto& pair : bucket.map_) {
        v.push_back(*pair.second.get());
        n_new++;
      }
      bucket.map_.clear();
    }
    return n_new;
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