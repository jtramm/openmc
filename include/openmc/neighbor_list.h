#ifndef OPENMC_NEIGHBOR_LIST_H
#define OPENMC_NEIGHBOR_LIST_H

#include <algorithm>
#include <cstdint>
#include <forward_list>
#include <mutex>
#include <assert.h>

#define NEIGHBOR_SIZE 50 // limited by fusion models

namespace openmc{

//==============================================================================
//! A threadsafe, dynamic container for listing neighboring cells.
//
//! This container is a reduced interface for a linked list with an added OpenMP
//! lock for write operations.  It allows for threadsafe dynamic growth; any
//! number of threads can safely read data without locks or reference counting.
//==============================================================================

class NeighborList
{
public:

  // Constructor
  NeighborList();

  // Attempt to add an element.
  //
  // If the relevant OpenMP lock is currently owned by another thread, this
  // function will return without actually modifying the data.  It has been
  // found that returning the transport calculation and possibly re-adding the
  // element later is slightly faster than waiting on the lock to be released.
  #ifdef OPENMC_OFFLOAD
#pragma omp declare target
#endif
  void push_back(int32_t new_elem);
  #ifdef OPENMC_OFFLOAD
#pragma omp end declare target
#endif
  
  int32_t list_[NEIGHBOR_SIZE];
};

} // namespace openmc
#endif // OPENMC_NEIGHBOR_LIST_H
