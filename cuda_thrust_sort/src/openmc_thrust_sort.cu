#include <thrust/sort.h>

// This macro is used to enable the "__host__ __device__" attributes
// for the EventQueueItem comparator in event.h. If those attributes
// are missing, the code will compile and run, but it will use some sort
// of default comparator that does not have the desired effect.
#define COMPILE_CUDA_COMPARATOR

#include "openmc/event.h"

namespace openmc{

void thrust_sort_MatE(EventQueueItem* begin, EventQueueItem* end)
{
  thrust::sort(thrust::device, begin, end, MatECmp());
}

void thrust_sort_CellSurf(EventQueueItem* begin, EventQueueItem* end)
{
  thrust::sort(thrust::device, begin, end, CellSurfCmp());
}

}
