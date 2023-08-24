#include<thrust/sort.h>

// This macro is used to enable the "__host__ __device__" attributes
// for the EventQueueItem comparator in event.h. If those attributes
// are missing, the code will compile and run, but it will use some sort
// of default comparator that does not have the desired effect.
#define COMPILE_CUDA_COMPARATOR

#include"openmc/event.h"


namespace openmc{

void device_sort_event_queue_item(EventQueueItem* begin, EventQueueItem* end)
{
  thrust::sort(thrust::device, begin, end);
}

struct CellCmp {
  __host__ __device__
  bool operator()(const EventQueueItem& o1, const EventQueueItem& o2) {
    if (o1.cell_id  == o2.cell_id) {
      return o1.surface_id < o2.surface_id;
    } else {
      return o1.cell_id < o2.cell_id;
    }
  }
};

void device_sort_event_queue_item_by_cell(EventQueueItem* begin, EventQueueItem* end)
{
  thrust::sort(thrust::device, begin, end, CellCmp());
}

}
