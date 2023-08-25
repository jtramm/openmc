#ifdef SYCL_SORT
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>

#include "openmc/event.h"

namespace openmc{

void SYCL_sort_MatE(EventQueueItem* begin, EventQueueItem* end)
{
  std::sort( oneapi::dpl::execution::dpcpp_default, begin, end, MatECmp());
}

void SYCL_sort_CellSurf(EventQueueItem* begin, EventQueueItem* end)
{
  std::sort( oneapi::dpl::execution::dpcpp_default, begin, end, CellSurfCmp());
}

} // end namespace openmc

#endif
