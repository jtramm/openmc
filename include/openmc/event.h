#ifndef OPENMC_EVENT_H
#define OPENMC_EVENT_H

//! \file event.h
//! \brief Event-based data structures and methods

#include "openmc/shared_array.h"

namespace openmc {

//==============================================================================
// Structs
//==============================================================================

// In the event-based model, instead of moving or sorting the particles
// themselves based on which event they need, a queue is used to store the
// index (and other useful info) for each event type.
// The EventQueueItem struct holds the relevant information about a particle needed
// for sorting the queue. For very high particle counts, a sorted queue has the
// potential to result in greatly improved cache efficiency. However, sorting
// will introduce some overhead due to the sorting process itself, and may not
// result in any benefits if not enough particles are present for them to achieve
// consistent locality improvements. 
struct EventQueueItem{
  int idx;        //!< particle index in event-based particle buffer
  int material;   //!< material that particle is in
  float E;        //!< particle energy
  int cell_id;    //!< cell ID
  int surface_id; //!< surface ID

  // Constructors
  EventQueueItem() = default;
  EventQueueItem(double energy, int buffer_idx, int cell_id, int surface_id) :
    idx(buffer_idx), material(0), E(static_cast<float>(energy)), cell_id(cell_id), surface_id(surface_id)  {}
  EventQueueItem(double energy, int mat, int buffer_idx, int cell_id, int surface_id) :
    idx(buffer_idx), material(mat), E(static_cast<float>(energy)), cell_id(cell_id), surface_id(surface_id) {}
};

// Enumeration used for specifying which way you want to sort a queue
enum class SortBy { material_energy, cell_surface };

// Comparators for sorting queues. The "G" variants are required
// for the parallel qsort host implementation in addition to the regular
// comparators.

struct MatECmp {
  #ifdef COMPILE_CUDA_COMPARATOR
  __host__ __device__
  #endif
  bool operator()(const EventQueueItem& o1, const EventQueueItem& o2) {
    if (o1.material == o2.material) {
      return o1.E < o2.E;
    } else {
      return o1.material < o2.material;
    }
  }
};

struct MatECmpG {
  #ifdef COMPILE_CUDA_COMPARATOR
  __host__ __device__
  #endif
  bool operator()(const EventQueueItem& o1, const EventQueueItem& o2) {
    if (o1.material == o2.material) {
      return o1.E > o2.E;
    } else {
      return o1.material > o2.material;
    }
  }
};

struct CellSurfCmp {
  #ifdef COMPILE_CUDA_COMPARATOR
  __host__ __device__
  #endif
  bool operator()(const EventQueueItem& o1, const EventQueueItem& o2) {
    if (o1.cell_id  == o2.cell_id) {
      return o1.surface_id < o2.surface_id;
    } else {
      return o1.cell_id < o2.cell_id;
    }
  }
};

struct CellSurfCmpG {
  #ifdef COMPILE_CUDA_COMPARATOR
  __host__ __device__
  #endif
  bool operator()(const EventQueueItem& o1, const EventQueueItem& o2) {
    if (o1.cell_id  == o2.cell_id) {
      return o1.surface_id > o2.surface_id;
    } else {
      return o1.cell_id > o2.cell_id;
    }
  }
};

//==============================================================================
// Global variable declarations
//==============================================================================

namespace simulation {

// Event queues. These use the special SharedArray type, rather than a normal
// vector, as they will be shared between threads and may be appended to at the
// same time. To facilitate this, the SharedArray thread_safe_append() method
// is provided which controls the append operations using atomics.

// Note: we only need to declare the xs queues as global items as the rest
// are only used within the lexical scope of a target construct. 
#ifdef OPENMC_OFFLOAD
#pragma omp declare target
#endif
extern SharedArray<EventQueueItem> calculate_fuel_xs_queue;
extern SharedArray<EventQueueItem> calculate_nonfuel_xs_queue;
extern SharedArray<EventQueueItem> advance_particle_queue;
extern SharedArray<EventQueueItem> surface_crossing_queue;
extern SharedArray<EventQueueItem> collision_queue;
extern SharedArray<EventQueueItem> revival_queue;

extern int current_source_offset;
#ifdef OPENMC_OFFLOAD
#pragma omp end declare target
#endif

extern int sort_counter;

} // namespace simulation

//==============================================================================
// Functions
//==============================================================================

//! Allocate space for the event queues and particle buffer
//
//! \param n_particles The number of particles in the particle buffer
void init_event_queues(int n_particles);

//! Free the event queues and particle buffer
void free_event_queues(void);

//! Enqueue a particle based on if it is in fuel or a non-fuel material
//
//! \param buffer_idx The particle's actual index in the particle buffer
void dispatch_xs_event(int buffer_idx);

//! Execute the initialization event for all particles
//
//! \param n_particles The number of particles in the particle buffer
void process_init_events(int n_particles);

//! Execute the calculate XS event for all particles in this event's buffer
//
//! \param queue A reference to the desired XS lookup queue
//void process_calculate_xs_events(SharedArray<EventQueueItem>& queue);
void process_calculate_xs_events_fuel();
void process_calculate_xs_events_nonfuel();

//! Execute the advance particle event for all particles in this event's buffer
void process_advance_particle_events();
bool depletion_rx_check();

//! Execute the surface crossing event for all particles in this event's buffer
void process_surface_crossing_events();

//! Execute the collision event for all particles in this event's buffer
void process_collision_events();

//! Execute the death event for all particles
//
//! \param n_particles The number of particles in the particle buffer
void process_death_events(int n_particles);

//! Execute the revival event for all particles in this event's buffer
void process_revival_events();

#ifdef CUDA_THRUST_SORT
//! Sort a queue on-device using CUDA Thrust
//
//! \param begin A pointer to the beginning of the queue
//! \param end A pointer to the end of the queue
void thrust_sort_MatE(EventQueueItem* begin, EventQueueItem* end);
void thrust_sort_CellSurf(EventQueueItem* begin, EventQueueItem* end);
#endif

#ifdef SYCL_SORT
//! Sort a queue on-device using Intel OneAPI DPL via SYCL interop
//
//! \param begin A pointer to the beginning of the queue
//! \param end A pointer to the end of the queue
void SYCL_sort_MatE(EventQueueItem* begin, EventQueueItem* end);
void SYCL_sort_CellSurf(EventQueueItem* begin, EventQueueItem* end);
#endif

} // namespace openmc

#endif // OPENMC_EVENT_H
