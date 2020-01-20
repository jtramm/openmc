#include "openmc/event.h"
#include "openmc/material.h"
#include "openmc/simulation.h"
#include "openmc/timer.h"


namespace openmc {

//==============================================================================
// Global variables
//==============================================================================

namespace simulation {

std::unique_ptr<QueueItem[]> calculate_fuel_xs_queue;
std::unique_ptr<QueueItem[]> calculate_nonfuel_xs_queue;
std::unique_ptr<QueueItem[]> advance_particle_queue;
std::unique_ptr<QueueItem[]> surface_crossing_queue;
std::unique_ptr<QueueItem[]> collision_queue;
std::unique_ptr<Particle[]>  particles;

int64_t calculate_fuel_xs_queue_length {0};
int64_t calculate_nonfuel_xs_queue_length {0};
int64_t advance_particle_queue_length {0};
int64_t surface_crossing_queue_length {0};
int64_t collision_queue_length {0};
int64_t dead_particle_count {0};

int64_t max_particles_in_flight {100000};

} // namespace simulation

//==============================================================================
// Non-member functions
//==============================================================================

void init_event_queues(int64_t n_particles)
{
  simulation::calculate_fuel_xs_queue =    std::make_unique<QueueItem[]>(n_particles);
  simulation::calculate_nonfuel_xs_queue = std::make_unique<QueueItem[]>(n_particles);
  simulation::advance_particle_queue =     std::make_unique<QueueItem[]>(n_particles);
  simulation::surface_crossing_queue =     std::make_unique<QueueItem[]>(n_particles);
  simulation::collision_queue =            std::make_unique<QueueItem[]>(n_particles);
  simulation::particles =                  std::make_unique<Particle[] >(n_particles);
}

void free_event_queues(void)
{
  simulation::calculate_fuel_xs_queue.reset();
  simulation::calculate_nonfuel_xs_queue.reset();
  simulation::advance_particle_queue.reset();
  simulation::surface_crossing_queue.reset();
  simulation::collision_queue.reset();
  simulation::particles.reset();
}

void enqueue_particle(QueueItem* queue, int64_t& length, Particle* p,
    int64_t buffer_idx, bool use_atomic)
{
  int64_t idx;
  if (use_atomic) {
    #pragma omp atomic capture
    idx = length++;
  } else {
    idx = length++;
  }

  queue[idx].idx = buffer_idx;
  queue[idx].E = p->E_;
  queue[idx].material = p->material_;
  queue[idx].type = p->type_;
}

void dispatch_xs_event(Particle* p)
{
  if (p->material_ == MATERIAL_VOID ||
      !model::materials[p->material_]->fissionable_) {
    p->next_event_ = Particle::EventType::calculate_nonfuel_xs;
    #pragma omp atomic
    simulation::calculate_nonfuel_xs_queue_length++;
  } else {
    p->next_event_ = Particle::EventType::calculate_fuel_xs;
    #pragma omp atomic
    simulation::calculate_fuel_xs_queue_length++;
  }
}

void process_init_events(int64_t n_particles, int64_t source_offset)
{
  simulation::time_event_init.start();
  //#pragma omp parallel for schedule(runtime) reduction(+:simulation::calculate_nonfuel_xs_queue_length, simulation::calculate_fuel_xs_queue_length)
  #pragma omp parallel for schedule(runtime)
  for (int64_t i = 0; i < n_particles; i++) {
    Particle* p = &simulation::particles[i];
    initialize_history(p, source_offset + i + 1);
    dispatch_xs_event(p);
  }
  simulation::time_event_init.stop();
}

void process_calculate_xs_events(Particle::EventType event, int64_t n_particles)
{
  simulation::time_event_calculate_xs.start();

  // TODO: If using C++17, perform a parallel sort of the queue
  // by particle type, material type, and then energy, in order to
  // improve cache locality and reduce thread divergence on GPU.
  //std::sort(queue, queue+n);

  //#pragma omp parallel for schedule(runtime) reduction(+:simulation::advance_particle_queue_length)
  #pragma omp parallel for schedule(runtime)
  for (int64_t i = 0; i < n_particles; i++) {
    Particle* p = &simulation::particles[i]; 
    if (p->next_event_ == event ) {
      p->event_calculate_xs();
      p->next_event_ = Particle::EventType::advance;
      #pragma omp atomic
      simulation::advance_particle_queue_length++;
    }
  }

  simulation::time_event_calculate_xs.stop();
}

void process_advance_particle_events(int64_t n_particles)
{
  simulation::time_event_advance_particle.start();

  //#pragma omp parallel for schedule(runtime) reduction(+:simulation::surface_crossing_queue_length, simulation::collision_queue_length)
  #pragma omp parallel for schedule(runtime)
  for (int64_t i = 0; i < n_particles; i++) {
    Particle* p = &simulation::particles[i]; 
    if (p->next_event_ == Particle::EventType::advance) {
      p->event_advance();
      if (p->collision_distance_ > p->boundary_.distance) {
        p->next_event_ = Particle::EventType::surface_crossing;
        #pragma omp atomic
        simulation::surface_crossing_queue_length++;
      } else {
        p->next_event_ = Particle::EventType::collision;
        #pragma omp atomic
        simulation::collision_queue_length++;
      }
    }
  }

  simulation::time_event_advance_particle.stop();
}

void process_surface_crossing_events(int64_t n_particles)
{
  simulation::time_event_surface_crossing.start();

  //#pragma omp parallel for schedule(runtime) reduction(+:simulation::calculate_nonfuel_xs_queue_length, simulation::calculate_fuel_xs_queue_length)
  #pragma omp parallel for schedule(runtime)
  for (int64_t i = 0; i < n_particles; i++) {
    Particle* p = &simulation::particles[i]; 
    if (p->next_event_ == Particle::EventType::surface_crossing) {
      p->event_cross_surface();
      p->event_revive_from_secondary();
      if (p->alive_) {
        dispatch_xs_event(p);
      } else {
        #pragma omp atomic
        simulation::dead_particle_count++;
      }
    }
  }
  
  simulation::time_event_surface_crossing.stop();
}

void process_collision_events(int64_t n_particles)
{
  simulation::time_event_collision.start();

  //#pragma omp parallel for schedule(runtime) reduction(+:simulation::calculate_nonfuel_xs_queue_length, simulation::calculate_fuel_xs_queue_length)
  #pragma omp parallel for schedule(runtime)
  for (int64_t i = 0; i < n_particles; i++) {
    Particle* p = &simulation::particles[i]; 
    if (p->next_event_ == Particle::EventType::collision) {
      p->event_collide();
      p->event_revive_from_secondary();
      if (p->alive_) {
        dispatch_xs_event(p);
      } else {
        #pragma omp atomic
        simulation::dead_particle_count++;
      }
    }
  }

  simulation::time_event_collision.stop();
}

void process_death_events(int64_t n_particles)
{
  simulation::time_event_death.start();
  #pragma omp parallel for schedule(runtime)
  for (int64_t i = 0; i < n_particles; i++) {
    Particle* p = &simulation::particles[i];
    p->event_death();
  }
  simulation::time_event_death.stop();
}

void stream_compaction(int64_t n_particles)
{
  std::sort(simulation::particles.get(), simulation::particles.get() + n_particles);
}

} // namespace openmc
