#ifndef OPENMC_BANK_H
#define OPENMC_BANK_H

#include <cstdint>
#include <vector>

#include "openmc/shared_array.h"
#include "openmc/particle.h"
#include "openmc/position.h"

namespace openmc {

//==============================================================================
// Global variables
//==============================================================================

namespace simulation {

extern std::vector<Particle::Bank> source_bank;
#ifdef OPENMC_OFFLOAD
#pragma omp declare target
#endif
extern Particle::Bank* device_source_bank;
#ifdef OPENMC_OFFLOAD
#pragma omp end declare target
#endif

extern SharedArray<Particle::Bank> surf_source_bank;

#ifdef OPENMC_OFFLOAD
#pragma omp declare target
#endif
extern SharedArray<Particle::Bank> fission_bank;
#ifdef OPENMC_OFFLOAD
#pragma omp end declare target
#endif

extern std::vector<int64_t> progeny_per_particle;

#ifdef OPENMC_OFFLOAD
#pragma omp declare target
#endif
extern int64_t* device_progeny_per_particle;
#ifdef OPENMC_OFFLOAD
#pragma omp end declare target
#endif

} // namespace simulation

//==============================================================================
// Non-member functions
//==============================================================================

void sort_fission_bank();

void free_memory_bank();

void init_fission_bank(int64_t max);

} // namespace openmc

#endif // OPENMC_BANK_H
