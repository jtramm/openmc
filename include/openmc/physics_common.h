//! \file physics_common.h
//! A collection of physics methods common to MG, CE, photon, etc.

#ifndef OPENMC_PHYSICS_COMMON_H
#define OPENMC_PHYSICS_COMMON_H

#include "openmc/particle.h"

namespace openmc {

//! \brief Performs the russian roulette operation for a particle
//! \param[in,out] p  Particle object
//! \param[in] weight_survive Weight assigned to particles that survive
#ifdef OPENMC_OFFLOAD
#pragma omp declare target
#endif
void russian_roulette(Particle& p, double weight_survive);
#ifdef OPENMC_OFFLOAD
#pragma omp end declare target
#endif

} // namespace openmc
#endif // OPENMC_PHYSICS_COMMON_H
