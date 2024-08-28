#ifndef OPENMC_MESSAGE_PASSING_H
#define OPENMC_MESSAGE_PASSING_H

#ifdef OPENMC_MPI
#include <mpi.h>
#endif

namespace openmc {
namespace mpi {

  #ifdef OPENMC_OFFLOAD
#pragma omp declare target
#endif
  extern int rank;
  extern int n_procs;
  extern bool master;
  #ifdef OPENMC_OFFLOAD
#pragma omp end declare target
#endif

#ifdef OPENMC_MPI
  extern MPI_Datatype bank;
  extern MPI_Comm intracomm;
#endif

} // namespace mpi
} // namespace openmc

#endif // OPENMC_MESSAGE_PASSING_H
