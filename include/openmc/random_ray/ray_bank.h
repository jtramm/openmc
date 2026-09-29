#ifndef OPENMC_RAY_BANK_H
#define OPENMC_RAY_BANK_H

#include <cstdint>
#include <unordered_map>

#include "openmc/particle_data.h"
#include "openmc/position.h"
#include "openmc/random_ray/random_ray.h"
#include "openmc/vector.h"

#ifdef OPENMC_DAGMC_ENABLED
#include "DagMC.hpp"
#endif

namespace openmc {

class DecompositionMap;
class FlatSourceDomain;

// Maximum number of DAGMC entity handles to send when exchanging rays
// between MPI ranks. This caps the RayHistory length to avoid sending
// variable-length vectors.
constexpr int MAX_N_HANDLES {5};

// Maximum number of times a random ray may be handed to another MPI rank
// before it is terminated. Bounds a ray ping-ponging across a subdomain
// boundary, which would otherwise hang every rank in the job.
constexpr int MAX_RAY_TRANSFERS {10000};

//! The fixed-size part of the state of a ray that is handed to another MPI
//! rank. The angular flux and the coordinate levels of the ray, whose sizes
//! are only known at runtime, are sent alongside it.
struct RayExchangeData {
  Position position; //!< Where the ray stopped (for diagnostics only)
  Direction direction;
  double distance_travelled;
  // Distance the ray still has to advance from the buffered geometry state to
  // reach `position`. Applied on the receiving rank via move_distance(), which
  // uses each coordinate level's own direction.
  double advance_distance;
  // Number of times this ray has been handed to another rank. A ray is handed
  // off before it registers a geometry event, so max_particle_events cannot
  // bound a forwarding loop.
  int n_transfers;
  int surface;
  bool is_active;
  uint64_t ray_id;
  int n_event; // Number of events (surface crossings) the ray has undergone

  // GeometryState scalar fields
  int n_coord;
  int cell_instance;
  int n_coord_last;
  int material;
  int material_last;
  double sqrtkT;
  double sqrtkT_last;

#ifdef OPENMC_DAGMC_ENABLED
  // DAGMC fields - fixed-size array to avoid variable-length vector
  Direction last_dir;
  moab::EntityHandle handles[MAX_N_HANDLES];
  int n_handles; // Actual number of valid handles (may be less than
                 // MAX_N_HANDLES)
#endif
};

/*
 * The RayBank class moves rays between MPI ranks. A ray that reaches a source
 * region owned by another rank is packed into a send buffer, and the buffers
 * are exchanged between transport rounds. The rays a rank receives are
 * restored to the state they were packed in and continue on that rank.
 */
class RayBank {
public:
  //----------------------------------------------------------------------------
  // Constructors
  explicit RayBank(DecompositionMap& map);

  //----------------------------------------------------------------------------
  // Methods

  //! Stop a ray and pack it for the rank that continues it. Threadsafe.
  //! \param ray The ray, which is terminated on this rank
  //! \param destination The rank that owns the region the ray is entering
  //! \param offset Distance from the position of the ray to the start of the
  //!   segment it was traversing
  //! \param partial Distance from the start of that segment to the point at
  //!   which the ray stops
  void hand_off(RandomRay& ray, int destination, double offset, double partial);

  //! Send the packed rays to their ranks and receive the rays that enter the
  //! subdomain of this rank, which replace the rays held by the bank.
  void exchange(FlatSourceDomain* domain);

  //! Whether any rank holds rays that have yet to be transported
  bool any_rays_alive() const;

  //----------------------------------------------------------------------------
  // Public data members

  // Rays received from other ranks, to be transported by this rank
  vector<RandomRay> rays_;

private:
  //----------------------------------------------------------------------------
  // Private methods
  void merge_thread_buffers();
  void communicate_counts();
  void communicate_rays();
  void restore_rays(FlatSourceDomain* domain);
  void restore(RandomRay& ray, FlatSourceDomain* domain,
    const RayExchangeData& data, const float* angular_flux,
    const LocalCoord* coord, const int* cell_last) const;

  //----------------------------------------------------------------------------
  // Private data members
  DecompositionMap& map_;
  int negroups_;
  int total_receiving_rays_ {0};

  // The rays bound for one rank: the fixed-size data of each ray, and the
  // angular flux and geometry state of all of them laid end to end
  struct SendBuffer {
    vector<RayExchangeData> ray_data;
    vector<float> angular_flux;
    vector<LocalCoord> coord;
    vector<int> cell_last;
  };

  // Send buffers keyed by destination rank. Each thread packs rays into
  // buffers of its own, which are merged before they are sent.
  vector<std::unordered_map<int, SendBuffer>> thread_buffers_;
  std::unordered_map<int, SendBuffer> send_buffers_;

  // Number of rays to be received from and sent to each rank
  vector<int> num_messages_receiving_;
  vector<int> num_messages_sending_;

  // Received ray data
  vector<RayExchangeData> received_ray_data_;
  vector<float> received_angular_flux_data_;
  vector<LocalCoord> received_coord_;
  vector<int> received_cell_last_;

}; // class RayBank

} // namespace openmc

#endif // OPENMC_RAY_BANK_H
