#ifndef OPENMC_RANDOM_RAY_DOMAIN_DECOMPOSITION_H
#define OPENMC_RANDOM_RAY_DOMAIN_DECOMPOSITION_H

#include <cstddef>
#include <cstdint>
#include <string>

#include "openmc/memory.h"
#include "openmc/position.h"
#include "openmc/timer.h"
#include "openmc/vector.h"

namespace openmc {

// Forward declarations
class DecompositionMap;
class FlatSourceDomain;
class RandomRay;
class RayBank;
class SourceRegionKey;

/*
 * The DomainDecomposition class is the single point of contact between the
 * random ray solver and its MPI domain decomposition. The solver calls the
 * methods below unconditionally. When the run is not decomposed (a build
 * without MPI, or a single MPI rank) every one of them is a no-op, so the
 * solver needs no knowledge of whether, or how, the problem is decomposed.
 *
 * When the run is decomposed, the spatial domain is partitioned into one
 * subdomain per MPI rank. Each rank stores and updates only the source
 * regions of its own subdomain, and a ray that reaches a source region owned
 * by another rank is handed to that rank to continue. The partitioning, the
 * exchange of rays and source regions between ranks, and the load balancing
 * are implemented by the classes this one owns and are not visible to the
 * solver.
 */
class DomainDecomposition {
public:
  //----------------------------------------------------------------------------
  // Constructors and destructor

  //! \param domain The source regions of this rank
  explicit DomainDecomposition(FlatSourceDomain* domain);
  ~DomainDecomposition();

  // The decomposition refers back to the domain that owns it
  DomainDecomposition(const DomainDecomposition&) = delete;
  DomainDecomposition& operator=(const DomainDecomposition&) = delete;

  //----------------------------------------------------------------------------
  // Methods

  //! Check that the settings of the run are compatible with a decomposed
  //! problem. Called before the solver is set up.
  static void validate_inputs();

  //! Whether the problem is decomposed over more than one MPI rank
  bool active() const { return active_; }

  //! Prepare for a solve. Called once at the start of each solve (a forward
  //! solve and an adjoint solve each count as one).
  void begin_solve();

  //! Transport sweep over all random rays of a batch, in which a ray that
  //! leaves the subdomain of this rank is handed to the rank it enters.
  //! \return The number of geometric intersections made on this rank
  uint64_t transport_sweep();

  //! Move source regions between ranks to even out the work of the ranks.
  //! Does nothing past the first few batches of a solve.
  void balance_load();

  //----------------------------------------------------------------------------
  // Hooks of the ray transport

  //! Hand a newly sampled ray to the rank that owns its starting position.
  //! \param ray The ray
  //! \param sr_key The source region the ray starts in
  //! \return Whether the ray was handed off, in which case it is terminated
  //!   on this rank
  bool hand_off_at_birth(RandomRay& ray, const SourceRegionKey& sr_key);

  //! Hand a ray to the rank that owns the source region it is about to
  //! cross, if that is not this rank.
  //! \param ray The ray
  //! \param sr_key The source region the ray is about to cross
  //! \param r Position at which the ray enters the source region
  //! \param distance Length of the crossing
  //! \param offset Distance from the position of the ray to the start of
  //!   the geometric segment the crossing is part of
  //! \param partial Distance from the start of that segment to \p r
  //! \return Whether the ray was handed off, in which case it is terminated
  //!   on this rank
  bool hand_off(RandomRay& ray, const SourceRegionKey& sr_key, Position r,
    double distance, double offset, double partial);

  //! Whether a crossing is too short to be attributed to a source region
  //! reliably when the problem is decomposed, and is to be skipped
  bool skip_crossing(double distance) const;

  //! Record the ray tracing work of a geometric segment, from which the load
  //! of each rank is estimated
  //! \param ray The ray
  //! \param sr The base source region the segment lies in
  //! \param n_mesh_bins The number of mesh bins the segment crosses
  void tally_ray_trace(const RandomRay& ray, int64_t sr, int n_mesh_bins);

  //----------------------------------------------------------------------------
  // Hooks of the plotting

  //! Whether a source region belongs to this rank, which holds its data
  bool is_local(const SourceRegionKey& sr_key) const;

  //! A quantity that describes the decomposition and is added to the plots of
  //! the solver
  struct PlotField {
    std::string name;
    vector<float> rank_values; //!< Value of the quantity on each rank
  };

  //! The quantities to add to the plots, none if the problem is not
  //! decomposed
  vector<PlotField> plot_fields() const;

  //----------------------------------------------------------------------------
  // Reductions over MPI ranks. Each replaces the value on every rank by the
  // combination of the values of all ranks.
  void sum(double& value) const;
  void sum(int64_t& value) const;
  void sum(uint64_t& value) const;
  void sum(double* values, size_t n) const;
  void sum(float* values, size_t n) const;
  void sum(int* values, size_t n) const;
  void max(double& value) const;
  void max(float& value) const;
  void min(float& value) const;

  //----------------------------------------------------------------------------
  // Reporting

  //! The transport sweep time to report, given the time this rank spent in
  //! its own sweeps
  double transport_time(double time_transport);

  //! Total time spent on decomposition tasks outside of the transport sweep
  double overhead_time(double time_transport);

  //! Gather the statistics reported at the end of a solve
  void finalize_statistics(double time_transport);

  //! Print the decomposition lines of the simulation statistics
  void print_statistics() const;

  //! Print the decomposition lines of the timing statistics
  void print_timing(double time_transport);

  //----------------------------------------------------------------------------
  // Accessors used by the decomposition classes themselves
  DecompositionMap& map() { return *map_; }
  const DecompositionMap& map() const { return *map_; }

  //----------------------------------------------------------------------------
  // Public data members

  // Timers of the decomposition tasks
  Timer time_ray_comms_;              //!< Exchange of rays between ranks
  Timer time_handling_;               //!< All decomposition tasks of the sweep
  Timer time_load_balance_;           //!< Load balancing
  Timer time_partition_;              //!< Generation of the initial partition
  Timer time_source_region_exchange_; //!< Exchange of contested regions
  Timer time_imbalance_;              //!< Waiting on the sweeps of other ranks

  // Number of rounds of ray exchange, summed over the batches of the solve
  uint64_t n_communication_rounds_ {0};

private:
  //----------------------------------------------------------------------------
  // Private data members
  bool active_ {false};
  bool partitioned_ {false}; //!< Has the domain been divided among the ranks?
  bool balancing_ {false};   //!< Is the load still being balanced?
  FlatSourceDomain* domain_; //!< The source regions of this rank
  unique_ptr<DecompositionMap> map_;
  unique_ptr<RayBank> bank_;

  // Statistics gathered by finalize_statistics()
  uint64_t avg_communication_rounds_ {0};
  double max_load_imbalance_ {0.0};
  int64_t n_regions_moved_ {0};  //!< Source regions moved by load balancing
  int64_t n_regions_merged_ {0}; //!< Source regions found by several ranks

}; // class DomainDecomposition

} // namespace openmc

#endif // OPENMC_RANDOM_RAY_DOMAIN_DECOMPOSITION_H
