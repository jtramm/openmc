#include "openmc/random_ray/domain_decomposition.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include <fmt/core.h>

#include "openmc/cell.h"
#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/message_passing.h"
#include "openmc/output.h"
#include "openmc/random_lcg.h"
#include "openmc/random_ray/decomposition_map.h"
#include "openmc/random_ray/flat_source_domain.h"
#include "openmc/random_ray/random_ray.h"
#include "openmc/random_ray/ray_bank.h"
#include "openmc/settings.h"
#include "openmc/simulation.h"

namespace openmc {

//==============================================================================
// DomainDecomposition implementation
//==============================================================================

DomainDecomposition::DomainDecomposition(FlatSourceDomain* domain)
  : domain_(domain)
{
#ifdef OPENMC_MPI
  active_ = mpi::n_procs > 1;
  if (active_) {
    map_ = make_unique<DecompositionMap>(this);
    bank_ = make_unique<RayBank>(*map_);
  }
#endif
}

DomainDecomposition::~DomainDecomposition() = default;

void DomainDecomposition::validate_inputs()
{
#ifdef OPENMC_MPI
  if (mpi::n_procs == 1)
    return;

  // Each rank scores the tallies of its own subdomain only, so a tally is
  // only complete once it has been summed over the ranks
  if (!settings::reduce_tallies) {
    fatal_error("The no_reduce setting is not supported in random ray mode "
                "when running on more than one MPI rank.");
  }
#endif
}

void DomainDecomposition::begin_solve()
{
  if (!active_)
    return;

#ifdef OPENMC_MPI
  // The timers and counters report on one solve at a time
  time_ray_comms_.reset();
  time_handling_.reset();
  time_load_balance_.reset();
  time_partition_.reset();
  time_source_region_exchange_.reset();
  time_imbalance_.reset();
  n_communication_rounds_ = 0;

  map_->initialize();
#endif
}

//==============================================================================
// Transport sweep and load balancing
//==============================================================================

uint64_t DomainDecomposition::transport_sweep()
{
  uint64_t n_intersections = 0;

#ifdef OPENMC_MPI
  // Divide the domain among the ranks. This is done once, and the division is
  // kept for any later solve.
  if (!partitioned_) {
    time_partition_.start();
    map_->partition(*domain_);
    time_partition_.stop();
    partitioned_ = true;
  }

  double start_time_transport = simulation::time_transport.elapsed();
  balancing_ = simulation::current_batch <= ITER_LOAD_BALANCE;

  time_handling_.start();

  // Sample the rays of this rank and transport those that start in its
  // subdomain. The others are handed to the ranks they start in, as are the
  // rays that leave the subdomain along the way.
  simulation::time_transport.start();
#pragma omp parallel for schedule(dynamic) reduction(+ : n_intersections)
  for (int i = 0; i < simulation::work_per_rank; i++) {
    uint64_t id = simulation::work_index[mpi::rank] + i;
    RandomRay ray(id, domain_);
    if (ray.alive()) {
      n_intersections += ray.transport_history_based_single_ray();
    }
  }
  simulation::time_transport.stop();

  // Move rays across ranks until they have all terminated
  while (true) {
    // Capture wait time resulting from other transport sweeps
    time_imbalance_.start();
    MPI_Barrier(mpi::intracomm);
    time_imbalance_.stop();

    // Send the rays that left the subdomain to the ranks they entered, and
    // receive the rays that entered it
    time_ray_comms_.start();
    bank_->exchange(domain_);
    MPI_Barrier(mpi::intracomm);
    time_ray_comms_.stop();
    n_communication_rounds_++;

    if (!bank_->any_rays_alive())
      break;

    simulation::time_transport.start();
    uint64_t n_intersections_round = 0;
#pragma omp parallel for schedule(dynamic) reduction(+ : n_intersections_round)
    for (int64_t i = 0; i < bank_->rays_.size(); i++) {
      n_intersections_round +=
        bank_->rays_[i].transport_history_based_single_ray();
    }
    n_intersections += n_intersections_round;
    simulation::time_transport.stop();
  }

  // Estimate the load of each rank from the ray tracing work of the batch
  if (balancing_) {
    double batch_transport_time =
      simulation::time_transport.elapsed() - start_time_transport;
    map_->calculate_rank_load(domain_, batch_transport_time);
  }

  time_handling_.stop();

  // Check if any new source regions discovered and if so, exchange
  // discovered cell data between ranks
  if (map_->any_discovered_source_regions(
        domain_->discovered_source_regions_)) {
    time_source_region_exchange_.start();
    map_->exchange_sr_info(domain_->discovered_source_regions_);
    MPI_Barrier(mpi::intracomm);
    time_source_region_exchange_.stop();
  }
#endif

  return n_intersections;
}

void DomainDecomposition::balance_load()
{
#ifdef OPENMC_MPI
  if (!active_ || simulation::current_batch > ITER_LOAD_BALANCE)
    return;

  time_load_balance_.start();
  map_->balance_load(domain_);
  MPI_Barrier(mpi::intracomm);
  time_load_balance_.stop();
#endif
}

//==============================================================================
// Hooks of the ray transport
//==============================================================================

bool DomainDecomposition::hand_off_at_birth(
  RandomRay& ray, const SourceRegionKey& sr_key)
{
#ifdef OPENMC_MPI
  if (!active_)
    return false;

  int owner =
    map_->find_owner(sr_key, ray.r(), domain_->discovered_source_regions_);
  if (owner == mpi::rank)
    return false;

  // The rank that owns the starting position sets the starting angular flux
  bank_->hand_off_at_birth(ray, owner);
  return true;
#else
  return false;
#endif
}

bool DomainDecomposition::hand_off(RandomRay& ray,
  const SourceRegionKey& sr_key, Position r, double distance, double offset,
  double partial)
{
#ifdef OPENMC_MPI
  if (!active_)
    return false;

  Position midpoint = r + ray.u() * (distance / 2.0);
  int owner =
    map_->find_owner(sr_key, midpoint, domain_->discovered_source_regions_);
  if (owner == mpi::rank)
    return false;

  bank_->hand_off(ray, owner, offset, partial);
  return true;
#else
  return false;
#endif
}

bool DomainDecomposition::skip_crossing(double distance) const
{
  // Very flat angles can result in very small crossings of a mesh bin. At the
  // boundary between two subdomains, such a crossing can make a ray bounce
  // back and forth between two ranks indefinitely.
  return active_ && distance <= TINY_BIT;
}

void DomainDecomposition::tally_ray_trace(
  const RandomRay& ray, int64_t sr, int n_mesh_bins)
{
#ifdef OPENMC_MPI
  // The counters are only read while the load is being balanced. Past that,
  // updating them would be pure contention between threads.
  if (!active_ || !balancing_)
    return;

  // The cost of finding the distance to the boundary grows with the number of
  // surfaces of the cells the ray is in
  uint64_t n_surfaces = 0;
  for (int i = 0; i < ray.n_coord(); i++) {
    n_surfaces += model::cells[ray.coord(i).cell()]->n_surfaces();
  }

  // Threads routinely transport rays through the same base source region, so
  // the counters are shared and must be updated atomically
#pragma omp atomic
  map_->num_base_source_region_RT_[sr] += n_surfaces;
#pragma omp atomic
  map_->num_mesh_bin_RT_[sr] += n_mesh_bins;
#endif
}

//==============================================================================
// Hooks of the plotting
//==============================================================================

bool DomainDecomposition::is_local(const SourceRegionKey& sr_key) const
{
#ifdef OPENMC_MPI
  if (active_) {
    // A source region that no ray has crossed belongs to no rank. The master
    // stands in for its owner.
    auto it = map_->subdomain_map_.find(sr_key);
    int owner = (it != map_->subdomain_map_.end()) ? it->second : 0;
    return owner == mpi::rank;
  }
#endif
  return true;
}

vector<DomainDecomposition::PlotField> DomainDecomposition::plot_fields() const
{
  vector<PlotField> fields;
#ifdef OPENMC_MPI
  if (active_) {
    // The subdomain of each rank, shown in a color of its own
    vector<float> colors(mpi::n_procs);
    for (int rank = 0; rank < mpi::n_procs; rank++) {
      colors[rank] = future_prn(10, rank);
    }
    fields.push_back({"rank_subdomains", colors});

    // The share of each rank in the time spent in transport sweeps
    const auto& load = map_->measured_rank_load_fractions_;
    fields.push_back({"measured_load", {load.begin(), load.end()}});
  }
#endif
  return fields;
}

//==============================================================================
// Reductions
//==============================================================================

#ifdef OPENMC_MPI
namespace {

template<typename T>
void reduce(T* data, size_t n, MPI_Datatype type, MPI_Op op)
{
  if (n > static_cast<size_t>(std::numeric_limits<int>::max())) {
    fatal_error(fmt::format("A reduction over MPI ranks of {} values exceeds "
                            "the maximum MPI count.",
      n));
  }
  MPI_Allreduce(
    MPI_IN_PLACE, data, static_cast<int>(n), type, op, mpi::intracomm);
}

} // namespace
#endif

void DomainDecomposition::sum(double& value) const
{
#ifdef OPENMC_MPI
  if (active_)
    reduce(&value, 1, MPI_DOUBLE, MPI_SUM);
#endif
}

void DomainDecomposition::sum(int64_t& value) const
{
#ifdef OPENMC_MPI
  if (active_)
    reduce(&value, 1, MPI_INT64_T, MPI_SUM);
#endif
}

void DomainDecomposition::sum(uint64_t& value) const
{
#ifdef OPENMC_MPI
  if (active_)
    reduce(&value, 1, MPI_UINT64_T, MPI_SUM);
#endif
}

void DomainDecomposition::sum(double* values, size_t n) const
{
#ifdef OPENMC_MPI
  if (active_)
    reduce(values, n, MPI_DOUBLE, MPI_SUM);
#endif
}

void DomainDecomposition::sum(float* values, size_t n) const
{
#ifdef OPENMC_MPI
  if (active_)
    reduce(values, n, MPI_FLOAT, MPI_SUM);
#endif
}

void DomainDecomposition::sum(int* values, size_t n) const
{
#ifdef OPENMC_MPI
  if (active_)
    reduce(values, n, MPI_INT, MPI_SUM);
#endif
}

void DomainDecomposition::max(double& value) const
{
#ifdef OPENMC_MPI
  if (active_)
    reduce(&value, 1, MPI_DOUBLE, MPI_MAX);
#endif
}

void DomainDecomposition::max(float& value) const
{
#ifdef OPENMC_MPI
  if (active_)
    reduce(&value, 1, MPI_FLOAT, MPI_MAX);
#endif
}

void DomainDecomposition::min(float& value) const
{
#ifdef OPENMC_MPI
  if (active_)
    reduce(&value, 1, MPI_FLOAT, MPI_MIN);
#endif
}

//==============================================================================
// Reporting
//==============================================================================

double DomainDecomposition::transport_time(double time_transport)
{
  if (!active_)
    return time_transport;

  // The sweep of a batch ends when the slowest rank finishes, so the time a
  // rank spends waiting on the others counts as transport time
  return time_transport + time_imbalance_.elapsed();
}

double DomainDecomposition::overhead_time(double time_transport)
{
  if (!active_)
    return 0.0;

  // The handling timer runs for the whole decomposed sweep, transport
  // included
  return time_handling_.elapsed() + time_partition_.elapsed() +
         time_load_balance_.elapsed() + time_source_region_exchange_.elapsed() -
         transport_time(time_transport);
}

void DomainDecomposition::finalize_statistics(double time_transport)
{
  if (!active_)
    return;

#ifdef OPENMC_MPI
  avg_communication_rounds_ = static_cast<uint64_t>(std::round(
    static_cast<double>(n_communication_rounds_) / settings::n_batches));

  // Every rank needs the measured load of all ranks, as it is plotted
  auto& fractions = map_->measured_rank_load_fractions_;
  MPI_Allgather(&time_transport, 1, MPI_DOUBLE, fractions.data(), 1, MPI_DOUBLE,
    mpi::intracomm);
  double load_sum = std::accumulate(fractions.begin(), fractions.end(), 0.0);
  for (auto& fraction : fractions) {
    fraction /= load_sum;
  }
  double max_load = *std::max_element(fractions.begin(), fractions.end());
  max_load_imbalance_ = (max_load - map_->target_load_) / map_->target_load_;
#endif
}

void DomainDecomposition::print_statistics() const
{
  if (!active_)
    return;

  fmt::print(" MPI Ranks                         = {}\n", mpi::n_procs);
  fmt::print(
    " Avg Ray Subdomain Crossings       = {}\n", avg_communication_rounds_);
  fmt::print(" Maximum Load Imbalance            = {:.2f}%\n",
    max_load_imbalance_ * 100.0);
}

void DomainDecomposition::print_timing(double time_transport)
{
  if (!active_)
    return;

  double time_overhead = overhead_time(time_transport);
  double time_other = time_overhead - time_source_region_exchange_.elapsed() -
                      time_partition_.elapsed() - time_ray_comms_.elapsed() -
                      time_load_balance_.elapsed();
  show_time("Decomposition handling", time_overhead, 1);
  show_time("Ray communication", time_ray_comms_.elapsed(), 2);
  show_time(
    "Exchanging contested SRs", time_source_region_exchange_.elapsed(), 2);
  show_time("Load balancing", time_load_balance_.elapsed(), 2);
  show_time("Generating Voronoi centers", time_partition_.elapsed(), 2);
  show_time("Other decomposition routines", time_other, 2);
}

} // namespace openmc
