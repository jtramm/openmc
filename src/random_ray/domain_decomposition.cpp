#include "openmc/random_ray/domain_decomposition.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include <fmt/core.h>

#include "openmc/error.h"
#include "openmc/message_passing.h"
#include "openmc/output.h"
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

DomainDecomposition::DomainDecomposition()
{
#ifdef OPENMC_MPI
  active_ = mpi::n_procs > 1;
  if (active_) {
    map_ = make_unique<DecompositionMap>(this);
  }
#endif
}

DomainDecomposition::~DomainDecomposition() = default;

void DomainDecomposition::begin_solve()
{
  if (!active_)
    return;

#ifdef OPENMC_MPI
  // The timers and counters report on one solve at a time
  time_ray_comms_.reset();
  time_ray_buffering_.reset();
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

uint64_t DomainDecomposition::transport_sweep(FlatSourceDomain& domain)
{
  uint64_t n_intersections_total = 0;

#ifdef OPENMC_MPI
  // Divide the domain among the ranks. This is done once, and the division is
  // kept for any later solve.
  if (!partitioned_) {
    time_partition_.start();
    map_->partition(domain);
    time_partition_.stop();
    partitioned_ = true;
  }

  double start_time_transport = simulation::time_transport.elapsed();
  double start_time_ray_buffering = time_ray_buffering_.elapsed();

  time_handling_.start();

  RayBank bank(*map_);

// Create rays and add them to ray bank
#pragma omp parallel for schedule(static)
  for (int i = 0; i < simulation::work_per_rank; i++) {
    uint64_t id = simulation::work_index[mpi::rank] + i;
    RandomRay ray(id, &domain);

    // Add ray to ray bank if it starts in my subdomain
    if (!ray.has_left_subdomain()) {
#pragma omp critical(raybank)
      {
        bank.my_ray_list_.push_back(ray);
      }
    }
    // Put ray straight into buffer if it starts outside my subdomain
    else {
#pragma omp critical(raybuffer)
      {
        bank.buffer_ray_data_to_send(ray, &domain);
      }
    }
  }

  // If no ray is alive at this stage, it means that all of them have been
  // buffered because they were sampled in foregin subdomain. This requires a
  // ray bank update here.
  if (!bank.is_any_ray_alive()) {
    bank.update(&domain);
  }

  // Move rays across ranks until they are terminated
  while (bank.is_any_ray_alive()) {

    // Start timer for transport
    simulation::time_transport.start();

    uint64_t n_intersections = 0;
#pragma omp parallel for schedule(dynamic) reduction(+ : n_intersections)
    for (int i = 0; i < bank.ray_bank_size(); i++) {
      RandomRay& ray = bank.my_ray_list_[i];
      n_intersections += ray.transport_history_based_single_ray();

      // If ray has left my subdomain, buffer ray state
      if (ray.has_left_subdomain()) {
#pragma omp critical(raybuffer)
        {
          time_ray_buffering_.start();
          bank.buffer_ray_data_to_send(ray, &domain);
          time_ray_buffering_.stop();
        }
      }
    }
    n_intersections_total += n_intersections;
    simulation::time_transport.stop();

    // Capture wait time resulting from other transport sweeps
    time_imbalance_.start();
    MPI_Barrier(mpi::intracomm);
    time_imbalance_.stop();

    // Update ray bank by communicating rays in buffer to new owner ranks and
    // removing terminated ranks
    time_ray_comms_.start();
    bank.update(&domain);
    MPI_Barrier(mpi::intracomm);
    time_ray_comms_.stop();

    n_communication_rounds_++;
  }

  // Calculate load per rank based on number of hits in each source region that
  // a rank owns
  double batch_ray_buffering_time =
    time_ray_buffering_.elapsed() - start_time_ray_buffering;
  double batch_transport_time = simulation::time_transport.elapsed() -
                                start_time_transport - batch_ray_buffering_time;

  // Calculate rank load fractions for load balancing
  if (simulation::current_batch <= ITER_LOAD_BALANCE) {
    map_->calculate_rank_load(&domain, batch_transport_time);
  }

  time_handling_.stop();

  // Check if any new source regions discovered and if so, exchange
  // discovered cell data between ranks
  if (map_->any_discovered_source_regions(domain.discovered_source_regions_)) {
    time_source_region_exchange_.start();
    map_->exchange_sr_info(domain.discovered_source_regions_);
    MPI_Barrier(mpi::intracomm);
    time_source_region_exchange_.stop();
  }
#endif

  return n_intersections_total;
}

void DomainDecomposition::balance_load(FlatSourceDomain& domain)
{
#ifdef OPENMC_MPI
  if (!active_ || simulation::current_batch > ITER_LOAD_BALANCE)
    return;

  time_load_balance_.start();
  map_->balance_load(&domain);
  MPI_Barrier(mpi::intracomm);
  time_load_balance_.stop();
#endif
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
  // rank spends waiting on the others counts as transport time, while the
  // time it spends buffering rays for other ranks does not
  return time_transport - time_ray_buffering_.elapsed() +
         time_imbalance_.elapsed();
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
  double time_sweep = time_transport - time_ray_buffering_.elapsed();
  auto& fractions = map_->measured_rank_load_fractions_;
  MPI_Allgather(&time_sweep, 1, MPI_DOUBLE, fractions.data(), 1, MPI_DOUBLE,
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
