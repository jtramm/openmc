#include "openmc/random_ray/ray_bank.h"

#include <cstring>
#include <limits>
#include <numeric>

#include <fmt/core.h>

#include "openmc/error.h"
#include "openmc/geometry.h"
#include "openmc/message_passing.h"
#include "openmc/mgxs_interface.h"
#include "openmc/openmp_interface.h"
#include "openmc/random_ray/decomposition_map.h"
#include "openmc/random_ray/flat_source_domain.h"
#include "openmc/random_ray/source_region.h"

#ifdef OPENMC_MPI

namespace openmc {

RayBank::RayBank(DecompositionMap& map) : map_(map)
{
  negroups_ = data::mg.num_energy_groups_;
  num_messages_receiving_.resize(mpi::n_procs, 0);
  num_messages_sending_.resize(mpi::n_procs, 0);
  thread_buffers_.resize(num_threads());
}

void RayBank::hand_off(
  RandomRay& ray, int destination, double offset, double partial)
{
  pack(ray, destination, offset, partial, false);
}

void RayBank::hand_off_at_birth(RandomRay& ray, int destination)
{
  pack(ray, destination, 0.0, 0.0, true);
}

void RayBank::pack(RandomRay& ray, int destination, double offset,
  double partial, bool is_newborn)
{
  if (destination == mpi::rank) {
    // Buffering a ray back to its own rank would send it straight back
    // unchanged on the next round and loop forever. Fail loudly instead.
    fatal_error(fmt::format("Ray {} at position ({:.5e}, {:.5e}, {:.5e}) is "
                            "being sent to the same rank {}. This indicates "
                            "an error in the decomposition map.",
      ray.id(), ray.r().x, ray.r().y, ray.r().z, destination));
  }

  // A ray that stops inside a segment, rather than at its start, is no longer
  // on the surface it last crossed
  if (partial > 0.0) {
    ray.surface() = 0;
  }

#ifdef OPENMC_DAGMC_ENABLED
  ray.history().rollback_last_intersection();
#endif

  // The geometry state of the ray is that of the last geometry update, so the
  // receiving rank has to advance the ray to the point at which it stops
  double advance_distance = offset + partial;

  RayExchangeData data;
  data.position = ray.r() + advance_distance * ray.u();
  data.direction = ray.u();
  data.distance_travelled = ray.distance_travelled_ + partial;
  data.advance_distance = advance_distance;
  data.n_transfers = ++ray.n_transfers_;
  data.surface = ray.surface();
  data.is_active = ray.is_active_;
  data.is_newborn = is_newborn;
  data.ray_id = ray.id();
  data.n_event = ray.n_event();

  // GeometryState scalar fields
  data.n_coord = ray.n_coord();
  data.cell_instance = ray.cell_instance();
  data.n_coord_last = ray.n_coord_last();
  data.material = ray.material();
  data.material_last = ray.material_last();
  data.sqrtkT = ray.sqrtkT();
  data.sqrtkT_last = ray.sqrtkT_last();

#ifdef OPENMC_DAGMC_ENABLED
  // Extract up to MAX_N_HANDLES from the ray history by rolling back
  data.last_dir = ray.last_dir();
  data.n_handles = 0;
  for (int i = 0; i < MAX_N_HANDLES; i++) {
    moab::EntityHandle handle;
    if (ray.history().get_last_intersection(handle) == moab::MB_SUCCESS) {
      data.handles[i] = handle;
      data.n_handles++;
      ray.history().rollback_last_intersection();
    } else {
      // No more handles in history
      break;
    }
  }
#endif

  // Each thread packs into buffers of its own, so no lock is needed
  SendBuffer& buffer = thread_buffers_[thread_num()][destination];
  buffer.ray_data.push_back(data);
  buffer.angular_flux.insert(buffer.angular_flux.end(),
    ray.angular_flux_.begin(), ray.angular_flux_.end());

  // LocalCoord is POD, so the coordinate levels can be sent as bytes. All
  // model::n_coord_levels of them are packed so that every ray has the same
  // size.
  for (int i = 0; i < model::n_coord_levels; i++) {
    buffer.coord.push_back(ray.coord(i));
    buffer.cell_last.push_back(ray.cell_last(i));
  }

  // Terminate the ray on this rank
  ray.wgt() = 0.0;
}

void RayBank::exchange(FlatSourceDomain* domain)
{
  // The rays held by the bank have either terminated or been packed for
  // another rank
  rays_.resize(0);

  merge_thread_buffers();
  communicate_counts();
  communicate_rays();
  restore_rays(domain);
}

void RayBank::merge_thread_buffers()
{
  for (auto& buffers : thread_buffers_) {
    for (auto& [rank, buffer] : buffers) {
      SendBuffer& merged = send_buffers_[rank];
      merged.ray_data.insert(
        merged.ray_data.end(), buffer.ray_data.begin(), buffer.ray_data.end());
      merged.angular_flux.insert(merged.angular_flux.end(),
        buffer.angular_flux.begin(), buffer.angular_flux.end());
      merged.coord.insert(
        merged.coord.end(), buffer.coord.begin(), buffer.coord.end());
      merged.cell_last.insert(merged.cell_last.end(), buffer.cell_last.begin(),
        buffer.cell_last.end());
    }
    buffers.clear();
  }
}

// Tells each rank how many rays to receive from who and how many rays to send
// to who
void RayBank::communicate_counts()
{
  // Ensure all values are zero in vector for receiving counts
  fill(num_messages_receiving_.begin(), num_messages_receiving_.end(), 0);
  fill(num_messages_sending_.begin(), num_messages_sending_.end(), 0);

  // Fill the sending counts
  for (auto& [rank, buffer] : send_buffers_) {
    num_messages_sending_[rank] = buffer.ray_data.size();
  }

  // Exchange message counts with all ranks
  MPI_Alltoall(num_messages_sending_.data(), 1, MPI_INT,
    num_messages_receiving_.data(), 1, MPI_INT, mpi::intracomm);

  total_receiving_rays_ = accumulate(
    num_messages_receiving_.begin(), num_messages_receiving_.end(), 0);
}

void RayBank::communicate_rays()
{
  // Get the maximum coordinate levels
  const int n_coord_max = model::n_coord_levels;

  // Allocate receiving buffers
  // int64_t intermediates: the products below overflow int well before the
  // ray counts themselves do, which would under-size the receive buffers and
  // let the MPI_Irecv calls write past their end
  int64_t n_recv = total_receiving_rays_;
  received_ray_data_.resize(n_recv);
  received_angular_flux_data_.resize(n_recv * negroups_);
  received_coord_.resize(n_recv * n_coord_max);
  received_cell_last_.resize(n_recv * n_coord_max);

  // Calculate total number of MPI requests needed
  // 4 messages per sending rank + 4 messages per receiving rank
  int num_send_ranks = send_buffers_.size();
  int num_recv_ranks = 0;
  for (int i = 0; i < mpi::n_procs; i++) {
    if (num_messages_receiving_[i] > 0)
      num_recv_ranks++;
  }

  // MPI counts are int. The byte products below are computed in size_t and
  // narrowed at the call, and because sender and receiver narrow identically
  // a wrapped count matches on both sides: MPI transfers the truncated
  // payload without error and the restarted rays read stale buffer contents.
  // Fail loudly instead.
  auto check_count = [](int64_t count, const char* what) {
    if (count > std::numeric_limits<int>::max()) {
      fatal_error(fmt::format("Random ray domain decomposition: {} message of "
                              "{} elements exceeds the maximum MPI count. Use "
                              "more MPI ranks or fewer rays per batch.",
        what, count));
    }
  };
  for (int r = 0; r < mpi::n_procs; r++) {
    int64_t n = num_messages_receiving_[r];
    check_count(n * static_cast<int64_t>(sizeof(RayExchangeData)), "ray data");
    check_count(n * negroups_, "angular flux");
    check_count(
      n * n_coord_max * static_cast<int64_t>(sizeof(LocalCoord)), "coord");
    check_count(n * n_coord_max, "cell_last");
  }

  int total_requests = num_send_ranks * 4 + num_recv_ranks * 4;
  vector<MPI_Request> requests(total_requests);
  int req_idx = 0;

  // Post all non-blocking receives first to allow for potential overlap
  // of communication and packing of send buffers
  int64_t recv_offset = 0;
  for (int sending_rank = 0; sending_rank < mpi::n_procs; sending_rank++) {
    int num_rays_receiving = num_messages_receiving_[sending_rank];
    if (num_rays_receiving == 0)
      continue;

    MPI_Irecv(&received_ray_data_[recv_offset],
      num_rays_receiving * sizeof(RayExchangeData), MPI_BYTE, sending_rank, 1,
      mpi::intracomm, &requests[req_idx++]);

    MPI_Irecv(&received_angular_flux_data_[recv_offset * negroups_],
      num_rays_receiving * negroups_, MPI_FLOAT, sending_rank, 2,
      mpi::intracomm, &requests[req_idx++]);

    MPI_Irecv(&received_coord_[recv_offset * n_coord_max],
      num_rays_receiving * n_coord_max * sizeof(LocalCoord), MPI_BYTE,
      sending_rank, 3, mpi::intracomm, &requests[req_idx++]);

    MPI_Irecv(&received_cell_last_[recv_offset * n_coord_max],
      num_rays_receiving * n_coord_max, MPI_INT, sending_rank, 4,
      mpi::intracomm, &requests[req_idx++]);

    recv_offset += num_rays_receiving;
  }

  // Post all non-blocking sends!
  for (auto& [receiving_rank, buffer] : send_buffers_) {
    int num_rays = buffer.ray_data.size();

    // Check neighbor list and add if not already known
    // Filter out rays that are sampled elsewhere
    bool has_transported_rays = false;
    for (int i = 0; i < num_rays; i++) {
      if (!buffer.ray_data[i].is_newborn) {
        has_transported_rays = true;
        break;
      }
    }
    if (has_transported_rays) {
      map_.my_neighbors_.insert(receiving_rank);
    }

    // Send all 4 data arrays - already packed, no intermediate copying!
    MPI_Isend(buffer.ray_data.data(), num_rays * sizeof(RayExchangeData),
      MPI_BYTE, receiving_rank, 1, mpi::intracomm, &requests[req_idx++]);

    MPI_Isend(buffer.angular_flux.data(), num_rays * negroups_, MPI_FLOAT,
      receiving_rank, 2, mpi::intracomm, &requests[req_idx++]);

    MPI_Isend(buffer.coord.data(), num_rays * n_coord_max * sizeof(LocalCoord),
      MPI_BYTE, receiving_rank, 3, mpi::intracomm, &requests[req_idx++]);

    MPI_Isend(buffer.cell_last.data(), num_rays * n_coord_max, MPI_INT,
      receiving_rank, 4, mpi::intracomm, &requests[req_idx++]);
  }

  // Wait for all communication to complete
  MPI_Waitall(req_idx, requests.data(), MPI_STATUSES_IGNORE);

  // Clear send buffers
  send_buffers_.clear();
}

void RayBank::restore_rays(FlatSourceDomain* domain)
{
  rays_.resize(received_ray_data_.size());

  const int n_coord_max = model::n_coord_levels;

#pragma omp parallel for
  for (int64_t i = 0; i < received_ray_data_.size(); i++) {
    restore(rays_[i], domain, received_ray_data_[i],
      &received_angular_flux_data_[i * negroups_],
      &received_coord_[i * n_coord_max], &received_cell_last_[i * n_coord_max]);
  }

  // Clear received data vectors
  received_ray_data_.resize(0);
  received_angular_flux_data_.resize(0);
  received_coord_.resize(0);
  received_cell_last_.resize(0);
}

// Returns a ray to the state in which it was packed on another rank
void RayBank::restore(RandomRay& ray, FlatSourceDomain* domain,
  const RayExchangeData& data, const float* angular_flux,
  const LocalCoord* coord, const int* cell_last) const
{
  ray.domain_ = domain;
  ray.distance_travelled_ = data.distance_travelled;
  ray.n_transfers_ = data.n_transfers;

  // A ray that keeps being handed between ranks without ever depositing is
  // a sign that two ranks disagree about who owns a region. Terminate it
  // rather than letting it hang every rank in the job.
  if (ray.n_transfers_ > MAX_RAY_TRANSFERS) {
    warning("Ray " + std::to_string(data.ray_id) +
            " was transferred between "
            "MPI ranks " +
            std::to_string(ray.n_transfers_) + " times, terminating ray.");
    ray.wgt() = 0.0;
    return;
  }
  ray.ntemperature_ = domain->ntemperature_;

  // Restore particle event counter from the transmitted ray
  // This preserves the event count across MPI rank boundaries
  ray.n_event() = data.n_event;

  ray.is_active_ = data.is_active;

  ray.wgt() = 1.0;

  // set identifier for particle
  ray.id() = data.ray_id;

  // Restore GeometryState scalar fields
  ray.n_coord() = data.n_coord;
  ray.cell_instance() = data.cell_instance;
  ray.n_coord_last() = data.n_coord_last;
  ray.material() = data.material;
  ray.material_last() = data.material_last;
  ray.sqrtkT() = data.sqrtkT;
  ray.sqrtkT_last() = data.sqrtkT_last;
  ray.surface() = data.surface;

  // Restore the coordinate levels. The vectors were already sized to
  // model::n_coord_levels in the GeometryState constructor.
  for (int i = 0; i < model::n_coord_levels; i++) {
    ray.coord(i) = coord[i];
    ray.cell_last(i) = cell_last[i];
  }

  ray.u() = data.direction;

  // The buffered coordinate levels hold the ray's state at the last geometry
  // update, so the ray still has to be advanced to the point at which it left
  // the previous subdomain. Use move_distance() rather than adding a single
  // top-level displacement to every level: a rotated universe or lattice fill
  // gives each level its own direction, and translating a lower level by the
  // top-level displacement would put it somewhere else entirely.
  ray.move_distance(data.advance_distance);

#ifdef OPENMC_DAGMC_ENABLED
  // Restore last_dir and rebuild the history by adding entities in reverse
  // order
  ray.last_dir() = data.last_dir;
  for (int i = data.n_handles - 1; i >= 0; i--) {
    ray.history().add_entity(data.handles[i]);
  }
#endif

  // Set particle type and energy (for random ray, these are not actually used)
  ray.type() = ParticleType::neutron();
  ray.E() = 0.0;

  // No need to call exhaustive_find_cell() since we have the full geometry
  // state! Just verify we have valid cell information
  if (ray.lowest_coord().cell() == C_NONE) {
    ray.mark_as_lost("Received particle " + std::to_string(ray.id()) +
                     " with invalid cell information");
  }

  // Set birth cell attribute if not set
  if (ray.cell_born() == C_NONE)
    ray.cell_born() = ray.lowest_coord().cell();

  if (data.is_newborn) {
    // Initialize ray's starting angular flux to starting location's isotropic
    // source
    SourceRegionKey sr_key = domain->lookup_source_region_key(ray);
    SourceRegionHandle srh =
      domain->get_subdivided_source_region_handle(sr_key, ray.r(), ray.u());

    if (!srh.is_numerical_fp_artifact_) {
      for (int g = 0; g < negroups_; g++) {
        ray.angular_flux_[g] = srh.source(g);
      }
    }
  } else {
    // Set ray's angular flux to value before subdomain change
    for (int g = 0; g < negroups_; g++) {
      ray.angular_flux_[g] = angular_flux[g];
    }
  }
}

bool RayBank::any_rays_alive() const
{
  int flag = rays_.empty() ? 0 : 1;
  MPI_Allreduce(MPI_IN_PLACE, &flag, 1, MPI_INT, MPI_MAX, mpi::intracomm);
  return flag > 0;
}

} // namespace openmc

#endif // OPENMC_MPI
