#include "openmc/random_ray/flat_source_domain.h"

#include "openmc/cell.h"
#include "openmc/geometry.h"
#include "openmc/material.h"
#include "openmc/message_passing.h"
#include "openmc/mgxs_interface.h"
#include "openmc/output.h"
#include "openmc/plot.h"
#include "openmc/random_ray/random_ray.h"
#include "openmc/simulation.h"
#include "openmc/tallies/filter.h"
#include "openmc/tallies/tally.h"
#include "openmc/tallies/tally_scoring.h"
#include "openmc/timer.h"

namespace openmc {

void FlatSourceRegion::merge(FlatSourceRegion& other)
{
  // Find the distance between the two positions
  double d = (other.position_ - position_).norm();
  // Find the normalized direction vector connecting the two positions
  Position u = (other.position_ - position_) / d;

  // compute the fraction of the total volume that the other FSR represents of
  // the pair
  double v = other.volume_ / (volume_ + other.volume_);

  // Move the position of the FSR to the new position (volume weighted point
  // between the two positions)
  position_ += u * d * v;

  // Add the volume to this FSR
  volume_t_ += other.volume_t_;

  // We have a problem: it's possible that the tally tasks are not the same, as
  // they might be in different cartesian mesh bins and thus different tally
  // regions. Given that the other is very small, we might be fine just
  // forgetting about its tasks, as its contributions would be small. However,
  // it's not great, as we are sort of stealing from peter to pay paul with
  // this. We fix the volume issue, but then we are stealing volume from a tally
  // space region. There is no good solution to this problem?

  // I think that the best solution is to just forget about the other's tally
  // tasks, as we need to protect transport and the eigenvalue -- the tallies
  // are not nearly as sensitive.

  // TODO: maybe I will warn if the tally tasks are not the same.

  // We discard all the flux data etc. The idea is that flux has units per
  // volume, so we can't just add them together. Also, by definition this
  // FSR had only like 1 or 0 hits for the last iteration or two, so its
  // flux estimate is garbage anyhow. Given that we are in the inactive
  // region, we don't really care here too much.
}

//==============================================================================
// FlatSourceDomain implementation
//==============================================================================

// Static variables
std::unordered_map<int32_t, int32_t> FlatSourceDomain::mesh_map_;
vector<unique_ptr<Mesh>> FlatSourceDomain::meshes_;

FlatSourceDomain::FlatSourceDomain() : negroups_(data::mg.num_energy_groups_)
{
  // Count the number of source regions, compute the cell offset
  // indices, and store the material type The reason for the offsets is that
  // some cell types may not have material fills, and therefore do not
  // produce FSRs. Thus, we cannot index into the global arrays directly
  for (auto&& c : model::cells) {
    if (c->type_ != Fill::MATERIAL) {
      source_region_offsets_.push_back(-1);
    } else {
      source_region_offsets_.push_back(n_source_regions_);
      n_source_regions_ += c->n_instances_;
      n_source_elements_ += c->n_instances_ * negroups_;
    }
  }
  // Initialize FSRs
  fsr_.assign(n_source_regions_, negroups_);

  // If in eigenvalue mode, set starting flux to guess of unity
  if (settings::run_mode == RunMode::EIGENVALUE) {
#pragma omp parallel for
    for (int i = 0; i < n_source_regions_; i++) {
      auto& fsr = fsr_[i];
      std::fill(fsr.scalar_flux_old_.begin(), fsr.scalar_flux_old_.end(), 1.0f);
    }
  }

  // Initialize material array
  int64_t source_region_id = 0;
  for (int i = 0; i < model::cells.size(); i++) {
    Cell& cell = *model::cells[i];
    if (cell.type_ == Fill::MATERIAL) {
      for (int j = 0; j < cell.n_instances_; j++) {
        fsr_[source_region_id++].material_ = cell.material(j);
      }
    }
  }

  // Sanity check
  if (source_region_id != n_source_regions_) {
    fatal_error("Unexpected number of source regions");
  }

  // The mesh has grid map has 5 dimensions:
  // 1. The mesh index
  // 2. z index
  // 3. y index
  // 4. x index
  // 5. Hash of the FSR
  int64_t n_total_bins mesh_hash_grid_.resize(meshes_.size());
  for (int d1 = 0; d1 < meshes_.size(); d1++) {
    Mesh* m = meshes_[d1].get();
    RegularMesh* mesh = dynamic_cast<RegularMesh*>(m);
    mesh_hash_grid_[d1].resize(mesh->shape_[2]);
    for (int d2 = 0; d2 < mesh->shape_[2]; d2++) {
      mesh_hash_grid_[d1][d2].resize(mesh->shape_[1]);
      for (int d3 = 0; d3 < mesh->shape_[1]; d3++) {
        mesh_hash_grid_[d1][d2][d3].resize(mesh->shape_[0]);
      }
    }
  }
}

void FlatSourceDomain::batch_reset()
{
// Reset scalar fluxes, iteration volume tallies, and region hit flags to
// zero
#pragma omp parallel for
  for (int i = 0; i < fsr_manifest_.size(); i++) {
    FlatSourceRegion& fsr = fsr_manifest_[i];
    std::fill(fsr.scalar_flux_new_.begin(), fsr.scalar_flux_new_.end(), 0.0f);
    fsr.volume_ = 0.0;
    fsr.was_hit_ = 0;
  }
}

void FlatSourceDomain::accumulate_iteration_flux()
{
#pragma omp parallel for
  for (int i = 0; i < fsr_manifest_.size(); i++) {
    FlatSourceRegion& fsr = fsr_manifest_[i];
    for (int e = 0; e < negroups_; e++) {
      fsr.scalar_flux_final_[e] += fsr.scalar_flux_new_[e];
    }
  }
}

// Compute new estimate of scattering + fission sources in each source region
// based on the flux estimate from the previous iteration.
void FlatSourceDomain::prepare_base_neutron_source(double k_eff)
{
  simulation::time_update_src.start();

  double inverse_k_eff = 1.0 / k_eff;

  // Temperature and angle indices, if using multiple temperature
  // data sets and/or anisotropic data sets.
  // TODO: Currently assumes we are only using single temp/single
  // angle data.
  const int t = 0;
  const int a = 0;

  // Add scattering source
#pragma omp parallel for
  for (int sr = 0; sr < n_source_regions_; sr++) {
    auto& fsr = fsr_[sr];
    int material = fsr.material_;

    for (int e_out = 0; e_out < negroups_; e_out++) {
      float sigma_t = data::mg.macro_xs_[material].get_xs(
        MgxsType::TOTAL, e_out, nullptr, nullptr, nullptr, t, a);
      float scatter_source = 0.0f;

      for (int e_in = 0; e_in < negroups_; e_in++) {
        float scalar_flux = fsr.scalar_flux_old_[e_in];

        float sigma_s = data::mg.macro_xs_[material].get_xs(
          MgxsType::NU_SCATTER, e_in, &e_out, nullptr, nullptr, t, a);
        scatter_source += sigma_s * scalar_flux;
      }

      fsr.source_[e_out] = scatter_source / sigma_t;
    }
  }

  if (settings::run_mode == RunMode::EIGENVALUE) {
    // Add fission source if in eigenvalue mode
#pragma omp parallel for
    for (int sr = 0; sr < n_source_regions_; sr++) {
      auto& fsr = fsr_[sr];
      int material = fsr.material_;

      for (int e_out = 0; e_out < negroups_; e_out++) {
        float sigma_t = data::mg.macro_xs_[material].get_xs(
          MgxsType::TOTAL, e_out, nullptr, nullptr, nullptr, t, a);
        float fission_source = 0.0f;

        for (int e_in = 0; e_in < negroups_; e_in++) {
          float scalar_flux = fsr.scalar_flux_old_[e_in];
          float nu_sigma_f = data::mg.macro_xs_[material].get_xs(
            MgxsType::NU_FISSION, e_in, nullptr, nullptr, nullptr, t, a);
          float chi = data::mg.macro_xs_[material].get_xs(
            MgxsType::CHI_PROMPT, e_in, &e_out, nullptr, nullptr, t, a);
          fission_source += nu_sigma_f * scalar_flux * chi;
        }
        fsr.source_[e_out] += fission_source * inverse_k_eff / sigma_t;
      }
    }
  } else {
// Add fixed source source if in fixed source mode
#pragma omp parallel for
    for (int i = 0; i < n_source_regions_; i++) {
      auto& fsr = fsr_[i];
      for (int e = 0; e < negroups_; e++) {
        fsr.source_[e] += fsr.fixed_source_[e];
      }
    }
  }

  simulation::time_update_src.stop();
}

// Compute new estimate of scattering + fission sources in each source region
// based on the flux estimate from the previous iteration.
void FlatSourceDomain::update_neutron_source(double k_eff)
{
  simulation::time_update_src.start();

  double inverse_k_eff = 1.0 / k_eff;

  // Temperature and angle indices, if using multiple temperature
  // data sets and/or anisotropic data sets.
  // TODO: Currently assumes we are only using single temp/single
  // angle data.
  const int t = 0;
  const int a = 0;

#pragma omp parallel for
  for (int i = 0; i < fsr_manifest_.size(); i++) {
    FlatSourceRegion& fsr = fsr_manifest_[i];
    for (int e_out = 0; e_out < negroups_; e_out++) {
      float sigma_t = data::mg.macro_xs_[fsr.material_].get_xs(
        MgxsType::TOTAL, e_out, nullptr, nullptr, nullptr, t, a);
      float scatter_source = 0.0f;

      for (int e_in = 0; e_in < negroups_; e_in++) {
        float scalar_flux = fsr.scalar_flux_old_[e_in];

        float sigma_s = data::mg.macro_xs_[fsr.material_].get_xs(
          MgxsType::NU_SCATTER, e_in, &e_out, nullptr, nullptr, t, a);
        scatter_source += sigma_s * scalar_flux;
      }
      fsr.source_[e_out] = scatter_source / sigma_t;
    }
  }

  if (settings::run_mode == RunMode::EIGENVALUE) {
// Add fission source if in eigenvalue mode
#pragma omp parallel for
    for (int i = 0; i < fsr_manifest_.size(); i++) {
      FlatSourceRegion& fsr = fsr_manifest_[i];
      for (int e_out = 0; e_out < negroups_; e_out++) {
        float sigma_t = data::mg.macro_xs_[fsr.material_].get_xs(
          MgxsType::TOTAL, e_out, nullptr, nullptr, nullptr, t, a);
        float fission_source = 0.0f;

        for (int e_in = 0; e_in < negroups_; e_in++) {
          float scalar_flux = fsr.scalar_flux_old_[e_in];
          float nu_sigma_f = data::mg.macro_xs_[fsr.material_].get_xs(
            MgxsType::NU_FISSION, e_in, nullptr, nullptr, nullptr, t, a);
          float chi = data::mg.macro_xs_[fsr.material_].get_xs(
            MgxsType::CHI_PROMPT, e_in, &e_out, nullptr, nullptr, t, a);
          fission_source += nu_sigma_f * scalar_flux * chi;
        }
        fsr.source_[e_out] += fission_source * inverse_k_eff / sigma_t;
      }
    }
  } else {
    // Add fixed source source if in fixed source mode
#pragma omp parallel for
    for (int i = 0; i < fsr_manifest_.size(); i++) {
      FlatSourceRegion& fsr = fsr_manifest_[i];
      for (int e = 0; e < negroups_; e++) {
        fsr.source_[e] += fsr.fixed_source_[e];
      }
    }
  }

  simulation::time_update_src.stop();
}

// Normalizes flux and updates simulation-averaged volume estimate
void FlatSourceDomain::normalize_scalar_flux_and_volumes(
  double total_active_distance_per_iteration)
{
  float normalization_factor = 1.0 / total_active_distance_per_iteration;
  double volume_normalization_factor =
    1.0 / (total_active_distance_per_iteration * simulation::current_batch);

  // Normalize Scalar flux to total distance traveled by all rays this
  // iteration Accumulate cell-wise ray length tallies collected this
  // iteration, then update the simulation-averaged cell-wise volume estimates
#pragma omp parallel for
  for (int i = 0; i < fsr_manifest_.size(); i++) {
    FlatSourceRegion& fsr = fsr_manifest_[i];
    fsr.volume_t_ += fsr.volume_;
    fsr.volume_i_ = fsr.volume_ * normalization_factor;
    fsr.volume_ = fsr.volume_t_ * volume_normalization_factor;
    for (int e = 0; e < negroups_; e++) {
      fsr.scalar_flux_new_[e] *= normalization_factor;
    }
  }
}

// Combines transport flux contributions and flat source contributions
// from the previous iteration to generate this iteration's estimate of
// scalar flux.
int64_t FlatSourceDomain::add_source_to_scalar_flux()
{
  int64_t n_hits = 0;

  // Temperature and angle indices, if using multiple temperature
  // data sets and/or anisotropic data sets.
  // TODO: Currently assumes we are only using single temp/single
  // angle data.
  const int t = 0;
  const int a = 0;

#pragma omp parallel for reduction(+ : n_hits)
  for (int i = 0; i < fsr_manifest_.size(); i++) {
    FlatSourceRegion& fsr = fsr_manifest_[i];

    // Check if this cell was hit this iteration
    if (fsr.was_hit_ > 0) {
      n_hits++;
    }

    double volume = fsr.volume_;
    double volume_i = fsr.volume_i_;
    int material = fsr.material_;
    for (int e = 0; e < negroups_; e++) {
      // There are three scenarios we need to consider:
      if (fsr.was_hit_ > 2) {
        // 1. If the FSR was hit this iteration, then the new flux is equal
        // to the flat source from the previous iteration plus the
        // contributions from rays passing through the source region
        // (computed during the transport sweep)
        float sigma_t = data::mg.macro_xs_[material].get_xs(
          MgxsType::TOTAL, e, nullptr, nullptr, nullptr, t, a);
        fsr.scalar_flux_new_[e] /= (sigma_t * volume);
        fsr.scalar_flux_new_[e] += fsr.source_[e];
      } else if (fsr.was_hit_ > 0) {
        float sigma_t = data::mg.macro_xs_[material].get_xs(
          MgxsType::TOTAL, e, nullptr, nullptr, nullptr, t, a);
        fsr.scalar_flux_new_[e] /= (sigma_t * volume_i);
        fsr.scalar_flux_new_[e] += fsr.source_[e];
      } else if (volume > 0.0) {
        // 2. If the FSR was not hit this iteration, but has been hit some
        // previous iteration, then we simply set the new scalar flux to be
        // equal to the contribution from the flat source alone.
        fsr.scalar_flux_new_[e] = fsr.source_[e];
      } else {
        // If the FSR was not hit this iteration, and it has never been hit
        // in any iteration (i.e., volume is zero), then we want to set this
        // to 0 to avoid dividing anything by a zero volume.
        fsr.scalar_flux_new_[e] = 0.f;
      }
    }
  }

  // Return the number of source regions that were hit this iteration
  return n_hits;
}

// Generates new estimate of k_eff based on the differences between this
// iteration's estimate of the scalar flux and the last iteration's estimate.
double FlatSourceDomain::compute_k_eff(double k_eff_old)
{
  double fission_rate_old = 0;
  double fission_rate_new = 0;

  // Temperature and angle indices, if using multiple temperature
  // data sets and/or anisotropic data sets.
  // TODO: Currently assumes we are only using single temp/single
  // angle data.
  const int t = 0;
  const int a = 0;

#pragma omp parallel for reduction(+ : fission_rate_old, fission_rate_new)
  for (int i = 0; i < fsr_manifest_.size(); i++) {
    FlatSourceRegion& fsr = fsr_manifest_[i];

    // If simulation averaged volume is zero, don't include this cell
    double volume = fsr.volume_;
    if (volume == 0.0) {
      continue;
    }

    int material = fsr.material_;

    double sr_fission_source_old = 0;
    double sr_fission_source_new = 0;

    for (int e = 0; e < negroups_; e++) {
      double nu_sigma_f = data::mg.macro_xs_[material].get_xs(
        MgxsType::NU_FISSION, e, nullptr, nullptr, nullptr, t, a);
      sr_fission_source_old += nu_sigma_f * fsr.scalar_flux_old_[e];
      sr_fission_source_new += nu_sigma_f * fsr.scalar_flux_new_[e];
    }

    fission_rate_old += sr_fission_source_old * volume;
    fission_rate_new += sr_fission_source_new * volume;
  }

  double k_eff_new = k_eff_old * (fission_rate_new / fission_rate_old);

  return k_eff_new;
}

// This function is responsible for generating a mapping between random
// ray flat source regions (cell instances) and tally bins. The mapping
// takes the form of a "TallyTask" object, which accounts for one single
// score being applied to a single tally. Thus, a single source region
// may have anywhere from zero to many tally tasks associated with it --
// meaning that the global "tally_task" data structure is in 2D. The outer
// dimension corresponds to the source element (i.e., each entry corresponds
// to a specific energy group within a specific source region), and the
// inner dimension corresponds to the tallying task itself. Mechanically,
// the mapping between FSRs and spatial filters is done by considering
// the location of a single known ray midpoint that passed through the
// FSR. I.e., during transport, the first ray to pass through a given FSR
// will write down its midpoint for use with this function. This is a cheap
// and easy way of mapping FSrs to spatial tally filters, but comes with
// the downside of adding the restriction that spatial tally filters must
// share boundaries with the physical geometry of the simulation (so as
// not to subdivide any FSR). It is acceptable for a spatial tally region
// to contain multiple FSRs, but not the other way around.

// TODO: In future work, it would be preferable to offer a more general
// (but perhaps slightly more expensive) option for handling arbitrary
// spatial tallies that would be allowed to subdivide FSRs.

// Besides generating the mapping structure, this function also keeps track
// of whether or not all flat source regions have been hit yet. This is
// required, as there is no guarantee that all flat source regions will
// be hit every iteration, such that in the first few iterations some FSRs
// may not have a known position within them yet to facilitate mapping to
// spatial tally filters. However, after several iterations, if all FSRs
// have been hit and have had a tally map generated, then this status will
// be passed back to the caller to alert them that this function doesn't
// need to be called for the remainder of the simulation.

void FlatSourceDomain::convert_source_regions_to_tallies()
{
  openmc::simulation::time_tallies.start();

  // Tracks if we've generated a mapping yet for all source regions.
  bool all_source_regions_mapped = true;

// Attempt to generate mapping for all source regions
#pragma omp parallel for
  for (int i = 0; i < fsr_manifest_.size(); i++) {
    FlatSourceRegion& fsr = fsr_manifest_[i];
    // If this source region has not been hit by a ray yet, then
    // we aren't going to be able to map it, so skip it.
    if (!fsr.position_recorded_) {
      all_source_regions_mapped = false;
      continue;
    }

    // A particle located at the recorded midpoint of a ray
    // crossing through this source region is used to estabilish
    // the spatial location of the source region
    Particle p;
    p.r() = fsr.position_;
    p.r_last() = fsr.position_;
    bool found = exhaustive_find_cell(p);

    // Loop over energy groups (so as to support energy filters)
    for (int e = 0; e < negroups_; e++) {

      // Set particle to the current energy
      p.g() = e;
      p.g_last() = e;
      p.E() = data::mg.energy_bin_avg_[p.g()];
      p.E_last() = p.E();

      // If this task has already been populated, we don't need to do
      // it again.
      if (fsr.tally_task_[e].size() > 0) {
        continue;
      }

      // Loop over all active tallies. This logic is essentially identical
      // to what happens when scanning for applicable tallies during
      // MC transport.
      for (auto i_tally : model::active_tallies) {
        Tally& tally {*model::tallies[i_tally]};

        // Initialize an iterator over valid filter bin combinations.
        // If there are no valid combinations, use a continue statement
        // to ensure we skip the assume_separate break below.
        auto filter_iter = FilterBinIter(tally, p);
        auto end = FilterBinIter(tally, true, &p.filter_matches());
        if (filter_iter == end)
          continue;

        // Loop over filter bins.
        for (; filter_iter != end; ++filter_iter) {
          auto filter_index = filter_iter.index_;
          auto filter_weight = filter_iter.weight_;

          // Loop over scores
          for (auto score_index = 0; score_index < tally.scores_.size();
               score_index++) {
            auto score_bin = tally.scores_[score_index];
            // If a valid tally, filter, and score cobination has been
            // found, then add it to the list of tally tasks for this source
            // element.
            fsr.tally_task_[e].emplace_back(
              i_tally, filter_index, score_index, score_bin);
          }
        }
      }
      // Reset all the filter matches for the next tally event.
      for (auto& match : p.filter_matches())
        match.bins_present_ = false;
    }
  }
  openmc::simulation::time_tallies.stop();

  mapped_all_tallies_ = all_source_regions_mapped;
}

// Tallying in random ray is not done directly during transport, rather,
// it is done only once after each power iteration. This is made possible
// by way of a mapping data structure that relates spatial source regions
// (FSRs) to tally/filter/score combinations. The mechanism by which the
// mapping is done (and the limitations incurred) is documented in the
// "convert_source_regions_to_tallies()" function comments above. The present
// tally function simply traverses the mapping data structure and executes
// the scoring operations to OpenMC's native tally result arrays.

void FlatSourceDomain::random_ray_tally()
{
  openmc::simulation::time_tallies.start();

  // Compute the volume weighted total strength of fixed sources throughout
  // the domain using up to date stochastic source region volumes. Note that
  // this value is different than the sum of the user input IndependentSource
  // object strengths, as the raw user input strengths do not account for the
  // volumes of each source. Computing this quantity is useful for normalizing
  // output in the same manner as would be expected in Monte Carlo mode.
  double inverse_source_strength = 1.0;
  if (settings::run_mode == RunMode::FIXED_SOURCE) {
    inverse_source_strength =
      1.0 / calculate_total_volume_weighted_source_strength();
  }

  // Temperature and angle indices, if using multiple temperature
  // data sets and/or anisotropic data sets.
  // TODO: Currently assumes we are only using single temp/single
  // angle data.
  const int t = 0;
  const int a = 0;

// We loop over all source regions and energy groups. For each
// element, we check if there are any scores needed and apply
// them.
#pragma omp parallel for
  for (int i = 0; i < fsr_manifest_.size(); i++) {
    FlatSourceRegion& fsr = fsr_manifest_[i];
    double volume = fsr.volume_;
    double factor = volume * inverse_source_strength;
    double material = fsr.material_;
    for (int e = 0; e < negroups_; e++) {
      double flux = fsr.scalar_flux_new_[e] * factor;
      for (auto& task : fsr.tally_task_[e]) {
        double score;
        switch (task.score_type) {

        case SCORE_FLUX:
          score = flux;
          break;

        case SCORE_TOTAL:
          score = flux * data::mg.macro_xs_[material].get_xs(
                           MgxsType::TOTAL, e, NULL, NULL, NULL, t, a);
          break;

        case SCORE_FISSION:
          score = flux * data::mg.macro_xs_[material].get_xs(
                           MgxsType::FISSION, e, NULL, NULL, NULL, t, a);
          break;

        case SCORE_NU_FISSION:
          score = flux * data::mg.macro_xs_[material].get_xs(
                           MgxsType::NU_FISSION, e, NULL, NULL, NULL, t, a);
          break;

        case SCORE_EVENTS:
          score = 1.0;
          break;

        default:
          fatal_error("Invalid score specified in tallies.xml. Only flux, "
                      "total, fission, nu-fission, and events are supported in "
                      "random ray mode.");
          break;
        }
        Tally& tally {*model::tallies[task.tally_idx]};
#pragma omp atomic
        tally.results_(task.filter_idx, task.score_idx, TallyResult::VALUE) +=
          score;
      }
    }
  }
}

// TODO: Make this work for AOS...
void FlatSourceDomain::all_reduce_replicated_source_regions()
{
#ifdef OPENMC_MPI

  // If we only have 1 MPI rank, no need
  // to reduce anything.
  if (mpi::n_procs <= 1)
    return;

  simulation::time_bank_sendrecv.start();

  // The "position_recorded" variable needs to be allreduced (and maxed),
  // as whether or not a cell was hit will affect some decisions in how the
  // source is calculated in the next iteration so as to avoid dividing
  // by zero. We take the max rather than the sum as the hit values are
  // expected to be zero or 1.
  MPI_Allreduce(MPI_IN_PLACE, position_recorded_.data(), n_source_regions_,
    MPI_INT, MPI_MAX, mpi::intracomm);

  // The position variable is more complicated to reduce than the others,
  // as we do not want the sum of all positions in each cell, rather, we
  // want to just pick any single valid position. Thus, we perform a gather
  // and then pick the first valid position we find for all source regions
  // that have had a position recorded. This operation does not need to
  // be broadcast back to other ranks, as this value is only used for the
  // tally conversion operation, which is only performed on the master rank.
  // While this is expensive, it only needs to be done for active batches,
  // and only if we have not mapped all the tallies yet. Once tallies are
  // fully mapped, then the position vector is fully populated, so this
  // operation can be skipped.

  // First, we broadcast the fully mapped tally status variable so that
  // all ranks are on the same page
  int mapped_all_tallies_i = static_cast<int>(mapped_all_tallies_);
  MPI_Bcast(&mapped_all_tallies_i, 1, MPI_INT, 0, mpi::intracomm);

  // Then, we perform the gather of position data, if needed
  if (simulation::current_batch > settings::n_inactive &&
      !mapped_all_tallies_i) {

    // Master rank will gather results and pick valid positions
    if (mpi::master) {
      // Initialize temporary vector for receiving positions
      std::vector<std::vector<Position>> all_position;
      all_position.resize(mpi::n_procs);
      for (int i = 0; i < mpi::n_procs; i++) {
        all_position[i].resize(n_source_regions_);
      }

      // Copy master rank data into gathered vector for convenience
      all_position[0] = position_;

      // Receive all data into gather vector
      for (int i = 1; i < mpi::n_procs; i++) {
        MPI_Recv(all_position[i].data(), n_source_regions_ * 3, MPI_DOUBLE, i,
          0, mpi::intracomm, MPI_STATUS_IGNORE);
      }

      // Scan through gathered data and pick first valid cell posiiton
      for (int sr = 0; sr < n_source_regions_; sr++) {
        if (position_recorded_[sr] == 1) {
          for (int i = 0; i < mpi::n_procs; i++) {
            if (all_position[i][sr].x != 0.0 || all_position[i][sr].y != 0.0 ||
                all_position[i][sr].z != 0.0) {
              position_[sr] = all_position[i][sr];
              break;
            }
          }
        }
      }
    } else {
      // Other ranks just send in their data
      MPI_Send(position_.data(), n_source_regions_ * 3, MPI_DOUBLE, 0, 0,
        mpi::intracomm);
    }
  }

  // For the rest of the source region data, we simply perform an all reduce,
  // as these values will be needed on all ranks for transport during the
  // next iteration.
  MPI_Allreduce(MPI_IN_PLACE, volume_.data(), n_source_regions_, MPI_DOUBLE,
    MPI_SUM, mpi::intracomm);

  MPI_Allreduce(MPI_IN_PLACE, was_hit_.data(), n_source_regions_, MPI_INT,
    MPI_SUM, mpi::intracomm);

  MPI_Allreduce(MPI_IN_PLACE, scalar_flux_new_.data(), n_source_elements_,
    MPI_FLOAT, MPI_SUM, mpi::intracomm);

  simulation::time_bank_sendrecv.stop();
#endif
}

// Outputs all basic material, FSR ID, multigroup flux, and
// fission source data to .vtk file that can be directly
// loaded and displayed by Paraview.
void FlatSourceDomain::output_to_vtk()
{
  // Rename .h5 plot filename(s) to .vtk filenames
  for (int p = 0; p < model::plots.size(); p++) {
    PlottableInterface* plot = model::plots[p].get();
    plot->path_plot() =
      plot->path_plot().substr(0, plot->path_plot().find_last_of('.')) + ".vtk";
  }

  // Print header information
  print_plot();

  // Outer loop over plots
  for (int p = 0; p < model::plots.size(); p++) {

    // Get handle to OpenMC plot object and extract params
    Plot* openmc_plot = dynamic_cast<Plot*>(model::plots[p].get());

    // Random ray plots only support voxel plots
    if (openmc_plot == nullptr) {
      warning(fmt::format("Plot {} is invalid plot type -- only voxel plotting "
                          "is allowed in random ray mode.",
        p));
      continue;
    } else if (openmc_plot->type_ != Plot::PlotType::voxel) {
      warning(fmt::format("Plot {} is invalid plot type -- only voxel plotting "
                          "is allowed in random ray mode.",
        p));
      continue;
    }

    int Nx = openmc_plot->pixels_[0];
    int Ny = openmc_plot->pixels_[1];
    int Nz = openmc_plot->pixels_[2];
    Position origin = openmc_plot->origin_;
    Position width = openmc_plot->width_;
    Position ll = origin - width / 2.0;
    double x_delta = width.x / Nx;
    double y_delta = width.y / Ny;
    double z_delta = width.z / Nz;
    std::string filename = openmc_plot->path_plot();

    // Perform sanity checks on file size
    uint64_t bytes = Nx * Ny * Nz * (negroups_ + 1 + 1 + 1) * sizeof(float);
    write_message(5, "Processing plot {}: {}... (Estimated size is {} MB)",
      openmc_plot->id(), filename, bytes / 1.0e6);
    if (bytes / 1.0e9 > 1.0) {
      warning("Voxel plot specification is very large (>1 GB). Plotting may be "
              "slow.");
    } else if (bytes / 1.0e9 > 100.0) {
      fatal_error("Voxel plot specification is too large (>100 GB). Exiting.");
    }

    // Relate voxel spatial locations to random ray source regions
    std::vector<FlatSourceRegion*> voxel_indices(Nx * Ny * Nz);
    std::vector<int> hits(Nx * Ny * Nz);

#pragma omp parallel for collapse(3)
    for (int z = 0; z < Nz; z++) {
      for (int y = 0; y < Ny; y++) {
        for (int x = 0; x < Nx; x++) {
          Position sample;
          sample.z = ll.z + z_delta / 2.0 + z * z_delta;
          sample.y = ll.y + y_delta / 2.0 + y * y_delta;
          sample.x = ll.x + x_delta / 2.0 + x * x_delta;
          Particle p;
          p.r() = sample;
          bool found = exhaustive_find_cell(p);
          int i_cell = p.lowest_coord().cell;
          int64_t source_region_idx =
            source_region_offsets_[i_cell] + p.cell_instance();
          auto& fsr = fsr_[source_region_idx];
          int i_mesh = fsr.mesh_;

          // If there is a mesh present, then we need to get the bin number
          // corresponding to the ray spatial location
          FlatSourceRegion* region;
          int hit_count = 0;
          if (i_mesh >= 0) {
            Mesh* mesh = meshes_[i_mesh].get();
            RegularMesh* rmesh = dynamic_cast<RegularMesh*>(mesh);
            if (rmesh == nullptr)
              fatal_error(
                "Only regular meshes are supported for random ray tracing.");
            int bin = rmesh->get_bin(p.r());
            RegularMesh::MeshIndex ijk = rmesh->get_indices_from_bin(bin);
            hit_count = hitmap[ijk[1] - 1][ijk[0] - 1];
            region = get_fsr(source_region_idx, bin);
          } else {
            region = get_fsr(source_region_idx, 0);
          }

          voxel_indices[z * Ny * Nx + y * Nx + x] = region;
          hits[z * Ny * Nx + y * Nx + x] = hit_count;
        }
      }
    }

    // Open file for writing
    FILE* plot = fopen(filename.c_str(), "w");

    // Write vtk metadata
    fprintf(plot, "# vtk DataFile Version 2.0\n");
    fprintf(plot, "Dataset File\n");
    fprintf(plot, "BINARY\n");
    fprintf(plot, "DATASET STRUCTURED_POINTS\n");
    fprintf(plot, "DIMENSIONS %d %d %d\n", Nx, Ny, Nz);
    fprintf(plot, "ORIGIN %lf %lf %lf\n", origin.x - width.x / 2.0,
      origin.y - width.y / 2.0, origin.z - width.z / 2.0);
    fprintf(plot, "SPACING %lf %lf %lf\n", x_delta, y_delta, z_delta);
    fprintf(plot, "POINT_DATA %d\n", Nx * Ny * Nz);

    // Plot multigroup flux data
    for (int g = 0; g < negroups_; g++) {
      fprintf(plot, "SCALARS flux_group_%d float\n", g);
      fprintf(plot, "LOOKUP_TABLE default\n");
      for (auto fsr : voxel_indices) {
        float flux = fsr->scalar_flux_final_[g];
        flux /= (settings::n_batches - settings::n_inactive);
        flux = flip_endianness<float>(flux);
        fwrite(&flux, sizeof(float), 1, plot);
      }
    }

    // Plot FSRs
    fprintf(plot, "SCALARS FSRs float\n");
    fprintf(plot, "LOOKUP_TABLE default\n");
    for (auto fsr : voxel_indices) {
      float value = future_prn(10, reinterpret_cast<uint64_t>(fsr));
      value = flip_endianness<float>(value);
      fwrite(&value, sizeof(float), 1, plot);
    }

    // Plot Materials
    fprintf(plot, "SCALARS Materials int\n");
    fprintf(plot, "LOOKUP_TABLE default\n");
    for (auto fsr : voxel_indices) {
      int mat = fsr->material_;
      mat = flip_endianness<int>(mat);
      fwrite(&mat, sizeof(int), 1, plot);
    }

    // Plot hitmap
    fprintf(plot, "SCALARS FSRs_per_mesh int\n");
    fprintf(plot, "LOOKUP_TABLE default\n");
    for (int bin : hits) {
      bin = flip_endianness<int>(bin);
      fwrite(&bin, sizeof(int), 1, plot);
    }

    // Plot fission source
    fprintf(plot, "SCALARS total_fission_source float\n");
    fprintf(plot, "LOOKUP_TABLE default\n");
    for (auto fsr : voxel_indices) {
      float total_fission = 0.0;
      int mat = fsr->material_;
      for (int g = 0; g < negroups_; g++) {
        float flux = fsr->scalar_flux_final_[g];
        flux /= (settings::n_batches - settings::n_inactive);
        float Sigma_f = data::mg.macro_xs_[mat].get_xs(
          MgxsType::FISSION, g, nullptr, nullptr, nullptr, 0, 0);
        total_fission += Sigma_f * flux;
      }
      total_fission = flip_endianness<float>(total_fission);
      fwrite(&total_fission, sizeof(float), 1, plot);
    }

    fclose(plot);
  }
}

void FlatSourceDomain::apply_fixed_source_to_source_region(
  Discrete* discrete, double strength_factor, int64_t source_region)
{
  const auto& discrete_energies = discrete->x();
  const auto& discrete_probs = discrete->prob();
  auto& fsr = fsr_[source_region];
  for (int e = 0; e < discrete_energies.size(); e++) {
    int g = data::mg.get_group_index(discrete_energies[e]);
    fsr.fixed_source_[g] += discrete_probs[e] * strength_factor;
  }
}

void FlatSourceDomain::apply_fixed_source_to_cell_instances(int32_t i_cell,
  Discrete* discrete, double strength_factor, int target_material_id,
  const vector<int32_t>& instances)
{
  Cell& cell = *model::cells[i_cell];
  if (cell.type_ != Fill::MATERIAL)
    return;

  for (int j : instances) {
    int cell_material_idx = cell.material(j);
    int cell_material_id = model::materials[cell_material_idx]->id();
    if (target_material_id == C_NONE ||
        cell_material_id == target_material_id) {
      int64_t source_region = source_region_offsets_[i_cell] + j;
      apply_fixed_source_to_source_region(
        discrete, strength_factor, source_region);
    }
  }
}

// OK - the basic issue is that the lowest level function (above) should
// execute over every instance, as it is the material level instance.

// HOWEVER - it really also should bnot be applying the fixed source
// to any cells which are not material filled. They should instead just
// be checked/ignored. I think I got away with it because my hierarchy was
// relatively flat. But in reality, I get this get_contained_cells list
// that has all the cells and their instances, but some may not be material
// filled

void FlatSourceDomain::apply_fixed_source_to_cell_and_children(int32_t i_cell,
  Discrete* discrete, double strength_factor, int32_t target_material_id)
{
  Cell& cell = *model::cells[i_cell];

  if (cell.type_ == Fill::MATERIAL) {
    vector<int> instances(cell.n_instances_);
    std::iota(instances.begin(), instances.end(), 0);
    apply_fixed_source_to_cell_instances(
      i_cell, discrete, strength_factor, target_material_id, instances);
  } else if (target_material_id == C_NONE) {
    std::unordered_map<int32_t, vector<int32_t>> cell_instance_list =
      cell.get_contained_cells(0, nullptr);
    for (const auto& pair : cell_instance_list) {
      int32_t i_child_cell = pair.first;
      apply_fixed_source_to_cell_instances(i_child_cell, discrete,
        strength_factor, target_material_id, pair.second);
    }
  }
}

void FlatSourceDomain::count_fixed_source_regions()
{
#pragma omp parallel for reduction(+ : n_fixed_source_regions_)
  for (int sr = 0; sr < n_source_regions_; sr++) {
    float total = 0.f;
    auto& fsr = fsr_[sr];
    for (int e = 0; e < negroups_; e++) {
      total += fsr.fixed_source_[e];
    }
    if (total != 0.f) {
      n_fixed_source_regions_++;
    }
  }
}

double FlatSourceDomain::calculate_total_volume_weighted_source_strength()
{
  double source_strength = 0.0;
#pragma omp parallel for reduction(+ : source_strength)
  for (int i = 0; i < fsr_manifest_.size(); i++) {
    FlatSourceRegion& fsr = fsr_manifest_[i];
    double volume = fsr.volume_;
    for (int e = 0; e < negroups_; e++) {
      source_strength += fsr.fixed_source_[e] * volume;
    }
  }
  return source_strength;
}

void FlatSourceDomain::convert_fixed_sources()
{
  // Compute total combined strength of all neutron/photon sources
  double total_strength = 0;
  for (int es = 0; es < model::external_sources.size(); es++) {
    total_strength += model::external_sources[es]->strength();
  }

  // Loop over external sources
  for (int es = 0; es < model::external_sources.size(); es++) {
    Source* s = model::external_sources[es].get();
    IndependentSource* is = dynamic_cast<IndependentSource*>(s);
    Discrete* energy = dynamic_cast<Discrete*>(is->energy());
    const std::unordered_set<int32_t>& domain_ids = is->domain_ids();

    double strength_factor = is->strength() / total_strength;

    if (is->domain_type() == IndependentSource::DomainType::MATERIAL) {
      for (int32_t material_id : domain_ids) {
        for (int i_cell = 0; i_cell < model::cells.size(); i_cell++) {
          apply_fixed_source_to_cell_and_children(
            i_cell, energy, strength_factor, material_id);
        }
      }
    } else if (is->domain_type() == IndependentSource::DomainType::CELL) {
      for (int32_t cell_id : domain_ids) {
        int32_t i_cell = model::cell_map[cell_id];
        apply_fixed_source_to_cell_and_children(
          i_cell, energy, strength_factor, C_NONE);
      }
    } else if (is->domain_type() == IndependentSource::DomainType::UNIVERSE) {
      for (int32_t universe_id : domain_ids) {
        int32_t i_universe = model::universe_map[universe_id];
        Universe& universe = *model::universes[i_universe];
        for (int32_t i_cell : universe.cells_) {
          apply_fixed_source_to_cell_and_children(
            i_cell, energy, strength_factor, C_NONE);
        }
      }
    }
  } // End loop over external sources

  // Temperature and angle indices, if using multiple temperature
  // data sets and/or anisotropic data sets.
  // TODO: Currently assumes we are only using single temp/single
  // angle data.
  const int t = 0;
  const int a = 0;

// Divide the fixed source term by sigma t (to save time when applying each
// iteration)
#pragma omp parallel for
  for (int sr = 0; sr < n_source_regions_; sr++) {
    auto& fsr = fsr_[sr];
    int material = fsr.material_;
    for (int e = 0; e < negroups_; e++) {
      float sigma_t = data::mg.macro_xs_[material].get_xs(
        MgxsType::TOTAL, e, nullptr, nullptr, nullptr, t, a);
      fsr.fixed_source_[e] /= sigma_t;
    }
  }
}

void FlatSourceDomain::apply_mesh_to_cell_instances(int32_t i_cell,
  int32_t mesh, int target_material_id, const vector<int32_t>& instances)
{
  Cell& cell = *model::cells[i_cell];
  if (cell.type_ != Fill::MATERIAL)
    return;
  for (int32_t j : instances) {
    // printf("applying to cell id %d instance %d of %d in found instances, %d
    // in "
    //        "actual cell instances\n",
    //   cell.id_, j, instances.size(), cell.n_instances_);
    int cell_material_idx = cell.material(j);
    int cell_material_id = model::materials[cell_material_idx]->id();
    if (target_material_id == C_NONE ||
        cell_material_id == target_material_id) {
      int64_t source_region = source_region_offsets_[i_cell] + j;
      fsr_[source_region].mesh_ = mesh;
    }
  }
}

void FlatSourceDomain::apply_mesh_to_cell_and_children(
  int32_t i_cell, int32_t mesh, int32_t target_material_id)
{
  Cell& cell = *model::cells[i_cell];

  if (cell.type_ == Fill::MATERIAL) {
    vector<int> instances(cell.n_instances_);
    std::iota(instances.begin(), instances.end(), 0);
    apply_mesh_to_cell_instances(i_cell, mesh, target_material_id, instances);
  } else if (target_material_id == C_NONE) {
    // printf("cell id %d, n instances %d\n", cell.id_, cell.n_instances_);
    for (int j = 0; j < cell.n_instances_; j++) {
      //   printf(
      //     "getting contained cells for cell id %d instance %d\n", cell.id_,
      //     j);
      std::unordered_map<int32_t, vector<int32_t>> cell_instance_list =
        cell.get_contained_cells(j, nullptr);
      for (const auto& pair : cell_instance_list) {
        int32_t i_child_cell = pair.first;
        apply_mesh_to_cell_instances(
          i_child_cell, mesh, target_material_id, pair.second);
      }
    }
  }
}

void FlatSourceDomain::apply_meshes()
{
  // Loop over external sources
  for (int32_t m = 0; m < meshes_.size(); m++) {
    Mesh* mesh = meshes_[m].get();
    RegularMesh* rm = dynamic_cast<RegularMesh*>(mesh);
    if (rm == nullptr) {
      fatal_error("Random ray FSR subdivide mesh must be of type RegularMesh.");
    }
    const std::unordered_set<int32_t>& domain_ids = rm->domain_ids();

    if (rm->domain_type() == RegularMesh::DomainType::MATERIAL) {
      for (int32_t material_id : domain_ids) {
        for (int i_cell = 0; i_cell < model::cells.size(); i_cell++) {
          apply_mesh_to_cell_and_children(i_cell, m, material_id);
        }
      }
    } else if (rm->domain_type() == RegularMesh::DomainType::CELL) {
      for (int32_t cell_id : domain_ids) {
        int32_t i_cell = model::cell_map[cell_id];
        apply_mesh_to_cell_and_children(i_cell, m, C_NONE);
      }
    } else if (rm->domain_type() == RegularMesh::DomainType::UNIVERSE) {
      for (int32_t universe_id : domain_ids) {
        int32_t i_universe = model::universe_map[universe_id];
        Universe& universe = *model::universes[i_universe];
        for (int32_t i_cell : universe.cells_) {
          apply_mesh_to_cell_and_children(i_cell, m, C_NONE);
        }
      }
    }
  } // End loop over source region meshes

  if (meshes_.size() > 0) {
    int x = dynamic_cast<RegularMesh*>(meshes_[0].get())->shape_[0];
    int y = dynamic_cast<RegularMesh*>(meshes_[0].get())->shape_[1];
    hitmap = vector<vector<int>>(y, vector<int>(x, 0));
  }
}

uint64_t hashPair(uint32_t a, uint32_t b)
{
  uint64_t hash = 0xcbf29ce484222325; // FNV-1a 64-bit offset basis
  uint64_t prime = 0x100000001b3;     // FNV-1a 64-bit prime

  // Hash the first integer
  hash ^= a;
  hash *= prime;

  // Hash the second integer
  hash ^= b;
  hash *= prime;

  // Mix the bits a bit more
  hash ^= hash >> 33;
  hash *= 0xff51afd7ed558ccd;
  hash ^= hash >> 33;
  hash *= 0xc4ceb9fe1a85ec53;
  hash ^= hash >> 33;

  return hash;
}

int hashbin(uint32_t a, uint32_t b)
{
  const int n_bins = 10;
  uint64_t hash = hashPair(a, b);
  return hash % N_FSR_HASH_BINS;
}

int emplaces = 0;

FlatSourceRegion* FlatSourceDomain::get_fsr(int64_t source_region, int bin)
{
  FlatSourceRegion& base_fsr = fsr_[source_region];
  if (base_fsr.mesh_ == C_NONE) {
    // Problem: this is no longer a valid FSR. How to handle?
    if (base_fsr.is_in_manifest_) {
      return &fsr_manifest_[base_fsr.manifest_index_];
    } else {
      return &fsr_[source_region];
    }
  }
  // Get hash and has controller bin
  uint64_t hash = hashPair(source_region, bin);
  int hash_bin = hash % N_FSR_HASH_BINS;
  SourceNode& node = controller_.nodes_[hash_bin];
  auto& map = node.fsr_map_;

  // Check if the FlatSourceRegion with this hash already exists
  auto it = map.find(hash);
  if (it == map.end()) {
    node.lock_.lock();

    auto& new_map = node.new_fsr_map_;
    auto it_new = new_map.find(hash);
    if (it_new != new_map.end()) {
      node.lock_.unlock();
      return &it_new->second;
    }
    // If not found, copy base FSR into new FSR
    auto result = new_map.emplace(std::make_pair(hash, fsr_[source_region]));
    node.lock_.unlock();

#pragma omp atomic
    n_subdivided_source_regions_++;

    if (fsr_[source_region].mesh_ != C_NONE) {
      Mesh* mesh = meshes_[fsr_[source_region].mesh_].get();
      RegularMesh* rmesh = dynamic_cast<RegularMesh*>(mesh);
      StructuredMesh::MeshIndex mesh_index = rmesh->get_indices_from_bin(bin);
      hitmap[mesh_index[1] - 1][mesh_index[0] - 1] += 1;
    }

    /*
    if( mesh_index[1]-1 ==0 && mesh_index[0]-1 == 0)
    {
      printf("Adding source region %d to bin %d (y=%d, x=%d) with material =
    %d\n", source_region, bin, mesh_index[1]-1, mesh_index[0]-1,
    fsr_[source_region].material_); if( fsr_[source_region].material_ == 0)
      {
        fatal_error("Added fuel to region that should be moderator");
      }
    }
    */

    // printf("Adding source region %d to bin %d (y=%d, x=%d) with material =
    // %d\n", source_region, bin, mesh_index[1]-1, mesh_index[0]-1,
    // fsr_[source_region].material_);

    // if( n_subdivided_source_regions_ > 15028) // 2x2 mesh analytical
    if (n_subdivided_source_regions_ > 198832) // 8x8 mesh analytical
    {
      Mesh* mesh = meshes_[fsr_[source_region].mesh_].get();
      RegularMesh* rmesh = dynamic_cast<RegularMesh*>(mesh);
      StructuredMesh::MeshIndex mesh_index = rmesh->get_indices_from_bin(bin);
      printf("source region = %ld, bin = %d, index x = %d, index y = %d\n",
        source_region, bin, mesh_index[0], mesh_index[1]);
      fatal_error("Too many subdivided source regions");
    }
    return &result.first->second;
  } else {
    // Otherwise, access the existing FlatSourceRegion
    return &fsr_manifest_[it->second];
  }
}

// FlatSourceRegion* FlatSourceDomain::get_fsr(int64_t source_region, int bin,
//  Position r0, Position r1, int ray_id, GeometryState& ip)
FlatSourceRegion* FlatSourceDomain::get_fsr(
  int64_t source_region, int bin, Position r0, Position r1, int ray_id)
{
  // This conditional will not be triggered right now due to the
  // presence of two function prototypes, but leaving it in for later
  FlatSourceRegion& base_fsr = fsr_[source_region];
  if (base_fsr.mesh_ == C_NONE) {
    if (base_fsr.is_in_manifest_) {
      return &fsr_manifest_[base_fsr.manifest_index_];
    } else {
      return &fsr_[source_region];
    }
  }
  // Get hash and has controller bin
  uint64_t hash = hashPair(source_region, bin);
  int hash_bin = hash % N_FSR_HASH_BINS;
  SourceNode& node = controller_.nodes_[hash_bin];
  auto& map = node.fsr_map_;

  // Check if the FlatSourceRegion with this hash already exists
  // Check if the FlatSourceRegion with this hash already exists
  auto it = map.find(hash);
  if (it == map.end()) {
    node.lock_.lock();
    auto& new_map = node.new_fsr_map_;
    auto it_new = new_map.find(hash);
    if (it_new != new_map.end()) {
      node.lock_.unlock();
      return &it_new->second;
    }
    // Before we do anything, check if sr's match...
    GeometryState p;
    p.r() = r0;
    bool found = exhaustive_find_cell(p);
    int i_cell = p.lowest_coord().cell;
    int64_t sr0 = source_region_offsets_[i_cell] + p.cell_instance();
    p.r() = r1;
    found = exhaustive_find_cell(p);
    i_cell = p.lowest_coord().cell;
    int64_t sr1 = source_region_offsets_[i_cell] + p.cell_instance();
    if (sr0 != sr1) {
      fatal_error("Source region mismatch");
    }
    if (sr0 != source_region) {
      source_region = sr0;
      node.lock_.unlock();
      printf("Saved SR mismatch!\n");
      // return get_fsr(source_region, bin, r0, r1, ray_id, ip); // Wow -- this
      // actually works!

      return get_fsr(
        source_region, bin, r0, r1, ray_id); // Wow -- this actually works!
      // We also don't care at all about this operation being slow. Appending
      // new FSRs is going to be pretty rare, so it's fine if this triggers
      // occasionally.
      // TODO: if this happens, I might just want to punt...
    }

    // Now we should check if the bins match....
    Mesh* mesh = meshes_[fsr_[source_region].mesh_].get();
    RegularMesh* rmesh = dynamic_cast<RegularMesh*>(mesh);
    int bin_r0 = rmesh->get_bin(r0);
    int bin_r1 = rmesh->get_bin(r1);
    if (bin_r0 != bin) {
      bin = bin_r0;
      node.lock_.unlock();
      printf("Saved bin mismatch!\n");
      // return get_fsr(source_region, bin, r0, r1, ray_id, ip);
      return get_fsr(source_region, bin, r0, r1, ray_id);

      // TODO: If this happens, I might just want to punt...
    }

    // If not found, copy base FSR into new FSR
    FlatSourceRegion& new_fsr = fsr_[source_region];
    new_fsr.source_region_ = source_region;
    new_fsr.bin_ = bin;
    auto result = new_map.emplace(std::make_pair(hash, new_fsr));
    node.lock_.unlock();

#pragma omp atomic
    n_subdivided_source_regions_++;

    if (fsr_[source_region].mesh_ != C_NONE) {
      Mesh* mesh = meshes_[fsr_[source_region].mesh_].get();
      RegularMesh* rmesh = dynamic_cast<RegularMesh*>(mesh);
      StructuredMesh::MeshIndex mesh_index = rmesh->get_indices_from_bin(bin);
      hitmap[mesh_index[1] - 1][mesh_index[0] - 1] += 1;
    }

    /*
    if( mesh_index[1]-1 ==0 && mesh_index[0]-1 == 0)
    {
      printf("Adding source region %d to bin %d (y=%d, x=%d) with material =
    %d\n", source_region, bin, mesh_index[1]-1, mesh_index[0]-1,
    fsr_[source_region].material_); if( fsr_[source_region].material_ == 0)
      {
        fatal_error("Added fuel to region that should be moderator");
      }
    }
    */

    // printf("Adding source region %d to bin %d (y=%d, x=%d) with material =
    // %d\n", source_region, bin, mesh_index[1]-1, mesh_index[0]-1,
    // fsr_[source_region].material_);

    // if( n_subdivided_source_regions_ > 15028) // 2x2 mesh analytical
    /*
    if (n_subdivided_source_regions_ > 198832) // 8x8 mesh analytical
    {
      Mesh* mesh = meshes_[fsr_[source_region].mesh_].get();
      RegularMesh* rmesh = dynamic_cast<RegularMesh*>(mesh);
      StructuredMesh::MeshIndex mesh_index = rmesh->get_indices_from_bin(bin);
      printf("source region = %ld, bin = %d, index x = %d, index y = %d\n",
        source_region, bin, mesh_index[0], mesh_index[1]);
      printf("Ray id = %d, position r0 = (%f, %f, %f), position r1 = (%f, %f, "
             "%f) pi position = (%f, %f, %f) length = %.6le\n",
        ray_id, r0.x, r0.y, r0.z, r1.x, r1.y, r1.z, ip.r().x, ip.r().y,
        ip.r().z, (r1 - r0).norm());
      int bin_r0 = rmesh->get_bin(r0);
      int bin_r1 = rmesh->get_bin(r1);
      int bin_ip = rmesh->get_bin(ip.r());
      printf("bin r0 = %d, bin r1 = %d, bin ip = %d\n", bin_r0, bin_r1, bin_ip);
      Particle p;

      p.r() = r0;
      bool found = exhaustive_find_cell(p);
      int i_cell = p.lowest_coord().cell;
      int64_t sr0 = source_region_offsets_[i_cell] + p.cell_instance();
      printf("source region r0 = %ld, input sr = %d\n", sr0, source_region);
      p.r() = r1;
      found = exhaustive_find_cell(p);
      i_cell = p.lowest_coord().cell;
      int64_t sr1 = source_region_offsets_[i_cell] + p.cell_instance();

      printf("source region r1 = %ld, input sr = %d\n", sr1, source_region);
      i_cell = ip.lowest_coord().cell;
      int64_t srip = source_region_offsets_[i_cell] + ip.cell_instance();
      printf("source region ip = %ld, input sr = %d\n", srip, source_region);
      fatal_error("Too many subdivided source regions");
    }
    */
    return &result.first->second;
  } else {
    // Otherwise, access the existing FlatSourceRegion
    return &fsr_manifest_[it->second];
  }
}

void FlatSourceDomain::swap_flux(void)
{
#pragma omp parallel for
  for (int i = 0; i < fsr_manifest_.size(); i++) {
    FlatSourceRegion& fsr = fsr_manifest_[i];
    fsr.scalar_flux_old_.swap(fsr.scalar_flux_new_);
  }
}

void FlatSourceDomain::mesh_hash_grid_add(
  int mesh_index, int bin, uint64_t hash)
{
  Mesh* mesh = meshes_[mesh_index].get();
  RegularMesh* rmesh = dynamic_cast<RegularMesh*>(mesh);
  StructuredMesh::MeshIndex ijk = rmesh->get_indices_from_bin(bin);
  mesh_hash_grid_[mesh_index][ijk[2] - 1][ijk[1] - 1][ijk[0] - 1].push_back(
    hash);
}

vector<uint64_t> FlatSourceDomain::mesh_hash_grid_get_neighbors(
  int mesh_index, int bin)
{
  Mesh* mesh = meshes_[mesh_index].get();
  RegularMesh* rmesh = dynamic_cast<RegularMesh*>(mesh);
  StructuredMesh::MeshIndex ijk = rmesh->get_indices_from_bin(bin);
  vector<uint64_t> neighbors;
  // Insert up, down, north, south, east, and west neighbors (if they exist!)
  if (ijk[2] < rmesh->shape_[2] - 1) {
    auto& up = mesh_hash_grid_[mesh_index][ijk[2] + 1][ijk[1]][ijk[0]];
    neighbors.insert(neighbors.end(), up.begin(), up.end());
  }
  if (ijk[2] > 0) {
    auto& down = mesh_hash_grid_[mesh_index][ijk[2] - 1][ijk[1]][ijk[0]];
    neighbors.insert(neighbors.end(), down.begin(), down.end());
  }
  if (ijk[1] < rmesh->shape_[1] - 1) {
    auto& north = mesh_hash_grid_[mesh_index][ijk[2]][ijk[1] + 1][ijk[0]];
    neighbors.insert(neighbors.end(), north.begin(), north.end());
  }
  if (ijk[1] > 0) {
    auto& south = mesh_hash_grid_[mesh_index][ijk[2]][ijk[1] - 1][ijk[0]];
    neighbors.insert(neighbors.end(), south.begin(), south.end());
  }
  if (ijk[0] < rmesh->shape_[0] - 1) {
    auto& east = mesh_hash_grid_[mesh_index][ijk[2]][ijk[1]][ijk[0] + 1];
    neighbors.insert(neighbors.end(), east.begin(), east.end());
  }
  if (ijk[0] > 0) {
    auto& west = mesh_hash_grid_[mesh_index][ijk[2]][ijk[1]][ijk[0] - 1];
    neighbors.insert(neighbors.end(), west.begin(), west.end());
  }
  return neighbors;
}

// For a given FSR, this function finds the closest neighbor FSR
// in the mesh to be used as a candidate for merging
FlatSourceRegion* FlatSourceDomain::get_closest_neighbor(FlatSourceRegion& fsr)
{
  // Get the mesh index and bin of the FSR
  int mesh_index = fsr.mesh_;
  Mesh* mesh = meshes_[mesh_index].get();
  RegularMesh* rmesh = dynamic_cast<RegularMesh*>(mesh);
  int bin = fsr.bin_;
  StructuredMesh::MeshIndex ijk = rmesh->get_indices_from_bin(bin);

  // Get the neighbors of the FSR
  vector<uint64_t> neighbors = mesh_hash_grid_get_neighbors(mesh_index, bin);

  // Loop over the neighbors and find the closest one
  double min_distance = std::numeric_limits<double>::max();
  FlatSourceRegion* closest_fsr = nullptr;
  for (auto& hash : neighbors) {
    // Get hash and has controller bin
    int hash_bin = hash % N_FSR_HASH_BINS;
    SourceNode& node = controller_.nodes_[hash_bin];
    auto& map = node.fsr_map_;

    FlatSourceRegion& neighbor_fsr = fsr_manifest_[map[hash]];
    if (fsr.source_region_ == neighbor_fsr.source_region_) {
      double distance = (fsr.position_ - neighbor_fsr.position_).norm();
      if (distance < min_distance) {
        min_distance = distance;
        closest_fsr = &neighbor_fsr;
      }
    }
  }
  return closest_fsr;
}

bool FlatSourceDomain::merge_fsr(FlatSourceRegion& fsr)
{
  // Get the closest neighbor to the FSR
  FlatSourceRegion* closest_fsr = get_closest_neighbor(fsr);
  if (closest_fsr == nullptr) {
    return false;
  }

  // Merge the FSRs
  for (int e = 0; e < negroups_; e++) {
    fsr.scalar_flux_new_[e] += closest_fsr->scalar_flux_new_[e];
  }
  return true;
}

void FlatSourceDomain::update_fsr_manifest(void)
{
  for (int bin = 0; bin < N_FSR_HASH_BINS; bin++) {
    SourceNode& node = controller_.nodes_[bin];
    auto& map = node.new_fsr_map_;
    for (auto& pair : map) {
      fsr_manifest_.push_back(pair.second);
      node.fsr_map_[pair.first] = fsr_manifest_.size() - 1;

      // Store hash of this FSR into the hash grid for
      // lookup later on if we need to merge low volume FSRs
      int mesh_id = pair.second.mesh_;
      mesh_hash_grid_add(mesh_id, pair.second.bin_, pair.first);
    }
    map.clear();
  }

  for (int sr = 0; sr < n_source_regions_; sr++) {
    FlatSourceRegion& fsr = fsr_[sr];
    if (fsr.position_recorded_) {
      if (!fsr.is_in_manifest_) {
        fsr_manifest_.push_back(fsr);
        fsr.is_in_manifest_ = true;
        fsr.manifest_index_ = fsr_manifest_.size() - 1;
        n_subdivided_source_regions_++;
      }
    }
  }
}

} // namespace openmc