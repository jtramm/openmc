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
  // I Actually don't think we should change the position, as it will
  // be used to populate the tally tasks.

  /*

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

  */

  // Add the volume to this FSR
  volume_t_ += other.volume_t_;
  volume_ += other.volume_;

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
  material_filled_cell_instance_.assign(n_source_regions_, negroups_);

  // If in eigenvalue mode, set starting flux to guess of unity
  if (settings::run_mode == RunMode::EIGENVALUE) {
#pragma omp parallel for
    for (int i = 0; i < n_source_regions_; i++) {
      auto& fsr = material_filled_cell_instance_[i];
      std::fill(fsr.scalar_flux_old_.begin(), fsr.scalar_flux_old_.end(), 1.0f);
    }
  }

  // Initialize material array
  int64_t source_region_id = 0;
  for (int i = 0; i < model::cells.size(); i++) {
    Cell& cell = *model::cells[i];
    if (cell.type_ == Fill::MATERIAL) {
      for (int j = 0; j < cell.n_instances_; j++) {
        //printf("setting material from cell ID %d, instance %d, to %d\n", i, j,
       //   cell.material(j));
        material_filled_cell_instance_[source_region_id++].material_ =
          cell.material(j);
      }
    }
  }

  // Sanity check
  if (source_region_id != n_source_regions_) {
    fatal_error("Unexpected number of source regions");
  }

  // Initialize tally volumes
  tally_volumes_.resize(model::tallies.size());
  for (int i = 0; i < model::tallies.size(); i++) {
    tally_volumes_[i] = model::tallies[i]->results();
  }
  // Initialize tally volumes
  tally_.resize(model::tallies.size());
  for (int i = 0; i < model::tallies.size(); i++) {
    tally_[i] = model::tallies[i]->results();
  }

  // Compute simulation domain volume based on ray source
  IndependentSource* is =
    dynamic_cast<IndependentSource*>(RandomRay::ray_source_.get());
  SpatialDistribution* space_dist = is->space();
  SpatialBox* sb = dynamic_cast<SpatialBox*>(space_dist);
  Position dims = sb->upper_right() - sb->lower_left();
  simulation_volume_ = dims.x * dims.y * dims.z;
}

// Set the 3D xtensor to zero for all tallies
void FlatSourceDomain::reset_tally_volumes()
{
#pragma omp parallel for
  for (auto& tensor : tally_volumes_) {
    tensor.fill(0.0); // Set all elements of the tensor to 0.0
  }
#pragma omp parallel for
  for (auto& tensor : tally_) {
    tensor.fill(0.0); // Set all elements of the tensor to 0.0
  }
}

void FlatSourceDomain::batch_reset()
{
// Reset scalar fluxes, iteration volume tallies, and region hit flags to
// zero
#pragma omp parallel for
  for (int i = 0; i < known_fsr_.size(); i++) {
    FlatSourceRegion& fsr = known_fsr_[i];
    std::fill(fsr.scalar_flux_new_.begin(), fsr.scalar_flux_new_.end(), 0.0f);
    fsr.volume_ = 0.0;
    fsr.was_hit_ = 0;
  }
}

void FlatSourceDomain::accumulate_iteration_flux()
{
#pragma omp parallel for
  for (int i = 0; i < known_fsr_.size(); i++) {
    FlatSourceRegion& fsr = known_fsr_[i];
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
    auto& fsr = material_filled_cell_instance_[sr];
    int material = fsr.material_;

    for (int e_out = 0; e_out < negroups_; e_out++) {
      //printf("material = %d, size of macro_xs_ = %d\n", material,
       // data::mg.macro_xs_.size());
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
      auto& fsr = material_filled_cell_instance_[sr];
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
      auto& fsr = material_filled_cell_instance_[i];
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
  for (int i = 0; i < known_fsr_.size(); i++) {
    FlatSourceRegion& fsr = known_fsr_[i];
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
    for (int i = 0; i < known_fsr_.size(); i++) {
      FlatSourceRegion& fsr = known_fsr_[i];
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
    for (int i = 0; i < known_fsr_.size(); i++) {
      FlatSourceRegion& fsr = known_fsr_[i];
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
  for (int i = 0; i < known_fsr_.size(); i++) {
    FlatSourceRegion& fsr = known_fsr_[i];
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
int64_t n_negative = 0;
#pragma omp parallel for reduction(+ : n_hits)
  for (int64_t i = 0; i < known_fsr_.size(); i++) {
    FlatSourceRegion& fsr = known_fsr_[i];
    if (fsr.is_merged_)
      continue;

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
      if (fsr.scalar_flux_new_[e] < 0.0) {
        n_negative++;
      }
    }
  }
  if (n_negative / (static_cast<double>(n_hits)*negroups_) > 0.01) {
    fatal_error("More than 1% of the scalar fluxes are negative. This may be a "
            "sign of a problem with the simulation.");
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
  for (int i = 0; i < known_fsr_.size(); i++) {
    FlatSourceRegion& fsr = known_fsr_[i];
    if (fsr.is_merged_)
      continue;

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
  for (int i = 0; i < known_fsr_.size(); i++) {
    FlatSourceRegion& fsr = known_fsr_[i];
    if (fsr.is_merged_)
      continue;

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

void FlatSourceDomain::initialize_tally_tasks(FlatSourceRegion& fsr)
{
  // A particle located at the recorded midpoint of a ray
  // crossing through this source region is used to establish
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

    // Loop over all active tallies. This logic is essentially identical
    // to what happens when scanning for applicable tallies during
    // MC transport.
    for (int i_tally = 0; i_tally < model::tallies.size(); i_tally++) {
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
          TallyTask task(i_tally, filter_index, score_index, score_bin);
          fsr.tally_task_[e].push_back(task);
          fsr.volume_task_.insert(task);
        }
      }
    }
    // Reset all the filter matches for the next tally event.
    for (auto& match : p.filter_matches())
      match.bins_present_ = false;
  }
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

  // Reset our tally volumes to zero
  reset_tally_volumes();

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
  for (int i = 0; i < known_fsr_.size(); i++) {
    FlatSourceRegion& fsr = known_fsr_[i];
    if (fsr.is_merged_)
      continue;

    // The fsr.volume_ is the unitless fractional simulation averaged volume
    // (i.e., it is the FSR's fraction of the overall simulation volume). The
    // simulation_volume_ is the total 3D physical volume in cm^3 of the entire
    // global simulation domain (as defined by the ray source box). Thus, the
    // FSR's true 3D spatial volume in cm^3 is found by multiplying its fraction
    // of the total volume by the total volume. Not important in eigenvalue
    // solves, but useful in fixed source solves for returning the flux shape
    // with a magnitude that makes sense relative to the fixed source strength.
    double volume = fsr.volume_ * simulation_volume_;

    double material = fsr.material_;
    for (int e = 0; e < negroups_; e++) {
      double flux = fsr.scalar_flux_new_[e] * volume;
      for (auto& task : fsr.tally_task_[e]) {
        double score;
        switch (task.score_type) {

        case SCORE_FLUX:
          score =
            flux; // This is not right, as I need to divide by total volume
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
        if (task.score_type == SCORE_FLUX) {
#pragma omp atomic
          tally_[task.tally_idx](
            task.filter_idx, task.score_idx, TallyResult::VALUE) += score;
        } else {
#pragma omp atomic
          tally.results_(task.filter_idx, task.score_idx, TallyResult::VALUE) +=
            score;
        }
      }
    } // end energy group loop
    for (const auto& task : fsr.volume_task_) {
      if (task.score_type == SCORE_FLUX) {
#pragma omp atomic
        tally_volumes_[task.tally_idx](
          task.filter_idx, task.score_idx, TallyResult::VALUE) += volume;
      }
    }
  } // end FSR loop

#pragma omp parallel for
  for (int i = 0; i < known_fsr_.size(); i++) {
    FlatSourceRegion& fsr = known_fsr_[i];
    if (fsr.is_merged_)
      continue;
    for (int e = 0; e < negroups_; e++) {
      for (auto& task : fsr.tally_task_[e]) {
        if (task.score_type == SCORE_FLUX) {
          Tally& tally {*model::tallies[task.tally_idx]};
          double vol_flux = tally_[task.tally_idx](
            task.filter_idx, task.score_idx, TallyResult::VALUE);
          double vol = tally_volumes_[task.tally_idx](
            task.filter_idx, task.score_idx, TallyResult::VALUE);
          if (vol > 0.0) {
#pragma omp atomic
            tally.results_(task.filter_idx, task.score_idx,
              TallyResult::VALUE) += vol_flux / vol;
          }
        }
      }
    }
  }

  openmc::simulation::time_tallies.stop();
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

    FlatSourceRegion default_fsr(negroups_);
    default_fsr.material_ = C_NONE;

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
          if (!found) {
            voxel_indices[z * Ny * Nx + y * Nx + x] = &default_fsr;
            hits[z * Ny * Nx + y * Nx + x] = 0;
            continue;
          }
          int i_cell = p.lowest_coord().cell;
          int64_t source_region_idx =
            source_region_offsets_[i_cell] + p.cell_instance();
          auto& fsr = material_filled_cell_instance_[source_region_idx];
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
            StructuredMesh::MeshIndex ijk = rmesh->get_indices_from_bin(bin);
            // hit_count = hitmap[ijk[1] - 1][ijk[0] - 1];
            hit_count = 0;
            region = get_fsr(source_region_idx, bin, p.r(), p.r(), 0);
          } else {
            region = get_fsr(source_region_idx, 0, p.r(), p.r(), 0);
          }
          if (region == nullptr) {
            fatal_error(
              fmt::format("Could not find FSR for position ({}, {}, {})",
                sample.x, sample.y, sample.z));
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
      float value;
      if (fsr == &default_fsr) {
        value = std::numeric_limits<double>::quiet_NaN();
      } else {
        value = future_prn(10, reinterpret_cast<uint64_t>(fsr));
      }
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
      if (mat != C_NONE) {
        for (int g = 0; g < negroups_; g++) {
          float flux = fsr->scalar_flux_final_[g];
          flux /= (settings::n_batches - settings::n_inactive);
          float Sigma_f = data::mg.macro_xs_[mat].get_xs(
            MgxsType::FISSION, g, nullptr, nullptr, nullptr, 0, 0);
          total_fission += Sigma_f * flux;
        }
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
  auto& fsr = material_filled_cell_instance_[source_region];
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
    auto& fsr = material_filled_cell_instance_[sr];
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
  for (int i = 0; i < known_fsr_.size(); i++) {
    FlatSourceRegion& fsr = known_fsr_[i];
    if (fsr.is_merged_)
      continue;
    double volume = fsr.volume_;
    for (int e = 0; e < negroups_; e++) {
      source_strength += fsr.fixed_source_[e] * volume;
    }
    // if( source_strength > 0.0)
    // printf("source strength in FSR %d is %.6le\n", i, source_strength);
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

    // double strength_factor = is->strength() / total_strength;
    double strength_factor = is->strength();

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
    auto& fsr = material_filled_cell_instance_[sr];
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
      material_filled_cell_instance_[source_region].mesh_ = mesh;
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

  /*  if (meshes_.size() > 0) {
     int x = dynamic_cast<RegularMesh*>(meshes_[0].get())->shape_[0];
     int y = dynamic_cast<RegularMesh*>(meshes_[0].get())->shape_[1];
     hitmap = vector<vector<int>>(y, vector<int>(x, 0));
   } */
}

// TODO:
// The bug here I think, (at least, it is A bug, and would explain the SHM
// fails, but serial glory), is that the locking is done with the assumption
// that once a pointer to a FSR is generated, that pointer is still valid.
// Unfortunately, that's not true, as the new map is storing the FSR directly.
// After we get a handle to it, another thread WILL come along and push more
// FSRs back to that new map, and the FSR will get moved. The thread still
// working with that FSR is then writing off in lala land and its updates are
// also toast.

// Solution: just make the new map a map of unique pointers, and copy those
// fuckers.

// FlatSourceRegion* FlatSourceDomain::get_fsr(int64_t source_region, int bin,
//  Position r0, Position r1, int ray_id, GeometryState& ip)
FlatSourceRegion* FlatSourceDomain::get_fsr(
  int64_t source_region, int bin, Position r0, Position r1, int ray_id)
{
  // This conditional will not be triggered right now due to the
  // presence of two function prototypes, but leaving it in for later
  FlatSourceRegion& base_fsr = material_filled_cell_instance_[source_region];
  if (base_fsr.mesh_ == C_NONE) {
    if (base_fsr.is_in_manifest_) {
      return &known_fsr_[base_fsr.manifest_index_];
    } else {
      return &material_filled_cell_instance_[source_region];
    }
  }
  FSRKey key(source_region, bin);

  // Check if the FlatSourceRegion with this hash already exists
  auto it = known_fsr_map_.find(key);
  if (it == known_fsr_map_.end()) {
    discovered_fsr_parallel_map_.lock(key);
    if (discovered_fsr_parallel_map_.contains(key)) {
      FlatSourceRegion* existing_fsr = &discovered_fsr_parallel_map_[key];
      discovered_fsr_parallel_map_.unlock(key);
      return existing_fsr;
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
      discovered_fsr_parallel_map_.unlock(key);
      printf("Saved SR mismatch!\n");
      // return get_fsr(source_region, bin, r0, r1, ray_id, ip); // Wow --
      // this actually works!

      return get_fsr(
        source_region, bin, r0, r1, ray_id); // Wow -- this actually works!
      // We also don't care at all about this operation being slow. Appending
      // new FSRs is going to be pretty rare, so it's fine if this triggers
      // occasionally.
      // TODO: if this happens, I might just want to punt...
    }

    // Now we should check if the bins match....
    Mesh* mesh =
      meshes_[material_filled_cell_instance_[source_region].mesh_].get();
    RegularMesh* rmesh = dynamic_cast<RegularMesh*>(mesh);
    int bin_r0 = rmesh->get_bin(r0);
    int bin_r1 = rmesh->get_bin(r1);
    if (bin_r0 != bin) {
      bin = bin_r0;
      discovered_fsr_parallel_map_.unlock(key);
      printf("Saved bin mismatch!\n");
      // return get_fsr(source_region, bin, r0, r1, ray_id, ip);
      return get_fsr(source_region, bin, r0, r1, ray_id);

      // TODO: If this happens, I might just want to punt...
    }

    // If not found, copy base FSR into new FSR
    // FlatSourceRegion& new_fsr = fsr_[source_region];
    // There's no lock over this stuff!

    // FlatSourceRegion* new_fsr = &discovered_fsr_parallel_map_[key];
    FlatSourceRegion* new_fsr = discovered_fsr_parallel_map_.emplace(
      key, material_filled_cell_instance_[source_region]);

    //*new_fsr = material_filled_cell_instance_[source_region];
    new_fsr->source_region_ = source_region;
    new_fsr->bin_ = bin;
    // new_fsr->position_ = r0;
    // new_fsr->position_recorded_ = 1;
    discovered_fsr_parallel_map_.unlock(key);

    // auto result = new_map.emplace(std::make_pair(
    // hash, std::make_unique<FlatSourceRegion>(fsr_[source_region])));

    // TODO: Bring back the hitmap?
    /*     if (material_filled_cell_instance_[source_region].mesh_ != C_NONE)
       { Mesh* mesh =
       meshes_[material_filled_cell_instance_[source_region].mesh_].get();
          RegularMesh* rmesh = dynamic_cast<RegularMesh*>(mesh);
          StructuredMesh::MeshIndex mesh_index =
       rmesh->get_indices_from_bin(bin); hitmap[mesh_index[1] -
       1][mesh_index[0]
       - 1] += 1;
        } */

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
      printf("Ray id = %d, position r0 = (%f, %f, %f), position r1 = (%f, %f,
    "
             "%f) pi position = (%f, %f, %f) length = %.6le\n",
        ray_id, r0.x, r0.y, r0.z, r1.x, r1.y, r1.z, ip.r().x, ip.r().y,
        ip.r().z, (r1 - r0).norm());
      int bin_r0 = rmesh->get_bin(r0);
      int bin_r1 = rmesh->get_bin(r1);
      int bin_ip = rmesh->get_bin(ip.r());
      printf("bin r0 = %d, bin r1 = %d, bin ip = %d\n", bin_r0, bin_r1,
    bin_ip); Particle p;

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
    return new_fsr;
  } else {
    // Otherwise, access the existing FlatSourceRegion
    return &known_fsr_[it->second];
  }
}

void FlatSourceDomain::swap_flux(void)
{
#pragma omp parallel for
  for (int i = 0; i < known_fsr_.size(); i++) {
    FlatSourceRegion& fsr = known_fsr_[i];
    fsr.scalar_flux_old_.swap(fsr.scalar_flux_new_);
  }
}

void FlatSourceDomain::mesh_hash_grid_add(
  int mesh_index, int bin, FSRKey fsr_key)
{
  Mesh* mesh = meshes_[mesh_index].get();
  RegularMesh* rmesh = dynamic_cast<RegularMesh*>(mesh);
  StructuredMesh::MeshIndex ijk = rmesh->get_indices_from_bin(bin);
  RegularMeshKey rm_key(mesh_index, ijk);
  mesh_hash_grid_[rm_key].push_back(fsr_key);
}

vector<FSRKey> FlatSourceDomain::mesh_hash_grid_get_neighbors(
  int mesh_index, int bin)
{
  Mesh* mesh = meshes_[mesh_index].get();
  RegularMesh* rmesh = dynamic_cast<RegularMesh*>(mesh);
  StructuredMesh::MeshIndex ijk_base = rmesh->get_indices_from_bin(bin);

  vector<FSRKey> neighbors;
  // Insert up, down, north, south, east, and west neighbors (if they exist!)
  if (ijk_base[2] < rmesh->shape_[2]) {
    StructuredMesh::MeshIndex ijk = ijk_base;
    ijk[2]++;
    RegularMeshKey rm_key(mesh_index, ijk);
    auto& up = mesh_hash_grid_[rm_key];
    neighbors.insert(neighbors.end(), up.begin(), up.end());
  }
  if (ijk_base[2] > 1) {
    StructuredMesh::MeshIndex ijk = ijk_base;
    ijk[2]--;
    RegularMeshKey rm_key(mesh_index, ijk);
    auto& down = mesh_hash_grid_[rm_key];
    neighbors.insert(neighbors.end(), down.begin(), down.end());
  }
  if (ijk_base[1] < rmesh->shape_[1]) {
    StructuredMesh::MeshIndex ijk = ijk_base;
    ijk[1]++;
    RegularMeshKey rm_key(mesh_index, ijk);
    auto& north = mesh_hash_grid_[rm_key];
    neighbors.insert(neighbors.end(), north.begin(), north.end());
  }
  if (ijk_base[1] > 1) {
    StructuredMesh::MeshIndex ijk = ijk_base;
    ijk[1]--;
    RegularMeshKey rm_key(mesh_index, ijk);
    auto& south = mesh_hash_grid_[rm_key];
    neighbors.insert(neighbors.end(), south.begin(), south.end());
  }
  if (ijk_base[0] < rmesh->shape_[0]) {
    StructuredMesh::MeshIndex ijk = ijk_base;
    ijk[0]++;
    RegularMeshKey rm_key(mesh_index, ijk);
    auto& east = mesh_hash_grid_[rm_key];
    neighbors.insert(neighbors.end(), east.begin(), east.end());
  }
  if (ijk_base[0] > 1) {
    StructuredMesh::MeshIndex ijk = ijk_base;
    ijk[0]--;
    RegularMeshKey rm_key(mesh_index, ijk);
    auto& west = mesh_hash_grid_[rm_key];
    neighbors.insert(neighbors.end(), west.begin(), west.end());
  }
  return neighbors;
}

// For a given FSR, this function finds the closest neighbor FSR
// in the mesh to be used as a candidate for merging
int64_t FlatSourceDomain::get_largest_neighbor(FlatSourceRegion& fsr)
{

  // Get the mesh index and bin of the FSR
  int mesh_index = fsr.mesh_;
  Mesh* mesh = meshes_[mesh_index].get();
  RegularMesh* rmesh = dynamic_cast<RegularMesh*>(mesh);
  int bin = fsr.bin_;
  StructuredMesh::MeshIndex ijk = rmesh->get_indices_from_bin(bin);

  // Get the neighbors of the FSR
  vector<FSRKey> neighbors = mesh_hash_grid_get_neighbors(mesh_index, bin);

  if (neighbors.size() == 0) {
    printf("mesh index = %d, bin = %d, Bins ijk = %d %d %d, sr = %ld\n",
      mesh_index, bin, ijk[0], ijk[1], ijk[2], fsr.source_region_);
  }

  // Loop over the neighbors and find the closest one
  double largest_volume = 0.0;
  int64_t largest_fsr_index = C_NONE;

  for (auto& key : neighbors) {
    int64_t neighbor_fsr_index = known_fsr_map_[key];
    FlatSourceRegion& neighbor_fsr = known_fsr_[neighbor_fsr_index];
    if (fsr.source_region_ == neighbor_fsr.source_region_) {
      double vol = neighbor_fsr.volume_t_;
      if (vol > largest_volume &&
          vol > fsr.volume_t_ * volume_merging_threshold_ &&
          !neighbor_fsr.is_merged_ && !neighbor_fsr.is_consumer_) {
        largest_volume = vol;
        largest_fsr_index = neighbor_fsr_index;
      }
    }
  }

  return largest_fsr_index;
}

// OK - to do for today:
// 1. Finish small volume merging class
// 2. Test on some different problems
// profit?
// 3. Add printout at the end showing different types of FSRs
// 4. Put in bug fixes as PRs (openmp)
// 5. Update issues with source assignments in base fixed source PR

// This class
bool FlatSourceDomain::merge_fsr(FlatSourceRegion& fsr)
{
  // Get the largest neighbor to the FSR
  int64_t largest_fsr_index = get_largest_neighbor(fsr);
  if (largest_fsr_index == C_NONE) {
    // printf("Merging of fsr failed\n");
    // fatal_error("Failed to merge small volume FSR");
    return false;
  }

  FlatSourceRegion& largest_fsr = known_fsr_[largest_fsr_index];
  // printf("Merging fsr of volume %.3le with larger FSR with volume %.3le\n",
  //   fsr.volume_t_, largest_fsr.volume_t_);

  // Merge FSR
  largest_fsr.merge(fsr);
  largest_fsr.is_consumer_ = true;

  // Mark merged FSR as being merged (to prevent contrib to k-eff and tallies)
  fsr.is_merged_ = true;

  // Point FSR hash to the FSR that it merged with
  FSRKey key(fsr.source_region_, fsr.bin_);

  known_fsr_map_[key] = largest_fsr_index;

  // Do we need to do anything with the deleted FSR?
  return true;
}

int64_t FlatSourceDomain::check_for_small_FSRs(void)
{
  // I vote that I just merge stuff and leave the dead FSRs there
  // Loop over fsr manifest
  int n_merges = 0;
  int n_prev_merges = 0;
  // #pragma omp parallel for reduction(+:n_merges, n_prev_merges)
  for (int i = 0; i < known_fsr_.size(); i++) {
    FlatSourceRegion& fsr = known_fsr_[i];
    if (fsr.mesh_ == C_NONE)
      continue;
    // We can lock the base source region, as no two FSRs that are
    // not from the same source region will ever interact.
    // fsr_[fsr.source_region_].lock_.lock();
    if (fsr.is_merged_) {
      n_prev_merges++;
      continue;
    }
    if (fsr.is_consumer_ || fsr.is_merge_failed_)
      continue;

    if (fsr.was_hit_ <= merging_threshold_) {
      fsr.no_hit_streak_++;
    } else {
      fsr.no_hit_streak_ = 0;
    }

    if (fsr.no_hit_streak_ >= streak_needed_to_merge_) {
      bool merge_success = merge_fsr(fsr);
      if (merge_success) {
        n_merges++;
        n_subdivided_source_regions_--;
        // printf(
        //     "Merge of FSR at %d index with volume = %.9le\n", i,
        //     fsr.volume_t_);
      } else {
        fsr.is_merge_failed_ = true;
      }
    }
    // fsr_[fsr.source_region_].lock_.unlock();
  }
  // printf("n_merges = %d, total merges to date = %d\n", n_merges,
  //   n_merges + n_prev_merges);
  return n_merges + n_prev_merges;
}

void FlatSourceDomain::update_fsr_manifest(void)
{
  int64_t starting_size = known_fsr_.size();

  // --------------------------------------------------------------------------!
  // Copy discovered mesh-subdivided FSRs into the known FSR vector and map
  // --------------------------------------------------------------------------!

  int64_t n_new_fsrs =
    discovered_fsr_parallel_map_.move_contents_into_vector(known_fsr_);
  for (int64_t i = starting_size; i < known_fsr_.size(); i++) {
    // Store the recently discovered FSRs in the known FSR map
    FSRKey key(known_fsr_[i].source_region_, known_fsr_[i].bin_);
    known_fsr_map_[key] = i;

    // Add the FSR to the mesh hash grid
    int64_t mesh_id = known_fsr_[i].mesh_;
    int64_t bin = known_fsr_[i].bin_;
    mesh_hash_grid_add(mesh_id, bin, key);
  }

  // --------------------------------------------------------------------------!
  // Copy discovered base FSRs into the known FSR vector
  // --------------------------------------------------------------------------!

  // For non-subdivided FSRS (i.e., FSRs with no mesh assigned to them),
  // we need to move them all from the base FSR vector to the
  // known FSR vector and mark them as discovered. They are not added
  // to the map, as they do not have a mesh bin or hash key.
  for (int sr = 0; sr < material_filled_cell_instance_.size(); sr++) {
    FlatSourceRegion& base_fsr = material_filled_cell_instance_[sr];
    if (base_fsr.position_recorded_) {
      if (!base_fsr.is_in_manifest_) {
        // If the FSR has been hit and has not yet been added to the list
        // of known physical FSRs, add it to the vector
        known_fsr_.push_back(base_fsr);

        // After copying the FSR into the known FSR vector, mark it as
        // being as such and point it to the correct location in the vector
        base_fsr.is_in_manifest_ = true;
        base_fsr.manifest_index_ = known_fsr_.size() - 1;
      }
    }
  }

  // --------------------------------------------------------------------------!
  // Initialize tally tasks for new FSRs
  // --------------------------------------------------------------------------!

  // Bookkeeping to keep track of total source regions
  n_subdivided_source_regions_ += known_fsr_.size() - starting_size;
  discovered_source_regions_ += known_fsr_.size() - starting_size;

#pragma omp parallel for
  for (int64_t i = starting_size; i < known_fsr_.size(); i++) {
    initialize_tally_tasks(known_fsr_[i]);
  }
}

} // namespace openmc