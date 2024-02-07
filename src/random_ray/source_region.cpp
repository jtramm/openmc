#include "openmc/random_ray/source_region.h"
#include "openmc/cell.h"
#include "openmc/material.h"
#include "openmc/mgxs_interface.h"
#include "openmc/source.h"
#include "openmc/random_ray/tally_convert.h"
#include "openmc/container_util.h"
#include "openmc/distribution.h"
#include "openmc/search.h"

namespace openmc {

//==============================================================================
// Global variables
//==============================================================================

namespace random_ray {

// Scalars
int64_t n_source_elements {0}; // Total number of source regions in the model
                               // times the number of energy groups
int64_t n_source_regions {0};  // Total number of source regions in the model

// 1D arrays representing values for each OpenMC "Cell"
std::vector<int64_t> source_region_offsets;

// 1D arrays reprenting values for all source regions
std::vector<OpenMPMutex> lock;
std::vector<int> material;
std::vector<int> position_recorded;
std::vector<Position> position;
std::vector<double> volume;
std::vector<double> volume_t;
std::vector<int> was_hit;

// 2D arrays stored in 1D representing values for all source regions x energy
// groups
std::vector<float> scalar_flux_new;
std::vector<float> scalar_flux_old;
std::vector<float> scalar_flux_final;
std::vector<float> source;
std::vector<float> fixed_source;

} // namespace random_ray

//==============================================================================
// Non-method functions
//==============================================================================

void initialize_source_regions()
{
  int negroups = data::mg.num_energy_groups_;

  // Count the number of source regions, compute the cell offset
  // indices, and store the material type The reason for the offsets is that
  // some cell types may not have material fills, and therefore do not
  // produce FSRs. Thus, we cannot index into the global arrays directly
  for (auto&& c : model::cells) {
    if (c->type_ != Fill::MATERIAL) {
      random_ray::source_region_offsets.push_back(-1);
    } else {
      random_ray::source_region_offsets.push_back(random_ray::n_source_regions);
      random_ray::n_source_regions += c->n_instances_;
      random_ray::n_source_elements += c->n_instances_ * negroups;
    }
  }

  // Initialize cell-wise arrays
  random_ray::lock.resize(random_ray::n_source_regions);
  random_ray::material.resize(random_ray::n_source_regions);
  random_ray::position_recorded.assign(random_ray::n_source_regions, 0);
  random_ray::position.resize(random_ray::n_source_regions);
  random_ray::volume.assign(random_ray::n_source_regions, 0.0);
  random_ray::volume_t.assign(random_ray::n_source_regions, 0.0);
  random_ray::was_hit.assign(random_ray::n_source_regions, 0);

  // Initialize element-wise arrays
  random_ray::scalar_flux_new.assign(random_ray::n_source_elements, 0.0);
  if (settings::run_mode == RunMode::EIGENVALUE) {
    random_ray::scalar_flux_old.assign(random_ray::n_source_elements, 1.0);
  } else {
    random_ray::scalar_flux_old.assign(random_ray::n_source_elements, 0.0);
  }
  random_ray::scalar_flux_final.assign(random_ray::n_source_elements, 0.0);
  random_ray::source.resize(random_ray::n_source_elements);
  random_ray::fixed_source.assign(random_ray::n_source_elements, 0.0);
  random_ray::tally_task.resize(random_ray::n_source_elements);

  // Initialize material array
  int64_t source_region_id = 0;
  for (int i = 0; i < model::cells.size(); i++) {
    Cell& cell = *model::cells[i];
    if (cell.type_ == Fill::MATERIAL) {
      for (int j = 0; j < cell.n_instances_; j++) {
        int material = cell.material(j);
        random_ray::material[source_region_id++] = material;
      }
    }
  }

  // Sanity check
  if (source_region_id != random_ray::n_source_regions) {
    fatal_error("Unexpected number of source regions");
  }
}

void transfer_fixed_sources(int sampling_source)
{
  int negroups = data::mg.num_energy_groups_;

  double total_strength = 0;
  for (int es = 0; es < model::external_sources.size(); es++) {

    // Don't use the random ray sampling source for sampling neutrons
    if (es == sampling_source) {
      continue;
    }

    Source* s = model::external_sources[es].get();

    // Check for independent source
    IndependentSource* is = dynamic_cast<IndependentSource*>(s);

    total_strength += is->strength();
  }
  printf("total source strength = %.3le\n", total_strength);

  // Loop over source regions
  // Loop over sources
  // Loop over egroups

  int sr = 0;
  // Loop over material-filled cells
  for (int i = 0; i < model::cells.size(); i++) {
    Cell& cell = *model::cells[i];
    if (cell.type_ != Fill::MATERIAL) {
      continue;
    }
    // Loop over material-filled cell instances
    for (int j = 0; j < cell.n_instances_; j++, sr++) {
      int material = cell.material(j);
      int material_id = model::materials[material]->id();

      // Loop over external sources
      for (int es = 0; es < model::external_sources.size(); es++) {

        // Don't use the random ray sampling source for sampling neutrons
        if (es == sampling_source) {
          continue;
        }

        Source* s = model::external_sources[es].get();

        // Check for independent source
        IndependentSource* is = dynamic_cast<IndependentSource*>(s);

        // Skip source if it is not independent, as this implies it is not
        // the random ray source
        if (is == nullptr) {
          fatal_error("Sources must be of independent type in random ray mode");
        }

        // TODO: validate that domain ids are not empty...
        bool found = false;
        if (is->domain_type() == IndependentSource::DomainType::MATERIAL) {
          if (contains(is->domain_ids(), material_id)) {
            found = true;
          }
        } else if (is->domain_type() == IndependentSource::DomainType::CELL) {
          vector<int32_t> cell_ids = cell.get_cell_and_parent_cell_ids(j);
          for (int32_t& cell_id: cell_ids) {
            if (contains(is->domain_ids(), cell_id)) {
              found = true;
            }
          }
        } else if (is->domain_type() == IndependentSource::DomainType::UNIVERSE) {
          vector<int32_t> universe_ids = cell.get_universe_and_parent_universe_ids(j);
          for (int32_t& universe_id: universe_ids) {
            if (contains(is->domain_ids(), universe_id)) {
              found = true;
            }
          }
        }

        if (found) {
          Distribution* d = is->energy();
          Discrete* dd = dynamic_cast<Discrete*>(d);
          if (dd == nullptr) {
            fatal_error("discrete distributions only!");
          }
          const auto& discrete_energies = dd->x();
          const auto& discrete_probs    = dd->prob();
          // Loop over discrete distribution energies
          for (int e = 0; e < discrete_energies.size(); e++) {
            int g = lower_bound_index(data::mg.rev_energy_bins_.begin(),
                data::mg.rev_energy_bins_.end(), discrete_energies[e]);
            g = data::mg.num_energy_groups_ - g - 1.;

            random_ray::fixed_source[sr * negroups + g] += discrete_probs[e] * is->strength() / total_strength;
            printf("Setting source region %d group %d, with prob %.3lf, strength %.3lf, and total strength %.3lf to:    %.3lf\n", sr, g, discrete_probs[e], is->strength(), total_strength, random_ray::fixed_source[sr * negroups + g]);
          } // End loop over discrete energies
        } // End found conditional
      } // End loop over external sources
    } // End loop over material-filled cell instances
  } // End loop over material-filled cells
}

} // namespace openmc
