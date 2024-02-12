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

void transfer_fixed_sources_bottom_up(int sampling_source)
{
  int negroups = data::mg.num_energy_groups_;

  // Compute total combined strength of all sources
  double total_strength = 0;
  for (int es = 0; es < model::external_sources.size(); es++) {

    // Don't use the random ray sampling source for sampling neutrons
    if (es == sampling_source) {
      continue;
    }

    Source* s = model::external_sources[es].get();
    IndependentSource* is = dynamic_cast<IndependentSource*>(s);

    total_strength += is->strength();
  }


  // TODO: Convert to downwards search rather than upwards
  // Current upwards search has F = 360k, with 1 source, with hierarchy size = O(5)
  // This essentially results in a complexity of a few million. Problem is all
  // the vector allocs/deallocs I think.
  // IF we convert to a downward, then we have 1 source, with a hierarchy size of O(5),
  // resulting in 5 stack traversals. 
  // Need to use std::unordered_map<int32_t, vector<int32_t>> Cell::get_contained_cells(

  // Counter to track the source region ID as we traverse cells/instances
  int sr = 0;
  
  // Now we loop over all FSRs. For each FSR (a specific material-filled
  // cell instance), we loop over all external sources and check if any of them
  // apply to that cell instance. If a match is found, add discrete source
  // distribution to the FSR's fixed source term.

  // Loop over material-filled cells
  for (int i = 0; i < model::cells.size(); i++) {

    Cell& cell = *model::cells[i];
    if (cell.type_ != Fill::MATERIAL) {
      continue;
    }

    // Loop over cell instances
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
        IndependentSource* is = dynamic_cast<IndependentSource*>(s);

        // For this external source, check to see if any of its domains match
        // this cell or any of its parents.
        bool is_match;
        if (is->domain_type() == IndependentSource::DomainType::MATERIAL) {
          is_match = contains(is->domain_ids(), material_id);
        } else if (is->domain_type() == IndependentSource::DomainType::CELL) {
          vector<int32_t> cell_ids = cell.get_cell_and_parent_cell_ids(j);
          is_match = has_matching_element(is->domain_ids(), cell_ids);
        } else {
          vector<int32_t> universe_ids = cell.get_universe_and_parent_universe_ids(j);
          is_match = has_matching_element(is->domain_ids(), universe_ids);
        }

        if (is_match) {
          Discrete* discrete = dynamic_cast<Discrete*>(is->energy());
          const auto& discrete_energies = discrete->x();
          const auto& discrete_probs    = discrete->prob();

          // Loop over discrete distribution energies
          for (int e = 0; e < discrete_energies.size(); e++) {
            int g = data::mg.get_group_index(discrete_energies[e]);
            random_ray::fixed_source[sr * negroups + g] += discrete_probs[e] * is->strength() / total_strength;
            printf("Setting source region %d group %d, with prob %.3lf, strength %.3lf, and total strength %.3lf to:    %.3lf\n", sr, g, discrete_probs[e], is->strength(), total_strength, random_ray::fixed_source[sr * negroups + g]);
          } // End loop over discrete energies
        } // End match conditional
      } // End loop over external sources
    } // End loop over material-filled cell instances
  } // End loop over material-filled cells
}

void apply_source_to_source_region(Discrete* discrete, double strength_factor, int64_t source_region)
{
  int negroups = data::mg.num_energy_groups_;
    
  const auto& discrete_energies = discrete->x();
  const auto& discrete_probs    = discrete->prob();

  // Loop over discrete distribution energies
  for (int e = 0; e < discrete_energies.size(); e++) {
    int g = data::mg.get_group_index(discrete_energies[e]);
    random_ray::fixed_source[source_region * negroups + g] += discrete_probs[e] * strength_factor;
  }
}

void apply_source_to_cell(int32_t i_cell, Discrete* discrete, double strength_factor)
{
  Cell& cell = *model::cells[i_cell];
    
  for (int j = 0; j < cell.n_instances_; j++) {
    int64_t source_region = random_ray::source_region_offsets[i_cell] + j;
    apply_source_to_source_region(discrete, strength_factor, source_region);
  }
}

void apply_source_to_cell_children(int32_t i_cell, Discrete* discrete, double strength_factor)
{
  Cell& cell = *model::cells[i_cell];
  
  std::unordered_map<int32_t, vector<int32_t>> cell_instance_list = cell.get_contained_cells(0, nullptr);
    
  for (const auto& pair : cell_instance_list) {
    int32_t i_child_cell = pair.first;
    Cell& child_cell = *model::cells[i_child_cell];
    if (child_cell.type_ == Fill::MATERIAL) {
      for (int32_t j : pair.second) {
        int64_t source_region = random_ray::source_region_offsets[i_cell] + j;
        apply_source_to_source_region(discrete, strength_factor, source_region);
      }
    }
  }
}

void transfer_fixed_sources(int sampling_source)
{
  int negroups = data::mg.num_energy_groups_;

  // Compute total combined strength of all sources
  double total_strength = 0;
  for (int es = 0; es < model::external_sources.size(); es++) {

    // Don't use the random ray sampling source for sampling neutrons
    if (es == sampling_source) {
      continue;
    }

    Source* s = model::external_sources[es].get();
    IndependentSource* is = dynamic_cast<IndependentSource*>(s);

    total_strength += is->strength();
  }

  // Loop over external sources
  for (int es = 0; es < model::external_sources.size(); es++) {

    // Don't use the random ray sampling source for sampling neutrons
    if (es == sampling_source) {
      continue;
    }

    Source* s = model::external_sources[es].get();
    IndependentSource* is = dynamic_cast<IndependentSource*>(s);
    const std::unordered_set<int32_t>& domain_ids = is->domain_ids();
    Discrete* discrete = dynamic_cast<Discrete*>(is->energy());
    double strength_factor = is->strength() / total_strength;

    if (is->domain_type() == IndependentSource::DomainType::MATERIAL) {
      for (int32_t material_id : domain_ids) {
        // I know the material ID. Now I want to find all material filled cells that match this
        for (int i_cell = 0; i_cell < model::cells.size(); i_cell++) {
          Cell& cell = *model::cells[i_cell];
          if (cell.type_ != Fill::MATERIAL) {
            continue;
          }
          //apply_source_to_cell(i, discrete, is->strength() / total_strength);
          // Loop over cell instances
          for (int j = 0; j < cell.n_instances_; j++) {
            int material = cell.material(j);
            int cell_material_id = model::materials[material]->id();
            if (material_id == cell_material_id) {
              int64_t source_region = random_ray::source_region_offsets[i_cell] + j;
              apply_source_to_source_region(discrete, strength_factor, source_region);
            }
          }
        }
      }
    } else if (is->domain_type() == IndependentSource::DomainType::CELL) {
      for (int32_t cell_id : domain_ids) {
        int32_t i_cell = model::cell_map[cell_id];
        Cell& cell = *model::cells[i_cell];
          
        // We can (and should) short circuit the logic if this is already a material filled cell.
        if (cell.type_ == Fill::MATERIAL) {
          apply_source_to_cell(i_cell, discrete, strength_factor);
        } else {
          // If we are not in a material filled cell, then we need to check cell IDs of all child cells downwards
          apply_source_to_cell_children(i_cell, discrete, strength_factor);
        }
      }
    } else if (is->domain_type() == IndependentSource::DomainType::UNIVERSE) {
      for (int32_t universe_id : domain_ids) {
        int32_t i_universe = model::universe_map[universe_id];
        Universe& universe = *model::universes[i_universe];

        for (int32_t cell_id : universe.cells_) {
          int32_t i_cell = model::cell_map[cell_id];
          Cell& cell = *model::cells[i_cell];

          // We can (and should) short circuit the logic if this is already a material filled cell.
          if (cell.type_ == Fill::MATERIAL) {
            apply_source_to_cell(i_cell, discrete, strength_factor);
          } else {
            apply_source_to_cell_children(i_cell, discrete, strength_factor);
          }
        } // End loop over cells within the target universe
      } // End loop over target universes
    }

  } // End loop over external sources
}

} // namespace openmc
