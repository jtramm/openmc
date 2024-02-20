#ifndef OPENMC_RANDOM_RAY_FLAT_SOURCE_DOMAIN_H
#define OPENMC_RANDOM_RAY_FLAT_SOURCE_DOMAIN_H

#include <vector>

#include "openmc/openmp_interface.h"
#include "openmc/position.h"
#include "openmc/source.h"

namespace openmc {
#define N_FSR_HASH_BINS 1000

//----------------------------------------------------------------------------
// Helper Structs

// A mapping object that is used to map between a specific random ray
// source region and an OpenMC native tally bin that it should score to
// every iteration.
struct TallyTask {
  int tally_idx;
  int filter_idx;
  int score_idx;
  int score_type;
  TallyTask(int tally_idx, int filter_idx, int score_idx, int score_type)
    : tally_idx(tally_idx), filter_idx(filter_idx), score_idx(score_idx),
      score_type(score_type)
  {}
};



class FlatSourceRegion {
public:
  FlatSourceRegion(int negroups) : tally_task_(negroups), scalar_flux_new_(negroups, 0.0f), scalar_flux_old_(negroups, 0.0f), scalar_flux_final_(negroups, 0.0f), source_(negroups), fixed_source_(negroups, 0.0f) {}
  
  FlatSourceRegion(const FlatSourceRegion& other) :
    material_(other.material_),
    position_recorded_(other.position_recorded_),
    position_(other.position_),
    volume_i_(other.volume_i_),
    volume_(other.volume_),
    volume_t_(other.volume_t_),
    was_hit_(other.was_hit_),
    mesh_(other.mesh_),
    scalar_flux_new_(other.scalar_flux_new_),
    scalar_flux_old_(other.scalar_flux_old_),
    scalar_flux_final_(other.scalar_flux_final_),
    source_(other.source_),
    fixed_source_(other.fixed_source_),
    tally_task_(other.tally_task_)
    {}

  OpenMPMutex lock_;
  int material_;
  int position_recorded_ {0};
  Position position_;
  double volume_i_ {0.0};
  double volume_ {0.0};
  double volume_t_ {0.0};
  int was_hit_ {0};
  int mesh_ {C_NONE};

  // 2D arrays with entry for each energy group
  vector<float> scalar_flux_new_;
  vector<float> scalar_flux_old_;
  vector<float> scalar_flux_final_;
  vector<float> source_;
  vector<float> fixed_source_;

    // Outer dimension is each energy group in the FSR, inner dimension is each tally operation that bin will perform
  vector<vector<TallyTask>> tally_task_; 

}; // class FlatSourceRegion 

// The source node receives the FSR ID + Mesh Bin index. It locks, and then 
// needs to decide if it should allocate another FSR or if there is one already
// present that works. Bascially, it needs some sort of map? Key + value pair
// should be the key is the hash and the value is the FSR itself.
class SourceNode {
public:
  SourceNode() = default;

  OpenMPMutex lock_;
  std::unordered_map<uint64_t, FlatSourceRegion> fsr_map_; // key is 64-bit hash, value is FSR itself
  
}; // class HashSourceController

class HashSourceController {
public:
  HashSourceController(int n_bins) : nodes_(n_bins) {};
  HashSourceController() : nodes_(N_FSR_HASH_BINS) {};

  vector<SourceNode> nodes_;

}; // class HashSourceController

/*
 * The FlatSourceDomain class encompasses data and methods for storing
 * scalar flux and source region for all flat source regions in a
 * random ray simulation domain.
 */

class FlatSourceDomain {
public:


  //----------------------------------------------------------------------------
  // Constructors
  FlatSourceDomain();

  //----------------------------------------------------------------------------
  // Methods
  void update_neutron_source(double k_eff);
  void prepare_base_neutron_source(double k_eff);
  double compute_k_eff(double k_eff_old);
  void normalize_scalar_flux_and_volumes(
    double total_active_distance_per_iteration);
  int64_t add_source_to_scalar_flux();
  double calculate_miss_rate();
  void batch_reset();
  void convert_source_regions_to_tallies();
  void random_ray_tally();
  void accumulate_iteration_flux();
  void output_to_vtk();
  void all_reduce_replicated_source_regions();
  void apply_fixed_source_to_source_region(
    Discrete* discrete, double strength_factor, int64_t source_region);
  void apply_fixed_source_to_cell_instances(int32_t i_cell, Discrete* discrete,
    double strength_factor, int target_material_id, const vector<int32_t>& instances);
  void apply_fixed_source_to_cell_and_children(int32_t i_cell,
    Discrete* discrete, double strength_factor, int32_t target_material_id);
  void convert_fixed_sources();
  void count_fixed_source_regions();
  double calculate_total_volume_weighted_source_strength();
  void swap_flux(void);

  void apply_mesh_to_cell_instances(int32_t i_cell,
  int32_t mesh, int target_material_id, const vector<int32_t>& instances);
  void apply_mesh_to_cell_and_children(int32_t i_cell,
  int32_t mesh, int32_t target_material_id);
  void apply_meshes();
  FlatSourceRegion* get_fsr(int64_t source_region, int bin);

  //----------------------------------------------------------------------------
  // Data members

  int negroups_;                  // Number of energy groups in simulation
  int64_t n_source_elements_ {0}; // Total number of source regions in the model
                                  // times the number of energy groups
  int64_t n_source_regions_ {0};  // Total number of source regions in the model
  int64_t n_fixed_source_regions_ {0}; // Total number of source regions with
                                       // non-zero fixed source terms

  bool mapped_all_tallies_ {false}; // If all source regions have been visited

  std::vector<FlatSourceRegion> fsr_;

  // 1D array representing source region starting offset for each OpenMC Cell
  // in model::cells
  std::vector<int64_t> source_region_offsets_;

  static std::unordered_map<int32_t, int32_t> mesh_map_;
  static vector<unique_ptr<Mesh>> meshes_;

  HashSourceController controller_;
  int64_t n_subdivided_source_regions_ {0};

}; // class FlatSourceDomain

//============================================================================
//! Non-Method Functions
//============================================================================

// Returns the inputted value with its
// endianness reversed. This is useful
// for conforming to the paraview VTK
// binary file format.
template<typename T>
T flip_endianness(T in)
{
  char* orig = reinterpret_cast<char*>(&in);
  char swapper[sizeof(T)];
  for (int i = 0; i < sizeof(T); i++) {
    swapper[i] = orig[sizeof(T) - i - 1];
  }
  T out = *reinterpret_cast<T*>(&swapper);
  return out;
}

template<typename T>
void parallel_fill(std::vector<T>& arr, T value)
{
#pragma omp parallel for schedule(static)
  for (int i = 0; i < arr.size(); i++) {
    arr[i] = value;
  }
}

} // namespace openmc

#endif // OPENMC_RANDOM_RAY_FLAT_SOURCE_DOMAIN_H
