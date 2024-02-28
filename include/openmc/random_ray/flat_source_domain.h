#ifndef OPENMC_RANDOM_RAY_FLAT_SOURCE_DOMAIN_H
#define OPENMC_RANDOM_RAY_FLAT_SOURCE_DOMAIN_H

#include <functional>
#include <vector>

#include "openmc/mesh.h"
#include "openmc/openmp_interface.h"
#include "openmc/position.h"
#include "openmc/random_ray/parallel_hash_map.h"
#include "openmc/source.h"

namespace openmc {

// This is the standard hash combine function from boost
// https://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine
inline void hash_combine(size_t& seed, const size_t& v)
{
  seed ^= (v + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

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
  TallyTask() = default;

  bool operator==(const TallyTask& other) const
  {
    return tally_idx == other.tally_idx && filter_idx == other.filter_idx &&
           score_idx == other.score_idx && score_type == other.score_type;
  }

  struct HashFunctor {
    size_t operator()(const TallyTask& task) const
    {
      size_t seed = 0;
      hash_combine(seed, task.tally_idx);
      hash_combine(seed, task.filter_idx);
      hash_combine(seed, task.score_idx);
      hash_combine(seed, task.score_type);
      return seed;
    }
  };
};

// Actually -- since we want to be able to use the base as an active
// FSR when we are discovering them in the first iteration, it is probably
// best to just get rid of the Base and make everything the same. Much less
// confusing than having two classes, where the second barely adds anything.

// I am drawn to the idea of not having to have ANY base FSRs, and just
// do everything dynamic, but it is very true that initializing the FSRs
// individually in a bottom up manner is going to be stupid and expensive.
// It's fine if it was needed, but I don't think there are going to be
// too many cases where only a few percent of FSRs are actually physical
// (absent any craziness with mesh overlaying)

class FlatSourceRegion {
public:
  FlatSourceRegion() = default;
  FlatSourceRegion(int negroups)
    : tally_task_(negroups), scalar_flux_new_(negroups, 0.0f),
      scalar_flux_old_(negroups, 0.0f), scalar_flux_final_(negroups, 0.0f),
      source_(negroups), fixed_source_(negroups, 0.0f)
  {}

  FlatSourceRegion(const FlatSourceRegion& other)
    : material_(other.material_), position_recorded_(other.position_recorded_),
      position_(other.position_), volume_i_(other.volume_i_),
      volume_(other.volume_), volume_t_(other.volume_t_),
      was_hit_(other.was_hit_), mesh_(other.mesh_),
      scalar_flux_new_(other.scalar_flux_new_),
      scalar_flux_old_(other.scalar_flux_old_),
      scalar_flux_final_(other.scalar_flux_final_), source_(other.source_),
      fixed_source_(other.fixed_source_), tally_task_(other.tally_task_),
      is_in_manifest_(other.is_in_manifest_), is_merged_(other.is_merged_),
      is_consumer_(other.is_consumer_), manifest_index_(other.manifest_index_),
      no_hit_streak_(other.no_hit_streak_),
      source_region_(other.source_region_), bin_(other.bin_)
  {}
  ~FlatSourceRegion() = default;

  void merge(FlatSourceRegion& other);

  OpenMPMutex lock_;
  int material_;
  int position_recorded_ {0};
  Position position_;
  double volume_i_ {0.0};
  double volume_ {0.0};
  double volume_t_ {0.0};
  int was_hit_ {0};
  int mesh_ {C_NONE};
  bool is_in_manifest_ {false};
  bool is_merged_ {false};
  bool is_consumer_ {false};
  bool is_merge_failed_ {false};
  int no_hit_streak_ {0};
  int64_t manifest_index_;
  int64_t source_region_;
  int64_t bin_;

  // 2D arrays with entry for each energy group
  vector<float> scalar_flux_new_;
  vector<float> scalar_flux_old_;
  vector<float> scalar_flux_final_;
  vector<float> source_;
  vector<float> fixed_source_;

  // Outer dimension is each energy group in the FSR, inner dimension is each
  // tally operation that bin will perform
  vector<vector<TallyTask>> tally_task_;
  std::unordered_set<TallyTask, TallyTask::HashFunctor> volume_task_;

}; // class FlatSourceRegion

/*
 * The FlatSourceDomain class encompasses data and methods for storing
 * scalar flux and source region for all flat source regions in a
 * random ray simulation domain.
 */

class RegularMeshKey {
public:
  int mesh_id;
  StructuredMesh::MeshIndex ijk;
  // Add a constructor that takes an int and a StructuredMesh::MeshIndex
  RegularMeshKey(int mesh_id, StructuredMesh::MeshIndex ijk)
    : mesh_id(mesh_id), ijk(ijk)
  {}
  RegularMeshKey() = default;

  bool operator==(const RegularMeshKey& other) const
  {
    return mesh_id == other.mesh_id && ijk[0] == other.ijk[0] &&
           ijk[1] == other.ijk[1] && ijk[2] == other.ijk[2];
  }

  struct HashFunctor {
    size_t operator()(const RegularMeshKey& key) const
    {
      size_t seed = 0;
      hash_combine(seed, key.mesh_id);
      hash_combine(seed, key.ijk[0]);
      hash_combine(seed, key.ijk[1]);
      hash_combine(seed, key.ijk[2]);
      return seed;
    }
  };
};

class FSRKey {
public:
  int64_t mfci;
  int64_t mesh_bin;
  FSRKey() = default;
  FSRKey(int64_t source_region, int64_t bin)
    : mfci(source_region), mesh_bin(bin)
  {}

  // Equality operator required by the unordered_map
  bool operator==(const FSRKey& other) const
  {
    return mfci == other.mfci && mesh_bin == other.mesh_bin;
  }

  // Hashing functor required by the unordered_map
  struct HashFunctor {
    size_t operator()(const FSRKey& key) const
    {
      size_t seed = 0;
      hash_combine(seed, key.mfci);
      hash_combine(seed, key.mesh_bin);
      return seed;
    }
  };
};

class FlatSourceDomain {
public:
  // Doing source/mesh prep up front:
  // maximize both init performance as well
  // as avoiding use of the map long term, but with the down

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
    double strength_factor, int target_material_id,
    const vector<int32_t>& instances);
  void apply_fixed_source_to_cell_and_children(int32_t i_cell,
    Discrete* discrete, double strength_factor, int32_t target_material_id);
  void convert_fixed_sources();
  void count_fixed_source_regions();
  double calculate_total_volume_weighted_source_strength();
  void swap_flux(void);

  void apply_mesh_to_cell_instances(int32_t i_cell, int32_t mesh,
    int target_material_id, const vector<int32_t>& instances);
  void apply_mesh_to_cell_and_children(
    int32_t i_cell, int32_t mesh, int32_t target_material_id);
  void apply_meshes();
  FlatSourceRegion* get_fsr(
    int64_t source_region, int bin, Position r0, Position r1, int ray_id);
  void update_fsr_manifest(void);
  void mesh_hash_grid_add(int mesh_index, int bin, FSRKey key);
  vector<FSRKey> mesh_hash_grid_get_neighbors(int mesh_index, int bin);
  int64_t get_largest_neighbor(FlatSourceRegion& fsr);
  bool merge_fsr(FlatSourceRegion& fsr);
  int64_t check_for_small_FSRs(void);
  void initialize_tally_tasks(FlatSourceRegion& fsr);
  void reset_tally_volumes();

  //----------------------------------------------------------------------------
  // Data members

  int negroups_;                  // Number of energy groups in simulation
  int64_t n_source_elements_ {0}; // Total number of source regions in the
                                  // model times the number of energy groups
  int64_t n_source_regions_ {0};  // Total number of source regions in the model
  int64_t n_fixed_source_regions_ {0}; // Total number of source regions with
                                       // non-zero fixed source terms

  double
    simulation_volume_; // Total physical volume of the simulation domain, as
                        // defined by the 3D box of the random ray source
  int64_t n_rays_sampled_ {0}; // Total number of rays sampled in the simulation
  int64_t n_rays_rejected_ {
    0}; // Total number of rays rejected in the simulation

  bool mapped_all_tallies_ {false}; // If all source regions have been visited

  // 1D array representing source region starting offset for each OpenMC Cell
  // in model::cells
  std::vector<int64_t> source_region_offsets_;

  static std::unordered_map<int32_t, int32_t> mesh_map_;
  static vector<unique_ptr<Mesh>> meshes_;

  int64_t n_subdivided_source_regions_ {0};
  int64_t discovered_source_regions_ {0};
  // vector<vector<int>> hitmap;

  // It would be nice to get around this one, as its sort of useless
  // but we need it in order to accelerate the lookup when there is no
  // mesh overlay. I.e., to skip all the unordered_map stuff.
  // Oh, actually, wait, this is bad. We can't really get by
  // with a base class here, as these get used the first iteration.
  vector<FlatSourceRegion> material_filled_cell_instance_;

  // Hypothetically, I could get rid of the above if just using teh same
  // mapping logic with 0 for the bin. Downside is slowness when there
  // is no mesh present.

  vector<FlatSourceRegion> known_fsr_;
  std::unordered_map<FSRKey, int64_t, FSRKey::HashFunctor> known_fsr_map_;

  ParallelMap<FSRKey, FlatSourceRegion, FSRKey::HashFunctor>
    discovered_fsr_parallel_map_;

  // This is a 5D vector, with dimensions:
  // 1. mesh index
  // 2. z index
  // 3. y index
  // 4. x index
  // 5. FSR hash
  // The first dimension is left as a vector for easy, access, the next
  // three dimensions are serialized, the 5th (hash) dimension
  // is left as a vector as it is dynamically updated.
  // vector<vector<vector<unt64_t>>> mesh_hash_grid_;
  std::unordered_map<RegularMeshKey, std::vector<FSRKey>,
    RegularMeshKey::HashFunctor>
    mesh_hash_grid_;

  //! Results for each bin -- the first dimension of the array is for the
  //! combination of filters (e.g. specific cell, specific energy group, etc.)
  //! and the second dimension of the array is for scores (e.g. flux, total
  //! reaction rate, fission reaction rate, etc.)
  vector<xt::xtensor<double, 3>> tally_volumes_;
  vector<xt::xtensor<double, 3>> tally_;

  // This is the number of hits that the FSR needs
  // to get in order to be counted as a "low hitter".
  // I.e., a value of 1 means that only getting hit 0 or 1
  // times in an iteration will mean that that iteration
  // will count towards its miss streak.
  const int merging_threshold_ = 1;

  // This is the number of iterations in a row that have
  // been at or under the threshold value. Once the streak is
  // met, the FSR will be merged. E.g.,
  const int streak_needed_to_merge_ = 3;

  // Only merge FSRs if they are this factor smaller
  // than the consuming FSR. E.g., a value of 10x means
  // that FSRs will not be merged unless the larger FSR
  // is at least 10x bigger than the small one.
  const double volume_merging_threshold_ = 10.0;

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
