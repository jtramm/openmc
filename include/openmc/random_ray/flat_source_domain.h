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
      no_hit_streak_(other.no_hit_streak_), source_region_(other.source_region_),
      bin_(other.bin_)
  {}

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

}; // class FlatSourceRegion

/*
 * The FlatSourceDomain class encompasses data and methods for storing
 * scalar flux and source region for all flat source regions in a
 * random ray simulation domain.
 */

struct MeshHashIndex {
  int mesh_id;
  StructuredMesh::MeshIndex ijk;
  // Add a constructor that takes an int and a StructuredMesh::MeshIndex
  MeshHashIndex(int mesh_id, StructuredMesh::MeshIndex ijk)
    : mesh_id(mesh_id), ijk(ijk)
  {}
  MeshHashIndex() = default;
};

// Custom hash function
struct MeshHashIndexHash {
  std::size_t operator()(const MeshHashIndex& mhi) const
  {
    std::size_t hash = std::hash<int>()(mhi.mesh_id);
    hash_combine(hash, std::hash<int>()(mhi.ijk[0]));
    hash_combine(hash, std::hash<int>()(mhi.ijk[1]));
    hash_combine(hash, std::hash<int>()(mhi.ijk[2]));
    return hash;
  }

private:
  // Combine the hash values of multiple variables
  template<typename T>
  void hash_combine(std::size_t& seed, const T& value) const
  {
    seed ^= std::hash<T>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
};

// Custom equality function
struct MeshHashIndexEqual {
  bool operator()(const MeshHashIndex& mhi1, const MeshHashIndex& mhi2) const
  {
    return mhi1.mesh_id == mhi2.mesh_id && mhi1.ijk[0] == mhi2.ijk[0] &&
           mhi1.ijk[1] == mhi2.ijk[1] && mhi1.ijk[2] == mhi2.ijk[2];
  }
};

class FlatSourceDomain {
public:
  // Doing source/mesh prep up front:
  // maximize both init performance as well
  // as avoiding use of the map long term, but with the down
  struct FSRKey {
    int64_t mfci;
    int64_t mesh_bin;
  };

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

  //----------------------------------------------------------------------------
  // Data members

  int negroups_;                  // Number of energy groups in simulation
  int64_t n_source_elements_ {0}; // Total number of source regions in the model
                                  // times the number of energy groups
  int64_t n_source_regions_ {0};  // Total number of source regions in the model
  int64_t n_fixed_source_regions_ {0}; // Total number of source regions with
                                       // non-zero fixed source terms

  bool mapped_all_tallies_ {false}; // If all source regions have been visited

  // 1D array representing source region starting offset for each OpenMC Cell
  // in model::cells
  std::vector<int64_t> source_region_offsets_;

  static std::unordered_map<int32_t, int32_t> mesh_map_;
  static vector<unique_ptr<Mesh>> meshes_;

  int64_t n_subdivided_source_regions_ {0};
  int64_t discovered_source_regions_ {0};
  vector<vector<int>> hitmap;

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
  std::unordered_map<FSRKey, int64_t> known_fsr_map_;

  struct HashFunctor {
    std::size_t operator()(const FSRKey& key) const
    {
      return 1;
      std::size_t h1 = std::hash<int64_t>{}(key.mfci);
      std::size_t h2 = std::hash<int64_t>{}(key.mesh_bin);
      return h1 ^ (h2 << 1);
    }
  };

  struct EqualFunctor {
    bool operator()(const FSRKey& lhs,
      const FSRKey& rhs) const
    {
      return lhs.mfci == rhs.mfci && lhs.mesh_bin == rhs.mesh_bin;
    }
  };

  ParallelMap<FSRKey, FlatSourceRegion, HashFunctor,
    EqualFunctor>
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
  std::unordered_map<MeshHashIndex, std::vector<FSRKey>, MeshHashIndexHash,
    MeshHashIndexEqual>
    mesh_hash_grid_;

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
