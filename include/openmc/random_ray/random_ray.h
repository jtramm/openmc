#ifndef OPENMC_RANDOM_RAY_H
#define OPENMC_RANDOM_RAY_H

#include "openmc/memory.h"
#include "openmc/particle.h"
#include "openmc/random_ray/flat_source_domain.h"
#include "openmc/random_ray/moment_matrix.h"
#include "openmc/source.h"
#include "openmc/vector.h"

namespace openmc {

class Segment{
public:
    int64_t sr;
    double distance;
    Position r;
    Direction u;
    int material;
    int is_active;
    int is_vac_end;
    int is_alive;
};

/*
 * The RandomRay class encompasses data and methods for transporting random rays
 * through the model. It is a small extension of the Particle class.
 */

// TODO: Inherit from GeometryState instead of Particle
class RandomRay : public Particle {
public:
  //----------------------------------------------------------------------------
  // Constructors
  RandomRay();
// RandomRay(uint64_t ray_id, FlatSourceDomain* domain);

//----------------------------------------------------------------------------
// Methods
  #pragma omp declare target
  void initialize_ray(
    uint64_t ray_id, Particle::Bank& site, uint64_t work_index);
  void event_advance_ray();
  void attenuate_flux(double distance, bool is_active);
  void attenuate_flux_flat_source(double distance, bool is_active);
  void attenuate_flux_linear_source(double distance, bool is_active);
  uint64_t transport_history_based_single_ray();
  #pragma omp end declare target

  void copy_ray_to_device();
  void update_from_device();
  void update_to_device();

  //----------------------------------------------------------------------------
  // Static data members
  #pragma omp declare target
  static double distance_inactive_;          // Inactive (dead zone) ray length
  static double distance_active_;            // Active ray length
  static RandomRaySourceShape source_shape_; // Flag for linear source
  #pragma omp end declare target

  static unique_ptr<IndependentSource>
    ray_source_; // Starting source for ray sampling

  //----------------------------------------------------------------------------
  // Public data members
  vector<float> angular_flux_;
  int max_segments_ = 10000;
  vector<Segment> segments_;

private:
  //----------------------------------------------------------------------------
  // Private data members
  vector<float> delta_psi_;
  vector<MomentArray> delta_moments_;

  int negroups_;

  double distance_travelled_ {0};
  bool is_active_ {false};
  bool is_alive_ {true};
}; // class RandomRay

} // namespace openmc

#endif // OPENMC_RANDOM_RAY_H
