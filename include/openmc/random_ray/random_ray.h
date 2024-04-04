#ifndef OPENMC_RANDOM_RAY_H
#define OPENMC_RANDOM_RAY_H

#include "openmc/particle.h"
#include "openmc/random_ray/flat_source_domain.h"
#include "openmc/source.h"

namespace openmc {

/*
 * The RandomRay class encompasses data and methods for transporting random rays
 * through the model. It is a small extension of the Particle class.
 */

// TODO: Inherit from GeometryState instead of Particle
class RandomRay : public Particle {
public:
  struct Intersection {
    double s;
    FlatSourceRegion* fsr;
    int material;
    bool is_active;
    bool vacuum_apply_at_end;
    double correction;
  };
  //----------------------------------------------------------------------------
  // Constructors
  RandomRay();
  RandomRay(uint64_t ray_id, FlatSourceDomain* domain);

  //----------------------------------------------------------------------------
  // Methods
  bool event_advance_ray();
  void attenuate_flux(double distance, double offset, bool is_active);
  void attenuate_flux_inner(
    double distance, bool is_active, FlatSourceRegion& fsr);
  void attenuate_flux_inner_non_void(
    double distance, bool is_active, FlatSourceRegion& fsr, int material, double correction=1.0);
  void attenuate_flux_inner_void(
    double distance, bool is_active, FlatSourceRegion& fsr, int material, double correction=1.0);
  void initialize_ray(uint64_t ray_id, FlatSourceDomain* domain);
  uint64_t transport_history_based_single_ray();

  //----------------------------------------------------------------------------
  // Static data members
  static double distance_inactive_;      // Inactive (dead zone) ray length
  static double distance_active_;        // Active ray length
  static unique_ptr<Source> ray_source_; // Starting source for ray sampling
  static bool ray_trace_mode_;           // Flag for ray tracing mode

  //----------------------------------------------------------------------------
  // Data members
  FlatSourceDomain* domain_ {nullptr}; // pointer to domain that has flat source
                                       // data needed for ray transport
  std::vector<float> angular_flux_;
  std::vector<float> delta_psi_;
  double distance_travelled_ {0};
  int negroups_;
  bool is_active_ {false};
  bool is_alive_ {true};
  vector<LocalCoord> coord_last_; //!< coordinates for all levels
  vector<Intersection> intersections_;
}; // class RandomRay

} // namespace openmc

#endif // OPENMC_RANDOM_RAY_H
