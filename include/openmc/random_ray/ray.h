#ifndef OPENMC_RANDOM_RAY_RAY_H
#define OPENMC_RANDOM_RAY_RAY_H

#include "openmc/particle.h"

namespace openmc {

#define NEGROUPS 7

struct Segment {
  double length_physical;
  //double length_corrected;
  int64_t cell;
  int material;
  //int is_end_vac;
  bool is_active;
  bool is_vac {0};
  /*
  float flux_in[NEGROUPS];
  float flux_out_true[NEGROUPS];
  float flux_out_adjusted[NEGROUPS];
  */
  Segment() = default;
  Segment(double length_phys, int64_t cell_val, int mat_val, bool is_act)
    : length_physical(length_phys),
    cell(cell_val),
    material(mat_val),
    is_active(is_act)
  {
  }
};

namespace random_ray {
extern std::vector<std::vector<Segment>> segments;
} // namespace random_ray

/*
 * The Ray class encompasses data and methods for transporting random rays
 * through the model. It is a small extension of the Particle class.
 */

class Ray : public Particle {
public:
  //==========================================================================
  // Constructors
  Ray();
  Ray(uint64_t index_source, uint64_t nrays, int iter);

  //==========================================================================
  // Methods
  void event_advance_ray(double distance_inactive, double distance_active);
  void attenuate_flux(double distance, bool is_active);
  void initialize_ray(uint64_t index_source, uint64_t nrays, int iter);
  uint64_t transport_history_based_single_ray(double distance_inactive, double distance_active);

  //==========================================================================
  // Data

  std::vector<float> angular_flux_;
  std::vector<float> delta_psi_;
  double distance_travelled_ {0};
  bool is_active_ {false};
  bool is_alive_ {true};
  bool is_rt_only_ {true};
  int curr_segment_ {0};
}; // class Ray

//==============================================================================
// Non-member functions
//==============================================================================
inline float cjosey_exponential(const float tau);
void attenuate_segment(Segment& s, std::vector<float>& angular_flux, std::vector<float>& delta_psi);

} // namespace openmc

#endif // OPENMC_RANDOM_RAY_RAY_H
