#include "openmc/random_ray/random_ray.h"

#include "openmc/geometry.h"
#include "openmc/mesh.h"
#include "openmc/message_passing.h"
#include "openmc/mgxs_interface.h"
#include "openmc/random_ray/flat_source_domain.h"
#include "openmc/search.h"
#include "openmc/settings.h"
#include "openmc/simulation.h"
#include "openmc/source.h"

namespace openmc {

//==============================================================================
// Non-method functions
//==============================================================================

// returns 1 - exp(-tau)
// Equivalent to -(_expm1f(-tau)), but faster
// Written by Colin Josey.
inline float cjosey_exponential(const float tau)
{
  const float c1n = -1.0000013559236386308f;
  const float c2n = 0.23151368626911062025f;
  const float c3n = -0.061481916409314966140f;
  const float c4n = 0.0098619906458127653020f;
  const float c5n = -0.0012629460503540849940f;
  const float c6n = 0.00010360973791574984608f;
  const float c7n = -0.000013276571933735820960f;

  const float c0d = 1.0f;
  const float c1d = -0.73151337729389001396f;
  const float c2d = 0.26058381273536471371f;
  const float c3d = -0.059892419041316836940f;
  const float c4d = 0.0099070188241094279067f;
  const float c5d = -0.0012623388962473160860f;
  const float c6d = 0.00010361277635498731388f;
  const float c7d = -0.000013276569500666698498f;

  float x = -tau;
  float num, den;

  den = c7d;
  den = den * x + c6d;
  den = den * x + c5d;
  den = den * x + c4d;
  den = den * x + c3d;
  den = den * x + c2d;
  den = den * x + c1d;
  den = den * x + c0d;

  num = c7n;
  num = num * x + c6n;
  num = num * x + c5n;
  num = num * x + c4n;
  num = num * x + c3n;
  num = num * x + c2n;
  num = num * x + c1n;
  num = num * x;

  const float exponential = num / den;
  return exponential;
}

//==============================================================================
// RandomRay implementation
//==============================================================================

// Static Variable Declarations
double RandomRay::distance_inactive_;
double RandomRay::distance_active_;
unique_ptr<Source> RandomRay::ray_source_;

RandomRay::RandomRay()
  : negroups_(data::mg.num_energy_groups_),
    angular_flux_(data::mg.num_energy_groups_),
    delta_psi_(data::mg.num_energy_groups_),
    coord_last_(model::n_coord_levels)
{}

RandomRay::RandomRay(uint64_t ray_id, FlatSourceDomain* domain) : RandomRay()
{
  initialize_ray(ray_id, domain);
}

// Transports ray until termination criteria are met
uint64_t RandomRay::transport_history_based_single_ray()
{
  while (alive()) {
    // Advance ray. If the ray exited the dead zone, we need to process
    // the active length as well so the function is called again.
    if (event_advance_ray()) {
      event_advance_ray();
    }
    if (!alive())
      break;
    event_cross_surface();
  }

  return n_event();
}

// Transports ray across a single source region
bool RandomRay::event_advance_ray()
{
  // Find the distance to the nearest boundary
  boundary() = distance_to_boundary(*this);
  double distance = boundary().distance;

  if (distance <= 0.0) {
    mark_as_lost("Negative transport distance detected for particle " +
                 std::to_string(id()));
    return false;
  }

  // Flag for end of inactive region (dead zone)
  bool dead_zone_exit = false;

  if (is_active_) {
    // If ray is in active region, then enforce limit of active ray length
    if (distance_travelled_ + distance >= distance_active_) {
      distance = distance_active_ - distance_travelled_;
      wgt() = 0.0;
    }
    distance_travelled_ += distance;
    attenuate_flux(distance, true);
  } else {
    // If ray is in dead zone, then enforce limit of dead zone length
    if (distance_travelled_ + distance >= distance_inactive_) {
      is_active_ = true;
      distance = distance_inactive_ - distance_travelled_;
      distance_travelled_ = 0.0;
      dead_zone_exit = true;
    } else {
      distance_travelled_ += distance;
    }
    attenuate_flux(distance, false);
  }

  // Advance particle
  for (int j = 0; j < n_coord(); ++j) {
    coord(j).r += distance * coord(j).u;
  }

  return dead_zone_exit;
}

void RandomRay::attenuate_flux(double distance, bool is_active)
{
  // Determine source region index etc.
  int i_cell = lowest_coord().cell;

  // The source region is the spatial region index
  int64_t source_region =
    domain_->source_region_offsets_[i_cell] + cell_instance();
  auto& fsr = domain_->fsr_[source_region];

  int i_mesh = fsr.mesh_;

  if (i_mesh >= 0) {
    Mesh* mesh = domain_->meshes_[i_mesh].get();
    RegularMesh* rmesh = dynamic_cast<RegularMesh*>(mesh);
    if (rmesh == nullptr)
      fatal_error("Only regular meshes are supported for random ray tracing.");
    
    vector<int> bins;
    vector<double> lengths;

    rmesh->bins_crossed(r(), r() + distance * u(), u(), bins, lengths);

    for( int i = 0; i < bins.size(); i++ ) {
      int bin = bins[i];
      double length = lengths[i] * distance;
      //Position intersect_point = r() + u() * length;

      //if (length > (distance/bins.size()) * 1.0e-3)
      if (length > 1.0e-5)
      {
        FlatSourceRegion* region = domain_->get_fsr(source_region, bin);
        attenuate_flux_inner(length, is_active, *region);
      }
    }
  } else { // If the FSR doesn't have a mesh, let's just say the bin is zero
    FlatSourceRegion* region = domain_->get_fsr(source_region, 0);
    attenuate_flux_inner(distance, is_active, *region);  
  }
  //attenuate_flux_inner(distance, is_active, domain_->fsr_[source_region]);  
}

// This function forms the inner loop of the random ray transport process.
// It is responsible for several tasks. Based on the incoming angular flux
// of the ray and the source term in the region, the outgoing angular flux
// is computed. The delta psi between the incoming and outgoing fluxes is
// contributed to the estimate of the total scalar flux in the source region.
// Additionally, the contribution of the ray path to the stochastically
// estimated volume is also kept track of. All tasks involving writing
// to the data for the source region are done with a lock over the entire
// source region.  Locks are used instead of atomics as all energy groups
// must be written, such that locking once is typically much more efficient
// than use of many atomic operations corresponding to each energy group
// individually (at least on CPU). Several other bookeeping tasks are also
// performed when inside the lock.
void RandomRay::attenuate_flux_inner(double distance, bool is_active, FlatSourceRegion& fsr)
{
  // The number of geometric intersections is counted for reporting purposes
  n_event()++;

  // The source element is the energy-specific region index
  int material = this->material();

  // Temperature and angle indices, if using multiple temperature
  // data sets and/or anisotropic data sets.
  // TODO: Currently assumes we are only using single temp/single
  // angle data.
  const int t = 0;
  const int a = 0;

  // MOC incoming flux attenuation + source contribution/attenuation equation
  for (int e = 0; e < negroups_; e++) {
    float sigma_t = data::mg.macro_xs_[material].get_xs(
      MgxsType::TOTAL, e, NULL, NULL, NULL, t, a);
    float tau = sigma_t * distance;
    float exponential = cjosey_exponential(tau); // exponential = 1 - exp(-tau)
    float new_delta_psi =
      (angular_flux_[e] - fsr.source_[e]) * exponential;
    delta_psi_[e] = new_delta_psi;
    angular_flux_[e] -= new_delta_psi;
  }

  // If ray is in the active phase (not in dead zone), make contributions to
  // source region bookkeeping
  if (is_active) {

    // Aquire lock for source region
    //omp_set_lock(&fsr.lock_);
    fsr.lock_.lock();

    // Accumulate delta psi into new estimate of source region flux for
    // this iteration
    for (int e = 0; e < negroups_; e++) {
      fsr.scalar_flux_new_[e] += delta_psi_[e];
    }

    // If the source region hasn't been hit yet this iteration,
    // indicate that it now has
    fsr.was_hit_++;

    // Accomulate volume (ray distance) into this iteration's estimate
    // of the source region's volume
    fsr.volume_ += distance;

    // Tally valid position inside the source region (e.g., midpoint of
    // the ray) if not done already
    if (!fsr.position_recorded_) {
      Position midpoint = r() + u() * (distance / 2.0);
      fsr.position_ = midpoint;
      fsr.position_recorded_ = 1;
    }

    // Release lock
   // omp_unset_lock(&fsr.lock_);
   fsr.lock_.unlock();
  }
}

void RandomRay::initialize_ray(uint64_t ray_id, FlatSourceDomain* domain)
{
  domain_ = domain;

  // Reset particle event counter
  n_event() = 0;

  if (distance_inactive_ <= 0.0)
    is_active_ = true;
  else
    is_active_ = false;

  wgt() = 1.0;

  // set identifier for particle
  id() = simulation::work_index[mpi::rank] + ray_id;

  // set random number seed
  int64_t particle_seed =
    (simulation::current_batch - 1) * settings::n_particles + id();
  init_particle_seeds(particle_seed, seeds());
  stream() = STREAM_TRACKING;

  // Sample from ray source distribution
  SourceSite site {ray_source_->sample(current_seed())};
  site.E = lower_bound_index(
    data::mg.rev_energy_bins_.begin(), data::mg.rev_energy_bins_.end(), site.E);
  site.E = negroups_ - site.E - 1.;
  from_source(&site);

  // Locate ray
  if (lowest_coord().cell == C_NONE) {
    if (!exhaustive_find_cell(*this)) {
      this->mark_as_lost(
        "Could not find the cell containing particle " + std::to_string(id()));
    }

    // Set birth cell attribute
    if (cell_born() == C_NONE)
      cell_born() = lowest_coord().cell;
  }

  // Initialize ray's starting angular flux to starting location's isotropic
  // source
  int i_cell = lowest_coord().cell;
  int64_t source_region_idx =
    domain_->source_region_offsets_[i_cell] + cell_instance();

  auto& fsr = domain->fsr_[source_region_idx];
  int i_mesh = fsr.mesh_;
  
  // If there is a mesh present, then we need to get the bin number corresponding
  // to the ray spatial location
  FlatSourceRegion* region;
  if (i_mesh >= 0) {
    Mesh* mesh = domain_->meshes_[i_mesh].get();
    RegularMesh* rmesh = dynamic_cast<RegularMesh*>(mesh);
    if (rmesh == nullptr)
      fatal_error("Only regular meshes are supported for random ray tracing.");
    int bin = rmesh->get_bin(r());
    region = domain_->get_fsr(source_region_idx, bin);
  } else {
    region = domain_->get_fsr(source_region_idx, 0);
  }

  for (int e = 0; e < negroups_; e++) {
    angular_flux_[e] = region->source_[e];
  }
}

} // namespace openmc
