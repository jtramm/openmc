#include "openmc/simulation.h"

#include "openmc/bank.h"
#include "openmc/cell.h"
#include "openmc/capi.h"
#include "openmc/container_util.h"
#include "openmc/eigenvalue.h"
#include "openmc/error.h"
#include "openmc/event.h"
#include "openmc/geometry.h"
#include "openmc/geometry_aux.h"
#include "openmc/material.h"
#include "openmc/message_passing.h"
#include "openmc/mgxs_interface.h"
#include "openmc/nuclide.h"
#include "openmc/output.h"
#include "openmc/particle.h"
#include "openmc/photon.h"
#include "openmc/random_lcg.h"
#include "openmc/settings.h"
#include "openmc/source.h"
#include "openmc/state_point.h"
#include "openmc/timer.h"
#include "openmc/tallies/derivative.h"
#include "openmc/tallies/filter.h"
#include "openmc/tallies/filter_mesh.h"
#include "openmc/tallies/tally.h"
#include "openmc/tallies/trigger.h"
#include "openmc/track_output.h"

#ifdef _OPENMP
#include <omp.h>
#endif
#include "xtensor/xview.hpp"

#ifdef OPENMC_MPI
#include <mpi.h>
#endif

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <string>


//==============================================================================
// C API functions
//==============================================================================

void initialize_cell_data()
{
  using namespace openmc;
  for( int i = 0; i < model::cells.size(); i++ )
  {
    Cell & c = *model::cells[i];
    if(c.type_ != Fill::MATERIAL)
      continue;
    int negroups = data::mg.num_energy_groups_;
    int nelements = c.n_instances_ * negroups;

    c.scalar_flux_new.resize(nelements);

    c.scalar_flux_old.resize(nelements);
    std::fill(c.scalar_flux_old.begin(), c.scalar_flux_old.end(), 1.0);

    c.source.resize(nelements);
    
    c.volume.resize(c.n_instances_);
    //std::fill(c.volume.begin(), c.volume.end(), 1.0);

    c.volume_t.resize(c.n_instances_);
    std::fill(c.volume_t.begin(), c.volume_t.end(), 0.0);

    c.was_hit.resize(c.n_instances_);
    std::fill(c.was_hit.begin(), c.was_hit.end(), 0);
    
    c.positions.resize(c.n_instances_);
    c.position_recorded.resize(c.n_instances_);
    std::fill(c.position_recorded.begin(), c.position_recorded.end(), 0);
  }
}

void set_scalar_flux_to_zero()
{
  using namespace openmc;
  for( int i = 0; i < model::cells.size(); i++ )
  {
    Cell & c = *model::cells[i];
    if(c.type_ != Fill::MATERIAL)
      continue;
    std::fill(c.scalar_flux_new.begin(), c.scalar_flux_new.end(), 0.0);
    std::fill(c.volume.begin(), c.volume.end(), 0.0);
  }
}
  
std::vector<float> Sigma_t_flat;
std::vector<float> Sigma_s_flat;
std::vector<float> nu_Sigma_f_flat;
std::vector<float> Chi_flat;

void prep_xs()
{
  using namespace openmc;
  int negroups = data::mg.num_energy_groups_;
  int zero = 0;

  for (auto& m : data::mg.macro_xs_) {
    for (int e = 0; e < negroups; e++) {
      float Sigma_t = m.get_xs(MgxsType::TOTAL,       e, NULL, NULL, NULL);
      Sigma_t_flat.push_back(Sigma_t);

      float nu_Sigma_f = m.get_xs(MgxsType::NU_FISSION, e, NULL,             NULL, NULL);
      nu_Sigma_f_flat.push_back(nu_Sigma_f);

      float Chi =     m.get_xs(MgxsType::CHI_PROMPT, e, &zero, NULL, NULL);
      Chi_flat.push_back(Chi);

      for (int ee = 0; ee < negroups; ee++) {
        float Sigma_s    = m.get_xs(MgxsType::NU_SCATTER, ee, &e, NULL, NULL);
        Sigma_s_flat.push_back(Sigma_s);
      }

    }
  }
}

void update_neutron_source(double k_eff)
{
  using namespace openmc;
  double inverse_k_eff = 1.0 / k_eff;
  int negroups = data::mg.num_energy_groups_;
  int ncells = 0;

  #pragma omp parallel
  {
    for( int i = 0; i < model::cells.size(); i++ )
    {
      Cell & cell = *model::cells[i];
      if(cell.type_ != Fill::MATERIAL)
        continue;
      int material = cell.material_[0];
      #pragma omp for schedule(static) nowait
      for( int c = 0; c < cell.n_instances_; c++ )
      {
        for( int energy_group_out = 0; energy_group_out < negroups; energy_group_out++ )
        {
          int out_idx = material * negroups + energy_group_out; 

          float Sigma_t = Sigma_t_flat[out_idx];

          float scatter_source = 0.0;
          float fission_source = 0.0;

          for( int energy_group_in = 0; energy_group_in < negroups; energy_group_in++ )
          {
            int in_idx = material * negroups * negroups + energy_group_out * negroups + energy_group_in; 
            int idx = material * negroups + energy_group_in; 

            float scalar_flux = cell.scalar_flux_old[c * negroups + energy_group_in];

            float Sigma_s = Sigma_s_flat[in_idx];
            float nu_Sigma_f = nu_Sigma_f_flat[idx];
            float Chi = Chi_flat[idx];

            scatter_source += Sigma_s    * scalar_flux;
            fission_source += nu_Sigma_f * scalar_flux * Chi;
          }

          fission_source *= inverse_k_eff;
          float new_isotropic_source = (scatter_source + fission_source)  / Sigma_t;
          cell.source[c * negroups + energy_group_out] = new_isotropic_source;
        }
      }
    }
  }
}

void old_update_neutron_source(double k_eff)
{
  using namespace openmc;
  double inverse_k_eff = 1.0 / k_eff;
  int negroups = data::mg.num_energy_groups_;
  int ncells = 0;

  #pragma omp parallel
  {
    for( int i = 0; i < model::cells.size(); i++ )
    {
      Cell & cell = *model::cells[i];
      if(cell.type_ != Fill::MATERIAL)
        continue;
      int material = cell.material_[0];
      #pragma omp for schedule(static) nowait
      for( int c = 0; c < cell.n_instances_; c++ )
      {
        for( int energy_group_out = 0; energy_group_out < negroups; energy_group_out++ )
        {
          float Sigma_t = data::mg.macro_xs_[material].get_xs(MgxsType::TOTAL,       energy_group_out, NULL, NULL, NULL);

          float scatter_source = 0.0;
          float fission_source = 0.0;

          for( int energy_group_in = 0; energy_group_in < negroups; energy_group_in++ )
          {
            float scalar_flux = cell.scalar_flux_old[c * negroups + energy_group_in];

            float Sigma_s    = data::mg.macro_xs_[material].get_xs(MgxsType::NU_SCATTER, energy_group_in, &energy_group_out, NULL, NULL);
            float nu_Sigma_f = data::mg.macro_xs_[material].get_xs(MgxsType::NU_FISSION, energy_group_in, NULL,             NULL, NULL);

            float Chi =     data::mg.macro_xs_[material].get_xs(MgxsType::CHI_PROMPT, energy_group_in, &energy_group_out, NULL, NULL);

            scatter_source += Sigma_s    * scalar_flux;
            fission_source += nu_Sigma_f * scalar_flux * Chi;
          }

          fission_source *= inverse_k_eff;
          float new_isotropic_source = (scatter_source + fission_source)  / Sigma_t;
          cell.source[c * negroups + energy_group_out] = new_isotropic_source;
        }
      }
    }
  }
}

uint64_t count_fsrs(void)
{
  using namespace openmc;
  uint64_t n_cells = 0;
  for( int i = 0; i < model::cells.size(); i++ )
    {
      Cell & cell = *model::cells[i];
      if(cell.type_ != Fill::MATERIAL)
        continue;

      n_cells += cell.n_instances_;
    }
  return n_cells;
}

void normalize_scalar_flux_and_volumes(double total_active_distance_per_iteration, int iter)
{
  using namespace openmc;
  int negroups = data::mg.num_energy_groups_;

  double normalization_factor =        1.0 /  total_active_distance_per_iteration;
  double volume_normalization_factor = 1.0 / (total_active_distance_per_iteration * iter);

  #pragma omp parallel
  {
    for( int i = 0; i < model::cells.size(); i++ )
    {
      Cell & cell = *model::cells[i];
      if(cell.type_ != Fill::MATERIAL)
        continue;
      #pragma omp for schedule(static) nowait
      for( int c = 0; c < cell.n_instances_; c++ )
      {
        for( int e = 0; e < negroups; e++ )
        {
          cell.scalar_flux_new[c * negroups + e] *= normalization_factor;
        }
        cell.volume_t[c] += cell.volume[c];
        cell.volume[c] = cell.volume_t[c] * volume_normalization_factor;
      }
    }
  }
}

void add_source_to_scalar_flux(void)
{
  using namespace openmc;
  int negroups = data::mg.num_energy_groups_;
  #pragma omp parallel
  {
    for( int i = 0; i < model::cells.size(); i++ )
    {
      Cell & cell = *model::cells[i];
      if(cell.type_ != Fill::MATERIAL)
        continue;
      int material = cell.material_[0];
      #pragma omp for schedule(static) nowait
      for( int c = 0; c < cell.n_instances_; c++ )
      {
        double volume = cell.volume[c];
        for( int e = 0; e < negroups; e++ )
        {
          float Sigma_t = data::mg.macro_xs_[material].get_xs(MgxsType::TOTAL, e, NULL, NULL, NULL);
          uint64_t idx = c * negroups + e;
          if (volume != 0)
            cell.scalar_flux_new[idx] /= (Sigma_t * volume);
          cell.scalar_flux_new[idx] += cell.source[idx];

          if( cell.was_hit[c] == 0 )
            cell.scalar_flux_new[idx] = cell.scalar_flux_old[idx];
        }
      }
    }
  }
}

double compute_k_eff(double k_eff_old)
{
  using namespace openmc;
  double fission_rate_old = 0;
  double fission_rate_new = 0;
  
  int negroups = data::mg.num_energy_groups_;
  #pragma omp parallel reduction(+: fission_rate_old, fission_rate_new)
  {
    for( int i = 0; i < model::cells.size(); i++ )
    {
      Cell & cell = *model::cells[i];
      if(cell.type_ != Fill::MATERIAL)
        continue;
      int material = cell.material_[0];
      #pragma omp for schedule(static) nowait
      for( int c = 0; c < cell.n_instances_; c++ )
      {
        double volume = cell.volume[c];
        if( volume == 0 )
          continue;
        double cell_fission_source_old = 0;
        double cell_fission_source_new = 0;
        for( int e = 0; e < negroups; e++ )
        {
          double nu_Sigma_f = data::mg.macro_xs_[material].get_xs(MgxsType::NU_FISSION, e, NULL, NULL, NULL);
          cell_fission_source_old += nu_Sigma_f * cell.scalar_flux_old[c * negroups + e];
          cell_fission_source_new += nu_Sigma_f * cell.scalar_flux_new[c * negroups + e];
        }
        fission_rate_old += cell_fission_source_old * volume;
        fission_rate_new += cell_fission_source_new * volume;
      }
    }
  }

  double k_eff_new = k_eff_old * (fission_rate_new / fission_rate_old);

  return k_eff_new;
}

void copy_scalar_fluxes(void)
{
  using namespace openmc;
  int negroups = data::mg.num_energy_groups_;
  #pragma omp parallel
  {
    for( int i = 0; i < model::cells.size(); i++ )
    {
      Cell & cell = *model::cells[i];
      if(cell.type_ != Fill::MATERIAL)
        continue;
      #pragma omp for schedule(static) nowait
      for( int c = 0; c < cell.n_instances_; c++ )
      {
        for( int e = 0; e < negroups; e++ )
        {
          uint64_t idx = c * negroups + e;
          cell.scalar_flux_old[idx] = cell.scalar_flux_new[idx];
        }
      }
    }
  }
}

void tally_fission_rates(void)
{
  using namespace openmc;

  int negroups = data::mg.num_energy_groups_;

  #pragma omp parallel
  {
    for( int i = 0; i < model::cells.size(); i++ )
    {
      Cell & cell = *model::cells[i];
      if(cell.type_ != Fill::MATERIAL)
        continue;
      int material = cell.material_[0];
      #pragma omp for schedule(static) nowait
      for( int c = 0; c < cell.n_instances_; c++ )
      {
        Position & p = cell.positions[c];
        for( int t = 0; t < model::tallies.size(); t++ )
        {
          uint64_t tally_filter_idx = model::tallies[t]->filters(0);

          const auto& filt_base = model::tally_filters[tally_filter_idx].get();
          auto* filt = dynamic_cast<MeshFilter*>(filt_base);
          int mesh_idx = filt->mesh();
          
          // If the distribcell is not inside this mesh, then don't do anything
          uint64_t tally_array_id = model::meshes[mesh_idx]->get_bin(p);
          if( tally_array_id == -1 )
            continue;
            
          double volume = cell.volume[c];

          for( int e = 0; e < negroups; e++ )
          {
            double Sigma_f = data::mg.macro_xs_[material].get_xs(MgxsType::FISSION, e, NULL, NULL, NULL);
            double phi = cell.scalar_flux_new[c * negroups + e];

            double score = Sigma_f * phi * volume;

            auto& tally {*model::tallies[0]};
        
            #pragma omp atomic
            tally.results_(tally_array_id, 0, TallyResult::VALUE) += score;
          }
        }
      }
    }
  }
  accumulate_tallies();
}

void initialize_ray(openmc::Particle & p, uint64_t index_source, uint64_t nrays, int iter)
{
  using namespace openmc;
  // set identifier for particle
  p.id_ = index_source;

  // Reset particle event counter
  p.n_event_ = 0;

  if( settings::ray_distance_inactive <= 0.0 )
    p.is_active_ = true;
  else
    p.is_active_ = false;

  // set random number seed
  int64_t particle_seed = (iter-1) * nrays + p.id_;
  init_particle_seeds(particle_seed, p.seeds_);
  p.stream_ = STREAM_TRACKING;
    
  // sample from external source distribution (should use box)
  auto site = sample_external_source(p.current_seed());
  p.from_source(&site);

  // Debugging
  /*
  Position & r = p.r();
  Direction & u = p.u();
  r.x = 0.350829492;
  r.y = -0.558578758;
  r.z = 0.692907163;
  u.x = -0.219246640;
  u.y = 0.875089877;
  u.z = 0.431449437;
  */
 //   Ray - Origin: [ 0.351, -0.559, 22.693] Direction: [-0.219,  0.875,  0.431]

  //Position r = p.r();
  //Direction u = p.u();
  //printf("Particle loc:[%.2f, %.2f, %.2f] dir:[%.2f, %.2f, %.2f]\n", r.x, r.y, r.z, u.x, u.y, u.z);
    
  // If the cell hasn't been determined based on the particle's location,
  // initiate a search for the current cell. This generally happens at the
  // beginning of the history and again for any secondary particles
  if (p.coord_[p.n_coord_ - 1].cell == C_NONE) {
    if (!exhaustive_find_cell(p)) {
      p.mark_as_lost("Could not find the cell containing particle "
        + std::to_string(p.id_));
      exit(1);
    }

    // Set birth cell attribute
    if (p.cell_born_ == C_NONE) p.cell_born_ = p.coord_[p.n_coord_ - 1].cell;
  }

  // Initialize ray's starting angular flux to starting location's isotropic source
  int coord_lvl = p.n_coord_ - 1;
  int i_cell = p.coord_[coord_lvl].cell;
  Cell& c {*model::cells[i_cell]};
  int negroups = data::mg.num_energy_groups_;
  int idx = p.cell_instance_ * negroups;
  
  for( int e = 0; e < negroups; e++ )
  {
    p.angular_flux_[e] = c.source[idx+e];
  }

}

uint64_t transport_history_based_single_ray(openmc::Particle& p, double distance_inactive, double distance_active)
{
  using namespace openmc;
  while (true) {
    p.event_advance_ray(distance_inactive, distance_active);
    if (!p.alive_)
      break;
    p.event_cross_surface();
  }

  return p.n_event_;
}

void print_inputs()
{
  using namespace openmc;
  header("RANDOM RAY INPUT SUMMARY", 3);
  printf("Rays per Iter             = %d\n", settings::n_particles);
  printf("Inactive Iters            = %d\n", settings::n_inactive);
  printf("Total Iters               = %d\n", settings::n_batches);
  printf("Active   Distance per Ray = %.3le [cm]\n", settings::ray_distance_active);
  printf("Inactive Distance per Ray = %.3le [cm]\n", settings::ray_distance_inactive);
  uint64_t n_fsrs = count_fsrs();
  printf("Number of FSRs (cells)    = %lu\n", n_fsrs);
  printf("Seed                      = %ld\n",openmc_get_seed()); 
}

double calculate_miss_rate(void)
{
  using namespace openmc;

  uint64_t n_cells = 0;
  uint64_t n_hits  = 0;

  // Note - not parallelized as this is over cells only, not cells x egroups
  for( int i = 0; i < model::cells.size(); i++ )
  {
    Cell & cell = *model::cells[i];
    if(cell.type_ != Fill::MATERIAL)
      continue;
    n_cells += cell.n_instances_;
    for( int c = 0; c < cell.n_instances_; c++ )
    {
      if( cell.was_hit[c] == 1 )
        n_hits++;
      cell.was_hit[c] = 0;
    }
  }

  uint64_t n_misses = n_cells - n_hits;
  double miss_rate = (double) n_misses / (double) n_cells;

  return miss_rate * 100.0;
}

// Random Ray Stuff
int openmc_run_random_ray()
{
  using namespace openmc;
  openmc::simulation::time_total.start();

  print_inputs();

  // Display header
  header("RANDOM RAY K EIGENVALUE SIMULATION", 3);
  print_columns();

  // Allocate tally results arrays if they're not allocated yet
  for (auto& t : model::tallies) {
    t->init_results();
  }
  
  // Reset global variables -- this is done before loading state point (as that
  // will potentially populate k_generation and entropy)
  simulation::current_batch = 0;
  simulation::current_gen = 1;
  simulation::n_realizations = 0;
  simulation::k_generation.clear();
  simulation::entropy.clear();
  openmc_reset();
  
  // Enable all tallies, and enforce 
  // Note: Currently, only tallies of mesh type that score fission are allowed
  for( int i = 0; i < model::tallies.size(); i++ )
  {
    auto& tally {*model::tallies[i]};
    assert(tally.scores_.size() == 1 && "Only a single fission score per mesh tally is supported in random ray mode.");
    assert(tally.scores_[0] == SCORE_FISSION && "Only fission scores are supported in random ray mode.");
    assert(tally.filters().size() == 1 && "Only a single mesh filter per tally is supported in random ray mode.");
    uint64_t tally_filter_idx = tally.filters(0);
    const auto& filt_base = model::tally_filters[tally_filter_idx].get();
    auto* filt = dynamic_cast<MeshFilter*>(filt_base);
    assert(filt != NULL && "Only mesh filter types are supported in random ray mode.");

    tally.active_ = true;
  }
  setup_active_tallies();

  double k_eff = 1.0;

  // Intialize Cell (FSR) data
  initialize_cell_data();

  uint64_t total_geometric_intersections = 0;

  int n_iters_total = settings::n_batches;
  int n_iters_inactive = settings::n_inactive;
  int n_iters_active = n_iters_total - n_iters_inactive;
  
  int nrays = settings::n_particles;
  double distance_active = settings::ray_distance_active;
  double distance_inactive = settings::ray_distance_inactive;
  double total_active_distance_per_iteration = distance_active * nrays;

  // Serialize Material XS data
  prep_xs();

  // Power Iteration Loop
  for( int iter = 1; iter <= n_iters_total; iter++ )
  {
    // Increment current batch
    simulation::current_batch++;

    // Update neutron source
    simulation::time_update_src.start();
    update_neutron_source(k_eff);
    simulation::time_update_src.stop();

    // Reset scalar and volumes flux to zero
    simulation::time_zero_flux.start();
    set_scalar_flux_to_zero();
    simulation::time_zero_flux.stop();
  
    // Start timer for transport
    simulation::time_transport.start();

    // Transport Sweep
    #pragma omp parallel for schedule(runtime) reduction(+:total_geometric_intersections)
    for( int i = 0; i < nrays; i++ )
    {
      Particle p;
      initialize_ray(p, i, nrays, iter);
      total_geometric_intersections += transport_history_based_single_ray(p, distance_inactive, distance_active);
    }
    
    // Start timer for transport
    simulation::time_transport.stop();

    // Normalize scalar flux and update volumes
    simulation::time_normalize_flux.start();
    normalize_scalar_flux_and_volumes(total_active_distance_per_iteration, iter);
    simulation::time_normalize_flux.stop();

    // Add source to scalar flux
    simulation::time_add_source_to_flux.start();
    add_source_to_scalar_flux();
    simulation::time_add_source_to_flux.stop();
    
    // Compute k-eff
    simulation::time_compute_keff.start();
    k_eff = compute_k_eff(k_eff);
    simulation::k_generation.push_back(k_eff);
    calculate_average_keff();
    simulation::time_compute_keff.stop();
    
    // Output status data
    if (mpi::master && settings::verbosity >= 7) {
      print_generation();
    }
    
    // Tally fission rates
    simulation::time_tally_fission_rates.start();
    if( iter > settings::n_inactive)
      tally_fission_rates();
    simulation::time_tally_fission_rates.stop();

    // Set phi_old = phi_new
    simulation::time_swap_fluxes.start();
    copy_scalar_fluxes();
    simulation::time_swap_fluxes.stop();

    double percent_missed = calculate_miss_rate();
    if( percent_missed > 0.01 )
      printf(" High FSR miss rate detected (%.4lf%%)! Consider increasing ray density by adding more particles and/or active distance.\n", percent_missed);

    if (k_eff > 2.0 || k_eff < 0.25 || !(std::isfinite(k_eff))) {
      fatal_error("Instability detected");
    }
  }
  
  openmc::simulation::time_total.stop();

  // display header block
  header("Timing Statistics", 6);
  printf(" Total time elapsed                = %.4le [s]\n", simulation::time_total.elapsed());
  printf(" Time in transport only            = %.3le [s]\n", simulation::time_transport.elapsed(), 1);

  printf(" Time in update src only           = %.3le [s]\n", simulation::time_update_src.elapsed(), 1);
  printf(" Time in zeroing flux only         = %.3le [s]\n", simulation::time_zero_flux.elapsed(), 1);
  printf(" Time in flux normalization only   = %.3le [s]\n", simulation::time_normalize_flux.elapsed(), 1);
  printf(" Time in add source to flux only   = %.3le [s]\n", simulation::time_add_source_to_flux.elapsed(), 1);
  printf(" Time in compte keff only          = %.3le [s]\n", simulation::time_compute_keff.elapsed(), 1);
  printf(" Time in fission rate tally only   = %.3le [s]\n", simulation::time_tally_fission_rates.elapsed(), 1);
  printf(" Time in flux swap only            = %.3le [s]\n", simulation::time_swap_fluxes.elapsed(), 1);

  printf(" Total Geometric Intersections     = %.4e\n", (double) total_geometric_intersections);
  int negroups = data::mg.num_energy_groups_;
  double total_integrations = (double) total_geometric_intersections * negroups;
  printf(" Total Integrations                = %.4e\n", total_integrations);
  printf(" Time per Integration              = %.4lf [ns]\n", simulation::time_transport.elapsed() * 1.0e9 / total_integrations);

  header("RESULTS", 3);
  fmt::print(" k-effective                       = {:.5f} +/- {:.5f}\n", simulation::keff, simulation::keff_std);
  
  // Write tally results to tallies.out
  if (settings::output_tallies && mpi::master) write_tallies();

  return 0;
}

// OPENMC_RUN encompasses all the main logic where iterations are performed
// over the batches, generations, and histories in a fixed source or k-eigenvalue
// calculation.

int openmc_run()
{
  openmc::simulation::time_total.start();
  openmc_simulation_init();

  int err = 0;
  int status = 0;
  while (status == 0 && err == 0) {
    err = openmc_next_batch(&status);
  }

  openmc_simulation_finalize();
  openmc::simulation::time_total.stop();
  return err;
}

int openmc_simulation_init()
{
  using namespace openmc;

  // Skip if simulation has already been initialized
  if (simulation::initialized) return 0;

  // Initialize nuclear data (energy limits, log grid)
  if (settings::run_CE) {
    initialize_data();
  }

  // Determine how much work each process should do
  calculate_work();

  // Allocate source, fission and surface source banks.
  allocate_banks();

  // If doing an event-based simulation, intialize the particle buffer
  // and event queues
  if (settings::event_based) {
    int64_t event_buffer_length = std::min(simulation::work_per_rank,
      settings::max_particles_in_flight);
    init_event_queues(event_buffer_length);
  }

  // Allocate tally results arrays if they're not allocated yet
  for (auto& t : model::tallies) {
    t->init_results();
  }

  // Set up material nuclide index mapping
  for (auto& mat : model::materials) {
    mat->init_nuclide_index();
  }

  // Reset global variables -- this is done before loading state point (as that
  // will potentially populate k_generation and entropy)
  simulation::current_batch = 0;
  simulation::k_generation.clear();
  simulation::entropy.clear();
  openmc_reset();

  // If this is a restart run, load the state point data and binary source
  // file
  if (settings::restart_run) {
    load_state_point();
    write_message("Resuming simulation...", 6);
  } else {
    // Only initialize primary source bank for eigenvalue simulations
    if (settings::run_mode == RunMode::EIGENVALUE) {
      initialize_source();
    }
  }

  // Display header
  if (mpi::master) {
    if (settings::run_mode == RunMode::FIXED_SOURCE) {
      header("FIXED SOURCE TRANSPORT SIMULATION", 3);
    } else if (settings::run_mode == RunMode::EIGENVALUE) {
      header("K EIGENVALUE SIMULATION", 3);
      if (settings::verbosity >= 7) print_columns();
    }
  }

  // Set flag indicating initialization is done
  simulation::initialized = true;
  return 0;
}

int openmc_simulation_finalize()
{
  using namespace openmc;

  // Skip if simulation was never run
  if (!simulation::initialized) return 0;

  // Stop active batch timer and start finalization timer
  simulation::time_active.stop();
  simulation::time_finalize.start();

  // Clear material nuclide mapping
  for (auto& mat : model::materials) {
    mat->mat_nuclide_index_.clear();
  }

  // Increment total number of generations
  simulation::total_gen += simulation::current_batch*settings::gen_per_batch;

#ifdef OPENMC_MPI
  broadcast_results();
#endif

  // Write tally results to tallies.out
  if (settings::output_tallies && mpi::master) write_tallies();

  // Deactivate all tallies
  for (auto& t : model::tallies) {
    t->active_ = false;
  }

  // Stop timers and show timing statistics
  simulation::time_finalize.stop();
  simulation::time_total.stop();
  if (mpi::master) {
    if (settings::verbosity >= 6) print_runtime();
    if (settings::verbosity >= 4) print_results();
  }
  if (settings::check_overlaps) print_overlap_check();

  // Reset flags
  simulation::initialized = false;
  return 0;
}

int openmc_next_batch(int* status)
{
  using namespace openmc;
  using openmc::simulation::current_gen;

  // Make sure simulation has been initialized
  if (!simulation::initialized) {
    set_errmsg("Simulation has not been initialized yet.");
    return OPENMC_E_ALLOCATE;
  }

  initialize_batch();

  // =======================================================================
  // LOOP OVER GENERATIONS
  for (current_gen = 1; current_gen <= settings::gen_per_batch; ++current_gen) {

    initialize_generation();

    // Start timer for transport
    simulation::time_transport.start();

    // Transport loop
    if (settings::event_based) {
      transport_event_based();
    } else {
      transport_history_based();
    }

    // Accumulate time for transport
    simulation::time_transport.stop();

    finalize_generation();
  }

  finalize_batch();

  // Check simulation ending criteria
  if (status) {
    if (simulation::current_batch == settings::n_max_batches) {
      *status = STATUS_EXIT_MAX_BATCH;
    } else if (simulation::satisfy_triggers) {
      *status = STATUS_EXIT_ON_TRIGGER;
    } else {
      *status = STATUS_EXIT_NORMAL;
    }
  }
  return 0;
}

bool openmc_is_statepoint_batch() {
  using namespace openmc;
  using openmc::simulation::current_gen;

  if (!simulation::initialized)
    return false;
  else
    return contains(settings::statepoint_batch, simulation::current_batch);
}

namespace openmc {

//==============================================================================
// Global variables
//==============================================================================

namespace simulation {

int current_batch;
int current_gen;
bool initialized {false};
double keff {1.0};
double keff_std;
double k_col_abs {0.0};
double k_col_tra {0.0};
double k_abs_tra {0.0};
double log_spacing;
int n_lost_particles {0};
bool need_depletion_rx {false};
int restart_batch;
bool satisfy_triggers {false};
int total_gen {0};
double total_weight;
int64_t work_per_rank;

const RegularMesh* entropy_mesh {nullptr};
const RegularMesh* ufs_mesh {nullptr};

std::vector<double> k_generation;
std::vector<int64_t> work_index;


} // namespace simulation

//==============================================================================
// Non-member functions
//==============================================================================

void allocate_banks()
{
  if (settings::run_mode == RunMode::EIGENVALUE) {
    // Allocate source bank
    simulation::source_bank.resize(simulation::work_per_rank);

    // Allocate fission bank
    init_fission_bank(3*simulation::work_per_rank);
  }

  if (settings::surf_source_write) {
    // Allocate surface source bank
    simulation::surf_source_bank.reserve(settings::max_surface_particles);
  }

}

void initialize_batch()
{
  // Increment current batch
  ++simulation::current_batch;

  if (settings::run_mode == RunMode::FIXED_SOURCE) {
    write_message(6, "Simulating batch {}", simulation::current_batch);
  }

  // Reset total starting particle weight used for normalizing tallies
  simulation::total_weight = 0.0;

  // Determine if this batch is the first inactive or active batch.
  bool first_inactive = false;
  bool first_active = false;
  if (!settings::restart_run) {
    first_inactive = settings::n_inactive > 0 && simulation::current_batch == 1;
    first_active = simulation::current_batch == settings::n_inactive + 1;
  } else if (simulation::current_batch == simulation::restart_batch + 1){
    first_inactive = simulation::restart_batch < settings::n_inactive;
    first_active = !first_inactive;
  }

  // Manage active/inactive timers and activate tallies if necessary.
  if (first_inactive) {
    simulation::time_inactive.start();
  } else if (first_active) {
    simulation::time_inactive.stop();
    simulation::time_active.start();
    for (auto& t : model::tallies) {
      t->active_ = true;
    }
  }

  // Add user tallies to active tallies list
  setup_active_tallies();
}

void finalize_batch()
{
  // Reduce tallies onto master process and accumulate
  simulation::time_tallies.start();
  accumulate_tallies();
  simulation::time_tallies.stop();

  // Reset global tally results
  if (simulation::current_batch <= settings::n_inactive) {
    xt::view(simulation::global_tallies, xt::all()) = 0.0;
    //printf("seting n_realizations to 0 (current_batch = %d, n_inactive = %d\n", simulation::current_batch, settings::n_inactive);
    simulation::n_realizations = 0;
  }

  // Check_triggers
  if (mpi::master) check_triggers();
#ifdef OPENMC_MPI
  MPI_Bcast(&simulation::satisfy_triggers, 1, MPI_C_BOOL, 0, mpi::intracomm);
#endif
  if (simulation::satisfy_triggers || (settings::trigger_on &&
      simulation::current_batch == settings::n_max_batches)) {
    settings::statepoint_batch.insert(simulation::current_batch);
  }

  // Write out state point if it's been specified for this batch and is not
  // a CMFD run instance
  if (contains(settings::statepoint_batch, simulation::current_batch)
      && !settings::cmfd_run) {
    if (contains(settings::sourcepoint_batch, simulation::current_batch)
        && settings::source_write && !settings::source_separate) {
      bool b = (settings::run_mode == RunMode::EIGENVALUE);
      openmc_statepoint_write(nullptr, &b);
    } else {
      bool b = false;
      openmc_statepoint_write(nullptr, &b);
    }
  }

  if (settings::run_mode == RunMode::EIGENVALUE) {
    // Write out a separate source point if it's been specified for this batch
    if (contains(settings::sourcepoint_batch, simulation::current_batch)
        && settings::source_write && settings::source_separate) {
      write_source_point(nullptr);
    }

    // Write a continously-overwritten source point if requested.
    if (settings::source_latest) {
      auto filename = settings::path_output + "source.h5";
      write_source_point(filename.c_str());
    }
  }

  // Write out surface source if requested.
  if (settings::surf_source_write && simulation::current_batch == settings::n_batches) {
    auto filename = settings::path_output + "surface_source.h5";
    write_source_point(filename.c_str(), true);
  }
}

void initialize_generation()
{
  if (settings::run_mode == RunMode::EIGENVALUE) {
    // Clear out the fission bank
    simulation::fission_bank.resize(0);

    // Count source sites if using uniform fission source weighting
    if (settings::ufs_on) ufs_count_sites();

    // Store current value of tracklength k
    simulation::keff_generation = simulation::global_tallies(
      GlobalTally::K_TRACKLENGTH, TallyResult::VALUE);
  }
}

void finalize_generation()
{
  auto& gt = simulation::global_tallies;

  // Update global tallies with the accumulation variables
  if (settings::run_mode == RunMode::EIGENVALUE) {
    gt(GlobalTally::K_COLLISION, TallyResult::VALUE) += global_tally_collision;
    gt(GlobalTally::K_ABSORPTION, TallyResult::VALUE) += global_tally_absorption;
    gt(GlobalTally::K_TRACKLENGTH, TallyResult::VALUE) += global_tally_tracklength;
  }
  gt(GlobalTally::LEAKAGE, TallyResult::VALUE) += global_tally_leakage;

  // reset tallies
  if (settings::run_mode == RunMode::EIGENVALUE) {
    global_tally_collision = 0.0;
    global_tally_absorption = 0.0;
    global_tally_tracklength = 0.0;
  }
  global_tally_leakage = 0.0;

  if (settings::run_mode == RunMode::EIGENVALUE) {
    // If using shared memory, stable sort the fission bank (by parent IDs)
    // so as to allow for reproducibility regardless of which order particles
    // are run in.
    sort_fission_bank();

    // Distribute fission bank across processors evenly
    synchronize_bank();

    // Calculate shannon entropy
    if (settings::entropy_on) shannon_entropy();

    // Collect results and statistics
    calculate_generation_keff();
    calculate_average_keff();

    // Write generation output
    if (mpi::master && settings::verbosity >= 7) {
      print_generation();
    }

  }
}

void initialize_history(Particle& p, int64_t index_source)
{
  // set defaults
  if (settings::run_mode == RunMode::EIGENVALUE) {
    // set defaults for eigenvalue simulations from primary bank
    p.from_source(&simulation::source_bank[index_source - 1]);
  } else if (settings::run_mode == RunMode::FIXED_SOURCE) {
    // initialize random number seed
    int64_t id = (simulation::total_gen + overall_generation() - 1)*settings::n_particles +
      simulation::work_index[mpi::rank] + index_source;
    uint64_t seed = init_seed(id, STREAM_SOURCE);
    // sample from external source distribution or custom library then set
    auto site = sample_external_source(&seed);
    p.from_source(&site);
  }
  p.current_work_ = index_source;

  // set identifier for particle
  p.id_ = simulation::work_index[mpi::rank] + index_source;

  // set progeny count to zero
  p.n_progeny_ = 0;

  // Reset particle event counter
  p.n_event_ = 0;

  // set random number seed
  int64_t particle_seed = (simulation::total_gen + overall_generation() - 1)
    * settings::n_particles + p.id_;
  init_particle_seeds(particle_seed, p.seeds_);

  // set particle trace
  p.trace_ = false;
  if (simulation::current_batch == settings::trace_batch &&
      simulation::current_gen == settings::trace_gen &&
      p.id_ == settings::trace_particle) p.trace_ = true;

  // Set particle track.
  p.write_track_ = false;
  if (settings::write_all_tracks) {
    p.write_track_ = true;
  } else if (settings::track_identifiers.size() > 0) {
    for (const auto& t : settings::track_identifiers) {
      if (simulation::current_batch == t[0] &&
          simulation::current_gen == t[1] &&
          p.id_ == t[2]) {
        p.write_track_ = true;
        break;
      }
    }
  }

  // Display message if high verbosity or trace is on
  if (settings::verbosity >= 9 || p.trace_) {
    write_message("Simulating Particle {}", p.id_);
  }

  // Add paricle's starting weight to count for normalizing tallies later
  #pragma omp atomic
  simulation::total_weight += p.wgt_;

  initialize_history_partial(p);
}

void initialize_history_partial(Particle& p)
{
  // Force calculation of cross-sections by setting last energy to zero
  if (settings::run_CE) {
    for (auto& micro : p.neutron_xs_) micro.last_E = 0.0;
  }

  // Prepare to write out particle track.
  if (p.write_track_) add_particle_track(p);

  // Every particle starts with no accumulated flux derivative.
  if (!model::active_tallies.empty())
  {
    p.flux_derivs_.resize(model::tally_derivs.size(), 0.0);
    std::fill(p.flux_derivs_.begin(), p.flux_derivs_.end(), 0.0);
  }

  // Allocate space for tally filter matches
  p.filter_matches_.resize(model::tally_filters.size());
}

int overall_generation()
{
  using namespace simulation;
  return settings::gen_per_batch*(current_batch - 1) + current_gen;
}

void calculate_work()
{
  // Determine minimum amount of particles to simulate on each processor
  int64_t min_work = settings::n_particles / mpi::n_procs;

  // Determine number of processors that have one extra particle
  int64_t remainder = settings::n_particles % mpi::n_procs;

  int64_t i_bank = 0;
  simulation::work_index.resize(mpi::n_procs + 1);
  simulation::work_index[0] = 0;
  for (int i = 0; i < mpi::n_procs; ++i) {
    // Number of particles for rank i
    int64_t work_i = i < remainder ? min_work + 1 : min_work;

    // Set number of particles
    if (mpi::rank == i) simulation::work_per_rank = work_i;

    // Set index into source bank for rank i
    i_bank += work_i;
    simulation::work_index[i + 1] = i_bank;
  }
}

void initialize_data()
{
  // Determine minimum/maximum energy for incident neutron/photon data
  data::energy_max = {INFTY, INFTY};
  data::energy_min = {0.0, 0.0};
  for (const auto& nuc : data::nuclides) {
    if (nuc->grid_.size() >= 1) {
      int neutron = static_cast<int>(Particle::Type::neutron);
      data::energy_min[neutron] = std::max(data::energy_min[neutron],
        nuc->grid_[0].energy.front());
      data::energy_max[neutron] = std::min(data::energy_max[neutron],
        nuc->grid_[0].energy.back());
    }
  }

  if (settings::photon_transport) {
    for (const auto& elem : data::elements) {
      if (elem->energy_.size() >= 1) {
        int photon = static_cast<int>(Particle::Type::photon);
        int n = elem->energy_.size();
        data::energy_min[photon] = std::max(data::energy_min[photon],
          std::exp(elem->energy_(1)));
        data::energy_max[photon] = std::min(data::energy_max[photon],
          std::exp(elem->energy_(n - 1)));
      }
    }

    if (settings::electron_treatment == ElectronTreatment::TTB) {
      // Determine if minimum/maximum energy for bremsstrahlung is greater/less
      // than the current minimum/maximum
      if (data::ttb_e_grid.size() >= 1) {
        int photon = static_cast<int>(Particle::Type::photon);
        int n_e = data::ttb_e_grid.size();
        data::energy_min[photon] = std::max(data::energy_min[photon],
          std::exp(data::ttb_e_grid(1)));
        data::energy_max[photon] = std::min(data::energy_max[photon],
          std::exp(data::ttb_e_grid(n_e - 1)));
      }
    }
  }

  // Show which nuclide results in lowest energy for neutron transport
  for (const auto& nuc : data::nuclides) {
    // If a nuclide is present in a material that's not used in the model, its
    // grid has not been allocated
    if (nuc->grid_.size() > 0) {
      double max_E = nuc->grid_[0].energy.back();
      int neutron = static_cast<int>(Particle::Type::neutron);
      if (max_E == data::energy_max[neutron]) {
        write_message(7, "Maximum neutron transport energy: {} eV for {}",
          data::energy_max[neutron], nuc->name_);
        if (mpi::master && data::energy_max[neutron] < 20.0e6) {
          warning("Maximum neutron energy is below 20 MeV. This may bias "
            "the results.");
        }
        break;
      }
    }
  }

  // Set up logarithmic grid for nuclides
  for (auto& nuc : data::nuclides) {
    nuc->init_grid();
  }
  int neutron = static_cast<int>(Particle::Type::neutron);
  simulation::log_spacing = std::log(data::energy_max[neutron] /
    data::energy_min[neutron]) / settings::n_log_bins;
}

#ifdef OPENMC_MPI
void broadcast_results() {
  // Broadcast tally results so that each process has access to results
  for (auto& t : model::tallies) {
    // Create a new datatype that consists of all values for a given filter
    // bin and then use that to broadcast. This is done to minimize the
    // chance of the 'count' argument of MPI_BCAST exceeding 2**31
    auto& results = t->results_;

    auto shape = results.shape();
    int count_per_filter = shape[1] * shape[2];
    MPI_Datatype result_block;
    MPI_Type_contiguous(count_per_filter, MPI_DOUBLE, &result_block);
    MPI_Type_commit(&result_block);
    MPI_Bcast(results.data(), shape[0], result_block, 0, mpi::intracomm);
    MPI_Type_free(&result_block);
  }

  // Also broadcast global tally results
  auto& gt = simulation::global_tallies;
  MPI_Bcast(gt.data(), gt.size(), MPI_DOUBLE, 0, mpi::intracomm);

  // These guys are needed so that non-master processes can calculate the
  // combined estimate of k-effective
  double temp[] {simulation::k_col_abs, simulation::k_col_tra,
    simulation::k_abs_tra};
  MPI_Bcast(temp, 3, MPI_DOUBLE, 0, mpi::intracomm);
  simulation::k_col_abs = temp[0];
  simulation::k_col_tra = temp[1];
  simulation::k_abs_tra = temp[2];
}

#endif

void free_memory_simulation()
{
  simulation::k_generation.clear();
  simulation::entropy.clear();
}

void transport_history_based_single_particle(Particle& p)
{
  while (true) {
    p.event_calculate_xs();
    p.event_advance();
    if (p.collision_distance_ > p.boundary_.distance) {
      p.event_cross_surface();
    } else {
      p.event_collide();
    }
    p.event_revive_from_secondary();
    if (!p.alive_)
      break;
  }
  p.event_death();
}

void transport_history_based()
{
  #pragma omp parallel for schedule(runtime)
  for (int64_t i_work = 1; i_work <= simulation::work_per_rank; ++i_work) {
    Particle p;
    initialize_history(p, i_work);
    transport_history_based_single_particle(p);
  }
}

void transport_event_based()
{
  int64_t remaining_work = simulation::work_per_rank;
  int64_t source_offset = 0;

  // To cap the total amount of memory used to store particle object data, the
  // number of particles in flight at any point in time can bet set. In the case
  // that the maximum in flight particle count is lower than the total number
  // of particles that need to be run this iteration, the event-based transport
  // loop is executed multiple times until all particles have been completed.
  while (remaining_work > 0) {
    // Figure out # of particles to run for this subiteration
    int64_t n_particles = std::min(remaining_work, settings::max_particles_in_flight);

    // Initialize all particle histories for this subiteration
    process_init_events(n_particles, source_offset);

    // Event-based transport loop
    while (true) {
      // Determine which event kernel has the longest queue
      int64_t max = std::max({
        simulation::calculate_fuel_xs_queue.size(),
        simulation::calculate_nonfuel_xs_queue.size(),
        simulation::advance_particle_queue.size(),
        simulation::surface_crossing_queue.size(),
        simulation::collision_queue.size()});

      // Execute event with the longest queue
      if (max == 0) {
        break;
      } else if (max == simulation::calculate_fuel_xs_queue.size()) {
        process_calculate_xs_events(simulation::calculate_fuel_xs_queue);
      } else if (max == simulation::calculate_nonfuel_xs_queue.size()) {
        process_calculate_xs_events(simulation::calculate_nonfuel_xs_queue);
      } else if (max == simulation::advance_particle_queue.size()) {
        process_advance_particle_events();
      } else if (max == simulation::surface_crossing_queue.size()) {
        process_surface_crossing_events();
      } else if (max == simulation::collision_queue.size()) {
        process_collision_events();
      }
    }

    // Execute death event for all particles
    process_death_events(n_particles);

    // Adjust remaining work and source offset variables
    remaining_work -= n_particles;
    source_offset += n_particles;
  }
}

} // namespace openmc
