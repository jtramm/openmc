//! \file output.h
//! Functions for ASCII output.

#ifndef OPENMC_OUTPUT_H
#define OPENMC_OUTPUT_H

#include <string>

#include "openmc/particle.h"

namespace openmc {

//! \brief Display the main title banner as well as information about the
//! program developers, version, and date/time which the problem was run.
void title();

//! Display a header block.
//
//! \param msg The main text of the header
//! \param level The lowest verbosity level at which this header is printed
void header(const char* msg, int level);

//! Retrieve a time stamp.
//
//! \return current time stamp (format: "yyyy-mm-dd hh:mm:ss")
std::string time_stamp();

//! Display the attributes of a particle.
void print_particle(Particle& p);

//! Display plot information.
void print_plot();

//! Display information regarding cell overlap checking.
void print_overlap_check();

//! Display information about command line usage of OpenMC
void print_usage();

//! Display current version and copright/license information
void print_version();

//! Display compile flags employed, etc
void print_build_info();

//! Display header listing what physical values will displayed
void print_columns();

//! Display information about a generation of neutrons
void print_generation();

//! Display time elapsed for various stages of a run
void print_runtime();

//! Display results for global tallies including k-effective estimators
void print_results();

//! Display timing results and global statistics for random ray simulations
void print_results_random_ray(uint64_t total_geometric_intersections,
  double avg_miss_rate, int negroups, int64_t mfci, int64_t max_regions,
  int64_t n_source_regions, int64_t n_fixed_source_regions);

void write_tallies();

} // namespace openmc
#endif // OPENMC_OUTPUT_H

//////////////////////////////////////
// Custom formatters
//////////////////////////////////////
namespace fmt {

template<typename T>
struct formatter<std::array<T, 2>> {
  template<typename ParseContext>
  constexpr auto parse(ParseContext& ctx)
  {
    return ctx.begin();
  }

  template<typename FormatContext>
  auto format(const std::array<T, 2>& arr, FormatContext& ctx)
  {
    return format_to(ctx.out(), "({}, {})", arr[0], arr[1]);
  }
};

} // namespace fmt
