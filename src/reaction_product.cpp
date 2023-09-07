#include "openmc/reaction_product.h"

#include <memory> // for unique_ptr
#include <string> // for string

#include <fmt/core.h>

#include "openmc/endf.h"
#include "openmc/error.h"
#include "openmc/hdf5_interface.h"
#include "openmc/particle.h"
#include "openmc/random_lcg.h"
#include "openmc/secondary_correlated.h"
#include "openmc/secondary_kalbach.h"
#include "openmc/secondary_nbody.h"
#include "openmc/secondary_uncorrelated.h"

namespace openmc {

//==============================================================================
// ReactionProduct implementation
//==============================================================================

ReactionProduct::ReactionProduct(hid_t group)
{
  // Read particle type
  std::string temp;
  read_attribute(group, "particle", temp);
  particle_ = str_to_particle_type(temp);

  // Read emission mode and decay rate
  read_attribute(group, "emission_mode", temp);
  if (temp == "prompt") {
    emission_mode_ = EmissionMode::prompt;
  } else if (temp == "delayed") {
    emission_mode_ = EmissionMode::delayed;
  } else if (temp == "total") {
    emission_mode_ = EmissionMode::total;
  }

  // Read decay rate for delayed emission
  if (emission_mode_ == EmissionMode::delayed) {
    if (attribute_exists(group, "decay_rate")) {
      read_attribute(group, "decay_rate", decay_rate_);
    } else if (particle_ == Particle::Type::neutron) {
      warning(fmt::format("Decay rate doesn't exist for delayed neutron "
        "emission ({}).", object_name(group)));
    }
  }

  // Read secondary particle yield
  yield_ = read_function(group, "yield");

  int n;
  read_attribute(group, "n_distribution", n);

  for (int i = 0; i < n; ++i) {
    std::string s {"distribution_"};
    s.append(std::to_string(i));
    hid_t dgroup = open_group(group, s.c_str());

    // Read applicability
    if (n > 1) {
      hid_t app = open_dataset(dgroup, "applicability");
      applicability_.emplace_back(app);
      close_dataset(app);
    }

    // Determine distribution type and read data
    read_attribute(dgroup, "type", temp);
    if (temp == "uncorrelated") {
      distribution_.push_back(std::make_unique<UncorrelatedAngleEnergy>(dgroup));
    } else if (temp == "correlated") {
      distribution_.push_back(std::make_unique<CorrelatedAngleEnergy>(dgroup));
    } else if (temp == "nbody") {
      distribution_.push_back(std::make_unique<NBodyPhaseSpace>(dgroup));
    } else if (temp == "kalbach-mann") {
      distribution_.push_back(std::make_unique<KalbachMann>(dgroup));
    }

    close_group(dgroup);
  }
}

void ReactionProduct::serialize(DataBuffer& buffer) const
{
  buffer.add(static_cast<int>(particle_));             // 4
  buffer.add(static_cast<int>(emission_mode_));        // 4
  buffer.add(decay_rate_);                             // 8

  // Write size of yield followed by yield itself
  size_t yield_size = buffer_nbytes(*yield_);
  buffer.add(yield_size);                              // 8
  yield_->serialize(buffer);                           // yield_size

  size_t n = 40 + yield_size + aligned(4*(applicability_.size() + distribution_.size()), 8);
  std::vector<int> locators;
  for (const auto& func : applicability_) {
    locators.push_back(n);
    n += buffer_nbytes(func);
  }
  for (const auto& d : distribution_) {
    locators.push_back(n);
    n += buffer_nbytes(*d);
  }

  buffer.add(applicability_.size());                   // 8
  buffer.add(distribution_.size());                    // 8
  buffer.add(locators);                                // 4 * (app + dist size)
  buffer.align(8);

  for (const auto& func : applicability_) {
    func.serialize(buffer);
  }

  for (const auto& d : distribution_) {
    d->serialize(buffer);
  }
}

ReactionProductFlat::ReactionProductFlat(const uint8_t* data) : data_(data)
{
  yield_size_ = *reinterpret_cast<const size_t*>(data_ + 16);
  n_applicability_ = *reinterpret_cast<const size_t*>(data_ + 24 + yield_size_);
  n_distribution_ = *reinterpret_cast<const size_t*>(data_ + 32 + yield_size_);
}

void ReactionProductFlat::sample(double E_in, double& E_out, double& mu,
  uint64_t* seed) const
{
  auto n = n_applicability_;
  if (n > 1) {
    double prob = 0.0;
    double c = prn(seed);
    for (int i = 0; i < n; ++i) {
      // Determine probability that i-th energy distribution is sampled
      prob += this->applicability(i)(E_in);

      // If i-th distribution is sampled, sample energy from the distribution
      if (c <= prob) {
        this->distribution(i).sample(E_in, E_out, mu, seed);
      }
    }
  } else {
    // If only one distribution is present, go ahead and sample it
    this->distribution(0).sample(E_in, E_out, mu, seed);
  }
}

Particle::Type ReactionProductFlat::particle() const
{
  return *reinterpret_cast<const Particle::Type*>(data_);
}

ReactionProduct::EmissionMode ReactionProductFlat::emission_mode() const
{
  return *reinterpret_cast<const ReactionProduct::EmissionMode*>(data_ + 4);
}

double ReactionProductFlat::decay_rate() const
{
  return *reinterpret_cast<const double*>(data_ + 8);
}

Function1DFlat ReactionProductFlat::yield() const
{
  return Function1DFlat(data_ + 24);
}

Tabulated1DFlat ReactionProductFlat::applicability(gsl::index i) const
{
  auto indices = reinterpret_cast<const int*>(data_ + 40 + yield_size_);
  size_t offset = indices[i];
  return Tabulated1DFlat(data_ + offset);
}

AngleEnergyFlat ReactionProductFlat::distribution(gsl::index i) const
{
  auto indices = reinterpret_cast<const int*>(data_ + 40 + yield_size_);
  size_t offset = indices[n_applicability_ + i];
  return AngleEnergyFlat(data_ + offset);
}

}
