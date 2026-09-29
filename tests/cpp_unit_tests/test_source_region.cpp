#include <catch2/catch_test_macros.hpp>

#include "openmc/constants.h"
#include "openmc/random_ray/flat_source_domain.h"
#include "openmc/random_ray/source_region.h"
#include "openmc/settings.h"
#include "openmc/vector.h"

using namespace openmc;

namespace {

// Sets the simulation settings that decide which fields of a source region
// are allocated, and restores them when it goes out of scope
class SettingsGuard {
public:
  SettingsGuard(RunMode run_mode, RandomRayVolumeEstimator estimator)
    : run_mode_(settings::run_mode),
      estimator_(FlatSourceDomain::resolved_volume_estimator_)
  {
    settings::run_mode = run_mode;
    FlatSourceDomain::resolved_volume_estimator_ = estimator;
  }
  ~SettingsGuard()
  {
    settings::run_mode = run_mode_;
    FlatSourceDomain::resolved_volume_estimator_ = estimator_;
  }

private:
  RunMode run_mode_;
  RandomRayVolumeEstimator estimator_;
};

// Gives every field of a source region a value of its own, all of which
// differ from the values a new source region starts with
void fill(SourceRegion& sr, double x)
{
  auto& s = sr.scalars_;
  s.material_ = 3;
  s.temperature_idx_ = 2;
  s.density_mult_ = x + 0.5;
  s.volume_ = x + 1.0;
  s.volume_t_ = x + 2.0;
  s.volume_sq_ = x + 3.0;
  s.volume_sq_t_ = x + 4.0;
  s.volume_naive_ = x + 5.0;
  s.position_recorded_ = 1;
  s.external_source_present_ = 1;
  s.is_small_ = 1;
  s.n_negative_batches_ = 4;
  s.converged_negative_ = 2;
  s.n_hits_ = 17;
  s.mesh_ = 1;
  s.key_ = {12, 34};
  s.parent_sr_ = 7;
  s.position_ = {x + 6.0, x + 7.0, x + 8.0};
  s.centroid_ = {x + 9.0, x + 10.0, x + 11.0};
  s.centroid_iteration_ = {x + 12.0, x + 13.0, x + 14.0};
  s.centroid_t_ = {x + 15.0, x + 16.0, x + 17.0};
  s.centroid_offset_ = {x + 18.0, x + 19.0, x + 20.0};
  s.mom_matrix_ = {x + 21.0, x + 22.0, x + 23.0, x + 24.0, x + 25.0, x + 26.0};
  s.mom_matrix_t_ = {
    x + 27.0, x + 28.0, x + 29.0, x + 30.0, x + 31.0, x + 32.0};
  sr.extent_ = {{x - 3.0, x - 2.0, x - 1.0}, {x + 1.0, x + 2.0, x + 3.0}};

  double v = x + 100.0;
  auto fill_array = [&v](auto& array) {
    for (auto& element : array) {
      element = v;
      v += 1.0;
    }
  };
  auto fill_moments = [&v](vector<MomentArray>& array) {
    for (auto& element : array) {
      element = {v, v + 1.0, v + 2.0};
      v += 3.0;
    }
  };
  fill_array(sr.scalar_flux_old_);
  fill_array(sr.scalar_flux_new_);
  fill_array(sr.source_);
  fill_array(sr.external_source_);
  fill_array(sr.scalar_flux_final_);
  fill_array(sr.scalar_flux_t_);
  fill_moments(sr.source_gradients_);
  fill_moments(sr.flux_moments_old_);
  fill_moments(sr.flux_moments_new_);
  fill_moments(sr.flux_moments_t_);
}

bool same(const Position& a, const Position& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

bool same(const MomentMatrix& a, const MomentMatrix& b)
{
  return a.a == b.a && a.b == b.b && a.c == b.c && a.d == b.d && a.e == b.e &&
         a.f == b.f;
}

bool same(const vector<MomentArray>& a, const vector<MomentArray>& b)
{
  if (a.size() != b.size())
    return false;
  for (int i = 0; i < a.size(); i++) {
    if (!same(a[i], b[i]))
      return false;
  }
  return true;
}

void require_same(const SourceRegion& a, const SourceRegion& b)
{
  const auto& s = a.scalars_;
  const auto& t = b.scalars_;
  REQUIRE(s.material_ == t.material_);
  REQUIRE(s.temperature_idx_ == t.temperature_idx_);
  REQUIRE(s.density_mult_ == t.density_mult_);
  REQUIRE(s.volume_ == t.volume_);
  REQUIRE(s.volume_t_ == t.volume_t_);
  REQUIRE(s.volume_sq_ == t.volume_sq_);
  REQUIRE(s.volume_sq_t_ == t.volume_sq_t_);
  REQUIRE(s.volume_naive_ == t.volume_naive_);
  REQUIRE(s.position_recorded_ == t.position_recorded_);
  REQUIRE(s.external_source_present_ == t.external_source_present_);
  REQUIRE(s.is_small_ == t.is_small_);
  REQUIRE(s.n_negative_batches_ == t.n_negative_batches_);
  REQUIRE(s.converged_negative_ == t.converged_negative_);
  REQUIRE(s.n_hits_ == t.n_hits_);
  REQUIRE(s.mesh_ == t.mesh_);
  REQUIRE(s.key_ == t.key_);
  REQUIRE(s.parent_sr_ == t.parent_sr_);
  REQUIRE(same(s.position_, t.position_));
  REQUIRE(same(s.centroid_, t.centroid_));
  REQUIRE(same(s.centroid_iteration_, t.centroid_iteration_));
  REQUIRE(same(s.centroid_t_, t.centroid_t_));
  REQUIRE(same(s.centroid_offset_, t.centroid_offset_));
  REQUIRE(same(s.mom_matrix_, t.mom_matrix_));
  REQUIRE(same(s.mom_matrix_t_, t.mom_matrix_t_));
  REQUIRE(same(a.extent_.min, b.extent_.min));
  REQUIRE(same(a.extent_.max, b.extent_.max));

  REQUIRE(a.scalar_flux_old_ == b.scalar_flux_old_);
  REQUIRE(a.scalar_flux_new_ == b.scalar_flux_new_);
  REQUIRE(a.source_ == b.source_);
  REQUIRE(a.external_source_ == b.external_source_);
  REQUIRE(a.scalar_flux_final_ == b.scalar_flux_final_);
  REQUIRE(a.scalar_flux_t_ == b.scalar_flux_t_);
  REQUIRE(same(a.source_gradients_, b.source_gradients_));
  REQUIRE(same(a.flux_moments_old_, b.flux_moments_old_));
  REQUIRE(same(a.flux_moments_new_, b.flux_moments_new_));
  REQUIRE(same(a.flux_moments_t_, b.flux_moments_t_));
}

} // namespace

TEST_CASE("Source regions are packed and unpacked without loss")
{
  // The run mode, the volume estimator and the source shape decide which
  // arrays of a source region are allocated
  struct Case {
    RunMode run_mode;
    RandomRayVolumeEstimator estimator;
    bool is_linear;
  };
  vector<Case> cases = {
    {RunMode::FIXED_SOURCE, RandomRayVolumeEstimator::ADAPTIVE, true},
    {RunMode::FIXED_SOURCE, RandomRayVolumeEstimator::HYBRID, false},
    {RunMode::EIGENVALUE, RandomRayVolumeEstimator::STRICT_ADAPTIVE, false},
    {RunMode::EIGENVALUE, RandomRayVolumeEstimator::NAIVE, true}};

  const int negroups = 3;

  for (const auto& c : cases) {
    SettingsGuard guard(c.run_mode, c.estimator);

    // Two source regions packed one after the other, as they are when they
    // are sent to another MPI rank
    SourceRegion first(negroups, c.is_linear);
    SourceRegion second(negroups, c.is_linear);
    fill(first, 1.0);
    fill(second, 1000.0);

    size_t n = first.packed_size();
    REQUIRE(second.packed_size() == n);

    vector<char> buffer(2 * n);
    char* end = first.pack(buffer.data());
    REQUIRE(end == buffer.data() + n);
    end = second.pack(end);
    REQUIRE(end == buffer.data() + 2 * n);

    SourceRegion first_copy(negroups, c.is_linear);
    SourceRegion second_copy(negroups, c.is_linear);
    const char* position = first_copy.unpack(buffer.data());
    REQUIRE(position == buffer.data() + n);
    position = second_copy.unpack(position);
    REQUIRE(position == buffer.data() + 2 * n);

    require_same(first, first_copy);
    require_same(second, second_copy);
  }
}

TEST_CASE("Source regions discovered by two MPI ranks are merged")
{
  SettingsGuard guard(RunMode::FIXED_SOURCE, RandomRayVolumeEstimator::HYBRID);

  const int negroups = 2;
  SourceRegion a(negroups, true);
  SourceRegion b(negroups, true);
  fill(a, 1.0);
  fill(b, 50.0);
  SourceRegion sum = a;
  sum.merge(b);

  // What the ranks tallied in the batch is added up
  REQUIRE(sum.scalars_.volume_ == a.scalars_.volume_ + b.scalars_.volume_);
  REQUIRE(
    sum.scalars_.volume_sq_ == a.scalars_.volume_sq_ + b.scalars_.volume_sq_);
  REQUIRE(sum.scalars_.volume_naive_ ==
          a.scalars_.volume_naive_ + b.scalars_.volume_naive_);
  REQUIRE(sum.scalars_.n_hits_ == a.scalars_.n_hits_ + b.scalars_.n_hits_);
  REQUIRE(same(sum.scalars_.centroid_iteration_,
    a.scalars_.centroid_iteration_ + b.scalars_.centroid_iteration_));
  REQUIRE(sum.scalars_.mom_matrix_.a ==
          a.scalars_.mom_matrix_.a + b.scalars_.mom_matrix_.a);
  for (int g = 0; g < negroups; g++) {
    REQUIRE(
      sum.scalar_flux_new_[g] == a.scalar_flux_new_[g] + b.scalar_flux_new_[g]);
    REQUIRE(same(sum.flux_moments_new_[g],
      a.flux_moments_new_[g] + b.flux_moments_new_[g]));
  }

  // The sampled extent is the union of the two
  REQUIRE(same(sum.extent_.min, a.extent_.min));
  REQUIRE(same(sum.extent_.max, b.extent_.max));

  // The source is a property of the region that both ranks hold in full
  REQUIRE(sum.source_ == a.source_);
  REQUIRE(sum.external_source_ == a.external_source_);
  REQUIRE(sum.scalar_flux_old_ == a.scalar_flux_old_);
}
