// Standalone validation harness for CORE-MATH fast-path functions.
// Compares our coremath:: functions against glibc's std:: equivalents.
//
// Uses bit-level stochastic sampling to avoid deterministic grid bias,
// with dense sampling near known sensitive regions.
//
// Build:
//   g++ -O2 -std=c++17 -ffp-contract=off -Iinclude \
//       tools/validate_coremath.cpp src/math/cr_*.cpp \
//       -o tools/validate_coremath -lm
//
// Run:
//   ./tools/validate_coremath               # default: level 1 (~1M pts/func)
//   ./tools/validate_coremath --level 2     # ~10M pts/func
//   ./tools/validate_coremath --level 3     # ~100M pts/func

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cfloat>
#include <algorithm>
#include <random>
#include <vector>
#include <chrono>

#include "openmc/coremath.h"

// ============================================================
// Bit-level helpers
// ============================================================

static uint64_t to_bits(double x) {
  uint64_t u;
  std::memcpy(&u, &x, sizeof(u));
  return u;
}

static double from_bits(uint64_t u) {
  double x;
  std::memcpy(&x, &u, sizeof(x));
  return x;
}

// ============================================================
// Bit-uniform random double generator
// ============================================================
// Samples uniformly in IEEE 754 representation space within [lo, hi].
// Every representable double in the range has equal probability.
// This avoids the clustering-near-zero bias of uniform_real_distribution.

static double random_double_bits(std::mt19937_64& rng, double lo, double hi) {
  uint64_t lo_bits = to_bits(lo);
  uint64_t hi_bits = to_bits(hi);
  if (lo_bits > hi_bits) std::swap(lo_bits, hi_bits);
  uint64_t range = hi_bits - lo_bits;
  if (range == 0) return lo;
  uint64_t r = rng() % (range + 1);
  return from_bits(lo_bits + r);
}

// For ranges that span zero or include negatives, sample positive and
// negative halves separately with equal probability.
static double random_double_bits_signed(std::mt19937_64& rng,
                                        double neg_lo, double pos_hi) {
  // neg_lo < 0 <= pos_hi
  // Negative doubles: bit pattern is 0x8000... for -0, increasing for more negative
  // We handle by sampling |x| then flipping sign
  double abs_neg = -neg_lo;
  // Weight by number of representable doubles in each half
  uint64_t n_neg = to_bits(abs_neg) - to_bits(0.0);
  uint64_t n_pos = to_bits(pos_hi) - to_bits(0.0);
  // Use 64-bit comparison to decide which half
  if (rng() % (n_neg + n_pos) < n_neg) {
    // Negative half: sample |x| in (0, |neg_lo|], negate
    return -random_double_bits(rng, DBL_MIN, abs_neg);
  } else {
    return random_double_bits(rng, 0.0, pos_hi);
  }
}

// ============================================================
// ULP distance computation
// ============================================================

static uint64_t ulp_distance(double a, double b) {
  if (std::isnan(a) || std::isnan(b)) return UINT64_MAX;
  if (a == b) return 0;

  uint64_t ua = to_bits(a);
  uint64_t ub = to_bits(b);

  // Convert to signed-magnitude integer where ULP distance = |ia - ib|
  int64_t ia = (ua >> 63) ? -(int64_t)(ua & 0x7FFFFFFFFFFFFFFFull) : (int64_t)ua;
  int64_t ib = (ub >> 63) ? -(int64_t)(ub & 0x7FFFFFFFFFFFFFFFull) : (int64_t)ub;

  int64_t diff = ia - ib;
  return (uint64_t)(diff < 0 ? -diff : diff);
}

// ============================================================
// Statistics tracker
// ============================================================

struct Stats {
  const char* name;
  uint64_t max_ulp = 0;
  double max_ulp_x = 0.0;
  double max_ulp_y = 0.0; // for 2-arg functions
  uint64_t total_ulp = 0;
  size_t count = 0;
  size_t gt1 = 0;
  size_t gt2 = 0;
  size_t gt4 = 0;
  size_t gt10 = 0;
  size_t gt100 = 0;
  size_t nan_mismatch = 0;
  size_t inf_mismatch = 0;

  void record(double x, double ours, double ref, double y = 0.0) {
    count++;
    if (std::isnan(ours) != std::isnan(ref)) { nan_mismatch++; return; }
    if (std::isinf(ours) != std::isinf(ref)) { inf_mismatch++; return; }
    if (std::isnan(ours) && std::isnan(ref)) return;
    if (std::isinf(ours) && std::isinf(ref)) {
      if ((ours > 0) != (ref > 0)) inf_mismatch++;
      return;
    }
    uint64_t d = ulp_distance(ours, ref);
    if (d == UINT64_MAX) return;
    total_ulp += d;
    if (d > max_ulp) { max_ulp = d; max_ulp_x = x; max_ulp_y = y; }
    if (d > 1) gt1++;
    if (d > 2) gt2++;
    if (d > 4) gt4++;
    if (d > 10) gt10++;
    if (d > 100) gt100++;
  }

  void print_row() const {
    printf("%-12s %12zu %10llu %10.2f %8zu %8zu %8zu %8zu %8zu",
           name, count, (unsigned long long)max_ulp,
           count > 0 ? (double)total_ulp / count : 0.0,
           gt1, gt2, gt4, gt10, gt100);
    if (nan_mismatch > 0) printf("  NaN=%zu", nan_mismatch);
    if (inf_mismatch > 0) printf("  Inf=%zu", inf_mismatch);
    printf("\n");
  }

  void print_detail(bool two_arg = false) const {
    printf("  %-10s  %12zu pts  max_ulp=%6llu  mean_ulp=%.2f  "
           ">1=%zu  >2=%zu  >4=%zu  >10=%zu  >100=%zu",
           name, count, (unsigned long long)max_ulp,
           count > 0 ? (double)total_ulp / count : 0.0,
           gt1, gt2, gt4, gt10, gt100);
    if (nan_mismatch > 0) printf("  NaN_mismatch=%zu", nan_mismatch);
    if (inf_mismatch > 0) printf("  Inf_mismatch=%zu", inf_mismatch);
    if (max_ulp > 0) {
      if (two_arg)
        printf("  worst_at=(%.17g, %.17g)", max_ulp_x, max_ulp_y);
      else
        printf("  worst_at=%.17g", max_ulp_x);
    }
    printf("\n");
  }
};

// ============================================================
// Per-function test routines
//
// Each function allocates its point budget as:
//   ~50% bit-uniform stochastic across full valid domain
//   ~50% concentrated in sensitive hot zones
// ============================================================

static Stats test_exp(size_t N, uint64_t seed) {
  Stats s; s.name = "exp";
  std::mt19937_64 rng(seed);
  size_t n_rand = N / 2;
  size_t n_hot  = N - n_rand;

  // Stochastic: full domain [-708.4, 709.8] in bit-uniform space
  for (size_t i = 0; i < n_rand; i++) {
    double x = random_double_bits_signed(rng, -708.4, 709.8);
    s.record(x, coremath::exp(x), std::exp(x));
  }
  // Hot zones: near 0, near overflow/underflow boundaries
  size_t per = n_hot / 4;
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits_signed(rng, -1.0, 1.0);
    s.record(x, coremath::exp(x), std::exp(x));
  }
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits_signed(rng, -0.001, 0.001);
    s.record(x, coremath::exp(x), std::exp(x));
  }
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits(rng, -708.5, -706.0);
    s.record(x, coremath::exp(x), std::exp(x));
  }
  for (size_t i = 0; i < n_hot - 3*per; i++) {
    double x = random_double_bits(rng, 708.0, 709.8);
    s.record(x, coremath::exp(x), std::exp(x));
  }
  return s;
}

static Stats test_expm1(size_t N, uint64_t seed) {
  Stats s; s.name = "expm1";
  std::mt19937_64 rng(seed);
  size_t n_rand = N / 2;
  size_t n_hot  = N - n_rand;

  for (size_t i = 0; i < n_rand; i++) {
    double x = random_double_bits_signed(rng, -708.4, 709.8);
    s.record(x, coremath::expm1(x), std::expm1(x));
  }
  // Hot: near 0 (cancellation in exp(x)-1)
  size_t per = n_hot / 3;
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits_signed(rng, -1.0, 1.0);
    s.record(x, coremath::expm1(x), std::expm1(x));
  }
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits_signed(rng, -1e-6, 1e-6);
    s.record(x, coremath::expm1(x), std::expm1(x));
  }
  for (size_t i = 0; i < n_hot - 2*per; i++) {
    double x = random_double_bits_signed(rng, -1e-15, 1e-15);
    s.record(x, coremath::expm1(x), std::expm1(x));
  }
  return s;
}

static Stats test_log(size_t N, uint64_t seed) {
  Stats s; s.name = "log";
  std::mt19937_64 rng(seed);
  size_t n_rand = N / 2;
  size_t n_hot  = N - n_rand;

  // Stochastic: full positive range
  for (size_t i = 0; i < n_rand; i++) {
    double x = random_double_bits(rng, DBL_MIN, DBL_MAX);
    s.record(x, coremath::log(x), std::log(x));
  }
  // Hot: near 1 (cancellation), near 0, denormals
  size_t per = n_hot / 4;
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits(rng, 0.9375, 1.0625);
    s.record(x, coremath::log(x), std::log(x));
  }
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits(rng, 0.999999, 1.000001);
    s.record(x, coremath::log(x), std::log(x));
  }
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits(rng, DBL_MIN, 1e-300);
    s.record(x, coremath::log(x), std::log(x));
  }
  for (size_t i = 0; i < n_hot - 3*per; i++) {
    double x = random_double_bits(rng, 0.5, 2.0);
    s.record(x, coremath::log(x), std::log(x));
  }
  return s;
}

static Stats test_log1p(size_t N, uint64_t seed) {
  Stats s; s.name = "log1p";
  std::mt19937_64 rng(seed);
  size_t n_rand = N / 2;
  size_t n_hot  = N - n_rand;

  // Domain: x > -1. Sample (-1+eps, large)
  for (size_t i = 0; i < n_rand; i++) {
    double x = random_double_bits_signed(rng, -0.9999999, 1e15);
    s.record(x, coremath::log1p(x), std::log1p(x));
  }
  // Hot: near 0
  size_t per = n_hot / 3;
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits_signed(rng, -0.5, 0.5);
    s.record(x, coremath::log1p(x), std::log1p(x));
  }
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits_signed(rng, -1e-8, 1e-8);
    s.record(x, coremath::log1p(x), std::log1p(x));
  }
  for (size_t i = 0; i < n_hot - 2*per; i++) {
    // Near -1 (log1p → -inf)
    double x = random_double_bits(rng, -0.9999999999, -0.99);
    s.record(x, coremath::log1p(x), std::log1p(x));
  }
  return s;
}

static Stats test_sin(size_t N, uint64_t seed) {
  Stats s; s.name = "sin";
  std::mt19937_64 rng(seed);
  size_t n_rand = N / 2;
  size_t n_hot  = N - n_rand;

  // Stochastic: moderate range (argument reduction stress)
  for (size_t i = 0; i < n_rand; i++) {
    double x = random_double_bits_signed(rng, -1e6, 1e6);
    s.record(x, coremath::sin(x), std::sin(x));
  }
  // Hot zones
  size_t per = n_hot / 4;
  // Near 0
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits_signed(rng, -0.01, 0.01);
    s.record(x, coremath::sin(x), std::sin(x));
  }
  // Near multiples of pi (sin ≈ 0, high relative error risk)
  for (size_t i = 0; i < per; i++) {
    int k = (int)(rng() % 201) - 100; // k in [-100, 100]
    double center = k * M_PI;
    double x = center + random_double_bits_signed(rng, -0.001, 0.001);
    s.record(x, coremath::sin(x), std::sin(x));
  }
  // Large arguments (stress argument reduction)
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits_signed(rng, -1e15, 1e15);
    s.record(x, coremath::sin(x), std::sin(x));
  }
  // Primary period
  for (size_t i = 0; i < n_hot - 3*per; i++) {
    double x = random_double_bits_signed(rng, -6.3, 6.3);
    s.record(x, coremath::sin(x), std::sin(x));
  }
  return s;
}

static Stats test_cos(size_t N, uint64_t seed) {
  Stats s; s.name = "cos";
  std::mt19937_64 rng(seed);
  size_t n_rand = N / 2;
  size_t n_hot  = N - n_rand;

  for (size_t i = 0; i < n_rand; i++) {
    double x = random_double_bits_signed(rng, -1e6, 1e6);
    s.record(x, coremath::cos(x), std::cos(x));
  }
  size_t per = n_hot / 4;
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits_signed(rng, -0.01, 0.01);
    s.record(x, coremath::cos(x), std::cos(x));
  }
  // Near pi/2 + k*pi (cos ≈ 0)
  for (size_t i = 0; i < per; i++) {
    int k = (int)(rng() % 201) - 100;
    double center = M_PI/2.0 + k * M_PI;
    double x = center + random_double_bits_signed(rng, -0.001, 0.001);
    s.record(x, coremath::cos(x), std::cos(x));
  }
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits_signed(rng, -1e15, 1e15);
    s.record(x, coremath::cos(x), std::cos(x));
  }
  for (size_t i = 0; i < n_hot - 3*per; i++) {
    double x = random_double_bits_signed(rng, -6.3, 6.3);
    s.record(x, coremath::cos(x), std::cos(x));
  }
  return s;
}

static Stats test_asin(size_t N, uint64_t seed) {
  Stats s; s.name = "asin";
  std::mt19937_64 rng(seed);
  size_t n_rand = N / 2;
  size_t n_hot  = N - n_rand;

  // Domain: [-1, 1]
  for (size_t i = 0; i < n_rand; i++) {
    double x = random_double_bits_signed(rng, -1.0, 1.0);
    s.record(x, coremath::asin(x), std::asin(x));
  }
  size_t per = n_hot / 3;
  // Near 0
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits_signed(rng, -0.001, 0.001);
    s.record(x, coremath::asin(x), std::asin(x));
  }
  // Near ±1 (derivative → ∞)
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits(rng, 0.999, 1.0);
    s.record(x, coremath::asin(x), std::asin(x));
  }
  for (size_t i = 0; i < n_hot - 2*per; i++) {
    double x = random_double_bits(rng, -1.0, -0.999);
    s.record(x, coremath::asin(x), std::asin(x));
  }
  return s;
}

static Stats test_acos(size_t N, uint64_t seed) {
  Stats s; s.name = "acos";
  std::mt19937_64 rng(seed);
  size_t n_rand = N / 2;
  size_t n_hot  = N - n_rand;

  for (size_t i = 0; i < n_rand; i++) {
    double x = random_double_bits_signed(rng, -1.0, 1.0);
    s.record(x, coremath::acos(x), std::acos(x));
  }
  size_t per = n_hot / 3;
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits_signed(rng, -0.001, 0.001);
    s.record(x, coremath::acos(x), std::acos(x));
  }
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits(rng, 0.999, 1.0);
    s.record(x, coremath::acos(x), std::acos(x));
  }
  for (size_t i = 0; i < n_hot - 2*per; i++) {
    double x = random_double_bits(rng, -1.0, -0.999);
    s.record(x, coremath::acos(x), std::acos(x));
  }
  return s;
}

static Stats test_atan(size_t N, uint64_t seed) {
  Stats s; s.name = "atan";
  std::mt19937_64 rng(seed);
  size_t n_rand = N / 2;
  size_t n_hot  = N - n_rand;

  for (size_t i = 0; i < n_rand; i++) {
    double x = random_double_bits_signed(rng, -1e10, 1e10);
    s.record(x, coremath::atan(x), std::atan(x));
  }
  size_t per = n_hot / 4;
  // Near 0
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits_signed(rng, -0.01, 0.01);
    s.record(x, coremath::atan(x), std::atan(x));
  }
  // Near the 0.00662 threshold (historical hotspot)
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits_signed(rng, -0.008, 0.008);
    s.record(x, coremath::atan(x), std::atan(x));
  }
  // Near ±1 (table lookup transition)
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits_signed(rng, -2.0, 2.0);
    s.record(x, coremath::atan(x), std::atan(x));
  }
  // Large values (asymptotic to ±π/2)
  for (size_t i = 0; i < n_hot - 3*per; i++) {
    double x = random_double_bits_signed(rng, -1e100, 1e100);
    s.record(x, coremath::atan(x), std::atan(x));
  }
  return s;
}

static Stats test_atan2(size_t N, uint64_t seed) {
  Stats s; s.name = "atan2";
  std::mt19937_64 rng(seed);
  // Split budget: half stochastic pairs, half near axes/diagonals
  size_t n_rand = N / 2;
  size_t n_hot  = N - n_rand;

  std::uniform_real_distribution<double> wide(-1e6, 1e6);
  for (size_t i = 0; i < n_rand; i++) {
    double y = wide(rng), x = wide(rng);
    s.record(y, coremath::atan2(y, x), std::atan2(y, x), x);
  }
  // Near axes (y≈0 or x≈0)
  size_t per = n_hot / 3;
  for (size_t i = 0; i < per; i++) {
    double y = random_double_bits_signed(rng, -1e-6, 1e-6);
    double x = wide(rng);
    s.record(y, coremath::atan2(y, x), std::atan2(y, x), x);
  }
  for (size_t i = 0; i < per; i++) {
    double y = wide(rng);
    double x = random_double_bits_signed(rng, -1e-6, 1e-6);
    if (x == 0.0 && y == 0.0) continue;
    s.record(y, coremath::atan2(y, x), std::atan2(y, x), x);
  }
  // Near diagonal (|y| ≈ |x|)
  for (size_t i = 0; i < n_hot - 2*per; i++) {
    double v = wide(rng);
    double perturb = random_double_bits_signed(rng, -1e-6, 1e-6);
    double y = v, x = v + perturb;
    s.record(y, coremath::atan2(y, x), std::atan2(y, x), x);
  }
  return s;
}

static Stats test_sinh(size_t N, uint64_t seed) {
  Stats s; s.name = "sinh";
  std::mt19937_64 rng(seed);
  size_t n_rand = N / 2;
  size_t n_hot  = N - n_rand;

  for (size_t i = 0; i < n_rand; i++) {
    double x = random_double_bits_signed(rng, -709.0, 709.0);
    s.record(x, coremath::sinh(x), std::sinh(x));
  }
  size_t per = n_hot / 3;
  // Near 0 (sinh(x) ≈ x, cancellation)
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits_signed(rng, -1.0, 1.0);
    s.record(x, coremath::sinh(x), std::sinh(x));
  }
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits_signed(rng, -0.001, 0.001);
    s.record(x, coremath::sinh(x), std::sinh(x));
  }
  // Near overflow
  for (size_t i = 0; i < n_hot - 2*per; i++) {
    double x = random_double_bits(rng, 708.0, 709.8);
    s.record(x, coremath::sinh(x), std::sinh(x));
  }
  return s;
}

static Stats test_erf(size_t N, uint64_t seed) {
  Stats s; s.name = "erf";
  std::mt19937_64 rng(seed);
  size_t n_rand = N / 2;
  size_t n_hot  = N - n_rand;

  for (size_t i = 0; i < n_rand; i++) {
    double x = random_double_bits_signed(rng, -6.0, 6.0);
    s.record(x, coremath::erf(x), std::erf(x));
  }
  size_t per = n_hot / 3;
  // Near 0
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits_signed(rng, -0.01, 0.01);
    s.record(x, coremath::erf(x), std::erf(x));
  }
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits_signed(rng, -1e-10, 1e-10);
    s.record(x, coremath::erf(x), std::erf(x));
  }
  // Transition region (erf goes from ~0 to ~1)
  for (size_t i = 0; i < n_hot - 2*per; i++) {
    double x = random_double_bits_signed(rng, -3.0, 3.0);
    s.record(x, coremath::erf(x), std::erf(x));
  }
  return s;
}

static Stats test_erfc(size_t N, uint64_t seed) {
  Stats s; s.name = "erfc";
  std::mt19937_64 rng(seed);
  size_t n_rand = N / 2;
  size_t n_hot  = N - n_rand;

  // Full range including large x where erfc is tiny
  for (size_t i = 0; i < n_rand; i++) {
    double x = random_double_bits_signed(rng, -6.0, 27.3);
    s.record(x, coremath::erfc(x), std::erfc(x));
  }
  size_t per = n_hot / 4;
  // Near 0
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits_signed(rng, -0.01, 0.01);
    s.record(x, coremath::erfc(x), std::erfc(x));
  }
  // Transition region
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits(rng, 1.0, 6.0);
    s.record(x, coremath::erfc(x), std::erfc(x));
  }
  // Large x (tiny erfc, historically problematic)
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits(rng, 20.0, 27.3);
    s.record(x, coremath::erfc(x), std::erfc(x));
  }
  // Very large x near limit
  for (size_t i = 0; i < n_hot - 3*per; i++) {
    double x = random_double_bits(rng, 25.0, 27.3);
    s.record(x, coremath::erfc(x), std::erfc(x));
  }
  return s;
}

static Stats test_lgamma(size_t N, uint64_t seed) {
  Stats s; s.name = "lgamma";
  std::mt19937_64 rng(seed);
  size_t n_rand = N / 2;
  size_t n_hot  = N - n_rand;

  // Positive domain
  for (size_t i = 0; i < n_rand; i++) {
    double x = random_double_bits(rng, DBL_MIN, 1e10);
    s.record(x, coremath::lgamma(x), std::lgamma(x));
  }
  size_t per = n_hot / 4;
  // Near positive root (~1.46)
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits(rng, 1.0, 2.0);
    s.record(x, coremath::lgamma(x), std::lgamma(x));
  }
  // Near 0 (pole)
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits(rng, DBL_MIN, 0.01);
    s.record(x, coremath::lgamma(x), std::lgamma(x));
  }
  // Negative non-integers (near poles and roots)
  for (size_t i = 0; i < per; i++) {
    int k = (int)(rng() % 50) + 1;
    double x = -(double)k - random_double_bits(rng, 0.001, 0.999);
    s.record(x, coremath::lgamma(x), std::lgamma(x));
  }
  // Specifically near known worst case x ≈ -3.955 (lgamma root)
  for (size_t i = 0; i < n_hot - 3*per; i++) {
    double x = random_double_bits(rng, -3.96, -3.94);
    s.record(x, coremath::lgamma(x), std::lgamma(x));
  }
  return s;
}

static Stats test_tgamma(size_t N, uint64_t seed) {
  Stats s; s.name = "tgamma";
  std::mt19937_64 rng(seed);
  size_t n_rand = N / 2;
  size_t n_hot  = N - n_rand;

  // Positive domain (main usage)
  for (size_t i = 0; i < n_rand; i++) {
    double x = random_double_bits(rng, 0.01, 171.6);
    s.record(x, coremath::tgamma(x), std::tgamma(x));
  }
  size_t per = n_hot / 4;
  // Near 0 and small positive
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits(rng, 0.001, 2.0);
    s.record(x, coremath::tgamma(x), std::tgamma(x));
  }
  // Near positive integers (factorial values)
  for (size_t i = 0; i < per; i++) {
    int k = (int)(rng() % 20) + 1;
    double x = (double)k + random_double_bits_signed(rng, -0.01, 0.01);
    if (x > 0.0) s.record(x, coremath::tgamma(x), std::tgamma(x));
  }
  // Negative non-integers
  for (size_t i = 0; i < per; i++) {
    int k = (int)(rng() % 170) + 1;
    double x = -(double)k - random_double_bits(rng, 0.001, 0.999);
    double ours = coremath::tgamma(x);
    double ref = std::tgamma(x);
    if (std::isfinite(ours) && std::isfinite(ref))
      s.record(x, ours, ref);
  }
  // Large arguments near overflow
  for (size_t i = 0; i < n_hot - 3*per; i++) {
    double x = random_double_bits(rng, 150.0, 171.6);
    s.record(x, coremath::tgamma(x), std::tgamma(x));
  }
  return s;
}

static Stats test_pow(size_t N, uint64_t seed) {
  Stats s; s.name = "pow";
  std::mt19937_64 rng(seed);
  size_t n_rand = N / 2;
  size_t n_hot  = N - n_rand;

  // Wide stochastic: random (base, exp) pairs
  for (size_t i = 0; i < n_rand; i++) {
    double x = random_double_bits(rng, 1e-10, 1e10);
    double y = random_double_bits_signed(rng, -50.0, 50.0);
    double ours = coremath::pow(x, y);
    double ref = std::pow(x, y);
    if (std::isfinite(ours) && std::isfinite(ref))
      s.record(x, ours, ref, y);
  }
  size_t per = n_hot / 3;
  // Near x=1 (cancellation in log(x))
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits(rng, 0.99, 1.01);
    double y = random_double_bits_signed(rng, -1000.0, 1000.0);
    double ours = coremath::pow(x, y);
    double ref = std::pow(x, y);
    if (std::isfinite(ours) && std::isfinite(ref))
      s.record(x, ours, ref, y);
  }
  // Near y=0 (pow → 1)
  for (size_t i = 0; i < per; i++) {
    double x = random_double_bits(rng, 1e-5, 1e5);
    double y = random_double_bits_signed(rng, -0.001, 0.001);
    double ours = coremath::pow(x, y);
    double ref = std::pow(x, y);
    if (std::isfinite(ours) && std::isfinite(ref))
      s.record(x, ours, ref, y);
  }
  // Small base, large exponent (near overflow/underflow)
  for (size_t i = 0; i < n_hot - 2*per; i++) {
    double x = random_double_bits(rng, 0.01, 100.0);
    double y = random_double_bits_signed(rng, -300.0, 300.0);
    double ours = coremath::pow(x, y);
    double ref = std::pow(x, y);
    if (std::isfinite(ours) && std::isfinite(ref))
      s.record(x, ours, ref, y);
  }
  return s;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
  int level = 1;
  for (int i = 1; i < argc; i++) {
    if (std::strcmp(argv[i], "--level") == 0 && i+1 < argc) {
      level = std::atoi(argv[++i]);
      if (level < 1) level = 1;
      if (level > 3) level = 3;
    }
  }

  // Points per function: level 1 = 1M, level 2 = 10M, level 3 = 100M
  size_t N;
  if (level == 1)      N = 1000000;
  else if (level == 2) N = 10000000;
  else                 N = 100000000;

  printf("CORE-MATH Fast-Path Validation (level %d, %zuM pts/func)\n", level, N/1000000);
  printf("Bit-uniform stochastic sampling (no deterministic grid bias)\n");
  printf("Comparing coremath:: vs std:: (glibc)\n");
  printf("=============================================\n\n");

  auto t0 = std::chrono::steady_clock::now();

  // Use different seeds per function to ensure independence.
  // Each function gets a distinct prime seed.
  Stats results[] = {
    test_exp(N, 100003),
    test_expm1(N, 200003),
    test_log(N, 300007),
    test_log1p(N, 400009),
    test_sin(N, 500009),
    test_cos(N, 600011),
    test_asin(N, 700001),
    test_acos(N, 800011),
    test_atan(N, 900001),
    test_atan2(N, 1000003),
    test_sinh(N, 1100009),
    test_erf(N, 1200007),
    test_erfc(N, 1300021),
    test_lgamma(N, 1400017),
    test_tgamma(N, 1500007),
    test_pow(N, 1600033),
  };

  bool two_arg[] = {
    false, false, false, false, false, false,
    false, false, false, true, false, false,
    false, false, false, true
  };

  auto t1 = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(t1 - t0).count();

  printf("%-12s %12s %10s %10s %8s %8s %8s %8s %8s\n",
         "Function", "Points", "Max ULP", "Mean ULP",
         ">1", ">2", ">4", ">10", ">100");
  printf("%-12s %12s %10s %10s %8s %8s %8s %8s %8s\n",
         "--------", "------", "-------", "--------",
         "--", "--", "--", "---", "----");

  for (size_t i = 0; i < sizeof(results)/sizeof(results[0]); i++) {
    results[i].print_row();
  }

  printf("\nDetailed worst cases:\n");
  for (size_t i = 0; i < sizeof(results)/sizeof(results[0]); i++) {
    results[i].print_detail(two_arg[i]);
  }

  printf("\nCompleted in %.1f seconds\n", elapsed);

  return 0;
}
