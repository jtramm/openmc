// Standalone validation harness for CORE-MATH fast-path functions.
// Compares our coremath:: functions against glibc's std:: equivalents.
//
// Build:
//   g++ -O2 -std=c++17 -ffp-contract=off -I../../include -Isrc/math \
//       tools/validate_coremath.cpp src/math/cr_*.cpp \
//       -o tools/validate_coremath -lm
//
// Run:
//   ./tools/validate_coremath

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cfloat>
#include <algorithm>
#include <random>
#include <vector>

#include "openmc/coremath.h"

// ============================================================
// ULP distance computation
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

// Compute ULP distance between two finite doubles with the same sign.
// Returns UINT64_MAX for special cases (NaN, different signs, etc.)
static uint64_t ulp_distance(double a, double b) {
  if (std::isnan(a) || std::isnan(b)) return UINT64_MAX;
  if (a == b) return 0;

  uint64_t ua = to_bits(a);
  uint64_t ub = to_bits(b);

  // Handle sign: convert to signed-magnitude integer representation
  // where ULP distance is just |ia - ib|
  int64_t ia = (ua >> 63) ? -(int64_t)(ua & 0x7FFFFFFFFFFFFFFFull) : (int64_t)ua;
  int64_t ib = (ub >> 63) ? -(int64_t)(ub & 0x7FFFFFFFFFFFFFFFull) : (int64_t)ub;

  int64_t diff = ia - ib;
  return (uint64_t)(diff < 0 ? -diff : diff);
}

// ============================================================
// Test point generators
// ============================================================

// Log-spaced positive values from lo to hi
static void gen_logspaced(std::vector<double>& pts, double lo, double hi, size_t n) {
  double log_lo = std::log(lo);
  double log_hi = std::log(hi);
  for (size_t i = 0; i < n; i++) {
    double t = (double)i / (double)(n - 1);
    pts.push_back(std::exp(log_lo + t * (log_hi - log_lo)));
  }
}

// Linear-spaced values from lo to hi
static void gen_linspaced(std::vector<double>& pts, double lo, double hi, size_t n) {
  for (size_t i = 0; i < n; i++) {
    double t = (double)i / (double)(n - 1);
    pts.push_back(lo + t * (hi - lo));
  }
}

// Random doubles in [lo, hi]
static void gen_random(std::vector<double>& pts, double lo, double hi, size_t n, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(lo, hi);
  for (size_t i = 0; i < n; i++) {
    pts.push_back(dist(rng));
  }
}

// Random doubles across full positive range (log-uniform)
static void gen_random_log(std::vector<double>& pts, double lo, double hi, size_t n, uint64_t seed) {
  std::mt19937_64 rng(seed);
  double log_lo = std::log(lo);
  double log_hi = std::log(hi);
  std::uniform_real_distribution<double> dist(log_lo, log_hi);
  for (size_t i = 0; i < n; i++) {
    pts.push_back(std::exp(dist(rng)));
  }
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
  size_t gt1 = 0;   // > 1 ULP
  size_t gt2 = 0;   // > 2 ULP
  size_t gt4 = 0;   // > 4 ULP
  size_t gt10 = 0;  // > 10 ULP
  size_t gt100 = 0; // > 100 ULP
  size_t nan_mismatch = 0; // one NaN, one not
  size_t inf_mismatch = 0; // one Inf, one not

  void record(double x, double ours, double ref, double y = 0.0) {
    count++;

    // Handle NaN/Inf mismatches
    if (std::isnan(ours) != std::isnan(ref)) {
      nan_mismatch++;
      return;
    }
    if (std::isinf(ours) != std::isinf(ref)) {
      inf_mismatch++;
      return;
    }
    if (std::isnan(ours) && std::isnan(ref)) return; // both NaN = OK
    if (std::isinf(ours) && std::isinf(ref)) {
      // Check sign
      if ((ours > 0) != (ref > 0)) inf_mismatch++;
      return;
    }

    uint64_t d = ulp_distance(ours, ref);
    if (d == UINT64_MAX) return;

    total_ulp += d;
    if (d > max_ulp) {
      max_ulp = d;
      max_ulp_x = x;
      max_ulp_y = y;
    }
    if (d > 1) gt1++;
    if (d > 2) gt2++;
    if (d > 4) gt4++;
    if (d > 10) gt10++;
    if (d > 100) gt100++;
  }

  void print(bool two_arg = false) const {
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
// Test each function
// ============================================================

static Stats test_exp() {
  Stats s; s.name = "exp";
  std::vector<double> pts;
  gen_linspaced(pts, -708.0, 709.0, 500000);
  gen_linspaced(pts, -1.0, 1.0, 200000);
  gen_linspaced(pts, -0.001, 0.001, 100000);
  gen_random(pts, -700.0, 700.0, 200000, 1);
  for (double x : pts) {
    s.record(x, coremath::exp(x), std::exp(x));
  }
  return s;
}

static Stats test_expm1() {
  Stats s; s.name = "expm1";
  std::vector<double> pts;
  gen_linspaced(pts, -1.0, 1.0, 500000);
  gen_linspaced(pts, -0.001, 0.001, 200000);
  gen_linspaced(pts, -708.0, 709.0, 200000);
  gen_random(pts, -700.0, 700.0, 100000, 2);
  for (double x : pts) {
    s.record(x, coremath::expm1(x), std::expm1(x));
  }
  return s;
}

static Stats test_log() {
  Stats s; s.name = "log";
  std::vector<double> pts;
  gen_logspaced(pts, 1e-308, 1e+308, 500000);
  gen_linspaced(pts, 0.5, 2.0, 300000);
  gen_linspaced(pts, 0.999, 1.001, 200000);
  gen_random_log(pts, 1e-300, 1e+300, 200000, 3);
  for (double x : pts) {
    s.record(x, coremath::log(x), std::log(x));
  }
  return s;
}

static Stats test_log1p() {
  Stats s; s.name = "log1p";
  std::vector<double> pts;
  gen_linspaced(pts, -0.999, 1e6, 500000);
  gen_linspaced(pts, -0.001, 0.001, 300000);
  gen_linspaced(pts, -1e-10, 1e-10, 200000);
  gen_random(pts, -0.5, 1e6, 200000, 4);
  for (double x : pts) {
    s.record(x, coremath::log1p(x), std::log1p(x));
  }
  return s;
}

static Stats test_sin() {
  Stats s; s.name = "sin";
  std::vector<double> pts;
  gen_linspaced(pts, -6.3, 6.3, 500000);
  gen_linspaced(pts, -0.001, 0.001, 200000);
  gen_linspaced(pts, -1e4, 1e4, 200000);
  gen_random(pts, -1e6, 1e6, 100000, 5);
  // Near multiples of pi
  for (int k = -100; k <= 100; k++) {
    gen_linspaced(pts, k * M_PI - 0.001, k * M_PI + 0.001, 1000);
  }
  for (double x : pts) {
    s.record(x, coremath::sin(x), std::sin(x));
  }
  return s;
}

static Stats test_cos() {
  Stats s; s.name = "cos";
  std::vector<double> pts;
  gen_linspaced(pts, -6.3, 6.3, 500000);
  gen_linspaced(pts, -0.001, 0.001, 200000);
  gen_linspaced(pts, -1e4, 1e4, 200000);
  gen_random(pts, -1e6, 1e6, 100000, 6);
  for (int k = -100; k <= 100; k++) {
    gen_linspaced(pts, k * M_PI - 0.001, k * M_PI + 0.001, 1000);
  }
  for (double x : pts) {
    s.record(x, coremath::cos(x), std::cos(x));
  }
  return s;
}

static Stats test_asin() {
  Stats s; s.name = "asin";
  std::vector<double> pts;
  gen_linspaced(pts, -1.0, 1.0, 1000000);
  gen_linspaced(pts, -0.001, 0.001, 200000);
  gen_linspaced(pts, 0.999, 1.0, 100000);
  gen_linspaced(pts, -1.0, -0.999, 100000);
  for (double x : pts) {
    s.record(x, coremath::asin(x), std::asin(x));
  }
  return s;
}

static Stats test_acos() {
  Stats s; s.name = "acos";
  std::vector<double> pts;
  gen_linspaced(pts, -1.0, 1.0, 1000000);
  gen_linspaced(pts, -0.001, 0.001, 200000);
  gen_linspaced(pts, 0.999, 1.0, 100000);
  gen_linspaced(pts, -1.0, -0.999, 100000);
  for (double x : pts) {
    s.record(x, coremath::acos(x), std::acos(x));
  }
  return s;
}

static Stats test_atan() {
  Stats s; s.name = "atan";
  std::vector<double> pts;
  gen_linspaced(pts, -1e6, 1e6, 500000);
  gen_linspaced(pts, -1.0, 1.0, 300000);
  gen_linspaced(pts, -0.001, 0.001, 200000);
  gen_random(pts, -1e10, 1e10, 200000, 7);
  for (double x : pts) {
    s.record(x, coremath::atan(x), std::atan(x));
  }
  return s;
}

static Stats test_atan2() {
  Stats s; s.name = "atan2";
  std::vector<double> pts_y, pts_x;
  gen_linspaced(pts_y, -10.0, 10.0, 1000);
  gen_linspaced(pts_x, -10.0, 10.0, 1000);
  for (double y : pts_y) {
    for (double x : pts_x) {
      if (x == 0.0 && y == 0.0) continue;
      s.record(y, coremath::atan2(y, x), std::atan2(y, x), x);
    }
  }
  // Random pairs
  std::mt19937_64 rng(8);
  std::uniform_real_distribution<double> dist(-1e6, 1e6);
  for (size_t i = 0; i < 200000; i++) {
    double y = dist(rng), x = dist(rng);
    s.record(y, coremath::atan2(y, x), std::atan2(y, x), x);
  }
  return s;
}

static Stats test_sinh() {
  Stats s; s.name = "sinh";
  std::vector<double> pts;
  gen_linspaced(pts, -709.0, 709.0, 500000);
  gen_linspaced(pts, -1.0, 1.0, 300000);
  gen_linspaced(pts, -0.001, 0.001, 200000);
  gen_random(pts, -700.0, 700.0, 200000, 9);
  for (double x : pts) {
    s.record(x, coremath::sinh(x), std::sinh(x));
  }
  return s;
}

static Stats test_erf() {
  Stats s; s.name = "erf";
  std::vector<double> pts;
  gen_linspaced(pts, -6.0, 6.0, 500000);
  gen_linspaced(pts, -0.001, 0.001, 200000);
  gen_linspaced(pts, -1.0, 1.0, 200000);
  gen_random(pts, -10.0, 10.0, 200000, 10);
  for (double x : pts) {
    s.record(x, coremath::erf(x), std::erf(x));
  }
  return s;
}

static Stats test_erfc() {
  Stats s; s.name = "erfc";
  std::vector<double> pts;
  gen_linspaced(pts, -6.0, 30.0, 500000);
  gen_linspaced(pts, 0.0, 1.0, 300000);
  gen_linspaced(pts, 1.0, 6.0, 200000);
  gen_random(pts, -5.0, 27.0, 200000, 11);
  for (double x : pts) {
    s.record(x, coremath::erfc(x), std::erfc(x));
  }
  return s;
}

static Stats test_lgamma() {
  Stats s; s.name = "lgamma";
  std::vector<double> pts;
  gen_logspaced(pts, 1e-10, 1e+10, 500000);
  gen_linspaced(pts, 0.5, 10.0, 300000);
  gen_linspaced(pts, 0.001, 0.5, 200000);
  gen_random_log(pts, 1e-5, 1e+5, 200000, 12);
  // Negative non-integer values
  for (int k = 1; k <= 50; k++) {
    gen_linspaced(pts, -(double)k - 0.999, -(double)k - 0.001, 5000);
  }
  for (double x : pts) {
    s.record(x, coremath::lgamma(x), std::lgamma(x));
  }
  return s;
}

static Stats test_tgamma() {
  Stats s; s.name = "tgamma";
  std::vector<double> pts;
  gen_linspaced(pts, 0.01, 171.0, 500000);
  gen_linspaced(pts, 0.001, 2.0, 300000);
  gen_linspaced(pts, -170.0, -0.01, 200000);
  // Avoid exact negative integers
  gen_random(pts, 0.1, 170.0, 200000, 13);
  for (double x : pts) {
    if (x <= 0.0 && x == std::floor(x)) continue; // skip poles
    s.record(x, coremath::tgamma(x), std::tgamma(x));
  }
  return s;
}

static Stats test_pow() {
  Stats s; s.name = "pow";
  // Grid of (x, y) pairs
  std::vector<double> bases, exponents;
  gen_logspaced(bases, 1e-10, 1e10, 500);
  gen_linspaced(exponents, -20.0, 20.0, 500);

  for (double x : bases) {
    for (double y : exponents) {
      double ours = coremath::pow(x, y);
      double ref = std::pow(x, y);
      if (std::isfinite(ours) && std::isfinite(ref)) {
        s.record(x, ours, ref, y);
      }
    }
  }

  // Near x=1
  gen_linspaced(bases, 0.99, 1.01, 500);
  gen_linspaced(exponents, -1000.0, 1000.0, 500);
  for (double x : bases) {
    for (double y : exponents) {
      double ours = coremath::pow(x, y);
      double ref = std::pow(x, y);
      if (std::isfinite(ours) && std::isfinite(ref)) {
        s.record(x, ours, ref, y);
      }
    }
  }

  // Random pairs
  std::mt19937_64 rng(14);
  std::uniform_real_distribution<double> base_dist(1e-5, 1e5);
  std::uniform_real_distribution<double> exp_dist(-50.0, 50.0);
  for (size_t i = 0; i < 200000; i++) {
    double x = base_dist(rng), y = exp_dist(rng);
    double ours = coremath::pow(x, y);
    double ref = std::pow(x, y);
    if (std::isfinite(ours) && std::isfinite(ref)) {
      s.record(x, ours, ref, y);
    }
  }
  return s;
}

// ============================================================
// Main
// ============================================================

int main() {
  printf("CORE-MATH Fast-Path Validation\n");
  printf("Comparing coremath:: vs std:: (glibc)\n");
  printf("=============================================\n\n");

  Stats results[] = {
    test_exp(),
    test_expm1(),
    test_log(),
    test_log1p(),
    test_sin(),
    test_cos(),
    test_asin(),
    test_acos(),
    test_atan(),
    test_atan2(),
    test_sinh(),
    test_erf(),
    test_erfc(),
    test_lgamma(),
    test_tgamma(),
    test_pow(),
  };

  bool two_arg[] = {
    false, false, false, false, false, false,
    false, false, false, true, false, false,
    false, false, false, true
  };

  printf("%-12s %12s %10s %10s %8s %8s %8s %8s %8s\n",
         "Function", "Points", "Max ULP", "Mean ULP",
         ">1", ">2", ">4", ">10", ">100");
  printf("%-12s %12s %10s %10s %8s %8s %8s %8s %8s\n",
         "--------", "------", "-------", "--------",
         "--", "--", "--", "---", "----");

  for (size_t i = 0; i < sizeof(results)/sizeof(results[0]); i++) {
    const Stats& s = results[i];
    printf("%-12s %12zu %10llu %10.2f %8zu %8zu %8zu %8zu %8zu",
           s.name, s.count, (unsigned long long)s.max_ulp,
           s.count > 0 ? (double)s.total_ulp / s.count : 0.0,
           s.gt1, s.gt2, s.gt4, s.gt10, s.gt100);
    if (s.nan_mismatch > 0) printf("  NaN=%zu", s.nan_mismatch);
    if (s.inf_mismatch > 0) printf("  Inf=%zu", s.inf_mismatch);
    printf("\n");
  }

  printf("\nDetailed worst cases:\n");
  for (size_t i = 0; i < sizeof(results)/sizeof(results[0]); i++) {
    results[i].print(two_arg[i]);
  }

  return 0;
}
