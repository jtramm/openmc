#ifndef OPENMC_COREMATH_H
#define OPENMC_COREMATH_H

//! \file coremath.h
//! \brief Correctly-rounded math functions from the CORE-MATH project.
//!
//! These functions provide bit-for-bit identical results across all platforms
//! and compilers, eliminating libm-dependent differences in transcendental
//! function implementations (e.g., glibc vs Apple libSystem).
//!
//! Source: https://core-math.gitlabpages.inria.fr/
//! License: MIT

double cr_exp(double);
double cr_expm1(double);
double cr_log(double);
double cr_log1p(double);
double cr_pow(double, double);
double cr_sin(double);
double cr_cos(double);
double cr_asin(double);
double cr_acos(double);
double cr_atan(double);
double cr_atan2(double, double);
double cr_sinh(double);
double cr_erf(double);
double cr_erfc(double);
double cr_lgamma(double);
double cr_tgamma(double);

namespace coremath {
  inline double exp(double x) { return cr_exp(x); }
  inline double expm1(double x) { return cr_expm1(x); }
  inline double log(double x) { return cr_log(x); }
  inline double log1p(double x) { return cr_log1p(x); }
  inline double pow(double x, double y) { return cr_pow(x, y); }
  inline double sin(double x) { return cr_sin(x); }
  inline double cos(double x) { return cr_cos(x); }
  inline double asin(double x) { return cr_asin(x); }
  inline double acos(double x) { return cr_acos(x); }
  inline double atan(double x) { return cr_atan(x); }
  inline double atan2(double y, double x) { return cr_atan2(y, x); }
  inline double sinh(double x) { return cr_sinh(x); }
  inline double erf(double x) { return cr_erf(x); }
  inline double erfc(double x) { return cr_erfc(x); }
  inline double lgamma(double x) { return cr_lgamma(x); }
  inline double tgamma(double x) { return cr_tgamma(x); }
} // namespace coremath

#endif // OPENMC_COREMATH_H
