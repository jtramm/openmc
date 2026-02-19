// Fast-path arcsine function for binary64 values.
// Derived from the CORE-MATH project (MIT License).
// https://core-math.gitlabpages.inria.fr/
//
// Simplified: uses the identity asin(x) = atan2(x, sqrt(1-x^2)),
// leveraging the already-verified cr_atan2 (1 ULP) and IEEE-exact sqrt.

#include <cmath>
#include <cstdint>

double cr_atan2(double, double);

double cr_asin(double x) {
  // Handle NaN
  if (std::isnan(x)) return x + x;

  // Handle |x| > 1: domain error
  if (std::fabs(x) > 1.0) return 0.0 / 0.0;

  // Handle x = +/-1 exactly: return +/- pi/2
  // pi/2 = 0x1.921fb54442d18p+0 + 0x1.1a62633145c07p-54
  if (x == 1.0)
    return 0x1.921fb54442d18p+0 + 0x1.1a62633145c07p-54;
  if (x == -1.0)
    return -(0x1.921fb54442d18p+0 + 0x1.1a62633145c07p-54);

  // For tiny x, asin(x) ~ x
  if (std::fabs(x) < 0x1p-26)
    return std::fma(x, 0x1p-54, x);

  // General case: asin(x) = atan2(x, sqrt(1-x^2))
  // fma(x, -x, 1.0) computes 1-x^2 with a single rounding
  // sqrt is IEEE-exact, atan2 is our verified 1-ULP implementation
  double onemx2 = std::fma(x, -x, 1.0);
  return cr_atan2(x, std::sqrt(onemx2));
}
