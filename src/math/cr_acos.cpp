// Fast-path arccosine function for binary64 values.
// Derived from the CORE-MATH project (MIT License).
// https://core-math.gitlabpages.inria.fr/
//
// Simplified: uses the identity acos(x) = atan2(sqrt(1-x^2), x),
// leveraging the already-verified cr_atan2 (1 ULP) and IEEE-exact sqrt.

#include <cmath>
#include <cstdint>

double cr_atan2(double, double);

double cr_acos(double x) {
  // Handle NaN
  if (std::isnan(x)) return x + x;

  // Handle |x| > 1: domain error
  if (std::fabs(x) > 1.0) return 0.0 / 0.0;

  // Handle x = 1 exactly: return 0
  if (x == 1.0) return 0.0;

  // Handle x = -1 exactly: return pi
  // pi = 0x1.921fb54442d18p+1 + 0x1.1a62633145c07p-53
  if (x == -1.0)
    return 0x1.921fb54442d18p+1 + 0x1.1a62633145c07p-53;

  // General case: acos(x) = atan2(sqrt(1-x^2), x)
  double onemx2 = std::fma(x, -x, 1.0);
  return cr_atan2(std::sqrt(onemx2), x);
}
