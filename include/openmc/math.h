#ifndef OPENMC_MATH_H
#define OPENMC_MATH_H

/*
 * Portable transcendental math functions for OpenMC.
 *
 * All OpenMC code should call these via the openmc:: namespace (e.g.,
 * openmc::sin, openmc::exp). This header maps them to the CORE-MATH
 * fast-path implementations, which produce bit-identical results across
 * all platforms and compilers.
 *
 * To switch to standard library or a vendor math library, replace the
 * include and using declarations below. For example:
 *
 *   #include <cmath>
 *   namespace openmc {
 *   using std::exp;
 *   using std::log;
 *   ...
 *   }
 */

#include "openmc/coremath.h"

namespace openmc {
using coremath::exp;
using coremath::expm1;
using coremath::log;
using coremath::log1p;
using coremath::pow;
using coremath::sin;
using coremath::cos;
using coremath::asin;
using coremath::acos;
using coremath::atan;
using coremath::atan2;
using coremath::sinh;
using coremath::erf;
using coremath::erfc;
using coremath::lgamma;
using coremath::tgamma;
} // namespace openmc

#endif // OPENMC_MATH_H
