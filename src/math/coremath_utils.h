// Shared utilities for CORE-MATH fast-path implementations.
// Derived from the CORE-MATH project (MIT License).
// https://core-math.gitlabpages.inria.fr/

#ifndef COREMATH_UTILS_H
#define COREMATH_UTILS_H

#include <cmath>
#include <cstdint>
#include <cfloat>

// Bit-level reinterpretation between double and uint64_t
union b64u64_u {
  double f;
  uint64_t u;
};

typedef int64_t i64;
typedef uint64_t u64;

// Round to nearest integer, breaking ties to even.
// Uses the 2^52 magic-number trick — purely IEEE 754 basic operations.
// Named coremath_roundeven to avoid collision with glibc's roundeven.
static inline double coremath_roundeven(double x) {
  double magic = std::copysign(0x1p52, x);
  return (std::fabs(x) < 0x1p52) ? (x + magic) - magic : x;
}

// Scale x by 2^i via exponent manipulation (assumes normal result)
static inline double as_ldexp(double x, i64 i) {
  b64u64_u ix;
  ix.f = x;
  ix.u += (uint64_t)i << 52;
  return ix.f;
}

// Set exponent to 0 (subnormal range)
static inline double as_todenormal(double x) {
  b64u64_u ix;
  ix.f = x;
  ix.u &= ~(u64)0 >> 12;
  return ix.f;
}

// Error-free addition: returns s = x + y, *e = error such that x + y = s + e exactly
static inline double fasttwosum(double x, double y, double *e) {
  double s = x + y, z = s - x;
  *e = y - z;
  return s;
}

// Double-double addition: (xh+xl) + (yh+yl) → (sh, *e)
static inline double fastsum(double xh, double xl, double yh, double yl, double *e) {
  double sl, sh = fasttwosum(xh, yh, &sl);
  *e = (xl + yl) + sl;
  return sh;
}

// Double-double multiply: (xh+xl) * (ch+cl) → (result, *l)
static inline double muldd(double xh, double xl, double ch, double cl, double *l) {
  double ahhh = ch * xh;
  *l = (ch * xl + cl * xh) + std::fma(ch, xh, -ahhh);
  return ahhh;
}

// Evaluate odd polynomial in double-double
static inline double opolydd(double xh, double xl, int n, const double c[][2], double *l) {
  int i = n - 1;
  double ch = c[i][0], cl = c[i][1];
  while (--i >= 0) {
    ch = muldd(xh, xl, ch, cl, &cl);
    double th = ch + c[i][0], tl = (c[i][0] - th) + ch;
    ch = th;
    cl += tl + c[i][1];
  }
  *l = cl;
  return ch;
}

// Portable 64x64 → 128-bit multiply using 32-bit schoolbook method.
// Returns high and low 64-bit halves.
static inline void mul64x64(uint64_t a, uint64_t b, uint64_t &hi, uint64_t &lo) {
  uint64_t a_lo = a & 0xFFFFFFFF, a_hi = a >> 32;
  uint64_t b_lo = b & 0xFFFFFFFF, b_hi = b >> 32;
  uint64_t p0 = a_lo * b_lo;
  uint64_t p1 = a_lo * b_hi;
  uint64_t p2 = a_hi * b_lo;
  uint64_t p3 = a_hi * b_hi;
  uint64_t mid = (p0 >> 32) + (p1 & 0xFFFFFFFF) + (p2 & 0xFFFFFFFF);
  lo = (mid << 32) | (p0 & 0xFFFFFFFF);
  hi = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
}

// Exact multiply: *h = a*b (rounded), *l = exact error
static inline void a_mul(double *h, double *l, double a, double b) {
  *h = a * b;
  *l = std::fma(a, b, -*h);
}

// Scale multiply: *h = s*(xh+xl) approximately, *l captures low bits
static inline void s_mul(double *h, double *l, double s, double xh, double xl) {
  *h = s * xh;
  *l = std::fma(s, xh, -*h) + s * xl;
}

#endif // COREMATH_UTILS_H
