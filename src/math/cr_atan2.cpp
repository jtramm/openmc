// Fast-path atan2(y,x) for binary64 values.
// Derived from the CORE-MATH project (MIT License).
// Original authors: Paul Zimmermann and Alexei Sibidanov.
// https://core-math.gitlabpages.inria.fr/

#include "coremath_utils.h"

// PI_H+PI_L approximates pi with error bounded by 2^-108.041
static const double PI_H = 0x1.921fb54442d18p+1;
static const double PI_L = 0x1.1a62633145c07p-53;
// PI_OVER2_H+PI_OVER2_L approximates pi/2 with error bounded by 2^-109.041
static const double PI_OVER2_H = 0x1.921fb54442d18p+0;
static const double PI_OVER2_L = 0x1.1a62633145c07p-54;

static const u64 MASK = 0x7fffffffffffffffull; // 2^63-1 (mask the sign bit)

static double as_atan2_special(double y0, double x0){
  b64u64_u iy, ix;
  iy.f = y0;
  ix.f = x0;
  u64 aiy = iy.u<<1, aix = ix.u<<1;

  if (aiy >= 0x7ffull<<53 || aix >= 0x7ffull<<53){ // NaN or Inf
    if (aiy > 0x7ffull<<53 || aix > 0x7ffull<<53)
      return y0 + x0;
    // Now neither y nor x is NaN, but at least one is +Inf or -Inf
    if (aiy == 0x7ffull<<53 && aix == 0x7ffull<<53){ // both y and x are +/-Inf
      static const double finf[][2] = {{0x1p-55, 0x1.921fb54442d18p-1}, {0x1p-54, 0x1.2d97c7f3321d2p+1}};
      // atan2 (+/-Inf,-Inf) = +/-3pi/4
      // atan2 (+/-Inf,+Inf) = +/-pi/4
      return std::copysign(finf[ix.u>>63][1], y0) + std::copysign(finf[ix.u>>63][0], y0);
    }
    // now only one of y and x is +/-Inf
    if (aix == 0x7ffull<<53) {
      if (x0 < 0)
        return std::copysign(PI_H, y0) + std::copysign(PI_L, y0);
      // atan2(+/-0,x) = +/-0 for x > 0
      // atan2(+/-y,+Inf) = +/-0 for finite y>0
      return std::copysign(0.0, y0);
    }
    // now y = +/-Inf
    // atan2(+/-Inf,x) = +/-pi/2 for finite x
    return std::copysign(PI_OVER2_H, y0) + std::copysign(PI_OVER2_L, y0);
  }

  if (aiy == 0 || aix == 0){
    if (aiy == 0 && aix == 0){
      if (ix.u == 0) // atan2(+/-0, +0) = +/-0
        return y0;
      // atan2(+/-0, -0) = +/-pi
      return (iy.u == 0) ? PI_H + PI_L : -PI_H - PI_L;
    }
    // only one of y and x is zero
    if (aiy==0){
      // atan2(+/-0,x) = +/-0 for x>0
      if (x0 > 0) return y0/x0;
      // atan2(+/-0,x) = +/-pi for x<0
      return (!(iy.u>>63)) ? PI_H + PI_L : -PI_H - PI_L;
    }
    // now only x is zero
    // atan2(y,+/-0) = -pi/2 for y<0
    // atan2(y,+/-0) = +pi/2 for y>0
    return (y0 > 0) ? PI_OVER2_H + PI_OVER2_L : -PI_OVER2_H - PI_OVER2_L;
  }
  return 0;
}

double cr_atan2(double y0, double x0){
  static const double asgn[2] = {0.0, -0.0};
  static const double T2[] = {
    0x0p+0, 0x1p-6, 0x1p-5, 0x1.8p-5, 0x1p-4, 0x1.4p-4, 0x1.8p-4, 0x1.cp-4,
    0x1p-3, 0x1.2p-3, 0x1.4p-3, 0x1.6p-3, 0x1.8p-3, 0x1.ap-3, 0x1.cp-3, 0x1.ep-3,
    0x1p-2, 0x1.1p-2, 0x1.2p-2, 0x1.3p-2, 0x1.4p-2, 0x1.5p-2, 0x1.6p-2, 0x1.7p-2,
    0x1.8p-2, 0x1.9p-2, 0x1.ap-2, 0x1.bp-2, 0x1.cp-2, 0x1.dp-2, 0x1.ep-2, 0x1.fp-2,
    0x1p-1, 0x1.08p-1, 0x1.1p-1, 0x1.18p-1, 0x1.2p-1, 0x1.28p-1, 0x1.3p-1, 0x1.38p-1,
    0x1.4p-1, 0x1.48p-1, 0x1.5p-1, 0x1.58p-1, 0x1.6p-1, 0x1.68p-1, 0x1.7p-1, 0x1.78p-1,
    0x1.8p-1, 0x1.88p-1, 0x1.9p-1, 0x1.98p-1, 0x1.ap-1, 0x1.a8p-1, 0x1.bp-1, 0x1.b8p-1,
    0x1.cp-1, 0x1.c8p-1, 0x1.dp-1, 0x1.d8p-1, 0x1.ep-1, 0x1.e8p-1, 0x1.fp-1, 0x1.f8p-1, 0x1p+0};
  static const double f2[][2] = {
    {0x0p+0, 0x0p+0}, {-0x1.95220c39d4dffp-53, 0x1.fff555bbb73p-7},
    {0x1.2542779d776dep-53, 0x1.ffd55bba976p-6}, {-0x1.6061bbe3de53cp-53, 0x1.7fb818430da4p-5},
    {-0x1.639269b0da47ep-53, 0x1.ff55bb72cfep-5}, {-0x1.4a7663af440f7p-55, 0x1.3f59f0e7c55ap-4},
    {0x1.d1824d59f9e13p-53, 0x1.7ee182602f1p-4}, {0x1.bef71e5340b31p-55, 0x1.be39ebe6f07cp-4},
    {0x1.b8cb225e627dp-53, 0x1.fd5ba9aac2f6p-4}, {0x1.b92de9bac94c2p-53, 0x1.1e1fafb04372p-3},
    {-0x1.d3cb89e62dafdp-54, 0x1.3d6eee8c6627p-3}, {-0x1.882a55960087ap-53, 0x1.5c9811e3ec27p-3},
    {0x1.1347b0b4f881dp-54, 0x1.7b97b4bce5bp-3}, {0x1.873d8079ed0d2p-53, 0x1.9a6a8e96c862p-3},
    {0x1.022f621a5c1cbp-54, 0x1.b90d7529260ap-3}, {0x1.9c648d1534598p-53, 0x1.d77d5df20573p-3},
    {-0x1.4ea9238610a08p-54, 0x1.f5b75f92c80ep-3}, {0x1.2c5c8e721970dp-53, 0x1.09dc597d8636p-2},
    {0x1.30ca4748b1bf9p-57, 0x1.18bf5a30bf178p-2}, {-0x1.20ef9ba6dbf9p-53, 0x1.278372057ef48p-2},
    {0x1.e69c5abb498d2p-53, 0x1.362773707ebc8p-2}, {0x1.a8a86f0ea9311p-54, 0x1.44aa436c2af08p-2},
    {0x1.db5336feef7fp-54, 0x1.530ad9951cd48p-2}, {0x1.9636a3aa3b84p-54, 0x1.614840309cfep-2},
    {0x1.1ce2a8c848b74p-55, 0x1.6f61941e4defp-2}, {-0x1.4b1bbd1ea6db3p-55, 0x1.7d5604b63b3f8p-2},
    {-0x1.4925e8b916e0bp-53, 0x1.8b24d394a1b28p-2}, {0x1.9e6c988fd0a77p-56, 0x1.98cd5454d6b18p-2},
    {-0x1.a49bd836a17p-53, 0x1.a64eec3cc24p-2}, {-0x1.ca3cf09c6b5f8p-53, 0x1.b3a911da65c7p-2},
    {-0x1.cc1ce70934c34p-56, 0x1.c0db4c94ec9fp-2}, {0x1.2e982ddf3872ap-55, 0x1.cde53432c135p-2},
    {-0x1.2ea406ee84d0fp-55, 0x1.dac670561bb5p-2}, {-0x1.de35847c81979p-53, 0x1.e77eb7f175a38p-2},
    {-0x1.a3992dc382a23p-57, 0x1.f40dd0b541418p-2}, {-0x1.b32c949c9d593p-55, 0x1.0039c73c1a40cp-1},
    {-0x1.d5b495f6349e6p-56, 0x1.0657e94db30dp-1}, {-0x1.f34582f6255fep-53, 0x1.0c6145b5b43dcp-1},
    {0x1.ed42511e3f11dp-54, 0x1.1255d9bfbd2a8p-1}, {-0x1.1cef189ff9e7fp-54, 0x1.1835a88be7c14p-1},
    {-0x1.928df287a668fp-58, 0x1.1e00babdefeb4p-1}, {-0x1.e3bde360c7ddbp-53, 0x1.23b71e2cc9e6cp-1},
    {0x1.bd86313ce4fdep-54, 0x1.2958e59308e3p-1}, {-0x1.8e8a85803cc1dp-53, 0x1.2ee628406cbccp-1},
    {-0x1.77ef7641c777fp-54, 0x1.345f01cce37bcp-1}, {0x1.b73ef3389d02fp-53, 0x1.39c391cd41718p-1},
    {0x1.ecf8b492644fp-56, 0x1.3f13fb89e96f4p-1}, {0x1.c1125fd3810c7p-53, 0x1.445065b795b54p-1},
    {0x1.2483350fe548bp-53, 0x1.4978fa3269eep-1}, {0x1.4a33dbeb3796cp-55, 0x1.4e8de5bb6ec04p-1},
    {-0x1.46edd2af69483p-53, 0x1.538f57b89062p-1}, {-0x1.2bcb93b18b52ap-53, 0x1.587d81f732fbcp-1},
    {0x1.0028e4bc5e7cap-57, 0x1.5d58987169b18p-1}, {0x1.ed487acaf1174p-53, 0x1.6220d115d7b8cp-1},
    {-0x1.2dd4dfd7d1777p-53, 0x1.66d663923e088p-1}, {0x1.2bfe3cf3b9d79p-54, 0x1.6b798920b3d98p-1},
    {-0x1.8c34d25aadef6p-56, 0x1.700a7c5784634p-1}, {-0x1.f426acf4d3bdbp-54, 0x1.748978fba8e1p-1},
    {-0x1.afe57dd9ff23p-53, 0x1.78f6bbd5d316p-1}, {-0x1.54fbef0e862abp-54, 0x1.7d528289fa094p-1},
    {0x1.90227758b11bap-54, 0x1.819d0b7158a4cp-1}, {0x1.16b66e7fc8b8cp-53, 0x1.85d69576cc2c4p-1},
    {-0x1.55b9a5e177a1bp-55, 0x1.89ff5ff57f1f8p-1}, {0x1.c27cfaa9f7a14p-53, 0x1.8e17aa99cc05cp-1},
    {0x1.1a62633145c07p-55, 0x1.921fb54442d18p-1}};
  static const double O[8][2] = {
    {0,0}, {0x1.921fb54442d18p+0,0x1.1a62633145c07p-54},
    {0,0}, {-0x1.921fb54442d18p+0,-0x1.1a62633145c07p-54},
    {0x1.921fb54442d18p+1,0x1.1a62633145c07p-53}, {0x1.921fb54442d18p+0,0x1.1a62633145c07p-54},
    {-0x1.921fb54442d18p+1,-0x1.1a62633145c07p-53}, {-0x1.921fb54442d18p+0,-0x1.1a62633145c07p-54}};

  b64u64_u iy, ix;
  iy.f = y0;
  ix.f = x0;
  u64 aiy = iy.u & MASK;
  if (aiy==0 || aiy>=0x7ffull<<52) return as_atan2_special(y0,x0);
  u64 aix = ix.u & MASK;
  if (aix==0 || aix>=0x7ffull<<52) return as_atan2_special(y0,x0);
  double ax = std::fabs(x0), ay = std::fabs(y0);
  double x = std::fmax(ax, ay), y = std::fmin(ax, ay);
  u64 sy = iy.u>>63, sx = ix.u>>63;
  u64 GT = aix<aiy;
  u64 dxy = (aix-aiy)^-GT;
  if (dxy>=53ull<<52){
    // exponent difference too large: return fast approximation directly
    // For very large or very small y/x, atan2 is close to 0 or pi/2 or pi
    // We compute via the refined mid-path below which will converge
  }
  b64u64_u sgn;
  sgn.f = asgn[GT^sx^sy];
  u64 kw = sx<<2|sy<<1|GT;
  b64u64_u jj;
  jj.f = y/x + (2 + 1/128.);
  i64 jt = ((jj.u>>(52-7))&127);
  double fh = f2[jt][1]*std::copysign(1.0,sgn.f);
  double fl = f2[jt][0]*std::copysign(1.0,sgn.f);
  fh += O[kw][0];
  fl += O[kw][1];
  if (x<0x1p-968){x *= 0x1p968; y *= 0x1p968;}
  if (x>0x1p1022){
    if (jt != 0){
      x *= 0x1p-1; y *= 0x1p-1;
    }
  }
  double t0 = T2[jt];
  double zn = std::fma(-t0,x,y), zd = std::fma(t0,y,x);
  double z = zn/zd;
  static const double b[] = {-0x1.55555555554d2p-2, 0x1.999999860e1cap-3, -0x1.248ad469844a1p-3};
  double z2 = z*z;
  z *= std::copysign(1.0,sgn.f);
  double dz = (z*z2)*(b[0] + z2*(b[1] + z2*b[2]));
  double eps = std::fabs(z)*0x1.051p-51 + 0x1p-90;
  double rh = fasttwosum(fh, z, &z);
  double rl = (fl + dz) + z;
  double lb = rh + (rl - eps), ub = rh + (rl + eps);
  if(lb!=ub){
    // Refined mid-path: use higher-precision argument reduction
    double dh = y*t0, dl = std::fma(y,t0,-dh), e, rdh;
    dh = fasttwosum(x, dh, &e);
    rdh = 1/dh;
    dl += e;
    double nh = x*t0, nl = std::fma(x,t0,-nh);
    double dt = y-nh, y1 = dt+nh;
    if (y1 == y){
      nh = fasttwosum(dt, -nl, &nl);
    } else {
      nh = fasttwosum(dt, (y - y1) - nl, &nl);
    }
    double zh = nh * rdh;
    z2 = zh*zh;
    double zl = rdh * (std::fma(dh, -zh, nh) + (nl - (nh*rdh)*dl));
    static const double b2[] =
      {-0x1.5555555555555p-2, 0x1.999999999755ep-3, -0x1.24924883596f8p-3, 0x1.c6f7d73531bc2p-4};
    zl += zh*z2*((b2[0] + z2*b2[1]) + (z2*z2)*(b2[2] + z2*b2[3]));
    zh *= std::copysign(1.0,sgn.f);
    zl *= std::copysign(1.0,sgn.f);
    fh = fastsum(fh,fl,zh,zl,&fl);
    ub = fh + fl;
  }
  return ub;
}
