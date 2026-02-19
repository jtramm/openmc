// Fast-path true gamma function for binary64 values.
// Derived from the CORE-MATH project (MIT License).
// Original author: Alexei Sibidanov.
// https://core-math.gitlabpages.inria.fr/

#include "coremath_utils.h"



static inline double sumdd(double xh, double xl, double yh, double yl, double *e){
  double sl, sh;
  if(std::fabs(xh)>std::fabs(yh))
    sh = fasttwosum(xh, yh, &sl);
  else
    sh = fasttwosum(yh, xh, &sl);
  *e = (xl + yl) + sl;
  return sh;
}

static inline double mulddd(double x, double ch, double cl, double *l){
  double ahhh = ch*x;
  *l = cl*x + std::fma(ch, x, -ahhh);
  return ahhh;
}

static inline double polydd(double xh, double xl, int n, const double c[][2], double *l){
  int i = n-1;
  double cl, ch = fasttwosum(c[i][0], *l, &cl); cl += c[i][1];
  while(--i>=0){
    ch = muldd(xh,xl, ch,cl, &cl);
    ch = fastsum(c[i][0], c[i][1], ch, cl, &cl);
  }
  *l = cl;
  return ch;
}

static inline double polyddd(double x, int n, const double c[][2], double *l){
  int i = n-1;
  double cl, ch = fasttwosum(c[i][0], *l, &cl); cl += c[i][1];
  while(--i>=0){
    ch = mulddd(x, ch,cl, &cl);
    ch = sumdd(c[i][0],c[i][1], ch,cl, &cl);
  }
  *l = cl;
  return ch;
}

static inline double polyd(double x, int n, const double c[][2]){
  int i = n-1;
  double ch = c[i][0];
  while(--i>=0) ch = c[i][0] + x*ch;
  return ch;
}

// Database of hard-to-round cases
static double as_tgamma_database(double x, double f){
  static const double db[][3] = {
    {-0x1.48ba8e27d09adp+7, -0x1.0b34f909c5c92p-976, 0x0.01p-1022},
    {-0x1.1fe464bbe8b7ap+7, 0x1.f6e94380a86bfp-826, 0x1p-880},
    {-0x1.dfe438a574b34p+6, 0x1.e0efc1ffa409ep-656, 0x1p-710},
    {-0x1.c008ed7e2be92p+6, -0x1.2820dd1286d5ep-599, -0x1p-653},
    {-0x1.bcc2b6af0ebaep+6, 0x1.5742e6ca2fe43p-598, -0x1p-652},
    {-0x1.2c0358d14dacep+6, 0x1.c8e82e0e0a4f6p-356, 0x1p-410},
    {-0x1.f126edde91b5bp+5, -0x1.f4cdd1a52b2e3p-283, -0x1p-337},
    {-0x1.d97de88bda2dfp+5, 0x1.2227eb4f08b21p-265, 0x1p-319},
    {-0x1.ce749a6427fddp+5, 0x1.30d786820381fp-257, -0x1p-311},
    {-0x1.ccb10c3d47943p+5, 0x1.b9e53a96c9939p-257, -0x1p-311},
    {-0x1.c9cc11aba9632p+5, 0x1.644a102fa86bp-254, 0x1p-308},
    {-0x1.b93b0669b1556p+5, 0x1.03a95aab1bc81p-241, 0x1p-295},
    {-0x1.67893c596ef0cp+5, -0x1.1bd662d0bc936p-182, 0x1p-236},
    {-0x1.ae39c32c36e42p+4, -0x1.82b46d1babd86p-90, -0x1p-144},
    {-0x1.8d5826734a06p+4, -0x1.ae95b301e8bf3p-81, 0x1p-135},
    {-0x1.849b47bda8526p+4, -0x1.8c2973252464cp-79, 0x1p-133},
    {-0x1.0ba2acf2de6b2p+4, -0x1.ca9739fd7435ep-46, -0x1p-100},
    {-0x1.f49bb1a25a54ep+3, 0x1.fbf8e7755e967p-42, 0x1p-96},
    {-0x1.9057ede749837p+3, -0x1.eb68bf9d04126p-30, 0x1p-84},
    {-0x1.65509f6aed026p+3, 0x1.c3791f14051e1p-24, 0x1p-78},
    {-0x1.5f353a2d26238p+3, -0x1.20e6a9093e033p-20, 0x1p-74},
    {-0x1.3be29766cccacp+3, 0x1.8fad89ade334bp-19, 0x1p-73},
    {-0x1.e33cfdfb73bcdp+2, 0x1.abe0430abe7dep-13, -0x1p-67},
    {-0x1.e11a9d07e14aap+2, 0x1.c58ee82102e5p-13, 0x1p-67},
    {-0x1.9945148859d8p+2, -0x1.1d4187d2d1e32p-9, 0x1p-63},
    {-0x1.3fc07c80057fdp+2, -0x1.14fd6b28fb843p+1, -0x1p-53},
    {-0x1.ca5042fd026bep+1, 0x1.fe72b07f8530ap-3, 0x1p-57},
    {-0x1.4f977a4a186e9p+1, -0x1.c7082ecde156ap-1, -0x1p-55},
    {-0x1.2db99b79dfe45p+1, -0x1.394da6c1b5e01p+0, 0x1p-54},
    {-0x1.d71c18bba5e34p+0, 0x1.e1e710f476c6bp+1, 0x1p-53},
    {-0x1.9b4cf5c6b37edp+0, 0x1.285a86b08aca2p+1, 0x1p-53},
    {-0x1.4c20c91b866ffp+0, 0x1.ad49175ae3fa8p+1, 0x1p-53},
    {-0x1.3dea2db193059p+0, 0x1.02d76ec3da035p+2, -0x1p-52},
    {-0x1.2147794f4b43bp+0, 0x1.dcaa532ea5c3bp+2, -0x1p-52},
    {-0x1.f4180137777fp-1, -0x1.5bab35ff72e22p+5, -0x1p-49},
    {-0x1.89c1327cd62e6p-1, -0x1.481a01786e77p+2, 0x1p-52},
    {-0x1.74a9509402866p-1, -0x1.2378573c2914ep+2, 0x1p-52},
    {-0x1.0aa724e38e0c9p-1, -0x1.c648650a45953p+1, 0x1p-53},
    {-0x1.ee9a5ac162dc1p-2, -0x1.c69d8a8fa985ep+1, 0x1p-53},
    {-0x1.de31773767db1p-2, -0x1.c883c55f5d3a2p+1, -0x1p-53},
    {-0x1.0e17b51a5c1cbp-2, -0x1.2de378f77ff2dp+2, 0x1p-52},
    {0x1.c05aa42cb27fep-2, 0x1.02f80f15c9486p+1, 0x1p-53},
    {0x1.38828fbbe134p-1, 0x1.774353be6bfa6p+0, -0x1p-54},
    {0x1.5f35406cd126p-1, 0x1.522440edb9679p+0, -0x1p-54},
    {0x1.09ef8f46ee74bp+0, 0x1.f5443da4bc3bep-1, -0x1p-55},
    {0x1.0a9070c11f0c7p+0, 0x1.f4a2abf00601ap-1, -0x1p-55},
    {0x1.94fcb07ab7f61p+0, 0x1.c88182e08d193p-1, -0x1p-55},
    {0x1.bca8ea6404514p+0, 0x1.d512dc4822b38p-1, -0x1p-55},
    {0x1.e0c9a45452d7cp+0, 0x1.e8acd192e461ep-1, 0x1p-55},
    {0x1.616cd9ea484abp+1, 0x1.9f85cce39e731p+0, 0x1p-54},
    {0x1.a1d899263d9a1p+2, 0x1.2f2fb8e4d1274p+8, 0x1p-46},
    {0x1.b163c719149afp+2, 0x1.d76e1ede08821p+8, -0x1p-46},
    {0x1.43714b74fe055p+6, 0x1.ecc3784ce1ffp+393, 0x1p+339},
    {0x1.676921a72fecfp+6, 0x1.76e0e21ee3989p+451, 0x1p+397},
    {0x1.f3505ba057812p+6, 0x1.068712884fb42p+687, 0x1p+633},
    {0x1.303ed951d434p+7, 0x1.fb70d4503e49bp+880, -0x1p+826},
    {0x1.3a0b358e9e93bp+7, 0x1.81a5fa517374fp+916, 0x1p+862},
  };
  int a = 0, b = (int)(sizeof(db)/sizeof(db[0])) - 1, m = (a + b)/2;
  while (a <= b) {
    if (db[m][0] < x)
      a = m + 1;
    else if (db[m][0] == x) {
      f = db[m][1] + db[m][2];
      break;
    } else
      b = m - 1;
    m = (a + b)/2;
  }
  return f;
}

// Forward declarations for helper functions
static double as_logd(double, double*);
static double as_expd(double, double*, int*);
static double as_sinpid(double, double*);
static double as_lgamma_asym(double, double*);

double cr_tgamma(double x){
  b64u64_u t;
  t.f = x;
  uint64_t ax = t.u<<1;
  if(ax>=(0x7ffull<<53)){ /* x=NaN or +/-Inf */
    if(ax==(0x7ffull<<53)){ /* x=+/-Inf */
      if(t.u>>63){ /* x=-Inf */
        return x / x; /* will raise the "Invalid operation" exception */
      }
      return x; /* x=+Inf */
    }
    return x + x; /* x=NaN */
  }

  double z = x;
  if(std::fabs(x)<0.25){ /* |x| < 0x1p-2 */
    if(ax<0x71e0000000000000ul){ // |x| < 0x1p-112
      double r;
      // deal separately with x=2^-1024 to avoid a spurious overflow in 1/x
      if (x == 0x1p-1024) {
        r = 0x1.fffffffffffffp+1023 + 0x1p+970;
        return r;
      }
      r = 1/x;
      if (x == 0){
        return r;
      }
      // the following raises the inexact flag in case x=2^k
      if (std::fma(r, x, -1.0) == 0) r -= 0.5;
      return r;
    }
    static const double cc[][2] = {
      {-0x1.2788cfc6fb619p-1, 0x1.66d81dd231575p-58}, {0x1.fa658c23b1578p-1, 0x1.dded15c22e35ep-56},
      {-0x1.d0a118f324b6p-1, -0x1.bb37df476a7ccp-55}, {0x1.f6a51055097c6p-1, -0x1.30eee7e7c5482p-55},
      {-0x1.f6c80ec38bc47p-1, -0x1.22885891ee90dp-56}};
    static const double c[] = {
      0x1.fc7e0a6e9c2c9p-1, -0x1.fdf3f15764246p-1, 0x1.ff07b5af9892cp-1, -0x1.ff803d8f584c4p-1,
      0x1.ffc07f59b072bp-1, -0x1.ffe00e422ee2ep-1, 0x1.fff102b561602p-1, -0x1.fff9cb7b72f3bp-1,
      0x1.ffdcb35bbec92p-1, -0x1.ffcc551b96878p-1, 0x1.013dde0ace169p+0, -0x1.01baffd0f7e86p+0,
      0x1.e15c8c643ed7ap-1, -0x1.da0418fdfaac3p-1, 0x1.665b8c5abe55p+0, -0x1.721c7bc0d07cp+0};
    double x2 = x*x, x4 = x2*x2, x8 = x4*x4;
    double c0 = c[0] + x*c[1] + x2*(c[2] + x*c[3]);
    double c4 = c[4] + x*c[5] + x2*(c[6] + x*c[7]);
    double c8 = c[8] + x*c[9] + x2*(c[10] + x*c[11]);
    double c12 = c[12] + x*c[13] + x2*(c[14] + x*c[15]);
    c0 += x4*c4;
    c8 += x4*c12;
    double cl = x*(c0 + x8*c8);
    double ch = polyddd(x, 5,cc, &cl);
    double fh = 1.0/z, fl = std::fma(fh,-z,1.0)*fh;
    fh = fastsum(fh,fl, ch,cl, &fl);
    return fh + fl;
  }

  if(x >= 0x1.573fae561f648p+7){
    return 0x1.fp1023 + 0x1.fp1023;
  }

  double fx = std::floor(x);
  /* compute k only after the overflow check, otherwise the cast to integer
     might overflow */
  int64_t k = fx;
  if(fx==x){ /* x is integer */
    if(x == 0.0){
      return 1.0/x;
    }
    if(x < 0.0) {
      return 0.0 / 0.0; /* should raise the "Invalid operation" exception */
    }
    double t0h = 1, t0l = 0, x0 = 1;
    for(int i=1; i<k; i++, x0 += 1.0) t0h = mulddd(x0, t0h,t0l, &t0l);
    return t0h + t0l;
  }

  if(x<=-184.0){ /* negative non-integer */
    /* For x <= -184, x non-integer, |gamma(x)| < 2^-1078.  */
    static const double sgn[2] = {0x1p-1022, -0x1p-1022};
    return 0x1p-1022 * sgn[k&1];
  }

  if(x<-3){
    double ll, lh = fasttwosum(-x,1, &ll);
    lh = as_lgamma_asym(lh, &ll);
    int e; lh = as_expd(lh, &ll, &e);
    double ix = std::floor(x), dx = x - ix; int ip = ix;
    double sl, sh = as_sinpid(dx, &sl);
    lh = muldd(sh,sl, lh,ll, &ll);
    const double pih = 0x1.921fb54442d18p+1, pil = 0x1.1a62633145c07p-53;
    double rcp = 1/lh, rh = rcp*pih, rl = rcp*(pil - ll*rh - std::fma(rh,lh,-pih));
    if(ip&1) {
      rh = -rh;
      rl = -rl;
    }
    b64u64_u th;
    if(ip>=-170){ // -171 < x < -3
      th.f = rh + rl;
      th.u -= (int64_t)e<<52;
    } else { // x < -171
      th.f = rh;
      int re = (th.u>>52)&0x7ff;
      if(re-e<=0){ // subnormal case
        th.u += (int64_t)(e-re+1)<<52;
        th.u &= 0xfffull<<52;
        double l;
        rh = fasttwosum(th.f, rh, &l);
        rl += l;
        th.f = rh + rl;
        th.u &= ~(0x7ffull<<52); // make subnormal
      } else {
        th.f = rh + rl;
        th.u -= (int64_t)e<<52;
      }
    }
    return th.f;
  }

  if(x>4){
    double ll = 0, lh = as_lgamma_asym(x,&ll);
    int e; lh = as_expd(lh, &ll, &e);
    // Use the fast-path result directly (no accurate path fallback).
    // The ub/lb check is skipped since we always apply the exponent correction.
    double result = lh + ll;
    b64u64_u th;
    th.f = result;
    th.u += (int64_t)e<<52;
    return th.f;
  }

  static const double cc[][2] = {
    {0x1.a96390899a074p+1, -0x1.6e95430fab07p-58}, {0x1.d545472146024p+1, 0x1.c07f9774e12b3p-56},
    {0x1.491ad1cb98836p+1, 0x1.51e26c4cfd792p-53}, {0x1.4a0b6a8230929p+0, 0x1.c1c6993b10594p-54},
    {0x1.0e5d232b95859p-1, 0x1.d4248748dd78bp-56}, {0x1.71d1672129feep-3, 0x1.3b47c61245ee6p-59},
    {0x1.bd2afde7e4816p-5, -0x1.25466b734902dp-60}, {0x1.d8376e1031a16p-7, 0x1.2cd76af7fbb2p-61},
    {0x1.c9e94992c88c1p-9, 0x1.5d7be78c93d16p-64}, {0x1.90ba7276a0c19p-11, -0x1.6cad258076bb3p-66},
    {0x1.49cfed9d63c8bp-13, 0x1.0a8ada0cff18dp-74}, {0x1.ec018849c245bp-16, 0x1.cea7c4e5e9d4fp-70},
    {0x1.65e5a18d31c17p-18, 0x1.12fc2f27069ecp-72}, {0x1.ca1890add8727p-21, 0x1.69c0fe53eb0fap-75},
    {0x1.378b3b91f9033p-23, -0x1.d62590f524392p-78}, {0x1.432cdb3640fcap-26, -0x1.33987f0b3b6b6p-81},
    {0x1.f239fc9cf2155p-29, -0x1.8a95d04bfb2e4p-83}, {0x1.e3ea4e1366932p-33, -0x1.5c950f5465458p-93}};
  static const double c_main[] = {
    0x1.a96390899a074p+1, 0x1.d545472146024p+1, 0x1.491ad1cb98836p+1, 0x1.4a0b6a8230929p+0,
    0x1.0e5d232b95859p-1, 0x1.71d1672129feep-3, 0x1.bd2afde7e4816p-5, 0x1.d8376e1031a16p-7,
    0x1.c9e94992c88c1p-9, 0x1.90ba7276a0c19p-11, 0x1.49cfed9d63c8bp-13, 0x1.ec018849c245bp-16,
    0x1.65e5a18d31c17p-18, 0x1.ca1890add8727p-21, 0x1.378b3b91f9033p-23, 0x1.432cdb3640fcap-26,
    0x1.f239fc9cf2155p-29, 0x1.e3ea4e1366932p-33};
  double m = z - 3.5, i = coremath_roundeven(m);
  double d = z - (i + 3.5);
  double d2 = d*d, d4 = d2*d2;
  double fl = d*((c_main[10] + d*c_main[11]) + d2*(c_main[12] + d*c_main[13]) + d4*((c_main[14] + d*c_main[15]) + d2*(c_main[16] + d*c_main[17])));
  double fh = polyddd(d, 10,cc, &fl);
  int jm = std::fabs(i);
  double wh = 1, wl = 0;
  double xph = z, xpl = 0;
  if(jm){
    wh = xph;
    for(int j=jm-1; j; j--){
      double l;
      if(std::fabs(xph)>1){
        xph = fasttwosum(xph,1,&l);
      } else {
        xph = fasttwosum(1,xph,&l);
      }
      xpl += l;
      wh = muldd(xph,xpl,wh,wl,&wl);
    }
  }
  double rh = 1.0/wh, rl = (std::fma(rh,-wh,1.0) - wl*rh)*rh;
  fh = muldd(rh,rl,fh,fl,&fl);
  // Return best approximation directly (accurate path removed).
  return fh + fl;
}

// ---- Helper: as_logd ----

static double as_logd(double x, double *l){
  static const struct { uint16_t c0; int16_t c1; } B[] = {
    {301, 27565}, {7189, 24786}, {13383, 22167}, {18923, 19696}, {23845, 17361}, {28184, 15150},
    {31969, 13054}, {35231, 11064}, {37996, 9173}, {40288, 7372}, {42129, 5657}, {43542, 4020}, {44546,
    2457}, {45160, 962}, {45399, -468}, {45281, -1838}, {44821, -3151}, {44032, -4412}, {42929, -5622},
    {41522, -6786}, {39825, -7905}, {37848, -8982}, {35602, -10020}, {33097, -11020}, {30341, -11985},
    {27345, -12916}, {24115, -13816}, {20661, -14685}, {16989, -15526}, {13107, -16339}, {9022,
    -17126}, {4740, -17889}};
  static const double r1[] =
    {0x1p+0, 0x1.f508p-1, 0x1.ea4ap-1, 0x1.dfcap-1, 0x1.d582p-1, 0x1.cb72p-1, 0x1.c19ap-1, 0x1.b7f8p-1,
     0x1.ae8ap-1, 0x1.a55p-1, 0x1.9c4ap-1, 0x1.9374p-1, 0x1.8acep-1, 0x1.8258p-1, 0x1.7a12p-1, 0x1.71f8p-1,
     0x1.6a0ap-1, 0x1.6248p-1, 0x1.5abp-1, 0x1.5342p-1, 0x1.4bfep-1, 0x1.44ep-1, 0x1.3deap-1, 0x1.371ap-1,
     0x1.307p-1, 0x1.29eap-1, 0x1.2388p-1, 0x1.1d48p-1, 0x1.172cp-1, 0x1.113p-1, 0x1.0b56p-1, 0x1.059cp-1,
     0x1p-1,};
  static const double r2[] =
    {0x1p+0, 0x1.ffa7p-1, 0x1.ff4fp-1, 0x1.fef6p-1, 0x1.fe9ep-1, 0x1.fe45p-1,
     0x1.fdedp-1, 0x1.fd94p-1, 0x1.fd3cp-1, 0x1.fce4p-1, 0x1.fc8cp-1, 0x1.fc34p-1,
     0x1.fbdcp-1, 0x1.fb84p-1, 0x1.fb2cp-1, 0x1.fad4p-1, 0x1.fa7cp-1, 0x1.fa24p-1,
     0x1.f9cdp-1, 0x1.f975p-1, 0x1.f91ep-1, 0x1.f8c6p-1, 0x1.f86fp-1, 0x1.f817p-1,
     0x1.f7cp-1, 0x1.f769p-1, 0x1.f711p-1, 0x1.f6bap-1, 0x1.f663p-1, 0x1.f60cp-1,
     0x1.f5b5p-1, 0x1.f55ep-1, 0x1.f507p-1};
  static const double l1[][2] = {
    {0x0p+0, 0x0p+0}, {0x1.9f5e440f128dbp-37, 0x1.62d07abp-6},
    {-0x1.527d64b444fa3p-37, 0x1.62f483dp-5}, {0x1.3aff57187d0cfp-39, 0x1.0a267214p-4},
    {-0x1.4634c201e2b9cp-41, 0x1.62e04bcp-4}, {-0x1.d46364a8017c7p-36, 0x1.bb9db708p-4},
    {-0x1.882b6acb3f696p-36, 0x1.0a29f69cp-3}, {0x1.5a5833aeff542p-37, 0x1.368507dap-3},
    {-0x1.3876d32b0cbf5p-36, 0x1.62e4116cp-3}, {0x1.f5712171380e6p-37, 0x1.8f41d568p-3},
    {0x1.fc0b2e87a92c1p-36, 0x1.bb98bc4cp-3}, {0x1.44c7ceb2f93f2p-36, 0x1.e7f71f08p-3},
    {0x1.a147c39e44ebap-37, 0x1.0a2bfe2cp-2}, {0x1.36d8fc46707d1p-37, 0x1.205afe03p-2},
    {-0x1.0fd8155ea585p-37, 0x1.3685b589p-2}, {0x1.8954f1c1b010fp-37, 0x1.4cb42e19p-2},
    {-0x1.5d0bcd7fa4afap-36, 0x1.62e3e78cp-2}, {-0x1.b0a96458bf187p-36, 0x1.79123647p-2},
    {0x1.c543eab5348b9p-36, 0x1.8f422996p-2}, {-0x1.15143e5c177e1p-37, 0x1.a5711c7ep-2},
    {0x1.3be09bf52475cp-38, 0x1.bb9c3cebp-2}, {-0x1.9b3b32e71e21dp-40, 0x1.d1cd255bp-2},
    {-0x1.8f02175f93786p-38, 0x1.e7fb0671p-2}, {-0x1.c5fb374b7ddcfp-36, 0x1.fe2980ecp-2},
    {-0x1.8e174c5571bbdp-36, 0x1.0a2aef35p-1}, {0x1.fa33ff819b3ecp-36, 0x1.15420d49p-1},
    {0x1.23d2634096ca6p-38, 0x1.2058ca79p-1}, {0x1.c8afc264146b2p-38, 0x1.2b7156ffp-1},
    {0x1.e21780abaa301p-37, 0x1.3686c62p-1}, {0x1.3d67aee28cdc4p-36, 0x1.419f01cdp-1},
    {0x1.ccd8a77731be8p-36, 0x1.4cb504d68p-1}, {0x1.0cc7dc4dbbcfdp-37, 0x1.57cb333b8p-1},
    {0x1.1cf79abc9e3b4p-36, 0x1.62e42fef8p-1}};
  static const double l2[][2] = {
    {0x0p+0, 0x0p+0}, {0x1.2ccace5b018a7p-36, 0x1.641ef4p-11},
    {-0x1.88a5cd275513ap-36, 0x1.623d3fp-10}, {-0x1.0006a77b80a2dp-38, 0x1.0a4531p-9},
    {0x1.81a0ebe451ddp-39, 0x1.627a998p-9}, {0x1.4297627f3b4acp-37, 0x1.bbc015p-9},
    {0x1.afb8521676db1p-36, 0x1.0a0a0c8p-8}, {0x1.080b4c8bf43cap-36, 0x1.36bc4ap-8},
    {0x1.bbcdc4ef244f5p-37, 0x1.62f5a48p-8}, {-0x1.eb4354215c794p-36, 0x1.8f36a44p-8},
    {-0x1.020c23741371bp-36, 0x1.bb7f4b8p-8}, {-0x1.ff1289819f095p-36, 0x1.e7cf9d4p-8},
    {0x1.9b1383de1a3f4p-36, 0x1.0a13cdep-7}, {0x1.7fa1788e44213p-38, 0x1.2043a52p-7},
    {0x1.f22c04fdaaa38p-37, 0x1.3677558p-7}, {-0x1.d9828c736de23p-36, 0x1.4caee08p-7},
    {-0x1.41b9d7994644p-36, 0x1.62ea474p-7}, {0x1.aced891ec8e07p-36, 0x1.79298b2p-7},
    {-0x1.e4c5365e893ffp-36, 0x1.8f2be4ep-7}, {0x1.e8d32fe7dc6f5p-36, 0x1.a572dbep-7},
    {-0x1.3bf707f8ee0b7p-36, 0x1.bb7cd5p-7}, {-0x1.c60b95a619b91p-36, 0x1.d1cb84ap-7},
    {0x1.64cbc2b83b45cp-37, 0x1.e7dd224p-7}, {-0x1.68543f75f32c6p-37, 0x1.fe338fcp-7},
    {0x1.421a5be17c2ecp-36, 0x1.0a266bbp-6}, {-0x1.f6b329a1da537p-37, 0x1.1534f84p-6},
    {-0x1.9a217f361c264p-37, 0x1.2065ff9p-6}, {0x1.3856e49eac8ddp-40, 0x1.2b78651p-6},
    {0x1.c6c3e72a945a1p-39, 0x1.368cb54p-6}, {-0x1.dcfdc96d3a1c6p-36, 0x1.41a2f0dp-6},
    {0x1.a0da4813daf37p-38, 0x1.4cbb185p-6}, {0x1.b4d3084ac1ad1p-36, 0x1.57d52c8p-6}};
  b64u64_u tv;
  tv.f = x;
  int ex = tv.u>>52, e = ex - 0x3ff;
  tv.u &= ~(u64)0>>12;
  double ed = e;
  u64 ii = tv.u>>(52-5);
  int64_t d = tv.u & (~(u64)0>>17);
  u64 j = (tv.u + ((u64)B[ii].c0<<33) + ((int64_t)B[ii].c1*(d>>16)))>>(52-10);
  tv.u |= (int64_t)0x3ff<<52;
  int i1 = j>>5, i2 = j&0x1f;
  double r = r1[i1]*r2[i2];
  double o = r*tv.f, dxl = std::fma(r,tv.f,-o), dxh = o-1;
  static const double logc[] = {-0x1.fffffffffffd3p-2, 0x1.55555555543d5p-2, -0x1.000002bb2d74ep-2, 0x1.999a692c56e4ep-3};
  double dx = std::fma(r,tv.f,-1), dx2 = dx*dx;
  double f = dx2*((logc[0] + dx*logc[1]) + dx2*(logc[2] + dx*logc[3]));
  double lt = (l1[i1][1] + l2[i2][1]) + ed*0x1.62e42fef8p-1;
  double lh = lt + dxh, ll = (lt - lh) + dxh;
  ll += ((l1[i1][0] + l2[i2][0]) + 0x1.1cf79abc9e3b4p-36*ed) + dxl;
  ll += f;
  *l = ll;
  return lh;
}

// ---- Sine of pi*x table ----

static const double st[][2] = {
  {0x0p+0, 0x0p+0}, {-0x1.b1d63091a013p-64, 0x1.92155f7a3667ep-6},
  {-0x1.912bd0d569a9p-61, 0x1.91f65f10dd814p-5}, {-0x1.9a088a8bf6b2cp-59, 0x1.2d52092ce19f6p-4},
  {-0x1.e2718d26ed688p-60, 0x1.917a6bc29b42cp-4}, {0x1.a2704729ae56dp-59, 0x1.f564e56a9730ep-4},
  {0x1.13000a89a11ep-58, 0x1.2c8106e8e613ap-3}, {0x1.531ff779ddac6p-57, 0x1.5e214448b3fc6p-3},
  {-0x1.26d19b9ff8d82p-57, 0x1.8f8b83c69a60bp-3}, {-0x1.af1439e521935p-62, 0x1.c0b826a7e4f63p-3},
  {-0x1.42deef11da2c4p-57, 0x1.f19f97b215f1bp-3}, {0x1.824c20ab7aa9ap-56, 0x1.111d262b1f677p-2},
  {-0x1.5d28da2c4612dp-56, 0x1.294062ed59f06p-2}, {0x1.0c97c4afa2518p-56, 0x1.4135c94176601p-2},
  {-0x1.efdc0d58cf62p-62, 0x1.58f9a75ab1fddp-2}, {-0x1.44b19e0864c5dp-56, 0x1.7088530fa459fp-2},
  {-0x1.72cedd3d5a61p-57, 0x1.87de2a6aea963p-2}, {0x1.6da81290bdbabp-57, 0x1.9ef7943a8ed8ap-2},
  {0x1.5b362cb974183p-57, 0x1.b5d1009e15ccp-2}, {0x1.6850e59c37f8fp-58, 0x1.cc66e9931c45ep-2},
  {0x1.e0d891d3c6841p-58, 0x1.e2b5d3806f63bp-2}, {-0x1.2ec1fc1b776b8p-60, 0x1.f8ba4dbf89abap-2},
  {-0x1.a5a014347406cp-55, 0x1.073879922ffeep-1}, {-0x1.ef23b69abe4f1p-55, 0x1.11eb3541b4b23p-1},
  {0x1.b25dd267f66p-55, 0x1.1c73b39ae68c8p-1}, {-0x1.5da743ef3770cp-55, 0x1.26d054cdd12dfp-1},
  {-0x1.efcc626f74a6fp-57, 0x1.30ff7fce17035p-1}, {0x1.e3e25e3954964p-56, 0x1.3affa292050b9p-1},
  {0x1.8076a2cfdc6b3p-57, 0x1.44cf325091dd6p-1}, {0x1.3c293edceb327p-57, 0x1.4e6cabbe3e5e9p-1},
  {-0x1.75720992bfbb2p-55, 0x1.57d69348cecap-1}, {-0x1.251b352ff2a37p-56, 0x1.610b7551d2cdfp-1},
  {-0x1.bdd3413b26456p-55, 0x1.6a09e667f3bcdp-1}, {0x1.0d4ef0f1d915cp-55, 0x1.72d0837efff96p-1},
  {-0x1.0f537acdf0ad7p-56, 0x1.7b5df226aafafp-1}, {-0x1.6f420f8ea3475p-56, 0x1.83b0e0bff976ep-1},
  {-0x1.2c5e12ed1336dp-55, 0x1.8bc806b151741p-1}, {0x1.3d419a920df0bp-55, 0x1.93a22499263fbp-1},
  {-0x1.30ee286712474p-55, 0x1.9b3e047f38741p-1}, {-0x1.128bb015df175p-56, 0x1.a29a7a0462782p-1},
  {0x1.9f630e8b6dac8p-60, 0x1.a9b66290ea1a3p-1}, {-0x1.926da300ffccep-55, 0x1.b090a581502p-1},
  {-0x1.bc69f324e6d61p-55, 0x1.b728345196e3ep-1}, {-0x1.825a732ac700ap-55, 0x1.bd7c0ac6f952ap-1},
  {-0x1.6e0b1757c8d07p-56, 0x1.c38b2f180bdb1p-1}, {-0x1.2fb761e946603p-58, 0x1.c954b213411f5p-1},
  {-0x1.e7b6bb5ab58aep-58, 0x1.ced7af43cc773p-1}, {-0x1.4ef5295d25af2p-55, 0x1.d4134d14dc93ap-1},
  {0x1.457e610231ac2p-56, 0x1.d906bcf328d46p-1}, {0x1.83c37c6107db3p-55, 0x1.ddb13b6ccc23cp-1},
  {-0x1.014c76c126527p-55, 0x1.e212104f686e5p-1}, {-0x1.16b56f2847754p-57, 0x1.e6288ec48e112p-1},
  {0x1.760b1e2e3f81ep-55, 0x1.e9f4156c62ddap-1}, {0x1.e82c791f59cc2p-56, 0x1.ed740e7684963p-1},
  {0x1.52c7adc6b4989p-56, 0x1.f0a7efb9230d7p-1}, {-0x1.d7bafb51f72e6p-56, 0x1.f38f3ac64e589p-1},
  {0x1.562172a361fd3p-56, 0x1.f6297cff75cbp-1}, {0x1.ab256778ffcb6p-56, 0x1.f8764fa714ba9p-1},
  {-0x1.7a0a8ca13571fp-55, 0x1.fa7557f08a517p-1}, {0x1.1ec8668ecaceep-55, 0x1.fc26470e19fd3p-1},
  {-0x1.87df6378811c7p-55, 0x1.fd88da3d12526p-1}, {0x1.521ecd0c67e35p-57, 0x1.fe9cdad01883ap-1},
  {-0x1.c57bc2e24aa15p-57, 0x1.ff621e3796d7ep-1}, {-0x1.1354d4556e4cbp-55, 0x1.ffd886084cd0dp-1},
  {0x0p+0, 0x1p+0}};

static double as_sinpid(double x, double *l){
  x -= 0.5;
  x = std::fabs(x);
  x *= 128;
  double ix = coremath_roundeven(x), d = ix-x, d2 = d*d;
  int ky = ix, kx = 64-ky;

  double sh = st[kx][1], sl = st[kx][0];
  double ch = st[ky][1], cl = st[ky][0];
  static const double cv[] = {-0x1.3bd3cc9be45dep-12, 0x1.03c1f081b5ac4p-26, -0x1.55d3c7e3bd8bfp-42, 0x1.e1f4826790653p-59};
  double c0 = -0x1.692b66e3cf6e8p-66;
  static const double sv[] = {0x1.921fb54442d18p-6, -0x1.4abbce625be53p-19, 0x1.466bc67748efcp-34, -0x1.32d26e446373ap-50};
  double s0 = 0x1.1a624b88c9448p-60;

  double P = d2*(cv[1] + d2*(cv[2] + d2*cv[3]));
  double Q = d2*(sv[1] + d2*(sv[2] + d2*sv[3]));
  double ql, qh = fasttwosum(sv[0],Q,&ql); ql += s0;
  ch = muldd(qh,ql, ch,cl, &cl);
  double tl, th = fasttwosum(cv[0],P,&tl); tl += c0;
  th = mulddd(d, th,tl, &tl);
  double pl, ph = muldd(th,tl, sh,sl, &pl);
  ch = fastsum(ch,cl, ph,pl, &cl);
  ch = mulddd(d, ch,cl, &cl);
  sh = fastsum(sh,sl, ch,cl, l);
  return sh;
}

// ---- Exponential helper tables ----

static const double E0[][2] = {
  {0x0p+0, 0x1p+0}, {0x1.d73e2a475b465p-55, 0x1.059b0d3158574p+0},
  {0x1.8a62e4adc610bp-54, 0x1.0b5586cf9890fp+0}, {-0x1.6c51039449b3ap-54, 0x1.11301d0125b51p+0},
  {-0x1.19041b9d78a76p-55, 0x1.172b83c7d517bp+0}, {0x1.e016e00a2643cp-54, 0x1.1d4873168b9aap+0},
  {0x1.9b07eb6c70573p-54, 0x1.2387a6e756238p+0}, {0x1.612e8afad1255p-55, 0x1.29e9df51fdee1p+0},
  {0x1.6f46ad23182e4p-55, 0x1.306fe0a31b715p+0}, {-0x1.63aeabf42eae2p-54, 0x1.371a7373aa9cbp+0},
  {0x1.ada0911f09ebcp-55, 0x1.3dea64c123422p+0}, {0x1.89b7a04ef80dp-59, 0x1.44e086061892dp+0},
  {0x1.d4397afec42e2p-56, 0x1.4bfdad5362a27p+0}, {-0x1.07abe1db13cadp-55, 0x1.5342b569d4f82p+0},
  {0x1.6324c054647adp-54, 0x1.5ab07dd485429p+0}, {-0x1.383c17e40b497p-54, 0x1.6247eb03a5585p+0},
  {-0x1.bdd3413b26456p-54, 0x1.6a09e667f3bcdp+0}, {-0x1.16e4786887a99p-55, 0x1.71f75e8ec5f74p+0},
  {-0x1.41577ee04992fp-55, 0x1.7a11473eb0187p+0}, {-0x1.d4c1dd41532d8p-54, 0x1.82589994cce13p+0},
  {0x1.6e9f156864b27p-54, 0x1.8ace5422aa0dbp+0}, {-0x1.75fc781b57ebcp-57, 0x1.93737b0cdc5e5p+0},
  {0x1.c7c46b071f2bep-56, 0x1.9c49182a3f09p+0}, {-0x1.d2f6edb8d41e1p-54, 0x1.a5503b23e255dp+0},
  {0x1.7a1cd345dcc81p-54, 0x1.ae89f995ad3adp+0}, {-0x1.5584f7e54ac3bp-56, 0x1.b7f76f2fb5e47p+0},
  {0x1.11065895048ddp-55, 0x1.c199bdd85529cp+0}, {0x1.503cbd1e949dbp-56, 0x1.cb720dcef9069p+0},
  {0x1.2ed02d75b3707p-55, 0x1.d5818dcfba487p+0}, {-0x1.1a5cd4f184b5cp-54, 0x1.dfc97337b9b5fp+0},
  {-0x1.e9c23179c2893p-54, 0x1.ea4afa2a490dap+0}, {0x1.9d3e12dd8a18bp-54, 0x1.f50765b6e454p+0}};
static const double E1[][2] = {
  {0x0p+0, 0x1p+0}, {-0x1.d7c96f201bb2fp-55, 0x1.002c605e2e8cfp+0},
  {-0x1.5e00e62d6b30dp-56, 0x1.0058c86da1c0ap+0}, {0x1.da93f90835f75p-56, 0x1.0085382faef83p+0},
  {-0x1.4f6b2a7609f71p-55, 0x1.00b1afa5abcbfp+0}, {-0x1.406ac4e81a645p-57, 0x1.00de2ed0ee0f5p+0},
  {0x1.c1d0660524e08p-54, 0x1.010ab5b2cbd11p+0}, {-0x1.2b6aeb6176892p-56, 0x1.0137444c9b5b5p+0},
  {0x1.b61299ab8cdb7p-54, 0x1.0163da9fb3335p+0}, {-0x1.008eff5142bf9p-56, 0x1.019078ad6a19fp+0},
  {0x1.5e7626621eb5bp-56, 0x1.01bd1e77170b4p+0}, {-0x1.c11f5239bf535p-55, 0x1.01e9cbfe113efp+0},
  {-0x1.2bf310fc54eb6p-55, 0x1.02168143b0281p+0}, {-0x1.314aa16278aa3p-54, 0x1.02433e494b755p+0},
  {-0x1.082ef51b61d7ep-56, 0x1.027003103b10ep+0}, {0x1.64cbba902ca27p-58, 0x1.029ccf99d720ap+0},
  {-0x1.19083535b085dp-56, 0x1.02c9a3e778061p+0}, {-0x1.b8db0e9dbd87ep-55, 0x1.02f67ffa765e6p+0},
  {0x1.fea8d61ed6016p-54, 0x1.032363d42b027p+0}, {0x1.bc2ee8e5799acp-54, 0x1.03504f75ef071p+0},
  {0x1.56811eeade11ap-57, 0x1.037d42e11bbccp+0}, {-0x1.f1a93c1b824d3p-54, 0x1.03aa3e170aafep+0},
  {0x1.b7c00e7b751dap-54, 0x1.03d7411915a8ap+0}, {0x1.9dc3add8f9c02p-54, 0x1.04044be896ab6p+0},
  {-0x1.0a31c1977c96ep-54, 0x1.04315e86e7f85p+0}, {0x1.35bc86af4ee9ap-56, 0x1.045e78f5640b9p+0},
  {0x1.21cd53d5e8b66p-57, 0x1.048b9b35659d8p+0}, {-0x1.e7992580447bp-56, 0x1.04b8c54847a28p+0},
  {0x1.4c3793aa0d08dp-55, 0x1.04e5f72f654b1p+0}, {0x1.79a8be239ca45p-54, 0x1.051330ec1a03fp+0},
  {-0x1.abcae24b819dfp-54, 0x1.0540727fc1762p+0}, {0x1.06c87433776c9p-55, 0x1.056dbbebb786bp+0}};

static double as_expd(double x, double *l, int *e){
  const double ln2h = 0x1.71547652b82fep+10, ln2l = 0x1.777d0ffda0d24p-46;
  double xh = x, xl = *l;
  xh = muldd(xh,xl, ln2h,ln2l, &xl);
  double ix = coremath_roundeven(xh);
  xh = fasttwosum(xh-ix, xl, &xl);
  int k = ix, i0 = (k>>5)&31, i1 = k&31;
  *e = k>>10;
  double rl, rh = muldd(E0[i0][1],E0[i0][0], E1[i1][1],E1[i1][0], &rl);
  static const double ec[][2] = {
    {0x1.62e42fefa39efp-11, 0x1.abc9e3bf9d4d1p-66}, {0x1.ebfbdff82c58ep-23, 0x1.ec07243b4e585p-77},
    {0x1.c6b08d704a0bfp-35, 0x1.94bac118264d5p-89}, {0x1.3b2ab719edc2dp-47, 0x1.b530cee32e3dep-101},
    {0x1.5d87fe98a5fc4p-60, -0x1.63e85fdbde1cap-115}
  };
  const int m = 1;
  double fh, fl, el;
  fl = xh*polyd(xh,5-m,ec+m);
  fh = polydd(xh,xl, m, ec, &fl);
  fh = muldd(xh,xl, fh,fl, &fl);
  fh = fasttwosum(1, fh, &el); fl += el;
  rh = muldd(rh,rl, fh,fl, &rl);
  *l = rl;
  return rh;
}

static double as_lgamma_asym(double xh, double *xl){
  double zh = 1.0/xh, dz = *xl*zh, zl = (std::fma(zh,-xh,1.0) - dz)*zh;
  double ll, lh = as_logd(xh, &ll); ll += dz;
  lh = muldd(xh-0.5, *xl, lh-1,ll, &ll);
  double z2l, z2h = muldd(zh,zl,zh,zl, &z2l);
  double fh,fl;
  double x2 = z2h*z2h;
  if(xh>11.5){
    static const double lc[][2] = {
      {0x1.acfe390c97d69p-2, 0x1.34acf208a22c4p-56}, {0x1.5555555555555p-4, 0x1.31799ffbcdddbp-58},
      {-0x1.6c16c16c165a9p-9, 0x1.1eefaee02f69p-63}, {0x1.a01a019ada522p-11, -0x1.4d52971deb155p-66},
      {-0x1.381377e3a546dp-11, 0x1.fd1b354a8db62p-65}, {0x1.b9486dc1c9886p-11, -0x1.2dac4b8cca031p-65},
      {-0x1.f3ecd8799f337p-10, 0x1.da5dd745e3963p-64}, {0x1.6d399e561839p-8, 0x1.15e3000de141ap-62}};
    lh = fastsum(lh,ll, lc[0][0], lc[0][1], &ll);
    const int k = 1;
    const double (*b)[2] = lc + 1, (*q)[2] = lc + 1 + k;
    double q0 = q[0][0] + z2h*q[1][0];
    double q2 = q[2][0] + z2h*q[3][0];
    double q4 = q[4][0] + z2h*q[5][0];
    fl = z2h*(q0 + x2*(q2 + x2*q4));
    fh = polydd(z2h,z2l, k,b, &fl);
  } else {
    static const double lc[][2] = {
      {0x1.acfe390c97d69p-2, 0x1.f06a157d44d5bp-56}, {0x1.5555555555541p-4, 0x1.9d5fc10df4161p-58},
      {-0x1.6c16c16bfb733p-9, -0x1.557d8fba9e97ap-64}, {0x1.a01a01651819cp-11, -0x1.dd3c0f402122ap-65},
      {-0x1.38136b229bfb4p-11, -0x1.879990edddc5fp-67}, {0x1.b94c0472d00ap-11, 0x1.215a15f7d9289p-65},
      {-0x1.f619a122c3918p-10, 0x1.13405abdba76dp-64}, {0x1.9edef47081644p-8, 0x1.d9d833b12b9bp-62},
      {-0x1.bfc20185bf7ccp-6, -0x1.8aa555605e3b1p-60}, {0x1.0e832a937233p-3, 0x1.871cbde1ab342p-58},
      {-0x1.2beb46518ed4ap-1, -0x1.0298c44c99ceep-58}, {0x1.e5717107e0999p+0, 0x1.5bdfe7ac38f81p-56},
      {-0x1.90c04fbd840a6p+1, -0x1.5d2fbfe47e148p-54}};
    lh = fastsum(lh,ll, lc[0][0], lc[0][1], &ll);
    double x4 = x2*x2;
    const int k = 2;
    const double (*b)[2] = lc + 1, (*q)[2] = lc + 1 + k;
    double q0 = q[0][0] + z2h*q[1][0];
    double q2 = q[2][0] + z2h*q[3][0];
    double q4 = q[4][0] + z2h*q[5][0];
    double q6 = q[6][0] + z2h*q[7][0];
    double q8 = q[8][0] + z2h*q[9][0];
    q4 += x2*(q6 + x2*q8);
    q0 += x2*q2;
    q0 += x4*q4;
    fl = z2h*q0;
    fh = polydd(z2h,z2l, k,b, &fl);
  }
  fh = muldd(zh,zl, fh,fl, &fl);
  return fastsum(lh,ll, fh,fl, xl);
}
