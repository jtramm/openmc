// Fast-path hyperbolic sine for binary64 values.
// Derived from the CORE-MATH project (MIT License).
// Original author: Alexei Sibidanov.
// https://core-math.gitlabpages.inria.fr/

#include "coremath_utils.h"

static inline double mulddd(double xh, double xl, double ch, double *l){
  double h = ch*xh;
  *l = std::fma(ch,xh, -h) + ch*xl;
  return h;
}

static inline double polydd(double xh, double xl, int n, const double c[][2], double *l){
  int i = n-1;
  double ch = c[i][0] + *l, cl = ((c[i][0] - ch) + *l) + c[i][1], e;
  while(--i>=0){
    ch = muldd(xh, xl, ch, cl, &cl);
    ch = fasttwosum(c[i][0], ch, &e);
    cl = (cl + c[i][1]) + e;
  }
  *l = cl;
  return ch;
}

static double as_exp_accurate(double x, double t, double th, double tl, double *l){
  static const double ch[][2] = {
    {0x1p+0, 0x1.6c16bd194535dp-94}, {0x1p-1, -0x1.8259d904fd34fp-93},
    {0x1.5555555555555p-3, 0x1.53e93e9f26e62p-57}};
  const double l2h = 0x1.62e42ffp-13, l2l = 0x1.718432a1b0e26p-47, l2ll = 0x1.9ff0342542fc3p-102;
  double dx = x - l2h*t, dxl = l2l*t, dxll = l2ll*t + std::fma(l2l,t,-dxl);
  double dxh = dx + dxl; dxl = ((dx - dxh) + dxl) + dxll;
  double fl = dxh*(0x1.5555555555555p-5 + dxh *(0x1.11111113e93e9p-7 + dxh *0x1.6c16c169400a7p-10));
  double fh = polydd(dxh,dxl,3,ch, &fl);
  fh = muldd(dxh,dxl,fh,fl,&fl);
  fh = muldd(th,tl,fh,fl,&fl);
  double zh = th + fh, zl = (th-zh) + fh;
  double uh = zh + tl, ul = ((zh-uh) + tl) + zl;
  double vh = uh + fl, vl = ((uh-vh) + fl) + ul;
  *l = vl;
  return vh;
}

double cr_sinh(double x){
  static const double t0[][2] = {
    {0x0p+0, 0x1p+0}, {-0x1.19083535b085ep-56, 0x1.02c9a3e778061p+0},
    {0x1.d73e2a475b466p-55, 0x1.059b0d3158574p+0}, {0x1.186be4bb285p-57, 0x1.0874518759bc8p+0},
    {0x1.8a62e4adc610ap-54, 0x1.0b5586cf9890fp+0}, {0x1.03a1727c57b52p-59, 0x1.0e3ec32d3d1a2p+0},
    {-0x1.6c51039449b3ap-54, 0x1.11301d0125b51p+0}, {-0x1.32fbf9af1369ep-54, 0x1.1429aaea92dep+0},
    {-0x1.19041b9d78a76p-55, 0x1.172b83c7d517bp+0}, {0x1.e5b4c7b4968e4p-55, 0x1.1a35beb6fcb75p+0},
    {0x1.e016e00a2643cp-54, 0x1.1d4873168b9aap+0}, {0x1.dc775814a8494p-55, 0x1.2063b88628cd6p+0},
    {0x1.9b07eb6c70572p-54, 0x1.2387a6e756238p+0}, {0x1.2bd339940e9dap-55, 0x1.26b4565e27cddp+0},
    {0x1.612e8afad1256p-55, 0x1.29e9df51fdee1p+0}, {0x1.0024754db41d4p-54, 0x1.2d285a6e4030bp+0},
    {0x1.6f46ad23182e4p-55, 0x1.306fe0a31b715p+0}, {0x1.32721843659a6p-54, 0x1.33c08b26416ffp+0},
    {-0x1.63aeabf42eae2p-54, 0x1.371a7373aa9cbp+0}, {-0x1.5e436d661f5e2p-56, 0x1.3a7db34e59ff7p+0},
    {0x1.ada0911f09ebcp-55, 0x1.3dea64c123422p+0}, {-0x1.ef3691c309278p-58, 0x1.4160a21f72e2ap+0},
    {0x1.89b7a04ef80dp-59, 0x1.44e086061892dp+0}, {0x1.3c1a3b69062fp-56, 0x1.486a2b5c13cdp+0},
    {0x1.d4397afec42e2p-56, 0x1.4bfdad5362a27p+0}, {-0x1.4b309d25957e4p-54, 0x1.4f9b2769d2ca7p+0},
    {-0x1.07abe1db13cacp-55, 0x1.5342b569d4f82p+0}, {0x1.9bb2c011d93acp-54, 0x1.56f4736b527dap+0},
    {0x1.6324c054647acp-54, 0x1.5ab07dd485429p+0}, {0x1.ba6f93080e65ep-54, 0x1.5e76f15ad2148p+0},
    {-0x1.383c17e40b496p-54, 0x1.6247eb03a5585p+0}, {-0x1.bb60987591c34p-54, 0x1.6623882552225p+0},
    {-0x1.bdd3413b26456p-54, 0x1.6a09e667f3bcdp+0}, {-0x1.bbe3a683c88aap-57, 0x1.6dfb23c651a2fp+0},
    {-0x1.16e4786887a9ap-55, 0x1.71f75e8ec5f74p+0}, {-0x1.0245957316dd4p-54, 0x1.75feb564267c9p+0},
    {-0x1.41577ee04993p-55, 0x1.7a11473eb0187p+0}, {0x1.05d02ba15797ep-56, 0x1.7e2f336cf4e62p+0},
    {-0x1.d4c1dd41532d8p-54, 0x1.82589994cce13p+0}, {-0x1.fc6f89bd4f6bap-54, 0x1.868d99b4492edp+0},
    {0x1.6e9f156864b26p-54, 0x1.8ace5422aa0dbp+0}, {0x1.5cc13a2e3976cp-55, 0x1.8f1ae99157736p+0},
    {-0x1.75fc781b57ebcp-57, 0x1.93737b0cdc5e5p+0}, {-0x1.d185b7c1b85dp-54, 0x1.97d829fde4e5p+0},
    {0x1.c7c46b071f2bep-56, 0x1.9c49182a3f09p+0}, {-0x1.359495d1cd532p-54, 0x1.a0c667b5de565p+0},
    {-0x1.d2f6edb8d41e2p-54, 0x1.a5503b23e255dp+0}, {0x1.0fac90ef7fd32p-54, 0x1.a9e6b5579fdbfp+0},
    {0x1.7a1cd345dcc82p-54, 0x1.ae89f995ad3adp+0}, {-0x1.2805e3084d708p-57, 0x1.b33a2b84f15fbp+0},
    {-0x1.5584f7e54ac3ap-56, 0x1.b7f76f2fb5e47p+0}, {0x1.23dd07a2d9e84p-55, 0x1.bcc1e904bc1d2p+0},
    {0x1.11065895048dep-55, 0x1.c199bdd85529cp+0}, {0x1.2884dff483cacp-54, 0x1.c67f12e57d14bp+0},
    {0x1.503cbd1e949dcp-56, 0x1.cb720dcef9069p+0}, {-0x1.cbc3743797a9cp-54, 0x1.d072d4a07897cp+0},
    {0x1.2ed02d75b3706p-55, 0x1.d5818dcfba487p+0}, {0x1.c2300696db532p-54, 0x1.da9e603db3285p+0},
    {-0x1.1a5cd4f184b5cp-54, 0x1.dfc97337b9b5fp+0}, {0x1.39e8980a9cc9p-55, 0x1.e502ee78b3ff6p+0},
    {-0x1.e9c23179c2894p-54, 0x1.ea4afa2a490dap+0}, {0x1.dc7f486a4b6bp-54, 0x1.efa1bee615a27p+0},
    {0x1.9d3e12dd8a18ap-54, 0x1.f50765b6e454p+0}, {0x1.74853f3a5931ep-55, 0x1.fa7c1819e90d8p+0}
  };
  static const double t1[][2] = {
    {0x0p+0, 0x1p+0}, {0x1.ae8e38c59c72ap-54, 0x1.000b175effdc7p+0},
    {-0x1.7b5d0d58ea8f4p-58, 0x1.00162f3904052p+0}, {0x1.4115cb6b16a8ep-54, 0x1.0021478e11ce6p+0},
    {-0x1.d7c96f201bb2ep-55, 0x1.002c605e2e8cfp+0}, {0x1.84711d4c35eap-54, 0x1.003779a95f959p+0},
    {-0x1.0484245243778p-55, 0x1.0042936faa3d8p+0}, {-0x1.4b237da2025fap-54, 0x1.004dadb113dap+0},
    {-0x1.5e00e62d6b30ep-56, 0x1.0058c86da1c0ap+0}, {0x1.a1d6cedbb948p-54, 0x1.0063e3a559473p+0},
    {-0x1.4acf197a00142p-54, 0x1.006eff583fc3dp+0}, {-0x1.eaf2ea42391a6p-57, 0x1.007a1b865a8cap+0},
    {0x1.da93f90835f76p-56, 0x1.0085382faef83p+0}, {-0x1.6a79084ab093cp-55, 0x1.00905554425d4p+0},
    {0x1.86364f8fbe8f8p-54, 0x1.009b72f41a12bp+0}, {-0x1.82e8e14e3110ep-55, 0x1.00a6910f3b6fdp+0},
    {-0x1.4f6b2a7609f72p-55, 0x1.00b1afa5abcbfp+0}, {-0x1.e1a258ea8f71ap-56, 0x1.00bcceb7707ecp+0},
    {0x1.4362ca5bc26f2p-56, 0x1.00c7ee448ee02p+0}, {0x1.095a56c919d02p-54, 0x1.00d30e4d0c483p+0},
    {-0x1.406ac4e81a646p-57, 0x1.00de2ed0ee0f5p+0}, {0x1.b5a6902767e08p-54, 0x1.00e94fd0398ep+0},
    {-0x1.91b206085932p-54, 0x1.00f4714af41d3p+0}, {0x1.427068ab22306p-55, 0x1.00ff93412315cp+0},
    {0x1.c1d0660524e08p-54, 0x1.010ab5b2cbd11p+0}, {-0x1.e7bdfb3204be8p-54, 0x1.0115d89ff3a8bp+0},
    {0x1.843aa8b9cbbc6p-55, 0x1.0120fc089ff63p+0}, {-0x1.34104ee7edae8p-56, 0x1.012c1fecd613bp+0},
    {-0x1.2b6aeb6176892p-56, 0x1.0137444c9b5b5p+0}, {0x1.a8cd33b8a1bb2p-56, 0x1.01426927f5278p+0},
    {0x1.2edc08e5da99ap-56, 0x1.014d8e7ee8d2fp+0}, {0x1.57ba2dc7e0c72p-55, 0x1.0158b4517bb88p+0},
    {0x1.b61299ab8cdb8p-54, 0x1.0163da9fb3335p+0}, {-0x1.90565902c5f44p-54, 0x1.016f0169949edp+0},
    {0x1.70fc41c5c2d54p-55, 0x1.017a28af25567p+0}, {0x1.4b9a6e145d76cp-54, 0x1.018550706ab62p+0},
    {-0x1.008eff5142bfap-56, 0x1.019078ad6a19fp+0}, {-0x1.77669f033c7dep-54, 0x1.019ba16628de2p+0},
    {-0x1.09bb78eeead0ap-54, 0x1.01a6ca9aac5f3p+0}, {0x1.371231477ece6p-54, 0x1.01b1f44af9f9ep+0},
    {0x1.5e7626621eb5ap-56, 0x1.01bd1e77170b4p+0}, {-0x1.bc72b100828a4p-54, 0x1.01c8491f08f08p+0},
    {-0x1.ce39cbbab8bbep-57, 0x1.01d37442d507p+0}, {0x1.16996709da2e2p-55, 0x1.01de9fe280ac8p+0},
    {-0x1.c11f5239bf536p-55, 0x1.01e9cbfe113efp+0}, {0x1.e1d4eb5edc6b4p-55, 0x1.01f4f8958c1c6p+0},
    {-0x1.afb99946ee3fp-54, 0x1.020025a8f6a35p+0}, {-0x1.8f06d8a148a32p-54, 0x1.020b533856324p+0},
    {-0x1.2bf310fc54eb6p-55, 0x1.02168143b0281p+0}, {-0x1.c95a035eb4176p-54, 0x1.0221afcb09e3ep+0},
    {-0x1.491793e46834cp-54, 0x1.022cdece68c4fp+0}, {-0x1.3e8d0d9c4909p-56, 0x1.02380e4dd22adp+0},
    {-0x1.314aa16278aa4p-54, 0x1.02433e494b755p+0}, {0x1.48daf888e965p-55, 0x1.024e6ec0da046p+0},
    {0x1.56dc8046821f4p-55, 0x1.02599fb483385p+0}, {0x1.45b42356b9d46p-54, 0x1.0264d1244c719p+0},
    {-0x1.082ef51b61d7ep-56, 0x1.027003103b10ep+0}, {0x1.2106ed0920a34p-56, 0x1.027b357854772p+0},
    {-0x1.fd4cf26ea5d0ep-54, 0x1.0286685c9e059p+0}, {-0x1.09f8775e78084p-54, 0x1.02919bbd1d1d8p+0},
    {0x1.64cbba902ca28p-58, 0x1.029ccf99d720ap+0}, {0x1.4383ef231d206p-54, 0x1.02a803f2d170dp+0},
    {0x1.4a47a505b3a46p-54, 0x1.02b338c811703p+0}, {0x1.e47120223468p-54, 0x1.02be6e199c811p+0},
  };

  const double s = 0x1.71547652b82fep+12;
  double ax = std::fabs(x);
  b64u64_u iax;
  iax.f = ax;
  u64 aix = iax.u;

  if (aix<0x3fd0000000000000ull){ // |x| < 0.25
    if (aix<0x3e57137449123ef7ull) {
      // |x| < x0 = 0x1.7137449123ef7p-26
      return std::fma(x,0x1p-55,x);
    }
    static const double c[] =
      {0x1.5555555555555p-3, 0x1.1111111111087p-7, 0x1.a01a01a12e1c3p-13,
       0x1.71de2e415aa36p-19, 0x1.aed2bff4269e6p-26};
    double x2 = x*x, x3 = x2*x, x4 = x2*x2,
      p = x3*((c[0] + x2*c[1]) + x4*((c[2] + x2*c[3]) + x4*c[4]));
    return x + p;
  }
  if (aix>0x408633ce8fb9f87dull){ // |x| >~ 710.47586
    if(aix>=0x7ff0000000000000ull) return x + x; // nan Inf
    return std::copysign(0x1p1023, x)*2.0; // overflow
  }

  // now 0.25 <= |x| < 710.47586
  double v0 = std::fma(ax, s, 0x1.8000002p+26);
  b64u64_u jt;
  jt.f = v0;
  b64u64_u v;
  v.f = v0;
  uint64_t tt = ~((1<<26)-1l);
  v.u &= tt;
  double t = v.f - 0x1.8p26;

  int64_t il = ((u64)jt.u<<14)>>40, jl = -il;
  int64_t i1 = il&0x3f, i0 = (il>>6)&0x3f, ie = il>>12;
  int64_t j1 = jl&0x3f, j0 = (jl>>6)&0x3f, je = jl>>12;
  b64u64_u sp, sm;
  sp.u = (u64)(1022 + ie)<<52;
  sm.u = (u64)(1022 + je)<<52;
  double t0h = t0[i0][1], t0l = t0[i0][0];
  double t1h = t1[i1][1], t1l = t1[i1][0];
  double th = t0h*t1h, tl = t0h*t1l + t1h*t0l + std::fma(t0h,t1h,-th);
  const double l2h = 0x1.62e42ffp-13, l2l = 0x1.718432a1b0e26p-47;
  double dx = (ax - l2h*t) + l2l*t, dx2 = dx*dx, mx = -dx;
  static const double ch[] = {0x1p+0, 0x1p-1, 0x1.5555555aaaaaep-3, 0x1.55555551c98cp-5};
  double pp = dx*((ch[0] + dx*ch[1]) + dx2*(ch[2] + dx*ch[3]));
  double rh, rl;
  if (aix>0x4014000000000000ull){ // |x| > 5
    if (aix>0x40425e4f7b2737faull){ // |x| >~ 36.736801
      sp.u = (u64)(1021 + ie)<<52;
      rh = th;
      rl = tl + th*pp;
      rh *= std::copysign(1.0, x);
      rl *= std::copysign(1.0, x);
      double e = 0.11e-18*th, lb = rh + (rl - e), ub = rh + (rl + e);
      if(lb == ub) return (lb*sp.f)*2;

      th = as_exp_accurate(ax, t, th, tl, &tl);
      th = fasttwosum(th, tl, &tl);
      th *= std::copysign(1.0, x);
      tl *= std::copysign(1.0, x);
      th += tl;
      th *= 2;
      th *= sp.f;
      return th;
    }
    // now 5 < |x| < 36.736801
    double q0h = t0[j0][1], q1h = t1[j1][1], qh = q0h*q1h;
    th *= sp.f;
    tl *= sp.f;
    qh *= sm.f;
    double pm = mx*((ch[0] + mx*ch[1]) + dx2*(ch[2] + mx*ch[3]));
    double em = qh + qh*pm;
    rh = th;
    rl = (tl - em) + th*pp;

    rh *= std::copysign(1.0, x);
    rl *= std::copysign(1.0, x);
    double e = 0.117e-18*rh, lb = rh + (rl - e), ub = rh + (rl + e);
    if(lb == ub) return lb;

    th = as_exp_accurate( ax, t, th, tl, &tl);
    if (aix>0x403f666666666666ull){ // |x| > 31.4
      rh = th - qh; rl = ((th - rh) - qh) + tl;
    } else { // 5 < |x| <= 31.4
      qh = q0h*q1h;
      double q0l = t0[j0][0], q1l = t1[j1][0];
      double ql = q0h*q1l + q1h*q0l + std::fma(q0h,q1h,-qh);
      qh *= sm.f;
      ql *= sm.f;
      qh = as_exp_accurate(-ax,-t, qh, ql, &ql);
      rh = th - qh; rl = (((th - rh) - qh) - ql) + tl;
    }
  } else { // 0.25 <= |x| <= 5
    double q0h = t0[j0][1], q0l = t0[j0][0];
    double q1h = t1[j1][1], q1l = t1[j1][0];
    double qh = q0h*q1h, ql = q0h*q1l + q1h*q0l + std::fma(q0h,q1h,-qh);
    th *= sp.f;
    tl *= sp.f;
    qh *= sm.f;
    ql *= sm.f;
    double pm = mx*((ch[0] + mx*ch[1]) + dx2*(ch[2] + mx*ch[3]));
    double fph = th, fpl = tl + th*pp;
    double fmh = qh, fml = ql + qh*pm;

    rh = fph - fmh;
    rl = ((fph - rh) - fmh) - fml + fpl;
    rh *= std::copysign(1.0, x);
    rl *= std::copysign(1.0, x);
    double e = 0.33e-18*rh, lb = rh + (rl - e), ub = rh + (rl + e);
    if(lb == ub) return lb;
    th = as_exp_accurate( ax, t, th, tl, &tl);
    qh = as_exp_accurate(-ax,-t, qh, ql, &ql);
    rh = th - qh;
    rl = ((th-rh)-qh) - ql + tl;
  }
  rh = fasttwosum(rh, rl, &rl);
  rh *= std::copysign(1.0, x);
  rl *= std::copysign(1.0, x);
  rh += rl;
  return rh;
}
