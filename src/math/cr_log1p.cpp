// Fast-path log(1+x) for binary64 values.
// Derived from the CORE-MATH project (MIT License).
// Original author: Alexei Sibidanov.
// https://core-math.gitlabpages.inria.fr/

#include "coremath_utils.h"

static inline double twosum(double xh, double ch, double *l){
  double s = xh + ch, d = s - xh;
  *l = (ch - d) + (xh + (d - s));
  return s;
}

static inline double mulddd(double x, double ch, double cl, double *l){
  double ahhh = ch*x;
  *l = cl*x + std::fma(ch, x, -ahhh);
  return ahhh;
}

static inline double polydd(double xh, double xl, int n, const double c[][2], double *l){
  int i = n-1;
  double ch = fasttwosum(c[i][0], *l, l), cl = c[i][1] + *l;
  while(--i>=0){
    ch = muldd(xh,xl, ch,cl, &cl);
    ch = fastsum(c[i][0],c[i][1], ch,cl, &cl);
  }
  *l = cl;
  return ch;
}

static inline double polyddd(double x, int n, const double c[][2], double *l){
  int i = n-1;
  double ch = fasttwosum(c[i][0], *l, l), cl = c[i][1] + *l;
  while(--i>=0){
    ch = mulddd(x, ch,cl, &cl);
    ch = fastsum(c[i][0],c[i][1], ch,cl, &cl);
  }
  *l = cl;
  return ch;
}

/*
  rf[64] and lf[64][2] are lookup tables for the fast path
  -ln(rf[][]) = lf[][1] + lf[][0]
  values are approximately from 1/sqrt(2) to sqrt(2)
*/
static const double rf[64] = {
  0x1.6816818p+0, 0x1.642c858p+0, 0x1.605816p+0, 0x1.5c98828p+0,
  0x1.58ed23p+0, 0x1.5555558p+0, 0x1.51d07e8p+0, 0x1.4e5e0a8p+0,
  0x1.4afd6ap+0, 0x1.47ae148p+0, 0x1.446f868p+0, 0x1.4141418p+0,
  0x1.3e22ccp+0, 0x1.3b13b1p+0, 0x1.381381p+0, 0x1.3521cf8p+0,
  0x1.323e348p+0, 0x1.2f684cp+0, 0x1.2c9fb5p+0, 0x1.29e4128p+0,
  0x1.27350b8p+0, 0x1.249249p+0, 0x1.21fb78p+0, 0x1.1f7048p+0,
  0x1.1cf06bp+0, 0x1.1a7b96p+0, 0x1.181181p+0, 0x1.15b1e6p+0,
  0x1.135c81p+0, 0x1.111111p+0, 0x1.0ecf568p+0, 0x1.0c9715p+0,
  0x1.0a68108p+0, 0x1.0842108p+0, 0x1.0624ddp+0, 0x1.041041p+0,
  0x1.020408p+0, 0x1p+0, 0x1.f81f82p-1, 0x1.f07c1fp-1,
  0x1.e9131a8p-1, 0x1.e1e1e2p-1, 0x1.dae6078p-1, 0x1.d41d42p-1,
  0x1.cd85688p-1, 0x1.c71c72p-1, 0x1.c0e07p-1, 0x1.bacf918p-1,
  0x1.b4e81b8p-1, 0x1.af286cp-1, 0x1.a98ef6p-1, 0x1.a41a418p-1,
  0x1.9ec8e98p-1, 0x1.9999998p-1, 0x1.948b1p-1, 0x1.8f9c19p-1,
  0x1.8acb91p-1, 0x1.8618618p-1, 0x1.8181818p-1, 0x1.7d05f4p-1,
  0x1.78a4c8p-1, 0x1.745d178p-1, 0x1.702e06p-1, 0x1.6c16c18p-1
};

static const double lf[64][2] = {
  {-0x1.f2f8281bade6ap-42, -0x1.5d5bde3994p-2}, {0x1.c2843fdd367a4p-42, -0x1.522ae0438cp-2},
  {-0x1.06c10c34c14bp-44, -0x1.4718dc171cp-2}, {0x1.cfa4e853f589p-43, -0x1.3c2526cb34p-2},
  {-0x1.ce3ac179bd856p-42, -0x1.314f1e0534p-2}, {-0x1.b91f82deb8122p-42, -0x1.269621934cp-2},
  {0x1.46bbb83d7163ep-42, -0x1.1bf995a9a8p-2}, {0x1.b842e5a74bdbp-42, -0x1.1178e84a8p-2},
  {-0x1.862715e5bb534p-42, -0x1.071385f4d4p-2}, {-0x1.9bcbcbea0cdf8p-42, -0x1.f991c6eb38p-3},
  {-0x1.01101cb605958p-43, -0x1.e530f1067p-3}, {0x1.0c38c81ad8f06p-42, -0x1.d10380b658p-3},
  {0x1.3aa40992a6d82p-42, -0x1.bd0874c3cp-3}, {0x1.30f68780ae82ep-42, -0x1.a93ed248bp-3},
  {-0x1.7d116989d098p-47, -0x1.95a5ac5f7p-3}, {-0x1.1e0012ba619cap-42, -0x1.823c150518p-3},
  {0x1.54535d5671858p-43, -0x1.6f0127cf58p-3}, {-0x1.ed87db3498128p-42, -0x1.5bf407b54p-3},
  {-0x1.aafde9c9fc39ap-42, -0x1.4913d94338p-3}, {-0x1.015868c234p-43, -0x1.365fca3158p-3},
  {0x1.eff33f502c226p-42, -0x1.23d7126cap-3}, {0x1.b8521e874d358p-43, -0x1.1178e7228p-3},
  {0x1.54d75afe84568p-43, -0x1.fe89129dcp-4}, {-0x1.1a813f3fa7c1ep-42, -0x1.da7278384p-4},
  {-0x1.6c6676f40963ep-42, -0x1.b6ac8afadp-4}, {-0x1.2620b7957a7a6p-42, -0x1.9335e4d59p-4},
  {0x1.f8ffee5598f38p-43, -0x1.700d2f4ebp-4}, {-0x1.fab0f5bf42ca2p-42, -0x1.4d311652p-4},
  {-0x1.7a3e970b1c3a8p-44, -0x1.2aa049247p-4}, {-0x1.d030435fecb5p-43, -0x1.08598a59ep-4},
  {0x1.35084a4fb8ab8p-43, -0x1.ccb7357dep-5}, {0x1.32f36d60b44c4p-43, -0x1.894aa1cap-5},
  {0x1.c1bcce5be811p-45, -0x1.466ae8a2ep-5}, {0x1.777740b18714ap-42, -0x1.0415d81e8p-5},
  {-0x1.955c057693d94p-43, -0x1.8492470c8p-6}, {0x1.4f71addb8bep-43, -0x1.020564894p-6},
  {-0x1.bcda4e198afbp-44, -0x1.01014f588p-7}, {0x1.cp-67, 0x0p+0},
  {-0x1.fe0df75092c5ep-42, 0x1.fc0a891p-7}, {0x1.98036ec7e0a1p-45, 0x1.f829b1e78p-6},
  {0x1.ba010f49e5ffp-42, 0x1.774593832p-5}, {-0x1.3ab13c266d328p-42, 0x1.f0a30a012p-5},
  {-0x1.71798573e45d4p-43, 0x1.341d78b1cp-4}, {0x1.ad32f072669fcp-42, 0x1.6f0d272e5p-4},
  {-0x1.54e391e16ea38p-43, 0x1.a926d434bp-4}, {-0x1.a302bbaf0559p-45, 0x1.e27074e2bp-4},
  {0x1.cb4cd66e31f3p-44, 0x1.0d77e8cd08p-3}, {-0x1.5b7a5bc474128p-44, 0x1.29552e92p-3},
  {-0x1.7062e8135f74p-46, 0x1.44d2b5e4b8p-3}, {0x1.3d4c88fe1f4bp-43, 0x1.5ff3060a78p-3},
  {-0x1.37b70004a6946p-42, 0x1.7ab890411p-3}, {-0x1.4a5885167c1ecp-42, 0x1.9525aa7f48p-3},
  {0x1.ff9d5953004acp-42, 0x1.af3c940008p-3}, {0x1.a21ec41d8219cp-43, 0x1.c8ff7cf9a8p-3},
  {-0x1.a322bf2f02ae8p-44, 0x1.e27075e2bp-3}, {0x1.f1548b8a33616p-42, 0x1.fb9186b5ep-3},
  {0x1.0e36401f7a006p-42, 0x1.0a324e0f38p-2}, {-0x1.9f1fa55382a8ap-42, 0x1.1675cacabcp-2},
  {-0x1.a69763deb096p-44, 0x1.22941fc0f8p-2}, {0x1.d30bc3ac91bdap-42, 0x1.2e8e2bee1p-2},
  {0x1.7a79cf4d73b28p-44, 0x1.3a64c59694p-2}, {0x1.ec345197b22dep-42, 0x1.4618bb81c4p-2},
  {-0x1.f4810a30aeba8p-44, 0x1.51aad7c2ep-2}, {0x1.394d2371c1d1cp-43, 0x1.5d1bdbbd8p-2}
};

double cr_log1p(double x){
  b64u64_u ix;
  ix.f = x;
  u64 ax = ix.u<<1;
  double ln1, ln0;
  /* logp1 is expected to be used for x near 0, where it is more accurate than
     log(1+x), thus we expect x near 0 */
  if(ax<0x7f60000000000000ull){ // |x| < 0.0625
    // check case x tiny first to avoid spurious underflow in x*x
    if(ax<0x7940000000000000ull){ // |x| < 0x1p-53
      if(!ax) return x;
      double res = std::fma(std::fabs(x), -0x1p-54, x);
      return res;
    }
    double x2 = x*x;
    if(ax<0x7e60000000000000ull){ // |x| < 0x1p-12
      ln1 = x;
      if(ax<0x7d43360000000000ull){ // |x| < 0x1.19bp-21
	static const double c[] = {-0x1.00000000001d1p-1, 0x1.55555555558f7p-2};
	ln0 = x2*(c[0] + x*c[1]);
      } else {
	static const double c[] =
	  {-0x1.ffffffffffffdp-2, 0x1.5555555555551p-2, -0x1.000000d5555e1p-2, 0x1.99999b442f73fp-3};
	ln0 = x2*((c[0]+x*c[1])+x2*(c[2]+x*c[3]));
      }
    } else {
      static const double c[] =
	{0x1.5555555555555p-2, -0x1p-2, 0x1.9999999999b41p-3, -0x1.555555555583bp-3,
	 0x1.24924923f39ep-3, -0x1.fffffffe42e43p-4, 0x1.c71c75511d70bp-4, -0x1.99999de10510fp-4,
	 0x1.7457e81b175f6p-4, -0x1.554fb43e54e0fp-4, 0x1.3ed68744f3d18p-4, -0x1.28558ad5a7ac4p-4};
      double x3 = x2*x, x4 = x2*x2, hx = -0.5*x;
      ln1 = std::fma(hx,x,x);
      ln0 = std::fma(hx,x,x-ln1);
      double f = ((c[0]+x*c[1])+x2*(c[2]+x*c[3])) +
	x4*(((c[4]+x*c[5])+x2*(c[6]+x*c[7])) + x4*((c[8]+x*c[9])+x2*(c[10]+x*c[11])));
      ln0 += x3*f;
    }
  } else { // |x| >= 0.0625
    static const double c[] = {
      -0x1.000000000003dp-1, 0x1.5555555554cf5p-2, -0x1.ffffffeca2939p-3, 0x1.99999a3661724p-3,
      -0x1.555d345bfe6fdp-3, 0x1.247b887a6e5edp-3};
    b64u64_u t, dt;
    if((i64)ix.u<0x4340000000000000ll && ix.u<0xbff0000000000000ull){
      t.f = fasttwosum(1.0, x, &dt.f);
    } else {
      if(ix.u<0x4690000000000000ull){ // x < 0x1p+106
	t.f = x; dt.f = 1;
      } else {
	if(ix.u<0x7ff0000000000000ull){ // x < 0x1p+1024
	  t.f = x; dt.f = 0;
	} else {
	  if(ax>0xffe0000000000000ull) return x + x; // nan
	  if(ix.u==0x7ff0000000000000ull) return x; // +inf
	  if(ix.u==0xbff0000000000000ull){ // -1
            return -1./0.0;
          }
	  return 0.0/0.0; // <-1
	}
      }
    }
    i64 j = t.u - 0x3fe6a00000000000ll, j1 = (j>>(52-6))&0x3f, je = (j>>52),
      eoff = (u64)je<<52;
    b64u64_u rs;
    rs.f = rf[j1];
    if(je<1022){
      rs.u -= eoff;
    } else {
      rs.u -= (i64)1021<<52;
      static const double sc[] = {0x1p-1, 0x1p-2, 0x1p-3};
      t.f *= sc[je-1022];
      dt.f *= sc[je-1022];
    }
    double dh = rs.f*t.f, dl = std::fma(rs.f,t.f,-dh) + rs.f*dt.f;
    double xl, xh = fasttwosum(dh-1.0, dl, &xl), x2 = xh*xh;
    xl += x2*((c[0] + xh*c[1]) + x2*((c[2] + xh*c[3]) + x2*(c[4] + xh*c[5])));
    double L1 = 0x1.62e42fefa4p-1*je, L0 = -0x1.8432a1b0e2634p-43*je;
    ln1 = lf[j1][1] + L1;
    ln0 = lf[j1][0] + L0;
    ln1 = fastsum(ln1, ln0, xh, xl, &ln0);
  }
  return ln1 + ln0;
}
