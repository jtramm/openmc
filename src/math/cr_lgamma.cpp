// Fast-path log-gamma function for binary64 values.
// Derived from the CORE-MATH project (MIT License).
// Original author: Alexei Sibidanov.
// https://core-math.gitlabpages.inria.fr/
//
// This is a stripped-down fast-path-only version. The accurate/refine
// paths have been removed; when the original code would call the
// accurate path, the fast result is returned directly.
// The global signgam variable is not set; this computes log(|gamma(x)|).

#include "coremath_utils.h"

typedef unsigned short ushort;

static inline double twosum(double x, double y, double *e){
  if(std::fabs(x)>std::fabs(y))
    return fasttwosum(x, y, e);
  else
    return fasttwosum(y, x, e);
}

static inline double sumdd(double xh, double xl, double yh, double yl, double *e){
  double sl, sh;
  char o = std::fabs(xh)>std::fabs(yh);
  if(o)
    sh = fasttwosum(xh, yh, &sl);
  else
    sh = fasttwosum(yh, xh, &sl);
  sl += xl + yl;
  *e = sl;
  return sh;
}

static inline double mulddd(double x, double ch, double cl, double *l){
  double ahhh = ch*x;
  *l = cl*x + std::fma(ch, x, -ahhh);
  return ahhh;
}

static inline double polydddfst(double x, int n, const double c[][2], double *l){
  int i = n-1;
  double cl, ch = fasttwosum(c[i][0], *l, &cl); cl += c[i][1];
  while(--i>=0){
    ch = mulddd(x, ch,cl, &cl);
    ch = fastsum(c[i][0],c[i][1], ch,cl, &cl);
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

// Forward declarations for fast-path helpers
static double as_logd(double, double*);
static double as_sinpipid(double, double*);

// Lookup tables for the piece-wise polynomial approximation in [0.5, 8.29541]
// range borders
static const unsigned ubrd[20] = {
  0x1ff0000, 0x1ff146c, 0x1ff2b7b, 0x1ff4532, 0x1ff614c, 0x1ff8310, 0x1ff93f7, 0x1ffa880, 0x1ffc05e,
  0x1ffdb73, 0x1fff8a5, 0x2001147, 0x2002703, 0x20041ac, 0x200622a, 0x20084d9, 0x2009ce7, 0x200ba2c,
  0x200ddd7, 0x20104ba};
// the region offset
static const double offs[19] = {
  0x1.146cd8p-1, 0x1.3fe898p-1, 0x1.70aea8p-1, 0x1.a67fcp-1, 0x1.e76db8p-1, 0x1.170838p+0, 0x1.3c78a8p+0,
  0x1.68df2p+0, 0x1.9bd14p+0, 0x1.d41868p+0, 0x1.0d9a64p+1, 0x1.384b8p+1, 0x1.68b06p+1, 0x1.a3d6dp+1,
  0x1.ebdd9p+1, 0x1.21c1p+2, 0x1.571368p+2, 0x1.9803e8p+2, 0x1.e74cc8p+2};
// polynomial coefficients low part
static const double cl[19][8] = {
  {-0x1.18ad63ca097e9p+2, 0x1.af8e15b715c51p+2, -0x1.56213b7191ba4p+3, 0x1.151f165a9425fp+4,
   -0x1.c826426e4b7cdp+4, 0x1.7c313095e4b75p+5, -0x1.44f3d7d848e78p+6, 0x1.13384c97ea99dp+7},
  {-0x1.0f58e76c8d235p+1, 0x1.67c3f6b7124f6p+1, -0x1.ec78d7d8185a3p+1, 0x1.588d6487de574p+2,
   -0x1.e9fbe8564220dp+2, 0x1.60dd913b80b5ep+3, -0x1.0465db7c895a6p+4, 0x1.7ca34d903fc3p+4},
  {-0x1.0c505555a86b2p+0, 0x1.33d3a22d1bb51p+0, -0x1.6d2a2457f05d4p+0, 0x1.bb1c77fad8b03p+0,
   -0x1.115210a553746p+1, 0x1.558e305fd694p+1, -0x1.b4f9a0654679fp+1, 0x1.1489e269cbf39p+2},
  {-0x1.1170ead9585bap-1, 0x1.10c67b04495d7p-1, -0x1.19de3af9dd349p-1, 0x1.2a34cd6e66472p-1,
   -0x1.40e2f93066eb8p-1, 0x1.5ddc21a559735p-1, -0x1.860a742d837aap-1, 0x1.ace7238771e7fp-1},
  {-0x1.be31df8f7d605p-3, 0x1.8e33a32c94cf7p-3, -0x1.6c62efd534bf3p-3, 0x1.53719e404a7d6p-3,
   -0x1.4074d1d083331p-3, 0x1.31c6c5226f2b3p-3, -0x1.2b062b8eedd9fp-3, 0x1.219431f82fbfcp-3},
  {-0x1.168e45409b785p-3, 0x1.a04e5759477fp-4, -0x1.43c620bb1d77fp-4, 0x1.027d79414ff7dp-4,
   -0x1.a46afc0776356p-5, 0x1.5a92c3ddb75f5p-5, -0x1.23d37b3b3e3b6p-5, 0x1.f66a6169fe8efp-6},
  {-0x1.2db051283fb7ap-4, 0x1.8afce072c9222p-5, -0x1.0dcc84e49a658p-5, 0x1.7af09a263459bp-6,
   -0x1.0f51524188551p-6, 0x1.8a1f8bbd73d04p-7, -0x1.24e389a08ab23p-7, 0x1.b5c2228d32783p-8},
  {-0x1.3fbb4a9e75e6dp-5, 0x1.6c40332da72cp-6, -0x1.b235e2a6ed724p-7, 0x1.0a9023dcf81a5p-7,
   -0x1.4e128ec28e27ap-8, 0x1.a90e4421d59e7p-9, -0x1.14e6becdb3889p-9, 0x1.68e6a1727f763p-10},
  {-0x1.5365e61675f08p-6, 0x1.4fd143859dc2cp-7, -0x1.5cb1d911bcabbp-8, 0x1.75a869d793508p-9,
   -0x1.9941834996ea5p-10, 0x1.c782075b40e2cp-11, -0x1.03ab2e70c9df2p-11, 0x1.26c9636a06719p-12},
  {-0x1.7145b3bd2da75p-7, 0x1.3e6749a0fe63p-8, -0x1.20ea7ae3d0208p-9, 0x1.0f17a25bb48f3p-10,
   -0x1.045c4de3c2101p-11, 0x1.fcc9430345441p-13, -0x1.fccc917990854p-14, 0x1.f41fbb5026b5p-15},
  {0x1.253f3fc844189p-9, -0x1.cadf5cc04da1bp-11, 0x1.73e9dbf6ed988p-12, -0x1.34f75abb9acfdp-13,
   0x1.05502104fc072p-14, -0x1.c015daee9145bp-16, 0x1.8af9ccda4578cp-17, -0x1.5b973bad98b6bp-18},
  {-0x1.7f80bfa6d705ep-9, 0x1.e416c7e5d3bb3p-11, -0x1.4361a69711e0fp-12, 0x1.c0beed7451d56p-14,
   -0x1.3fc0c220552b8p-15, 0x1.d09a0850c9ad6p-17, -0x1.5b403dca4645p-18, 0x1.07c1349c4989ap-19},
  {-0x1.8bd8d36b6b68p-10, 0x1.ab590101636b6p-12, -0x1.e980083cba776p-14, 0x1.23c1bb53d55c4p-15,
   -0x1.65be06922ac5p-17, 0x1.bfd7d06279b09p-19, -0x1.2127a54c7c981p-20, 0x1.79ae1f9c24de8p-22},
  {-0x1.8db1211cc179cp-11, 0x1.6c24488bdc8e9p-13, -0x1.62882b2ca3c96p-15, 0x1.67e36a9a5f89cp-17,
   -0x1.785b8294e4cf2p-19, 0x1.925c40ccc4611p-21, -0x1.bccb0b78110bcp-23, 0x1.f05d365676624p-25},
  {-0x1.879379df4c28cp-12, 0x1.2e193030ccfd1p-14, -0x1.f0900c0b3c1fcp-17, 0x1.aa304a80f3ce4p-19,
   -0x1.795cdc082b2dfp-21, 0x1.56025546a45a8p-23, -0x1.4137abecddaa9p-25, 0x1.303d852294977p-27},
  {-0x1.7b7a8dbd38635p-13, 0x1.eab932219e072p-16, -0x1.528291d0efb42p-18, 0x1.e86163ded7066p-21,
   -0x1.6be3b6446506ap-23, 0x1.15d4d64fd204ap-25, -0x1.b8820763832efp-28, 0x1.5ff781e6e4e19p-30},
  {-0x1.6b1f37f261621p-14, 0x1.87d88455d6443p-17, -0x1.c3a69901a7d7cp-20, 0x1.107e1d8456499p-22,
   -0x1.53f4b8e0d8be8p-25, 0x1.b3016ba9fadffp-28, -0x1.2175c3eb445d8p-30, 0x1.841e8df7f84e7p-33},
  {-0x1.57cce7fdc9fe7p-15, 0x1.347c6b65ace16p-18, -0x1.27eb9df26d911p-21, 0x1.296c0dcf0b476p-24,
   -0x1.354fd38659786p-27, 0x1.4a2dbe1c4af19p-30, -0x1.6f1651636db1p-33, 0x1.9b1457d56445ap-36},
  {-0x1.423d487c99e54p-16, 0x1.df4d1f34022a3p-20, -0x1.7d54e9cd7e7eap-23, 0x1.3e131c44c6382p-26,
   -0x1.12afb6bfa8c14p-29, 0x1.e7412bd9ebd87p-33, -0x1.c2ab005ebc13bp-36, 0x1.a3bbacb6ee6b7p-39},
};
// polynomial coefficients high part
static const double ch[19][13][2] = {
  {{0x1.fdbd7c56b02b5p-2, -0x1.9f8c66985b6f3p-56}, {-0x1.c771ed8981f3ep+0, 0x1.8d8b72ce9b19dp-54},
  {0x1.1558ba7c0144dp+1, 0x1.4fc1fa0f0451cp-53}, {-0x1.1fa938f4d4b53p+1, -0x1.f29beb3ca3738p-53},
  {0x1.7f7469f6781efp+1, -0x1.b59ce1aa03545p-53}},
  {{0x1.71c14e711391ep-2, 0x1.2ad5eb4fb4f59p-60}, {-0x1.740c890bd54d3p+0, -0x1.6978dab8a116p-55},
   {0x1.b38de2e957c18p+0, -0x1.aba2b91749902p-55}, {-0x1.7ab358c51c087p+0, 0x1.46a8f1bc5883bp-55},
   {0x1.af1b63b322b6dp+0, 0x1.2d98d261df8f3p-55}},
  {{0x1.e53b12b3407e2p-3, 0x1.97cb2965d31b5p-57}, {-0x1.2a144e9a8b92ep+0, -0x1.bbf90d2717ba5p-54},
   {0x1.5adc4ef58621ep+0, 0x1.d41b3282f1d5bp-54}, {-0x1.fb259e2817239p-1, 0x1.a19b744867ccbp-55},
   {0x1.ee43256a6bfd3p-1, 0x1.880c7ca4d6687p-55}},
  {{0x1.0719312af823cp-3, -0x1.77ca1d8b99601p-57}, {-0x1.d11f75dc5be7dp-1, 0x1.997295e7f58d5p-57},
   {0x1.18a58180335ddp+0, -0x1.9e4f675e9e244p-58}, {-0x1.5aea0e9166a08p-1, 0x1.4799eb996a78bp-55},
   {0x1.22b448094c052p-1, -0x1.221db12561423p-56}},
  {{0x1.3c3b637596f8dp-1, -0x1.051b18f5744bap-56}, {-0x1.b9ccef0d71197p-1, -0x1.cf98e73bfb3d7p-55},
   {0x1.c55517304ef35p-2, 0x1.dfe2299217a1ap-57}, {-0x1.4230fb2a20b13p-2, -0x1.8eb1c5690348fp-57},
   {0x1.03aa1691c1841p-2, 0x1.a0e14e4b5a96cp-57}},
  {{-0x1.752403c835a4dp-5, 0x1.a3a43faf6ecccp-59}, {-0x1.c0be76051e3a5p-2, -0x1.c737cd3ea73d9p-57},
   {0x1.73c36ef7bf402p-1, -0x1.40c4dff8e4c1ep-56}, {-0x1.458cec1d1393dp-2, -0x1.c7f148cf356efp-56},
   {0x1.8ec2d305516c4p-3, 0x1.9566535c9eabp-57}},
  {{-0x1.85361b993719fp-4, -0x1.dc41ac35a716fp-58}, {-0x1.f3e2bae2cdf7dp-3, 0x1.6d5cae27956a4p-57},
   {0x1.3745220b46975p-1, 0x1.56d68f9018bb8p-60}, {-0x1.d29172b1a4407p-3, -0x1.93fc4238117bdp-58},
   {0x1.ef0f914e4a75bp-4, -0x1.6f0339a5cbb3ap-58}},
  {{-0x1.ec2ab5aa5843ap-4, -0x1.adc658df2c1c1p-62}, {-0x1.a6243a7f3534cp-5, 0x1.0dc0b707b85abp-59},
   {0x1.04116f85f23a3p-1, 0x1.517c0b25b9233p-57}, {-0x1.4bb33f1abe408p-3, 0x1.cc0c1f637cea4p-58},
   {0x1.2ecfafae59f8fp-4, 0x1.3c57c7651ae8ap-58}},
  {{-0x1.c8928613eb4f5p-4, 0x1.55f36a43c02bcp-62}, {0x1.1151b40dad4e9p-3, 0x1.67907a753aa66p-57},
   {0x1.b46b0b78660acp-2, -0x1.9bcdfa3bbcd41p-56}, {-0x1.d9cd6009ac89dp-4, -0x1.69c4d18a5c993p-59},
   {0x1.73b079d35c37cp-5, -0x1.4d3891ecef09ep-59}},
  {{-0x1.00ad2093da6e4p-4, -0x1.cbf7cf885033p-58}, {0x1.391f431d39831p-2, 0x1.8fb94bb0e7df5p-56},
   {0x1.71d5a6e677f1cp-2, 0x1.d1dc12aaa3806p-59}, {-0x1.57f6fbf9108c1p-4, 0x1.4e341fb4cef78p-61},
   {0x1.d1e33efae7a1dp-6, 0x1.c4938a6deffbep-60}},
  {{0x1.d344dabcc201ep-2, 0x1.574f453e55614p-56}, {0x1.3c3a02b015763p-2, -0x1.342e3d6a27dfap-56},
   {-0x1.f5d49f62ecfd6p-5, -0x1.07444b43ab601p-60}, {0x1.22abe7bbdf628p-6, 0x1.2cb184651725ap-63},
   {-0x1.8b52066552f48p-8, 0x1.bc2dbb1b8365dp-62}},
  {{0x1.f22e8b160e053p-3, 0x1.89c03c62a66d7p-57}, {0x1.58ae0ae32162p-1, -0x1.594df075ee813p-56},
   {0x1.028e87f2859fdp-2, 0x1.bf1ead4dde3d4p-58}, {-0x1.55b4949f3971ap-5, -0x1.2cfd594571487p-59},
   {0x1.4cfe08a2baa09p-7, 0x1.495ab3aeecafp-62}},
  {{0x1.104861734d948p-1, 0x1.32e74856dbad8p-56}, {0x1.b2445e9d82006p-1, 0x1.6e48e474ddfbfp-55},
   {0x1.b352d20042182p-3, 0x1.a8ac4f9b7c938p-60}, {-0x1.e6c5b3585790ep-6, -0x1.31a8ef26cbf2ep-60},
   {0x1.93111b206dab4p-8, -0x1.aa3ae79b1707p-63}},
  {{0x1.eed49cf014c0bp-1, 0x1.bca14c01f79aep-55}, {0x1.0718fe597659bp+0, 0x1.7d14012138c17p-55},
   {0x1.6c89e19ff8e58p-3, 0x1.12dfe29d6e296p-59}, {-0x1.56a9890298c3ap-6, -0x1.2181516eb15d6p-61},
   {0x1.deaa0ec93f6d9p-9, 0x1.e4d7a3e816168p-63}},
  {{0x1.990530fe5fa37p+0, -0x1.cf639a3a54f76p-56}, {0x1.35e029ece68dp+0, -0x1.e3db2cbb514ebp-60},
   {0x1.301f23426a05fp-3, -0x1.b5ec346a456bcp-57}, {-0x1.de5b0dd5127b5p-7, 0x1.371374acf777fp-61},
   {0x1.1843ded6af0f6p-9, -0x1.7779056d714p-64}},
  {{0x1.3ef64cb5ced7bp+1, 0x1.c3c21b0562715p-54}, {0x1.654a3f497c726p+0, 0x1.3331f28ee09bbp-54},
   {0x1.f9f5117f295a1p-4, -0x1.ee3d2bb334106p-58}, {-0x1.4bb07b47ebf8dp-7, -0x1.64c2c019b90b5p-61},
   {0x1.449d9854bac59p-10, -0x1.d0a2827bf227p-64}},
  {{0x1.de185c1178ad9p+1, 0x1.d477f1a273bfcp-55}, {0x1.9539397e34b21p+0, 0x1.9743cc0cd10f2p-54},
   {0x1.a3e2c09f7886dp-4, -0x1.17f6c25e05338p-59}, {-0x1.c98eb5fc97ce2p-8, 0x1.0a5104a9f402dp-63},
   {0x1.74b50213890abp-11, 0x1.ff0ae56647adp-65}},
  {{0x1.5c2be39a4c6fdp+2, 0x1.ff2814687494cp-52}, {0x1.c59e5d40889c7p+0, 0x1.299ee0827992ap-55},
   {0x1.5bbf97b18270ep-4, -0x1.d04ddc6346897p-60}, {-0x1.3a2d0322cf70ep-8, 0x1.53fe131154027p-65},
   {0x1.a8c6d657c0cfdp-12, -0x1.b402fb82b45efp-66}},
  {{0x1.f07834a362b11p+2, -0x1.738a86a953af8p-52}, {0x1.f68034cafc0d3p+0, 0x1.b8d6c9e2cd7d4p-56},
   {0x1.1f68e6efd00fap-4, -0x1.6083738e28e87p-61}, {-0x1.ad889b8da1552p-9, 0x1.1325e8a48689dp-64},
   {0x1.e0ae44f526429p-13, -0x1.997df9412e4aap-67}},
};

// sin(pi*x)/pi lookup table
static const double stpi[][2] = {
  {0x0p+0,0x0p+0}, {0x1.c14eff99a3ff1p-64,0x1.fff2d746c8895p-8},
  {-0x1.8c4d4c1bbe38bp-62,0x1.ffcb5e52d1f36p-7}, {-0x1.08ef2408930ebp-61,0x1.7fa7329846febp-6},
  {-0x1.14daa07929354p-60,0x1.ff2d8cc5320c7p-6}, {0x1.d845cf264d016p-60,0x1.3f3289bb44643p-5},
  {-0x1.43aa63f69aceap-60,0x1.7e9d144d37f33p-5}, {-0x1.bc90382ed68a4p-59,0x1.bdcc9ea69fc93p-5},
  {0x1.0fbc215a3c756p-60,0x1.fcb76a6ecccabp-5}, {0x1.72b75e84ab5e2p-58,0x1.1da9e1f36c497p-4},
  {-0x1.20d100fccf991p-59,0x1.3ccc01b453709p-4}, {-0x1.f7aac846eccfdp-63,0x1.5bbd477204bep-4},
  {-0x1.17799578a6651p-59,0x1.7a78edace5e27p-4}, {0x1.0c85deb5bb812p-58,0x1.98fa372a35c37p-4},
  {-0x1.67d2eb81bbf36p-60,0x1.b73c6faf2275cp-4}, {-0x1.14b2141507a9dp-63,0x1.d53aecba7bfp-4},
  {-0x1.8939cffeb036cp-58,0x1.f2f10e3ce6d42p-4}, {-0x1.2f3fbb178d1c5p-57,0x1.082d1fa7b9738p-3},
  {0x1.08479c62d3d77p-57,0x1.16b8fb743c879p-3}, {-0x1.894149dc3b5f7p-57,0x1.2519dc47527b3p-3},
  {-0x1.44dad213ab344p-60,0x1.334d8a850758dp-3}, {-0x1.2d415416bae28p-58,0x1.4151d589a490fp-3},
  {-0x1.2e0d0b51ed237p-57,0x1.4f24940025067p-3}, {0x1.e8045a3cf3213p-57,0x1.5cc3a43788a3p-3},
  {0x1.be4e50e1bf91fp-57,0x1.6a2cec76fa4bp-3}, {0x1.b1e18c1f7f635p-62,0x1.775e5b50bb365p-3},
  {-0x1.10946c1f6f484p-63,0x1.8455e7f3c6e5ap-3}, {0x1.291a88889a4e6p-59,0x1.9111927c231cfp-3},
  {-0x1.bedd6f9a25da4p-57,0x1.9d8f6441cf80bp-3}, {-0x1.ce108006670c7p-57,0x1.a9cd702648a97p-3},
  {0x1.4c65624119572p-61,0x1.b5c9d2e092baap-3}, {-0x1.a26e2a2682111p-57,0x1.c182b347bfc21p-3},
  {0x1.fce159c2bb59bp-59,0x1.ccf6429be6621p-3}, {-0x1.59b1cffa69603p-58,0x1.d822bccd7d86ep-3},
  {0x1.677083288397ap-57,0x1.e30668c31224ep-3}, {0x1.9a49696faa0ecp-57,0x1.ed9f989d4c415p-3},
  {0x1.ca323e77a3345p-58,0x1.f7eca9f938c6fp-3}, {-0x1.c702625d3863bp-57,0x1.00f6031866f76p-2},
  {0x1.180cf0e52237dp-56,0x1.05ce114cd024ap-2}, {-0x1.4be56fec860b9p-56,0x1.0a7dc060df5eep-2},
  {0x1.b5d970e5d9d07p-58,0x1.0f045755560d9p-2}, {0x1.4c32e06c67499p-58,0x1.1361238136929p-2},
  {-0x1.b512d49aedaa1p-56,0x1.179378ad51274p-2}, {-0x1.161478130996dp-58,0x1.1b9ab12ed2518p-2},
  {-0x1.25feb091e921fp-59,0x1.1f762e00ced83p-2}, {0x1.3750bc95dae67p-56,0x1.232556dcc945fp-2},
  {-0x1.257966a1044c5p-56,0x1.26a79a522d332p-2}, {-0x1.ac6af78c05e44p-57,0x1.29fc6ddcbcb72p-2},
  {0x1.71dbd64ba4f95p-56,0x1.2d234df9ec8c9p-2}, {0x1.020107d2c17bp-57,0x1.301bbe3d2b9c7p-2},
  {-0x1.9d4016f0b15c4p-56,0x1.32e5496312cfcp-2}, {0x1.f557b51b587ccp-56,0x1.357f81637a329p-2},
  {0x1.c88cee9bad9f9p-57,0x1.37e9ff82709ecp-2}, {0x1.ebde6bb284e87p-56,0x1.3a246460134f7p-2},
  {-0x1.8b1d8c40ffea3p-56,0x1.3c2e580742edap-2}, {0x1.de48797b477f2p-56,0x1.3e0789fb33cf7p-2},
  {-0x1.8bc6105a80fa5p-56,0x1.3fafb143d754bp-2}, {-0x1.9f4cc680744f3p-56,0x1.41268c791c743p-2},
  {0x1.2ed295e9d0ef2p-60,0x1.426be1cd05c06p-2}, {-0x1.4a98b72ed3789p-60,0x1.437f7f1493531p-2},
  {-0x1.5c080cdd72ddfp-56,0x1.446139cf7f413p-2}, {-0x1.b00c622ae015ep-57,0x1.4510ef2ecb654p-2},
  {0x1.8dd5ec4960646p-56,0x1.458e841a1f7dap-2}, {0x1.e1f89d1adcbc6p-56,0x1.45d9e533f6cacp-2},
  {-0x1.6b01ec5417056p-56,0x1.45f306dc9c883p-2}};

// as_sinpipid: fast-path sin(pi*x)/pi for x in (0,1)
static double as_sinpipid(double x, double *l){
  x -= 0.5;
  double ax = std::fabs(x);
  double sx = ax*128;
  double ix = coremath_roundeven(sx);
  int ky = (int)ix, kx = 64-ky;
  if(kx<2){
    static const double c[2] = {-0x1.a51a6625307d3p+0, -0x1.16cc8f2044a4ap-55};
    static const double ccl[] = {0x1.9f9cb402bc42ap-1, -0x1.86a8e46ddf78dp-3, 0x1.ac644e7aa33e6p-6};
    double z = 0.5-ax, z2 = z*z, z2l = std::fma(z,z,-z2);
    double fl = z2*(ccl[0] + z2*(ccl[1] + z2*(ccl[2]))), fh = fasttwosum(c[0], fl, &fl), e;
    fl += c[1];
    fh = muldd(z2,z2l, fh,fl, &fl);
    fh = mulddd(z, fh,fl, &fl);
    fh = fasttwosum(z, fh, &e);
    fl += e;
    *l = fl;
    return fh;
  }
  double d = ix-sx, d2 = d*d;

  double sh = stpi[kx][1], sl = stpi[kx][0];
  double chv = stpi[ky][1], clv = stpi[ky][0];

  static const double c[] = {-0x1.3bd3cc9be45dep-12, 0x1.03c1f081b5ac4p-26, -0x1.55d3c7e3bd8bfp-42, 0x1.e1f4826790653p-59};
  double c0 = -0x1.692b66e3cf6e8p-66;
  static const double s[] = {0x1.921fb54442d18p-6, -0x1.4abbce625be53p-19, 0x1.466bc67748efcp-34, -0x1.32d26e446373ap-50};
  double s0 = 0x1.1a624b88c9448p-60;

  double P = d2*(c[1] + d2*(c[2] + d2*c[3]));
  double Q = d2*(s[1] + d2*(s[2] + d2*s[3]));

  double ql, qh = fasttwosum(s[0],Q,&ql); ql += s0;
  chv = muldd(qh,ql, chv,clv, &clv);
  double tl, th = fasttwosum(c[0],P,&tl); tl += c0;
  th = mulddd(d, th,tl, &tl);
  double pl, ph = muldd(th,tl, sh,sl, &pl);
  chv = fastsum(chv,clv, ph,pl, &clv);
  chv = mulddd(d, chv,clv, &clv);
  sh = fastsum(sh,sl, chv,clv, l);
  return sh;
}

// as_logd: fast-path log for the lgamma function
static double as_logd(double x, double *l){
  static const struct {ushort c0; short c1;} B[] = {
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
  b64u64_u t;
  t.f = x;
  int ex = t.u>>52;
  if(ex==0){
    int k = __builtin_clzll(t.u);
    t.u <<= k-11;
    ex -= k-12;
  }
  int e = ex - 0x3ff;
  t.u &= ~(u64)0>>12;
  double ed = e;
  u64 i = t.u>>(52-5);
  int64_t d = t.u & (~(u64)0>>17);
  u64 j = (t.u + ((u64)B[i].c0<<33) + ((int64_t)B[i].c1*(d>>16)))>>(52-10);
  t.u |= (int64_t)0x3ff<<52;
  int i1 = j>>5, i2 = j&0x1f;
  double r = r1[i1]*r2[i2];
  double o = r*t.f, dxl = std::fma(r,t.f,-o), dxh = o-1;
  static const double c[] = {-0x1.fffffffffffd3p-2, 0x1.55555555543d5p-2, -0x1.000002bb2d74ep-2, 0x1.999a692c56e4ep-3};
  double dx = std::fma(r,t.f,-1), dx2 = dx*dx;
  double f = dx2*((c[0] + dx*c[1]) + dx2*(c[2] + dx*c[3]));
  double lt = (l1[i1][1] + l2[i2][1]) + ed*0x1.62e42fef8p-1;
  double lh = lt + dxh, ll = (lt - lh) + dxh;
  ll += ((l1[i1][0] + l2[i2][0]) + 0x1.1cf79abc9e3b4p-36*ed) + dxl;
  ll += f;
  *l = ll;
  return lh;
}

double cr_lgamma(double x){
  b64u64_u t;
  t.f = x;
  uint64_t nx = t.u<<1;
  if(nx >= 0xfeaea9b24f16a34cull){
    // |x| >= 0x1.006df1bfac84ep+1015
    if(t.u == 0x7f5754d9278b51a6ull) return 0x1.ffffffffffffep+1023 - 0x1p+969;
    if(t.u == 0x7f5754d9278b51a7ull) return 0x1.fffffffffffffp+1023 - 0x1p+969;
    if(nx>=(0x7ffull<<53)){ /* x=NaN or +/-Inf */
      if(nx==(0x7ffull<<53)) /* x=+/-Inf */
        return std::fabs(x); /* +Inf */
      return x + x; /* NaN */
    }
    if(t.u>>63)
      return 1.0/0.0; // huge negative integer
    else
      return 0x1.fp1023 * 0x1.fp1023; // overflow
  }
  double fx = std::floor(x);
  if(fx==x){ /* x is integer */
    if(x <= 0.0) {
      return 1.0/0.0;
    }
    if(x==1.0 || x==2.0) {
      return 0.0;
    }
  }
  unsigned au = nx>>38;
  double fh, fl, eps;
  if(au < ubrd[0]){ // |x|<0.5
    double ll, lh = as_logd(std::fabs(x), &ll);
    if(au<0x1da0000){ // |x|<0x1p-75
      fh = -lh;
      fl = -ll;
      eps = 1.5e-22;
    } else if(au<0x1fd0000){ // |x|<0.03125
      static const double c0[][2] = {
        {-0x1.2788cfc6fb619p-1, 0x1.6cb9a4ff7c53bp-58}, {0x1.a51a6625307d3p-1, 0x1.18722054895e9p-56},
        {-0x1.9a4d55beab2d7p-2, -0x1.74ded0474fe66p-63}, {0x1.151322ac7d848p-2, 0x1.825b3df1d5722p-56}};
      static const double q[] = {
        -0x1.a8b9c17aa5d3dp-3, 0x1.5b40cb100b9bfp-3, -0x1.2703a1e13bcbcp-3, 0x1.010b36b6afdc1p-3,
        -0x1.c8062dd09ec62p-4, 0x1.9a018c7345316p-4, -0x1.7578ea8068cc4p-4, 0x1.566b51c990008p-4};
      double z = x, z2 = z*z, z4 = z2*z2;
      double q0 = q[0] + z*q[1], q2 = q[2] + z*q[3], q4 = q[4] + z*q[5], q6 = q[6] + z*q[7];
      fl = z*((q0 + z2*q2) + z4*(q4 + z2*q6));
      fh = polydddfst(z, 4, c0, &fl);
      fh = mulddd(x, fh,fl, &fl);
      fh = sumdd(-lh,-ll, fh,fl, &fl);
      eps = 1.5e-22;
    } else {
      double xl;
      t.f = fasttwosum(1,t.f, &xl);
      au = t.u>>37;
      unsigned ou = au - ubrd[0];
      int j = ((0x157ced865ul - ou*0x150d)*ou + 0x128000000000)>>45;
      j -= au < ubrd[j];
      double z = (t.f - offs[j]) + xl, z2 = z*z, z4 = z2*z2;
      const double *q = cl[j];
      double q0 = q[0] + z*q[1], q2 = q[2] + z*q[3], q4 = q[4] + z*q[5], q6 = q[6] + z*q[7];
      fl = z*((q0 + z2*q2) + z4*(q4 + z2*q6));
      fh = polydddfst(z, 5, ch[j], &fl);
      if(j==4){ // treat the region around the root at 1
        z = -x;
        fh = mulddd(z, fh,fl, &fl);
      }
      eps = std::fabs(fh)*8.3e-20;
      fh = sumdd(-lh,-ll, fh,fl, &fl);
      eps += std::fabs(lh)*5e-22;
    }
  } else { // |x| >= 0.5
    double ax = std::fabs(x);
    if(au>=ubrd[19]) {  // |x|>=8.29541 asymptotic expansion
      double ll, lh = as_logd(ax, &ll);
      lh -= 1;
      if(au>=0x2198000){ // x >= 0x1p52
        if(au>=0x3fabaa6) lh = fasttwosum(lh,ll,&ll);
        double hlh = lh*0.5;
        lh = mulddd(ax, lh,ll, &ll);
        ll -= hlh;
      } else {
        lh = mulddd(ax-0.5, lh,ll, &ll);
      }
      static const double c[][2] = {
        {0x1.acfe390c97d6ap-2, -0x1.1d9792ced423ap-58}, {0x1.55555555554c1p-4, -0x1.0143af34001bdp-59}};
      static const double q[] = {
        -0x1.6c16c1697de08p-9, 0x1.a019f47b230fdp-11, -0x1.380aab821e42ep-11,0x1.b617d2c5b5b66p-11,
        -0x1.a7fd66a05ccfcp-10};
      lh = fastsum(lh,ll, c[0][0], c[0][1], &ll);
      if(ax<0x1p100){
        double zh = 1.0/ax, zl = std::fma(zh,-ax, 1.0)*zh;
        double z2h = zh*zh, z4h = z2h*z2h;
        double q0 = q[0] + z2h*q[1], q2 = q[2] + z2h*q[3], q4 = q[4];
        fl = z2h*(q0 + z4h*(q2 + z4h*q4));
        fh = fasttwosum(c[1][0], fl, &fl); fl += c[1][1];
        fh = muldd(fh,fl, zh,zl, &fl);
      } else {
        fh = 0;
        fl = 0;
      }
      fh = fastsum(lh,ll, fh,fl, &fl);
      eps = std::fabs(fh)*4.5e-20;
    } else {// x in [0.5, 8.29541] range
      unsigned ou = au - ubrd[0];
      int j = ((0x157ced865ul - ou*0x150d)*ou + 0x128000000000)>>45;
      j -= au < ubrd[j];
      double z = ax - offs[j], z2 = z*z, z4 = z2*z2;
      const double *q = cl[j];
      double q0 = q[0] + z*q[1], q2 = q[2] + z*q[3], q4 = q[4] + z*q[5], q6 = q[6] + z*q[7];
      fl = z*((q0 + z2*q2) + z4*(q4 + z2*q6));
      fh = polydddfst(z, 5, ch[j], &fl);
      if(j==4){ // treat the region around the root at 1
        z = 1 - ax;
        fh = mulddd(z, fh,fl, &fl);
      }
      if(j==10){ // treat the region around the root at 2
        z = ax - 2;
        fh = mulddd(z, fh,fl, &fl);
      }
      eps = std::fabs(fh)*8.3e-20 + 1e-24;
    }
    if(t.u>>63){ // x<0 so use reflection formula
      double sl, sh = as_sinpipid(x - std::floor(x), &sl);
      sh = mulddd(-x, sh,sl, &sl);
      double ll, lh = as_logd(sh, &ll);
      ll += sl/sh;
      fh = -sumdd(fh,fl,lh,ll, &fl);
      fl = -fl;
      eps += std::fabs(lh)*4e-22;
    }
  }
  // Return fast result directly (accurate path removed)
  double ub = fh + (fl + eps);
  return ub;
}
