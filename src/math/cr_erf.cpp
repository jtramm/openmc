// Fast-path erf function for binary64 values.
// Derived from the CORE-MATH project (MIT License).
// https://core-math.gitlabpages.inria.fr/

#include "coremath_utils.h"

// CH+CL is a double-double approximation of 2/sqrt(pi) to nearest
static const double CH = 0x1.20dd750429b6dp+0;
static const double CL = 0x1.1ae3a914fed8p-56;

// Add a + b, such that *hi + *lo approximates a + b.
// Assumes |a| >= |b|.
static inline void fast_two_sum(double *hi, double *lo, double a, double b) {
  double e;
  *hi = a + b;
  e = *hi - a;
  *lo = b - e;
}

/* for 1 <= i < 95, p = C[i-1] is a degree-10 polynomial approximation of
   erf(i/16+1/32+x) for -1/32 <= x <= 1/32, where the coefficients of
   degree 0 and 1 are double-double: the coefficient of degree 0 is p[0]+p[1],
   that of degree 1 is p[2]+p[3], that of degree 2 is p[4], ..., that of
   degree 10 is p[12].
   Table generated with print_table(1,95,1/32,10,1,rel=true) from erf.sage.
*/
static const double C[94][13] = {
   {0x1.b0081148a873ap-4, -0x1.f0295f16ba5d8p-61, 0x1.1e565bca400d4p+0, -0x1.62d0ac26c78d3p-54, -0x1.ad8189af6013dp-4, -0x1.7712743c42914p-2, 0x1.aafd4760d7634p-5, 0x1.ba14988b4127ep-4, -0x1.1afcdb244078ap-6, -0x1.9d72ee25cf211p-6, 0x1.19502f7beca8fp-8, 0x1.3b955bfd46624p-8, -0x1.a4e2d4d32228bp-11}, /* i=1 69.005 */
   {0x1.662a0bdf7a89fp-3, -0x1.ef7bc5856c2d4p-59, 0x1.19e5e92b964abp+0, 0x1.cca4dec08a64p-57, -0x1.605f63767bdd6p-3, -0x1.6582e9b69c9a9p-2, 0x1.5aa32b580ec64p-4, 0x1.97594c2593d3ep-4, -0x1.c69c62749fb7fp-6, -0x1.6fa7f611aacdcp-6, 0x1.bf1e628a4606ep-8, 0x1.0e50e4329e8a9p-8, -0x1.68ca9c1954b4cp-10}, /* i=2 70.057 */
   {0x1.f190aa85540e2p-3, -0x1.e522ac9f718e6p-57, 0x1.135e3075d076bp+0, -0x1.e2d8ed30e4a48p-57, -0x1.e1e4d4ce2ccfbp-3, -0x1.4c04e66e0d59bp-2, 0x1.d2855d59988e8p-4, 0x1.659a35f29781ap-4, -0x1.2cf6266a634c8p-5, -0x1.2ef4180b1f3fap-6, 0x1.23199a6da60e3p-7, 0x1.9e80d13a3368cp-9, -0x1.ba4e4eff641ddp-10}, /* i=3 70.889 */
   {0x1.3c9aa8b84bedap-2, 0x1.38ec27d3e582p-58, 0x1.0ae54fa490723p+0, -0x1.d016b7bc67433p-54, -0x1.2c41f99922807p-2, -0x1.2b900b640a201p-2, 0x1.1c6c7eef8fa14p-3, 0x1.277ad7822021ep-4, -0x1.66c9b2023b9dfp-5, -0x1.bf7e7b4e8559ep-7, 0x1.53005de4b5751p-7, 0x1.0737c6ba405fp-9, -0x1.06ccc916b15dcp-9}, /* i=4 70.343 */
   {0x1.7e15944d9d3e4p-2, -0x1.95f819cf77862p-57, 0x1.00abcf3e187a9p+0, 0x1.5860d868dc542p-55, -0x1.60ec3cf561a89p-2, -0x1.05599bafe4eccp-2, 0x1.451ef6280e70fp-3, 0x1.c06c6e434be6fp-5, -0x1.8e2d73679096fp-5, -0x1.0ea4a60550d9cp-7, 0x1.6c911882cc99cp-7, 0x1.8c65a9990353bp-11, -0x1.1e8a88301a7b5p-9}, /* i=5 69.074 */
   {0x1.bccfec24855b8p-2, -0x1.472ab1c2b898cp-56, 0x1.e9d5a8e4c934ep-1, -0x1.9a002a2814a72p-56, -0x1.8dfd9939e37afp-2, -0x1.b588d8dc5bb96p-3, 0x1.62338788aee97p-3, 0x1.26cf85bc6dff9p-5, -0x1.a1bcaa91da902p-5, -0x1.5b4a7d42d0f64p-9, 0x1.6edef7de2b68dp-7, -0x1.037b458e2da8cp-11, -0x1.e8d6001a54334p-10}, /* i=6 70.183 */
   {0x1.f86faa9428f9dp-2, 0x1.9996c0c376e32p-56, 0x1.cfc41e36c7df9p-1, -0x1.9be994724ea34p-56, -0x1.b2c7dc535b619p-2, -0x1.5a9de93f9c0d5p-3, 0x1.7317958d24aaep-3, 0x1.133e02ab7d777p-6, -0x1.a155bbde32db8p-5, 0x1.72049c0cc8525p-9, 0x1.5adde5c722d85p-7, -0x1.b0a7ec5dc80fcp-10, -0x1.aa9393b806535p-10}, /* i=7 70.135 */
   {0x1.1855a5fd3dd5p-1, 0x1.8f6964e67d61ap-55, 0x1.b3aafcc27502ep-1, -0x1.a9dd26edea8a2p-56, -0x1.cee5ac8e9c531p-2, -0x1.fa02983c853d1p-4, 0x1.77cd75ec731p-3, -0x1.fa6f82f9333b7p-10, -0x1.8e0db5528e559p-5, 0x1.00bf7062212bcp-7, 0x1.3319e670adc9fp-7, -0x1.58833e091aa36p-9, -0x1.8f99b6e81e8f5p-10}, /* i=8 69.816 */
   {0x1.32a54cb8db67bp-1, -0x1.96221f7e18978p-57, 0x1.96164fafd8de3p-1, 0x1.0887f82841accp-56, -0x1.e23a7ea0d187ep-2, -0x1.3f5ee1564be49p-4, 0x1.70e469de06907p-3, -0x1.3da6878ae6fd8p-6, -0x1.6a0d076468415p-5, 0x1.8cf081f1fc304p-7, 0x1.f6d62866525e6p-8, -0x1.b93149d5701a4p-9, -0x1.1a6c1a9f7ea73p-10}, /* i=9 70.229 */
   {0x1.4b13713ad3513p-1, 0x1.e944ee1b212e4p-57, 0x1.7791b886e7403p-1, -0x1.da43cb53d911cp-57, -0x1.ecef42310f844p-2, -0x1.15c3c5ce705dfp-5, 0x1.5f6890affa468p-3, -0x1.1da642fabd4dap-5, -0x1.385991202c7ebp-5, 0x1.fa4f37fc7c6d4p-7, 0x1.7156b4e430998p-8, -0x1.f546a4377d648p-9, -0x1.32e4e5abb1e1ap-11}, /* i=10 70.743 */
   {0x1.61955607dd15dp-1, 0x1.98ff39319ab83p-55, 0x1.58a445da7c74cp-1, 0x1.08ec8e156809bp-55, -0x1.ef6c246a12e7ep-2, 0x1.e83e0da03048p-8, 0x1.44cc65df8bfc7p-3, -0x1.87d3c8dd62c82p-5, -0x1.f9271a8a1d4e2p-6, 0x1.225234c1c0a0ep-6, 0x1.c0b0e055a0c48p-9, -0x1.0585251f84919p-8, -0x1.85bfb02436e0fp-13}, /* i=11 70.022 */
   {0x1.762870f720c6fp-1, 0x1.118b1ba6da9a7p-55, 0x1.39ccc1b136d5ap-1, 0x1.faa9371c0dd8p-58, -0x1.ea4feea4e5addp-2, 0x1.715e595343353p-5, 0x1.22cdbdb4cdd0cp-3, -0x1.da50ae547e69ep-5, -0x1.75578f87f217dp-6, 0x1.353319c65f251p-6, 0x1.39db53a2d03d5p-10, -0x1.fc0364ce1787p-9, 0x1.272bc18b0f2cep-12}, /* i=12 70.545 */
   {0x1.88d1cd474a2ep-1, 0x1.6f571ada77d52p-55, 0x1.1b7e98fe26217p-1, 0x1.952bd607eb12ep-56, -0x1.de65a22ce0587p-2, 0x1.40686a3f3dc2bp-4, 0x1.f6b0cb6926c42p-4, -0x1.09c7caecd317dp-4, -0x1.da668f759eaeap-7, 0x1.364e72035e80ap-6, -0x1.d421975736447p-11, -0x1.cc98454e96141p-9, 0x1.a8860fdf17259p-11}, /* i=13 71.537 */
   {0x1.999d4192a5715p-1, -0x1.c888a5759a92cp-55, 0x1.fc3ee5d1524bp-2, -0x1.27e60faac0278p-58, -0x1.cc990045b293fp-2, 0x1.b37338e6ac814p-4, 0x1.a0d11fe9ba61ap-4, -0x1.19bb2ca3816bap-4, -0x1.a0b7d94791f03p-8, 0x1.274a59774d5e6p-6, -0x1.64adea7b36f57p-9, -0x1.83684bd8ef173p-9, 0x1.38905afd229ffp-10}, /* i=14 70.033 */
   {0x1.a89c850b7d54dp-1, -0x1.e2752ebf0cd02p-55, 0x1.c40b0729ed548p-2, -0x1.c4c1c4927306dp-56, -0x1.b5eaaef09de9dp-2, 0x1.0847c7dad86afp-3, 0x1.47de0a4f796cap-4, -0x1.1d9de8b54a3ecp-4, 0x1.33252fb810c7cp-10, 0x1.0ab3e329ded2fp-6, -0x1.12d82076274edp-8, -0x1.287bb4a78d728p-9, 0x1.57d31bd574dap-10}, /* i=15 70.519 */
   {0x1.b5e62fce16095p-1, 0x1.bc3cff4400364p-56, 0x1.8eed36b886d93p-2, 0x1.ea7e17b96436dp-56, -0x1.9b64a06e4b1p-2, 0x1.2bb6e2c74d4fep-3, 0x1.dee322c062364p-5, -0x1.169960d5a983dp-4, 0x1.feab4ad0bfc14p-8, 0x1.c76eb94b07a5fp-7, -0x1.584474ae8f994p-8, -0x1.88df75be9251fp-10, 0x1.4edef5031709p-10}, /* i=16 72.437 */
   {0x1.c194b1d49a184p-1, -0x1.6770a58b27668p-57, 0x1.5d4fd33729015p-2, -0x1.6db7d76e9e97bp-56, -0x1.7e0f4f0454d97p-2, 0x1.444bc66c35bc4p-3, 0x1.356dbb543255p-5, -0x1.0643de6e8c574p-4, 0x1.b2e1f789415e4p-7, 0x1.6ba6d9f4af32fp-7, -0x1.8138bf4573a6ap-8, -0x1.7e6e52a583322p-11, 0x1.0f87322fa18a3p-10}, /* i=17 70.211 */
   {0x1.cbc54b476248dp-1, 0x1.1a5083b01ec0dp-55, 0x1.2f7cc3fe6f423p-2, 0x1.9fbb4b774e85dp-56, -0x1.5ee8429e30a49p-2, 0x1.52a8395f9627p-3, 0x1.313759f199499p-6, -0x1.dcf844d90282cp-5, 0x1.1e45f25ab54a1p-6, 0x1.091cb68a58665p-7, -0x1.8ea40b0ac8b7bp-8, -0x1.6b91b1bf985f2p-17, 0x1.158d9c0e1c327p-10}, /* i=18 73.007 */
   {0x1.d4970f9ce00d9p-1, -0x1.56704209fca7p-56, 0x1.059f59af7a906p-2, -0x1.0ce27da57f153p-56, -0x1.3eda354ddd5ffp-2, 0x1.57b85ad436067p-3, 0x1.8e90c2a157e8dp-10, -0x1.a2893b28f4033p-5, 0x1.4d6af4484a1cbp-6, 0x1.4ccee8c8b1f57p-8, -0x1.83304b9e2e312p-8, 0x1.40cb679d0a832p-11, 0x1.d6b5f4bdef24bp-11}, /* i=19 75.628 */
   {0x1.dc29fb60715afp-1, 0x1.ab029f047a087p-55, 0x1.bf8e1b1ca2279p-3, 0x1.0426e10a38p-65, -0x1.1eb7095e57e16p-2, 0x1.549ea6f7a013fp-3, -0x1.b10f20d110552p-7, -0x1.61420b5b34a55p-5, 0x1.677b7ea46c6f2p-6, 0x1.24f9940ffd84p-9, -0x1.6304445e5f6cap-8, 0x1.222fabfa75bbp-10, 0x1.fdcf55be3c03ep-12}, /* i=20 70.096 */
   {0x1.e29e22a89d766p-1, 0x1.bcc9d569ed217p-55, 0x1.7bd5c7df3fe9cp-3, 0x1.488f3b06e1394p-57, -0x1.fe674493fde22p-3, 0x1.4a9feacf7e222p-3, -0x1.a0082c90a1b0dp-6, -0x1.1cf0e7655f99ap-5, 0x1.6e3396f04262p-6, -0x1.3a2d2cdd5650dp-12, -0x1.334add14b9a31p-8, 0x1.7e12864580191p-10, 0x1.dae75c3e2be46p-12}, /* i=21 74.177 */
   {0x1.e812fc64db369p-1, 0x1.3c66a6a23d9a5p-55, 0x1.3fda6bc016994p-3, 0x1.586ddaff31a18p-57, -0x1.c1cb27861fc79p-3, 0x1.3b1051230b982p-3, -0x1.1e645a2a638ffp-5, -0x1.b1f643b14fd89p-6, 0x1.64297d7a66c2p-6, -0x1.3e365adfbccaep-9, -0x1.f2aa2b3ef5ec2p-9, 0x1.b3339ee2c8c49p-10, 0x1.0ef571022311p-13}, /* i=22 71.398 */
   {0x1.eca6ccd709544p-1, 0x1.f3de8f195347p-57, 0x1.0b3f52ce8c383p-3, 0x1.d1234b508bcfbp-57, -0x1.8885019f5df29p-3, 0x1.274275fc87eaep-3, -0x1.57f7386bfd263p-5, -0x1.30769f45aaa8bp-6, 0x1.4c8231709cfeep-6, -0x1.0c2c99c75913fp-8, -0x1.7514483efc09p-9, 0x1.c3ebcf121a533p-10, 0x1.de2f1801b848p-17}, /* i=23 73.855 */
   {0x1.f0762fde45ee6p-1, 0x1.9c3612a14fb77p-55, 0x1.bb1c972f23e5p-4, 0x1.ba69c564971e1p-58, -0x1.5341e3c0177b6p-3, 0x1.107929f6e7528p-3, -0x1.7e1b362eacfe6p-5, -0x1.73b61e487b8a9p-7, 0x1.2aa763e0343a9p-6, -0x1.59a388fd2272dp-8, -0x1.eea3c7f50e8dep-10, 0x1.b5026fd87d0cap-10, -0x1.0f2c660125dc6p-12}, /* i=24 71.363 */
   {0x1.f39bc242e43e6p-1, -0x1.dbae0fd9b967dp-55, 0x1.6c7e64e7281cbp-4, 0x1.aa87392dc4c2p-58, -0x1.2274b86833f6ep-3, 0x1.efb890e5b6633p-4, -0x1.92c7dbb880b5cp-5, -0x1.4547708842f2bp-8, 0x1.02047ab6c08c4p-6, -0x1.888355239e9ecp-8, -0x1.0313bb85e86e1p-10, 0x1.8ced9ddf3d834p-10, -0x1.2d520499bd799p-12}, /* i=25 73.472 */
   {0x1.f62fe80272419p-1, -0x1.b7c2d17fc31d3p-55, 0x1.297db960e4f63p-4, -0x1.22bea9385fad9p-58, -0x1.ecb83b087b37bp-4, 0x1.bce18363bbbb9p-4, -0x1.985aaf97891cbp-5, 0x1.cd95f2aa8601ap-12, 0x1.ab9d43270d20fp-7, -0x1.9b93410d46789p-8, -0x1.9b530b472cadfp-13, 0x1.52f54de527458p-10, -0x1.6844d43c7d693p-12}, /* i=26 72.129 */
   {0x1.f848acb544e95p-1, -0x1.b27aa2c376c3cp-55, 0x1.e1d4cf1e2450ap-5, -0x1.783e14555c1e9p-59, -0x1.9e12e1fde7354p-4, 0x1.8a27806de834fp-4, -0x1.91674e13a339ap-5, 0x1.3bc75e8f9d448p-8, 0x1.51b4d09ac47b8p-7, -0x1.96dc7b5f9bd66p-8, 0x1.e16520532bde9p-12, 0x1.0e742b323f434p-10, -0x1.ac319bfed91d4p-12}, /* i=27 72.979 */
   {0x1.f9f9ba8d3c733p-1, 0x1.cd5790ff03ab3p-55, 0x1.83298d717210ep-5, 0x1.740e2b04276bfp-59, -0x1.58d101f909971p-4, 0x1.58f1456f7db5ep-4, -0x1.808d17b33b814p-5, 0x1.0c1bdce673b1p-7, 0x1.f5ff1c06e9df2p-8, -0x1.7f26b8865f398p-8, 0x1.f87060e6f646p-11, 0x1.8c6056bea9223p-11, -0x1.e3499a90b84f5p-12}, /* i=28 73.403 */
   {0x1.fb54641aebbc9p-1, -0x1.79975513f67e7p-55, 0x1.34ac36ad8dafep-5, 0x1.902fb5363d36p-63, -0x1.1c8ec267fe9e2p-4, 0x1.2a52c5d83c05p-4, -0x1.68541b2c0582cp-5, 0x1.5afe422155ad5p-7, 0x1.56303c111cd8ap-8, -0x1.597ead749c06ap-8, 0x1.57b0870a7b4cfp-10, 0x1.ffc0efb0ac024p-12, -0x1.9e3ea349ab39ep-12}, /* i=29 73.624 */
   {0x1.fc67bcf2d7b8fp-1, -0x1.0d2748f976e8cp-55, 0x1.e85c449e377f3p-6, -0x1.cb7ccd2616394p-60, -0x1.d177f166cce53p-5, 0x1.fe23b75845cdfp-5, -0x1.4b120f9dde895p-5, 0x1.8d9906d138bd5p-7, 0x1.9201b7e469e83p-9, -0x1.2aceacb2954fp-8, 0x1.8d4e8140dc518p-10, 0x1.00a33f7e93047p-12, -0x1.72b7adfeee575p-12}, /* i=30 74.601 */
   {0x1.fd40bd6d7a785p-1, 0x1.60d45e630998fp-55, 0x1.7f5188610ddc8p-6, -0x1.60e8565137ecbp-60, -0x1.7954423f89a51p-5, 0x1.af5baae337ae6p-5, -0x1.2ad77b77d17dcp-5, 0x1.a7b8c4a8d53fep-7, 0x1.4593adc5d737ap-10, -0x1.ef1cf14455c9cp-9, 0x1.a1a04ce289b4bp-10, 0x1.3d14f37840954p-15, -0x1.50b861df174eep-12}, /* i=31 73.251 */
   {0x1.fdea6e062d0c9p-1, -0x1.64c70f379f67p-56, 0x1.2a875b5ffab56p-6, 0x1.531231987c3b8p-63, -0x1.2f3178cd7aa03p-5, 0x1.68d1c45b96efep-5, -0x1.09648dd332653p-5, 0x1.ad8b148089c02p-7, -0x1.f00fa01e6ca19p-13, -0x1.8718785b346p-9, 0x1.9a7b0da775387p-10, -0x1.090258ede6532p-13, -0x1.b3980b454d442p-13}, /* i=32 73.521 */
   {0x1.fe6e1742f7cf6p-1, -0x1.cebced8a49e04p-55, 0x1.cd5ec93c12432p-7, -0x1.bb85326a5eff3p-61, -0x1.e2ff3aaae31e4p-6, 0x1.2aa4e5824252p-5, -0x1.d049824fc44dbp-6, 0x1.a34eda0fc336ep-7, -0x1.682d8d1801582p-10, -0x1.239bf51e17ea8p-9, 0x1.7e761274bf059p-10, -0x1.01e715d70d49fp-12, -0x1.4d89f3d9c30d5p-13}, /* i=33 76.244 */
   {0x1.fed37386190fbp-1, 0x1.72b1549ea44eep-55, 0x1.61beae53b72b7p-7, 0x1.401790f84b248p-64, -0x1.7d6193f2417adp-6, 0x1.e947279e4a43bp-6, -0x1.9060301092cdcp-6, 0x1.8d14d4bdaa7f4p-7, -0x1.1f795ac88038p-9, -0x1.9222edb6bd145p-10, 0x1.53f95c7b01615p-10, -0x1.529b07d094e1dp-12, -0x1.5b533d0382e2p-14}, /* i=34 74.706 */
   {0x1.ff20e0a7ba8c2p-1, -0x1.03f86c5a13f78p-57, 0x1.0d1d69569b82dp-7, -0x1.a5e866bd1366ep-62, -0x1.2a8ca0dc14852p-6, 0x1.8cc071b719c43p-6, -0x1.54a148886e917p-6, 0x1.6e91361df3c9ep-7, -0x1.65c02e0d08291p-9, -0x1.e94b0adc3b1cap-11, 0x1.210781b57b089p-10, -0x1.7b88f8c82fbffp-12, -0x1.68df27e9a1688p-15}, /* i=35 74.851 */
   {0x1.ff5b8fb26f5f6p-1, -0x1.7e917ec20b615p-55, 0x1.9646f35a76624p-8, -0x1.f771f32fd191bp-62, -0x1.cf68ed932f081p-7, 0x1.3e8735b5b73b1p-6, -0x1.1e1611aabcbeap-6, 0x1.4afd8cd100d7p-7, -0x1.8c72005b1cfcfp-9, -0x1.c6a7216b336aap-12, 0x1.d577412afc2e2p-11, -0x1.836a0c0e10a99p-12, 0x1.a8f39f410252ap-19}, /* i=36 75.151 */
   {0x1.ff87b1913e853p-1, -0x1.3ca98afc58454p-56, 0x1.30499b503957fp-8, -0x1.d1eabb1c04f5p-64, -0x1.6496420203331p-7, 0x1.fa73d7eb1b70dp-7, -0x1.daa3005c2d3fep-7, 0x1.250942c31c3adp-7, -0x1.997578dc240a8p-9, -0x1.3904177639e63p-15, 0x1.6a6ed488a1f54p-11, -0x1.71cf0c5789c7dp-12, 0x1.43cb84231ab1cp-15}, /* i=37 75.856 */
   {0x1.ffa89fe5b3625p-1, 0x1.934b2bcb7f9a3p-55, 0x1.c4412bf4b8f0bp-9, -0x1.bbcc9dca4ec6p-67, -0x1.100f34713740dp-7, 0x1.8ebda0768e8e6p-7, -0x1.850c68e8e5c3cp-7, 0x1.fdac8346071b3p-8, -0x1.929de70d00321p-9, 0x1.10c7101bc52d8p-12, 0x1.070f7e89ec1e2p-11, -0x1.4e4b3dcf4f08dp-12, 0x1.f0d43b9869b19p-15}, /* i=38 75.476 */
   {0x1.ffc10194fcb64p-1, 0x1.ea14750ac9b59p-55, 0x1.4d78bba8ca5fdp-9, 0x1.4d9a93566b5b4p-65, -0x1.9ba107a459ce4p-8, 0x1.36f273fbd909bp-7, -0x1.3b38708f7bef7p-7, 0x1.b3fdff1de2112p-8, -0x1.7d55d55d262d8p-9, 0x1.eae5e05e74fccp-12, 0x1.5ebc1e53214a9p-12, -0x1.1fd7c1cd5d63ep-12, 0x1.49559a04c8568p-14}, /* i=39 76.483 */
   {0x1.ffd2eae369a07p-1, -0x1.83b09df7f7db4p-57, 0x1.e7f232d9e263p-10, 0x1.a26ac725599e5p-64, -0x1.34c7442de142bp-8, 0x1.e066bed09942fp-8, -0x1.f914f2c60b9bbp-8, 0x1.6f4662f6be13bp-8, -0x1.5e664591d6604p-9, 0x1.3a1598d880f36p-11, 0x1.965b2e78a4544p-13, -0x1.d8db42b193729p-13, 0x1.449172919598ep-14}, /* i=40 76.603 */
   {0x1.ffdff92db56e5p-1, -0x1.8aeef4ee0690ap-56, 0x1.6235fbd7a4345p-10, -0x1.11380fe434056p-65, -0x1.cb5e029ba8f3dp-9, 0x1.6fa4c7ef470e9p-8, -0x1.903a08305eebp-8, 0x1.30f12c83fdb23p-8, -0x1.39d769a774af1p-9, 0x1.5d79439ceaefdp-11, 0x1.5326883e7dfebp-14, -0x1.7199782285958p-13, 0x1.47181c8911603p-14}, /* i=41 77.827 */
   {0x1.ffe96a78a04a9p-1, -0x1.2816fe4528f9bp-55, 0x1.fe41cd9bb4eeep-11, 0x1.e3be508cae7ecp-66, -0x1.52d7b2896626ap-9, 0x1.16c192d8803dcp-8, -0x1.39bfce9b4ecc2p-8, 0x1.f376a554e5decp-9, -0x1.12e67cb7aa486p-9, 0x1.66d6e460b1614p-11, -0x1.54f70e4bde32bp-18, -0x1.10e125571fe1ep-13, 0x1.2842d46eb9f29p-14}, /* i=42 78.426 */
   {0x1.fff0312b010b5p-1, 0x1.155dec9cdc96bp-55, 0x1.6caa0d3582fe9p-11, -0x1.97d95851163fcp-67, -0x1.efb729f4be121p-10, 0x1.a2da7cec01564p-9, -0x1.e6c27ad2b1cep-9, 0x1.93b1f34b17723p-9, -0x1.d8179cd2ad34fp-10, 0x1.5cf51e0add9bbp-11, -0x1.16d8f4b5119c7p-14, -0x1.768557564f5f5p-14, 0x1.f4fc9dde73f24p-15}, /* i=43 78.437 */
   {0x1.fff50456dab8cp-1, -0x1.a197a986f0dep-58, 0x1.0295ef6591848p-11, -0x1.262bd83520706p-66, -0x1.679880e93e5c4p-10, 0x1.37d38e3a705afp-9, -0x1.75b371a264745p-9, 0x1.4231c3bfe3e65p-9, -0x1.8e184d4921105p-10, 0x1.45d5b5a7f77fap-11, -0x1.bf8ece4afedd2p-14, -0x1.ccd677aaa82f7p-15, 0x1.9e5241d5b6b15p-15}, /* i=44 80.203 */
   {0x1.fff86cfd3e657p-1, -0x1.2e06adb26f84ep-56, 0x1.6be02102b352p-12, 0x1.448bcfd3cfe0cp-68, -0x1.02b15777eb7c5p-10, 0x1.cc1d886874d5bp-10, -0x1.1bff70664651dp-9, 0x1.fc0f76c943696p-10, -0x1.4a22286622d3ep-10, 0x1.268887688a6e6p-11, -0x1.0fa2692fd7da2p-13, -0x1.cc13d1a82f742p-16, 0x1.4153e6537aae5p-15}, /* i=45 79.393 */
   {0x1.fffad0b901755p-1, 0x1.70d5c9a92b65cp-57, 0x1.fc0d55470cf51p-13, -0x1.6f2b03553d4c8p-67, -0x1.7121aff59f6a1p-11, 0x1.506d6992fc8ffp-10, -0x1.ab596015fc183p-10, 0x1.8bdd79a098723p-10, -0x1.0d88da9deb868p-10, 0x1.031cdd07e4507p-11, -0x1.22fc41430a37dp-13, -0x1.b5cc9546afcecp-18, 0x1.d7ea1c7b8fdb6p-16}, /* i=46 79.358 */
   {0x1.fffc7a37857d2p-1, -0x1.97b30fd4b6b48p-56, 0x1.5feada379d8b7p-13, -0x1.0546c4da57036p-67, -0x1.05304df546ed8p-11, 0x1.e79c081b79ebcp-11, -0x1.3e5dc1062db15p-10, 0x1.30eb20ccc1f98p-10, -0x1.b1b06c20a060dp-11, 0x1.bd52fbd55e0efp-12, -0x1.214afb8835b23p-13, 0x1.19ae9d16650ap-17, 0x1.42d933ee154fdp-16}, /* i=47 79.5 */
   {0x1.fffd9fdeabccep-1, 0x1.0c43c3bc59762p-55, 0x1.e3bcf436a1a95p-14, -0x1.6458a28a3f9b6p-69, -0x1.6e95311166825p-12, 0x1.5e3edf674e2dbp-11, -0x1.d5be6d15abe3ap-11, 0x1.d07da13e640c2p-11, -0x1.58106cc648748p-11, 0x1.76c840985e5ebp-12, -0x1.111de112b1a2ep-13, 0x1.315fc34053fbdp-16, 0x1.939439a75a553p-17}, /* i=48 80.809 */
   {0x1.fffe68f4fa777p-1, 0x1.2f21786b7644p-60, 0x1.49e17724f4d41p-14, 0x1.747684f0023e4p-69, -0x1.fe48c44d2ab81p-13, 0x1.f2bd95d72a532p-12, -0x1.57389188a71a9p-11, 0x1.5decc4058f7a1p-11, -0x1.0d559cf0f2957p-11, 0x1.3583904af6f83p-12, -0x1.efd7979333337p-14, 0x1.904cf9fa5c1f6p-16, 0x1.a13a094bd56a2p-18}, /* i=49 81.559 */
   {0x1.fffef1960d85dp-1, -0x1.f7cc78053f6adp-55, 0x1.be6abbb10a5aap-15, -0x1.e50b219d40126p-70, -0x1.60403819b22b8p-13, 0x1.5fff1dde5305ep-12, -0x1.f0c93c73e7f42p-12, 0x1.04cbf67af6c26p-11, -0x1.a04893510426cp-12, 0x1.f66b51a7bc4ap-13, -0x1.b410d7f2fd319p-14, 0x1.b99f9eb427956p-16, 0x1.f26fcffb14441p-20}, /* i=50 82.307 */
   {0x1.ffff4db27f146p-1, 0x1.ddecdd5e1d408p-55, 0x1.2bb5cc22e5db6p-15, 0x1.c5112eca8acdep-70, -0x1.e258948829ed1p-14, 0x1.ec8a8e59d9d5bp-13, -0x1.6425722b9f3cdp-12, 0x1.80a83a7103b4bp-12, -0x1.3dbb9374004f9p-12, 0x1.913b301d37bdep-13, -0x1.7563b0d94459fp-14, 0x1.bc01eea9a10bep-16, -0x1.3df26463df6a5p-20}, /* i=51 82.355 */
   {0x1.ffff8b500e77cp-1, -0x1.1014e1f83ed4cp-56, 0x1.8f4ccca7fc90dp-16, 0x1.a5d4ec8b9de43p-70, -0x1.478cffe1cd2edp-14, 0x1.559f04ad4de62p-13, -0x1.f9e163b15c466p-13, 0x1.18bda8b8c1315p-12, -0x1.df381bd3c058ep-13, 0x1.3b94f531bb6bep-13, -0x1.385f32481ed94p-14, 0x1.a414bd2b7cb3cp-16, -0x1.ac2bbe30f8767p-19}, /* i=52 83.754 */
   {0x1.ffffb43555b5fp-1, 0x1.c17f83b8d73a2p-55, 0x1.07ebd2a2d2844p-16, 0x1.d1bbdc704f49bp-70, -0x1.b93e442837f52p-15, 0x1.d5cf1514977f3p-14, -0x1.63f5eb46877fdp-13, 0x1.95a0411e668b1p-13, -0x1.652e5f2a88269p-13, 0x1.e950ddb7f5444p-14, -0x1.ffeb9383bdb3dp-15, 0x1.7c24392346fddp-16, -0x1.1f3b3254d723p-18}, /* i=53 83.181 */
   {0x1.ffffcf23ff5fcp-1, -0x1.b18a8b25039c4p-55, 0x1.5a2adfa0b4bc4p-17, 0x1.eb6d61aaaf95cp-71, -0x1.26c8826ed9e85p-15, 0x1.40473571d5383p-14, -0x1.f057dbf365c0ap-14, 0x1.2217929fed933p-13, -0x1.07324014ddb42p-13, 0x1.762758a56d654p-14, -0x1.9ba250c662e9p-15, 0x1.4c25759179e3dp-16, -0x1.3e800358f1a7bp-18}, /* i=54 83.812 */
   {0x1.ffffe0bd3e852p-1, -0x1.d7ece4ab5315p-58, 0x1.c282cd3957edap-18, 0x1.eb3cf4fd1428p-73, -0x1.86ad6df7ba401p-16, 0x1.b0f313eeb65a6p-15, -0x1.56e457745d637p-14, 0x1.9ad1f65a78253p-14, -0x1.7f92ad8542929p-14, 0x1.1a5578c0d30b3p-14, -0x1.4548d876bb0a3p-15, 0x1.19e60bf53b25ap-16, -0x1.3f1745170e2d3p-18}, /* i=55 85.046 */
   {0x1.ffffec2641a9ep-1, -0x1.e7ba4fdaaa8c8p-55, 0x1.22df298214423p-18, -0x1.a9d49552152a4p-74, -0x1.00c902a4d5e27p-16, 0x1.22234eb745941p-15, -0x1.d57a2be01db67p-15, 0x1.200c2ffad65f1p-14, -0x1.147585d43f49ap-14, 0x1.a4b07aec797e9p-15, -0x1.f9d088bbeff64p-16, 0x1.d2b2be4e42422p-17, -0x1.2bb57c0cf2941p-18}, /* i=56 84.87 */
   {0x1.fffff37d63a36p-1, -0x1.753e3241c01bp-57, 0x1.74adc8f4064d3p-19, 0x1.de8a904d5c372p-73, -0x1.4ed4228b3da96p-17, 0x1.81918baca1979p-16, -0x1.3e81c09c29601p-15, 0x1.9004afed1bde9p-15, -0x1.8a40e183ee3fcp-15, 0x1.359242a8b8c58p-15, -0x1.834b953bcb845p-16, 0x1.79e345fb0b20dp-17, -0x1.0bb2d323900cep-18}, /* i=57 85.252 */
   {0x1.fffff82cdcf1bp-1, 0x1.046bbe9897fd5p-55, 0x1.d9c73698fb1dcp-20, 0x1.88de36481dfb5p-74, -0x1.b11017e7d5893p-18, 0x1.fc0dfadc2c6d6p-17, -0x1.ac4e1aa499ac6p-16, 0x1.131810ab2e2e3p-15, -0x1.1629d94abc864p-15, 0x1.c22a71036c259p-16, -0x1.244452f74de31p-16, 0x1.2bf17664310c1p-17, -0x1.cd1b31a8349bep-19}, /* i=58 84.869 */
   {0x1.fffffb248c39dp-1, 0x1.9b9a41713558cp-55, 0x1.2acee2f5ecdb8p-20, -0x1.2d1692a9a105cp-76, -0x1.15cc5700a2341p-18, 0x1.4be757b934819p-17, -0x1.1d6ab6f8cbf7cp-16, 0x1.76c5a3035bdabp-16, -0x1.847332578dfacp-16, 0x1.437f23f8d25ffp-16, -0x1.b305e625a092dp-17, 0x1.d3886ff986fefp-18, -0x1.81f2189b385a2p-19}, /* i=59 85.208 */
   {0x1.fffffd01f36afp-1, -0x1.d41915db812efp-55, 0x1.75fa8dbc84becp-21, 0x1.a5cd79572a1a6p-76, -0x1.6186d9fc357c5p-19, 0x1.ae02322e08822p-18, -0x1.79082befd50cap-17, 0x1.f9c26e211b174p-17, -0x1.0c768235c378bp-16, 0x1.cba7164e1064fp-17, -0x1.3f75c28c31ac8p-17, 0x1.663fcfff77e44p-18, -0x1.3a6da35f36ee6p-19}, /* i=60 85.253 */
   {0x1.fffffe2ba0ea5p-1, -0x1.26cd7908cba2bp-55, 0x1.d06ad6ecdf971p-22, -0x1.020b74d9d30fbp-76, -0x1.be46aa879edb2p-20, 0x1.143860c49d129p-18, -0x1.edabcbc3e620dp-18, 0x1.52139c87e9c82p-17, -0x1.6f567cd982028p-17, 0x1.42ebd266abd62p-17, -0x1.cf2f0c6adfb3ep-18, 0x1.0e2c0ed67786cp-18, -0x1.f50cb81b9b19p-20}, /* i=61 85.484 */
   {0x1.fffffee3cc32cp-1, 0x1.e429188c949b8p-56, 0x1.1e1e857adc568p-22, 0x1.2439f8a1649bbp-76, -0x1.1769ce59fb2c8p-20, 0x1.5fe5d47560794p-19, -0x1.405da04875e51p-18, 0x1.bfc96a938083dp-18, -0x1.f19ff5e59cbe9p-18, 0x1.c0c4d50d275bfp-18, -0x1.4b9df120462aep-18, 0x1.916640ee35de4p-19, -0x1.874483d99c37ep-20}, /* i=62 85.73 */
   {0x1.ffffff54dab72p-1, -0x1.a443df643729ap-55, 0x1.5dcd669f2cd34p-23, -0x1.ceb1ec59e0c28p-78, -0x1.5b11cbd1ee799p-21, 0x1.bc91a6b1c1839p-20, -0x1.9c2c5d12dfa2cp-19, 0x1.25d1e3c70364fp-18, -0x1.4dbe26c88e4f7p-18, 0x1.347bb8350b422p-18, -0x1.d51d3280da8ap-19, 0x1.25ed8e5b466b5p-19, -0x1.2b9c5d3390919p-20}, /* i=63 86.036 */
   {0x1.ffffff99b79d2p-1, -0x1.58ff1c425f8dep-56, 0x1.a854ea14102a9p-24, -0x1.21745e4b4fcb3p-78, -0x1.aba593e8384aep-22, 0x1.167c252a45678p-20, -0x1.06d78ca0424a3p-19, 0x1.7e0f59fcfa53dp-19, -0x1.bb4d48383b847p-19, 0x1.a39f3ad9a397fp-19, -0x1.47e836879c374p-19, 0x1.a89244d14b829p-20, -0x1.c33e15a6dbe37p-21}, /* i=64 86.403 */
   {0x1.ffffffc355dfdp-1, 0x1.88cb60fd4511cp-57, 0x1.febc107d5efabp-25, -0x1.d9ed10902067cp-81, -0x1.055a3c70279a4p-22, 0x1.59ff37766e9a7p-21, -0x1.4c53adb9dcc4dp-20, 0x1.ec49242997849p-20, -0x1.23927ad6ac54fp-19, 0x1.1a6e0676c7463p-19, -0x1.c5239f6a88a96p-20, 0x1.2e991308bf6fap-20, -0x1.4e276c09fe81bp-21}, /* i=65 86.718 */
   {0x1.ffffffdc4ad7ap-1, -0x1.d75de787812d4p-55, 0x1.30f93c3699079p-25, -0x1.8f941ab38e9dap-80, -0x1.3ce2f890bb01dp-23, 0x1.aa5010863c83bp-22, -0x1.a08ef1ca1636p-21, 0x1.3a4a6af3cafacp-20, -0x1.7be1e832218fp-20, 0x1.784775c30c386p-20, -0x1.3593046482ce3p-20, 0x1.a9d448178fbfdp-21, -0x1.e77bb85451c65p-22}, /* i=66 87.121 */
   {0x1.ffffffeb24467p-1, 0x1.bff89ef33d6ddp-55, 0x1.6961b8d641d07p-26, -0x1.74a7fc97b1544p-80, -0x1.7d2510f1f969dp-24, 0x1.0476b165ac852p-22, -0x1.02d3a3b9d195ep-21, 0x1.8db3567bef1dfp-21, -0x1.ea3ef4e3a126bp-21, 0x1.f03b0861a59acp-21, -0x1.a250ca467705ap-21, 0x1.27e9995f6dfcdp-21, -0x1.5e77b673c6d74p-22}, /* i=67 87.557 */
   {0x1.fffffff3e8892p-1, 0x1.befbf8d294678p-58, 0x1.a8e405e651ab7p-27, 0x1.167a2d8cf6b18p-84, -0x1.c6c40e5083698p-25, 0x1.3ba47a17512fdp-23, -0x1.3ee334beef6ecp-22, 0x1.f2bf9e6c43e99p-22, -0x1.395c08ac8e281p-21, 0x1.43ee4b521ccadp-21, -0x1.178f0deeb9b2p-21, 0x1.964e51b0f0532p-22, -0x1.f0cc4ecca5c2fp-23}, /* i=68 87.991 */
   {0x1.fffffff90b2e3p-1, -0x1.d82d94a90f1e4p-56, 0x1.efac5187b2864p-28, 0x1.f1301ae680614p-83, -0x1.0d229044adeeep-25, 0x1.7b5bc9db47dp-24, -0x1.8588212e670c2p-23, 0x1.35f42db1989fap-22, -0x1.8cd98865c4ffp-22, 0x1.a2b8587c48078p-22, -0x1.71aa2de99af9cp-22, 0x1.13a89805c15d9p-22, -0x1.5b53ca1bcf01ap-23}, /* i=69 88.463 */
   {0x1.fffffffc0748fp-1, 0x1.6ef7a9caef28p-57, 0x1.1edfa3c5f5ccbp-28, 0x1.368f60e2e6cfap-83, -0x1.3c025a6810c37p-26, 0x1.c42f78a0989adp-25, -0x1.d7c6c3583c6e3p-24, 0x1.7dd6ccb5c93b4p-23, -0x1.f1ec2f699fdccp-23, 0x1.0bf7a04407a8cp-22, -0x1.e3aafe6dfd4ep-23, 0x1.71bc3a55b63f4p-23, -0x1.df66b11724e7cp-24}, /* i=70 88.939 */
   {0x1.fffffffdbff2ap-1, 0x1.49438981099b2p-56, 0x1.4979ac8b28928p-29, -0x1.c2f44bcf3ce52p-83, -0x1.7015eec37753ap-27, 0x1.0b487791590cfp-25, -0x1.1b44b64c3c995p-24, 0x1.d23ff3ef8dd83p-24, -0x1.357d673d1ccfcp-23, 0x1.53a563ce0e9e3p-23, -0x1.3921106a960f6p-23, 0x1.ea527d318f96ep-24, -0x1.46bd6cea7103dp-24}, /* i=71 89.458 */
   {0x1.fffffffebc1a9p-1, 0x1.e0e5facabfab4p-56, 0x1.77756ec9f78fbp-30, 0x1.e20366d0e0306p-85, -0x1.a9530780ca70cp-28, 0x1.3962ecb10df65p-26, -0x1.51494525dee64p-25, 0x1.1a2961b90efbp-24, -0x1.7d35cd0b404bfp-24, 0x1.aa596d9d73afbp-24, -0x1.91493d8d43ba2p-24, 0x1.4184505343c2dp-24, -0x1.b7d977f1a3402p-25}, /* i=72 89.993 */
   {0x1.ffffffff4b453p-1, 0x1.59b25048a61ccp-55, 0x1.a887bd2b4404fp-31, -0x1.2556d8ad4dd44p-87, -0x1.e78be33fb01dap-29, 0x1.6c6ef0b68629ep-27, -0x1.8e36e9a44c497p-26, 0x1.5286ee37c531ep-25, -0x1.d146395886537p-25, 0x1.090902855d5fp-24, -0x1.fd0d1e8fcb6dfp-25, 0x1.a10f65c3c5a7bp-25, -0x1.24888c323daf3p-25}, /* i=73 90.536 */
   {0x1.ffffffff9bec8p-1, -0x1.6755054654b62p-56, 0x1.dc479de0ef004p-32, -0x1.c3434581af3b8p-86, -0x1.1535aee3eb1b2p-29, 0x1.a4547ed264758p-28, -0x1.d2308d0dead0fp-27, 0x1.929d46a9a7edcp-26, -0x1.195dbfd4afd19p-25, 0x1.46630f49ccd2fp-25, -0x1.3fa4637c64ebcp-25, 0x1.0b98a6e0cfc02p-25, -0x1.8093f032972f3p-26}, /* i=74 91.092 */
   {0x1.ffffffffc901cp-1, 0x1.9c951c943961cp-57, 0x1.0916f04b6e18dp-32, 0x1.1bdf9650721eap-87, -0x1.38b90f78fbe14p-30, 0x1.e0d7765326885p-29, -0x1.0e9760d0ac127p-27, 0x1.daad91166722dp-27, -0x1.513c51b9838edp-26, 0x1.8e27fb85ba534p-26, -0x1.8d6f6bd99eaffp-26, 0x1.53c31e52fff08p-26, -0x1.f3bfd31796bcp-27}, /* i=75 91.685 */
   {0x1.ffffffffe202dp-1, 0x1.a54841f566a61p-55, 0x1.24caf2c32af16p-33, 0x1.02e3358112fa1p-87, -0x1.5dfa962d49548p-31, 0x1.10ca1ff2af812p-29, -0x1.377c7e98dd9b4p-28, 0x1.156649e0b5dd2p-27, -0x1.9092f4db426c5p-27, 0x1.e12a29b227972p-27, -0x1.e94e18d5271a9p-27, 0x1.aae38927ee69bp-27, -0x1.41121b0293be1p-27}, /* i=76 92.286 */
   {0x1.ffffffffefc57p-1, -0x1.8225a9658ef84p-57, 0x1.40dfd87456f4fp-34, -0x1.a6d5c55f8e63bp-88, -0x1.848f101ce14c8p-32, 0x1.32fed47f8dd28p-30, -0x1.638ff4a6975f2p-29, 0x1.416d25168a6b8p-28, -0x1.d78fb22f58668p-28, 0x1.2009c6b4e61eap-27, -0x1.2a459e59c850bp-27, 0x1.096a3e8dac0eap-27, -0x1.97fba69de37d8p-28}, /* i=77 92.91 */
   {0x1.fffffffff748ep-1, 0x1.ae15e36044aacp-57, 0x1.5ce9ab1670dd6p-35, 0x1.cc9bbfb723fc4p-91, -0x1.abf69bd9866f7p-33, 0x1.56ae1e8abbbbfp-31, -0x1.927ca04d1a7a8p-30, 0x1.713d3b07d7a36p-29, -0x1.1318f5d7d717bp-28, 0x1.55ab94fdfd1f4p-28, -0x1.68216fb90717ap-28, 0x1.46ad5ce577d65p-28, -0x1.0065a20073e81p-28}, /* i=78 93.545 */
   {0x1.fffffffffb5bp-1, -0x1.50fb19119064fp-55, 0x1.7872d9fa10ab2p-36, -0x1.7760afdf543a4p-90, -0x1.d39eaac4a0b47p-34, 0x1.7b67ab8af33d6p-32, -0x1.c3ced54e694eap-31, 0x1.a4875d8a47f12p-30, -0x1.3e213e6f5c296p-29, 0x1.919137301f897p-29, -0x1.aea6bd9b3493p-29, 0x1.8e06e4ab5925fp-29, -0x1.3ed1d979421b2p-29}, /* i=79 94.198 */
   {0x1.fffffffffd8b3p-1, -0x1.5182469c211ep-57, 0x1.92ff33023d5c3p-37, -0x1.2932180032bd1p-91, -0x1.fae4fe28d12ddp-35, 0x1.a0a80964d6e97p-33, -0x1.f6f47be478e2ap-32, 0x1.dad968cdacb13p-31, -0x1.6ca68a8bfdb81p-30, 0x1.d3a79e5305b4ap-30, -0x1.fe1534ebf69c7p-30, 0x1.e01ee76d92779p-30, -0x1.883ed9069f3fdp-30}, /* i=80 94.867 */
   {0x1.fffffffffeb6p-1, -0x1.4d3f53e684bf8p-56, 0x1.ac0f5f322937ap-38, -0x1.8e8ab19224e58p-92, -0x1.108dc99cf03e5p-35, 0x1.c5db17016a0c6p-34, -0x1.159f41ea079c3p-32, 0x1.09ced3e9b7204p-31, -0x1.9e4dace0668p-31, 0x1.0dd5e0e9749b6p-30, -0x1.2b3aa6599d0b5p-30, 0x1.1eb5e8e4ffe8fp-30, -0x1.dd8955967ed31p-31}, /* i=81 95.555 */
   {0x1.ffffffffff542p-1, 0x1.b57ed63ed811p-57, 0x1.c324c20e337e5p-39, 0x1.53fd8abf42ed9p-93, -0x1.22c6b11327305p-36, 0x1.ea5f66f89cbd4p-35, -0x1.2ff1e0a81bedcp-33, 0x1.270ddbd8e501fp-32, -0x1.d2992b5c25c93p-32, 0x1.3492d76bdf266p-31, -0x1.5bc7361853ddep-31, 0x1.53121ae3f1d2ep-31, -0x1.1fb0f7e3f242bp-31}, /* i=82 96.256 */
   {0x1.ffffffffffa73p-1, -0x1.6fead614b7934p-56, 0x1.d7c593130dd16p-40, -0x1.8e78574fe0514p-95, -0x1.33c1e2f16e037p-37, 0x1.06c53fdc74764p-35, -0x1.4a029a87915acp-34, 0x1.44bd86238ff0dp-33, -0x1.0474ac3a80072p-32, 0x1.5db2a89e9bc47p-32, -0x1.906f4b51a7f75p-32, 0x1.8d189784c1f5p-32, -0x1.571a4760f483dp-32}, /* i=83 96.973 */
   {0x1.ffffffffffd27p-1, 0x1.19e1a84064c56p-56, 0x1.e9810295890f9p-41, 0x1.f998d55766fdbp-95, -0x1.43262ab4b77b2p-38, 0x1.1756eae580a28p-36, -0x1.6359d5b0d251ep-35, 0x1.626391bd58994p-34, -0x1.203efc6c9f556p-33, 0x1.88c0b111be9p-33, -0x1.c8ca211a38811p-33, 0x1.cc911f684d612p-33, -0x1.950e3edf09a71p-33}, /* i=84 97.706 */
   {0x1.ffffffffffe8dp-1, 0x1.e766e2c801398p-58, 0x1.f7f338086a87bp-42, -0x1.dfa0c27b527ep-98, -0x1.509f766d9f287p-39, 0x1.268e278ede221p-37, -0x1.7b7b43e9a1b0ep-36, 0x1.7f7aadab6b398p-35, -0x1.3c3cc6aafba0bp-34, 0x1.b52c69b4ab6dep-34, -0x1.0222c438d1182p-33, 0x1.0888e14314f83p-33, -0x1.d96aaea63b362p-34}, /* i=85 98.453 */
   {0x1.fffffffffff45p-1, -0x1.5948eec884df5p-55, 0x1.01647ba79874ep-42, 0x1.6d5d39dabc3p-103, -0x1.5be1cf20840dcp-40, 0x1.3418096320dafp-38, -0x1.91e9beb94b447p-37, 0x1.9b762261756a7p-36, -0x1.57f320a630c91p-35, 0x1.e24b78ce82b11p-35, -0x1.2112fff5c77aap-34, 0x1.2cfdd93a41786p-34, -0x1.11ea1f35b4d2bp-34}, /* i=86 99.218 */
   {0x1.fffffffffffa2p-1, 0x1.d07509a1a944p-57, 0x1.04e15ecc7f401p-43, -0x1.0858e34f7a6a6p-98, -0x1.64ac1f9b95f96p-41, 0x1.3fa8302ade993p-39, -0x1.a62b70897719ep-38, 0x1.b5c619266e9fp-37, -0x1.72de32129cbb8p-36, 0x1.07ae94305c398p-35, -0x1.40c45a9e95152p-35, 0x1.533d127efdf16p-35, -0x1.39dc242ba4cdap-35}, /* i=87 99.994 */
   {0x1.fffffffffffd1p-1, 0x1.3b6fc0b729759p-55, 0x1.065b9616170e1p-44, -0x1.49459f5147526p-99, -0x1.6acaa58a8be12p-42, 0x1.48fb92d0947e7p-40, -0x1.b7ce1a1ea8ea5p-39, 0x1.cddc552bbebebp-38, -0x1.8c751cc1a5784p-37, 0x1.1dc79b52007bp-36, -0x1.60b3d17e7714cp-36, 0x1.7ac1d379afc28p-36, -0x1.641ca84798564p-36}, /* i=88 100.787 */
   {0x1.fffffffffffe9p-1, -0x1.5fe91226dd51p-58, 0x1.05ca50205d279p-45, -0x1.7a281f9edb8e6p-99, -0x1.6e18ec0d42451p-43, 0x1.4fdb051100a15p-41, -0x1.c66b3f3fe565ep-40, 0x1.e331281475b54p-39, -0x1.a42e6965b2b9ap-38, 0x1.3301ef493196p-37, -0x1.804fcc1524d74p-37, 0x1.a2ef0c13a3daap-37, -0x1.9028a915f98d3p-37}, /* i=89 101.591 */
   {0x1.ffffffffffff5p-1, -0x1.238f8ed17d9b3p-55, 0x1.0330f0fd69931p-46, 0x1.a2c00e0c6dcbap-100, -0x1.6e8334c65749dp-44, 0x1.541d561058477p-42, -0x1.d1ac042ada69ep-41, 0x1.f54864c5a530ep-40, -0x1.b984c73c1d301p-39, 0x1.46ec7009c291fp-38, -0x1.9efc2df73776p-38, 0x1.cb12ac38f37cap-38, -0x1.bd54fcd67b8d4p-38}, /* i=90 102.407 */
   {0x1.ffffffffffffbp-1, -0x1.efa4d64f59f62p-55, 0x1.fd3de10d6287ap-48, -0x1.e1fdae91c5cfep-102, -0x1.6c073be0916e6p-45, 0x1.55a8eab9e129ap-43, -0x1.d94c87c1bc304p-42, 0x1.01db0818bec24p-40, -0x1.cbfbe4c0ef6eep-40, 0x1.59179d8c519c4p-39, -0x1.bc172710440bdp-39, 0x1.f26a4f726814ep-39, -0x1.ead889e052555p-39}, /* i=91 103.247 */
   {0x1.ffffffffffffdp-1, 0x1.6be96953fe014p-55, 0x1.f05e82aae2be2p-49, -0x1.070a8237b4337p-103, -0x1.66b44c6d7ddb6p-46, 0x1.5474bd9d072f3p-44, -0x1.dd1e8c33100ccp-43, 0x1.0711486984913p-41, -0x1.db2522b66a6cep-41, 0x1.6919a06329739p-40, -0x1.d6fe8f87926e8p-40, 0x1.0c1488010ff5cp-39, -0x1.0bf9fa407e9abp-39}, /* i=92 104.083 */
   {0x1.fffffffffffffp-1, -0x1.0fecc5ed770dep-55, 0x1.e00e9148a1d52p-50, 0x1.f7a503c7a2ad8p-107, -0x1.5eaaa4200e355p-47, 0x1.5088b6566fcedp-45, -0x1.dd0b48e0f634ep-44, 0x1.0a27116d7478ep-42, -0x1.e6a3e1d5c214fp-42, 0x1.769249755a4bcp-41, -0x1.ef16049050b69p-41, 0x1.1dbf2744f66dbp-40, -0x1.21c636bd8f5a9p-40}, /* i=93 104.933 */
   {0x1.fffffffffffffp-1, 0x1.989c6c5d51227p-55, 0x1.ccaaea71ab11p-51, 0x1.152f323a1f3b4p-107, -0x1.541a2f15eb476p-48, 0x1.49fd53e85cdf3p-46, -0x1.d9144beee6b4ap-45, 0x1.0b09b02f533a1p-43, -0x1.ee312fcf48076p-43, 0x1.812ed2f01f60ap-42, -0x1.01e6391f47ad7p-41, 0x1.2dce8f6b8c896p-41, -0x1.365d5011db0dfp-41}, /* i=94 105.705 */
};

/* Assuming 0 <= z <= 0x1.7afb48dc96626p+2, put in h+l an approximation
   of erf(z). Return err the maximal relative error:
   |(h + l)/erf(z) - 1| < err*|h+l| */
static double
cr_erf_fast(double *h, double *l, double z)
{
  double th, tl;
  /* we split [0,0x1.7afb48dc96626p+2] into intervals i/16 <= z < (i+1)/16,
     and for each interval, we use a minimax polynomial:
     * for i=0 (0 <= z < 1/16) we use a polynomial evaluated at zero,
       since if we evaluate in the middle 1/32, we will get bad accuracy
       for tiny z, and moreover z-1/32 might not be exact
     * for 1 <= i <= 94, we use a polynomial evaluated in the middle of
       the interval, namely i/16+1/32
  */
  if (z < 0.0625) /* z < 1/16 */
  {
    /* the following is a degree-11 minimax polynomial for erf(x) on [0,1/16]
       generated by Sollya, with double-double coefficients for degree 1 and 3,
       and double coefficients for degrees 5 to 11 (file erf0.sollya).
       The maximal relative error is 2^-68.935. */
    static const double c0[] = {
      0x1.20dd750429b6dp+0, 0x1.1ae3a7862d9c4p-56,  /* degree 1 */
      -0x1.812746b0379e7p-2, 0x1.f1a64d72722a2p-57, /* degree 3 */
      0x1.ce2f21a042b7fp-4,                         /* degree 5 */
      -0x1.b82ce31189904p-6,                        /* degree 7 */
      0x1.565bbf8a0fe0bp-8,                         /* degree 9 */
      -0x1.bf9f8d2c202e4p-11};                      /* degree 11 */
    double z2h, z2l, z4;
    a_mul(&z2h, &z2l, z, z);
    z4 = z2h * z2h;
    double c9 = std::fma(c0[7], z2h, c0[6]);
    double c5 = std::fma(c0[5], z2h, c0[4]);
    c5 = std::fma(c9, z4, c5);
    /* compute c0[2] + c0[3] + z2h*c5 */
    a_mul(&th, &tl, z2h, c5);
    fast_two_sum(h, l, c0[2], th);
    *l += tl + c0[3];
    /* compute c0[0] + c0[1] + (z2h + z2l)*(h + l) */
    double h_copy = *h;
    a_mul(&th, &tl, z2h, *h);
    tl += std::fma(z2h, *l, c0[1]);
    fast_two_sum(h, l, c0[0], th);
    *l += std::fma(z2l, h_copy, tl);
    /* multiply (h,l) by z */
    a_mul(h, &tl, *h, z);
    *l = std::fma(*l, z, tl);
    return 0x1.78p-69;
  }
  double v = std::floor(16.0 * z);
  uint32_t i = (uint32_t)(16.0 * z);
  /* i/16 <= z < (i+1)/16 */
  z = (z - 0.03125) - 0.0625 * v;
  /* now |z| <= 1/32 */
  const double *c = C[i-1];
  double z2 = z*z, z4 = z2*z2;
  /* the degree-10 coefficient is c[12] */
  double c9 = std::fma(c[12], z, c[11]);
  double c7 = std::fma(c[10], z, c[9]);
  double c5 = std::fma(c[8], z, c[7]);
  double c3h, c3l;
  /* c3h, c3l <- c[5] + z*c[6] */
  fast_two_sum(&c3h, &c3l, c[5], z * c[6]);
  c7 = std::fma(c9, z2, c7);
  /* c3h, c3l <- c3h, c3l + c5*z2 */
  fast_two_sum(&c3h, &tl, c3h, c5 * z2);
  c3l += tl;
  /* c3h, c3l <- c3h, c3l + c7*z4 */
  fast_two_sum(&c3h, &tl, c3h, c7 * z4);
  c3l += tl;
  /* c2h, c2l <- c[4] + z*(c3h + c3l) */
  double c2h, c2l;
  a_mul(&th, &tl, z, c3h);
  fast_two_sum(&c2h, &c2l, c[4], th);
  c2l += std::fma(z, c3l, tl);
  /* compute c[2] + c[3] + z*(c2h + c2l) */
  a_mul(&th, &tl, z, c2h);
  fast_two_sum(h, l, c[2], th);
  *l += tl + std::fma(z, c2l, c[3]);
  /* compute c[0] + c[1] + z*(h + l) */
  a_mul(&th, &tl, z, *h);
  tl = std::fma(z, *l, tl);
  fast_two_sum(h, l, c[0], th);
  *l += tl + c[1];
  return 0x1.11p-69;
}

double
cr_erf(double x)
{
  double z = std::fabs(x);
  b64u64_u t;
  t.f = z;
  uint64_t ux = t.u;
  /* erf(x) rounds to +/-1 for RNDN for |x| > 0x1.7afb48dc96626p+2 */
  if (ux > 0x4017afb48dc96626ull) // |x| > 0x1.7afb48dc96626p+2
  {
    double os = std::copysign(1.0, x);
    uint64_t MASK = (uint64_t)0x7ff0000000000000ull; // encoding of +Inf
    if (ux > MASK)
      return x + x; /* NaN */
    if (ux == MASK)
      return os; /* +/-Inf */
    return os - 0x1p-54 * os;
  }
  double h, l, err;
  /* now |x| <= 0x1.7afb48dc96626p+2 */
  if (z < 0x1p-61)
  {
    /* for x=-0 the code below returns +0 which is wrong */
    if (x == 0)
      return x;
    /* tiny x: erf(x) ~ 2/sqrt(pi) * x + O(x^3), where the ratio of the O(x^3)
       term to the main term is in x^2/3, thus less than 2^-123 */
    double y = CH * x; /* tentative result */
    /* scale x by 2^106 to get out the subnormal range */
    double sx = x * 0x1p106;
    a_mul(&h, &l, CH, sx);
    l = std::fma(CL, sx, l);
    /* now compute the residual h + l - y */
    l += h - y * 0x1p106; /* h-y*2^106 is exact since h and y are very close */
    double res = std::fma(l, 0x1p-106, y);
    return res;
  }
  /* now z >= 2^-61 */
  err = cr_erf_fast(&h, &l, z);
  b64u64_u u;
  u.f = h;
  b64u64_u v;
  v.f = l;
  t.f = x;
  uint64_t SIGN_MASK = (uint64_t)0x8000000000000000ull;
  u.u ^= t.u & SIGN_MASK;
  v.u ^= t.u & SIGN_MASK;
  double left = u.f + std::fma(err, -u.f, v.f);
  /* Fast path only: return left without checking accurate path */
  return left;
}
