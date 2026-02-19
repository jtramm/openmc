// Fast-path sine function for binary64 values.
// Derived from the CORE-MATH project (MIT License).
// Original authors: Paul Zimmermann and Tom Hubrecht.
// https://core-math.gitlabpages.inria.fr/
//
// Accurate path removed; returns fast-path result directly when the
// fast path cannot guarantee correct rounding.
// All 128-bit integer arithmetic replaced with mul64x64() from
// coremath_utils.h for portability.

#include "coremath_utils.h"

/* This table approximates 1/(2pi) downwards with precision 1280:
   1/(2*pi) ~ T[0]/2^64 + T[1]/2^128 + ... + T[i]/2^((i+1)*64) + ...
   Computed with computeT() from sin.sage. */
static const uint64_t T[20] = {
  0x28be60db9391054a, // i=0
   0x7f09d5f47d4d3770,
   0x36d8a5664f10e410,
   0x7f9458eaf7aef158,
   0x6dc91b8e909374b8,
   0x1924bba82746487, // i=5
   0x3f877ac72c4a69cf,
   0xba208d7d4baed121,
   0x3a671c09ad17df90,
   0x4e64758e60d4ce7d,
   0x272117e2ef7e4a0e, // i=10
   0xc7fe25fff7816603,
   0xfbcbc462d6829b47,
   0xdb4d9fb3c9f2c26d,
   0xd3d18fd9a797fa8b,
   0x5d49eeb1faf97c5e, // i=15
   0xcf41ce7de294a4ba,
   0x9afed7ec47e35742,
   0x1580cc11bf1edaea,
   0xfc33ef0826bd0d87, // i=19
};

/* Table generated with ./buildSC 15 using accompanying buildSC.c.
   For each i, 0 <= i < 256, xi=i/2^11+SC[i][0], with
   SC[i][1] and SC[i][2] approximating sin2pi(xi) and cos2pi(xi)
   respectively, both with 53+15 bits of accuracy. */
static const double SC[256][3] = {
   {0x0p+0, 0x0p+0, 0x1p+0}, /* 0 */
   {-0x1.c0f6cp-35, 0x1.921f892b900fep-9, 0x1.ffff621623fap-1}, /* 1 */
   {-0x1.9c7935ep-35, 0x1.921f0ea27ce01p-8, 0x1.fffd8858eca2ep-1}, /* 2 */
   {-0x1.d14d1acp-34, 0x1.2d96af779b0bbp-7, 0x1.fffa72c986392p-1}, /* 3 */
   {-0x1.dba8f6a8p-33, 0x1.921d1ce2d0a1cp-7, 0x1.fff62169dddaap-1}, /* 4 */
   {0x1.a6b7cdfp-32, 0x1.f6a29bdb7377p-7, 0x1.fff0943c02419p-1}, /* 5 */
   {0x1.b49618dp-33, 0x1.2d936d1506f3dp-6, 0x1.ffe9cb44829cp-1}, /* 6 */
   {-0x1.398d6fcp-35, 0x1.5fd4d1e21de6dp-6, 0x1.ffe1c687174b1p-1}, /* 7 */
   {-0x1.e9e9a8c8p-31, 0x1.9215597791e0ap-6, 0x1.ffd886097afcfp-1}, /* 8 */
   {-0x1.34e844cp-32, 0x1.c454f2e9480c7p-6, 0x1.ffce09ce95933p-1}, /* 9 */
   {-0x1.989a8a4p-32, 0x1.f693709b94f92p-6, 0x1.ffc251dfbac0cp-1}, /* 10 */
   {0x1.04a9b99p-30, 0x1.146860e69a571p-5, 0x1.ffb55e40a5c43p-1}, /* 11 */
   {-0x1.56947cp-36, 0x1.2d865748774adp-5, 0x1.ffa72efff95d1p-1}, /* 12 */
   {-0x1.c348768p-35, 0x1.46a396d34121ap-5, 0x1.ff97c420a8451p-1}, /* 13 */
   {0x1.9e80552p-32, 0x1.5fc00e6e4c65cp-5, 0x1.ff871dacd8761p-1}, /* 14 */
   {0x1.3f11d74p-34, 0x1.78dbaa97099ebp-5, 0x1.ff753bb18af95p-1}, /* 15 */
   {0x1.c039af4p-33, 0x1.91f65fc0abc0ap-5, 0x1.ff621e370ca7ap-1}, /* 16 */
   {0x1.53e1f8p-35, 0x1.ab101bf74ac2ep-5, 0x1.ff4dc54b00181p-1}, /* 17 */
   {0x1.114a649p-29, 0x1.c428d7de920e9p-5, 0x1.ff3830f2e9043p-1}, /* 18 */
   {0x1.adf0ef4p-31, 0x1.dd40723a3cdfbp-5, 0x1.ff21614b9d9adp-1}, /* 19 */
   {-0x1.d21f5918p-30, 0x1.f656e1e9e59cdp-5, 0x1.ff09565e83d77p-1}, /* 20 */
   {-0x1.4f54d708p-30, 0x1.07b612d6be078p-4, 0x1.fef0102c634e3p-1}, /* 21 */
   {-0x1.1efec9ap-30, 0x1.1440118ba7bdp-4, 0x1.fed58ecf342dap-1}, /* 22 */
   {0x1.cc17ba88p-29, 0x1.20c96cf0a7eedp-4, 0x1.feb9d24646fa6p-1}, /* 23 */
   {0x1.121dbe4p-33, 0x1.2d5209628edfp-4, 0x1.fe9cdacf99cffp-1}, /* 24 */
   {-0x1.9ecf61p-34, 0x1.39d9f103bf7f7p-4, 0x1.fe7ea854e6b08p-1}, /* 25 */
   {-0x1.04ede8ep-31, 0x1.466116c629e5cp-4, 0x1.fe5f3af4ee201p-1}, /* 26 */
   {-0x1.1821cecp-31, 0x1.52e773c9920c7p-4, 0x1.fe3e92c0e4108p-1}, /* 27 */
   {0x1.cdec726p-31, 0x1.5f6d02131f0b2p-4, 0x1.fe1cafc7f1a24p-1}, /* 28 */
   {-0x1.edece4dp-31, 0x1.6bf1b2653648cp-4, 0x1.fdf99233c230cp-1}, /* 29 */
   {-0x1.2aa4d1cp-31, 0x1.787585bc45f0fp-4, 0x1.fdd53a01d11d9p-1}, /* 30 */
   {0x1.d461592p-32, 0x1.84f871e32cf68p-4, 0x1.fdafa74f16482p-1}, /* 31 */
   {0x1.f0cbd728p-29, 0x1.917a71d3d2956p-4, 0x1.fd88da29f302ep-1}, /* 32 */
   {-0x1.583247p-30, 0x1.9dfb6c9865b06p-4, 0x1.fd60d2e14a6b1p-1}, /* 33 */
   {-0x1.2e81bf4p-30, 0x1.aa7b706bfdbbap-4, 0x1.fd3791484ff5p-1}, /* 34 */
   {-0x1.13941418p-28, 0x1.b6fa680a05c27p-4, 0x1.fd0d15a4b8471p-1}, /* 35 */
   {0x1.71098ffp-30, 0x1.c3785eba12b42p-4, 0x1.fce15fceddccfp-1}, /* 36 */
   {-0x1.c3519e8p-32, 0x1.cff53302f059p-4, 0x1.fcb4703b969e1p-1}, /* 37 */
   {0x1.2f522a5p-27, 0x1.dc70fb84af16ep-4, 0x1.fc8646987fc1dp-1}, /* 38 */
   {-0x1.ae9bed8p-33, 0x1.e8eb7f8a589e2p-4, 0x1.fc56e3b91ca3ap-1}, /* 39 */
   {0x1.f8868b2p-30, 0x1.f564e87d2330fp-4, 0x1.fc264701f9a09p-1}, /* 40 */
   {-0x1.b07985f8p-29, 0x1.00ee8835051f4p-3, 0x1.fbf47105f7439p-1}, /* 41 */
   {0x1.cbdaa94p-30, 0x1.072a05e1d4d8ep-3, 0x1.fbc16172a9e36p-1}, /* 42 */
   {0x1.37c5b908p-28, 0x1.0d64df9619f0dp-3, 0x1.fb8d18b635327p-1}, /* 43 */
   {-0x1.068b5fc8p-28, 0x1.139f09bc617f5p-3, 0x1.fb5797351da85p-1}, /* 44 */
   {-0x1.8ea66818p-29, 0x1.19d8919fa4ec8p-3, 0x1.fb20dc7da8affp-1}, /* 45 */
   {0x1.6278ceb8p-28, 0x1.2011719d50b87p-3, 0x1.fae8e8bd4427fp-1}, /* 46 */
   {-0x1.096df84p-29, 0x1.264993433763ap-3, 0x1.faafbcbfca356p-1}, /* 47 */
   {0x1.9b2534fp-29, 0x1.2c810967bbf7p-3, 0x1.fa7557d8d987ep-1}, /* 48 */
   {0x1.215b4ep-34, 0x1.32b7bfa25c91bp-3, 0x1.fa39bac71954bp-1}, /* 49 */
   {-0x1.94db891p-30, 0x1.38edb9d29b39dp-3, 0x1.f9fce56700a6dp-1}, /* 50 */
   {0x1.7727f7b8p-29, 0x1.3f22f7c3cce3ap-3, 0x1.f9bed7b8c8d8cp-1}, /* 51 */
   {-0x1.0cb33038p-29, 0x1.45576971dd53p-3, 0x1.f97f925d53c83p-1}, /* 52 */
   {-0x1.9071106p-31, 0x1.4b8b175c71e22p-3, 0x1.f93f14feb8022p-1}, /* 53 */
   {0x1.62741e78p-29, 0x1.51bdfa7ea30d5p-3, 0x1.f8fd5fe3efac8p-1}, /* 54 */
   {0x1.f8e16d0cp-28, 0x1.57f00e80e6e12p-3, 0x1.f8ba733a1ceb1p-1}, /* 55 */
   {-0x1.76acbcap-31, 0x1.5e2143b7bc1c2p-3, 0x1.f8764fad5e9bfp-1}, /* 56 */
   {-0x1.0a0f73ap-30, 0x1.6451a76411746p-3, 0x1.f830f4ad232d8p-1}, /* 57 */
   {0x1.ca11d1bcp-28, 0x1.6a8135d7bd143p-3, 0x1.f7ea625eb5af7p-1}, /* 58 */
   {-0x1.02f23628p-29, 0x1.70afd74071191p-3, 0x1.f7a299d3f182ap-1}, /* 59 */
   {0x1.b34dcb8p-29, 0x1.76dda08544b5cp-3, 0x1.f7599a1ac7ecdp-1}, /* 60 */
   {0x1.161ff4p-32, 0x1.7d0a7bf2d4abap-3, 0x1.f70f64322da74p-1}, /* 61 */
   {-0x1.c49b8b4p-31, 0x1.83366ddb3de23p-3, 0x1.f6c3f7e7c2707p-1}, /* 62 */
   {0x1.21da851p-29, 0x1.8961743b1429p-3, 0x1.f6775552a6ba2p-1}, /* 63 */
   {0x1.ac63edap-30, 0x1.8f8b851098588p-3, 0x1.f6297cef0cdd6p-1}, /* 64 */
   {0x1.27ef489cp-27, 0x1.95b4a5b9f2cebp-3, 0x1.f5da6e7820551p-1}, /* 65 */
   {0x1.ae8937p-30, 0x1.9bdcc07900146p-3, 0x1.f58a2b0689c82p-1}, /* 66 */
   {0x1.eb48c7ep-29, 0x1.a203e4a4f950ep-3, 0x1.f538b1d392049p-1}, /* 67 */
   {-0x1.bfd282fp-29, 0x1.a829ffaad0d79p-3, 0x1.f4e603d51f1aap-1}, /* 68 */
   {0x1.7ccf638p-29, 0x1.ae4f1fa80e1b5p-3, 0x1.f492204c5ef9ep-1}, /* 69 */
   {-0x1.2435c578p-28, 0x1.b4732b72ebc86p-3, 0x1.f43d0890e1e72p-1}, /* 70 */
   {0x1.0293fecp-30, 0x1.ba9634155f866p-3, 0x1.f3e6bbb6c2ea4p-1}, /* 71 */
   {-0x1.7bb1f92p-29, 0x1.c0b82461f65ep-3, 0x1.f38f3ae6f9afcp-1}, /* 72 */
   {0x1.27aaebcp-29, 0x1.c6d906faacf65p-3, 0x1.f3368589e17a2p-1}, /* 73 */
   {-0x1.2e2bcd5p-27, 0x1.ccf8c3f74a6c9p-3, 0x1.f2dc9cfb5fa74p-1}, /* 74 */
   {-0x1.6f070acp-30, 0x1.d31773ba218a8p-3, 0x1.f2817fd4d045bp-1}, /* 75 */
   {0x1.469adfcp-29, 0x1.d935004779e57p-3, 0x1.f2252f59c122dp-1}, /* 76 */
   {0x1.4f51c18p-32, 0x1.df5164301377ap-3, 0x1.f1c7abdeaa3efp-1}, /* 77 */
   {0x1.78e44dap-29, 0x1.e56ca4202807cp-3, 0x1.f168f51c5d5d5p-1}, /* 78 */
   {0x1.49bb5f8p-32, 0x1.eb86b4a1b7e9bp-3, 0x1.f1090bc4b68p-1}, /* 79 */
   {-0x1.67ba541p-28, 0x1.f19f9369d5e93p-3, 0x1.f0a7effdc937fp-1}, /* 80 */
   {0x1.c0cab95p-29, 0x1.f7b74ab7219d2p-3, 0x1.f045a1219e594p-1}, /* 81 */
   {-0x1.2b77e32p-30, 0x1.fdcdc0ca3288dp-3, 0x1.efe220cf5c751p-1}, /* 82 */
   {-0x1.e0d8cbp-33, 0x1.01f18054c8362p-2, 0x1.ef7d6e54c347dp-1}, /* 83 */
   {-0x1.ecd5b9cp-29, 0x1.04fb7f6d35d68p-2, 0x1.ef178a6f9a987p-1}, /* 84 */
   {0x1.eb24de5p-29, 0x1.0804e1d369ff2p-2, 0x1.eeb074934fdfp-1}, /* 85 */
   {0x1.4a897c4p-30, 0x1.0b0d9d7b0d042p-2, 0x1.ee482e14bcdep-1}, /* 86 */
   {0x1.336c376p-30, 0x1.0e15b555e7becp-2, 0x1.eddeb6908ca8cp-1}, /* 87 */
   {-0x1.3952d9p-31, 0x1.111d25efd48b8p-2, 0x1.ed740e7eb8dd6p-1}, /* 88 */
   {0x1.fc2a5d4p-31, 0x1.1423ef5c7e1bdp-2, 0x1.ed0835dc24e89p-1}, /* 89 */
   {0x1.a88ed37p-29, 0x1.172a0eb8361dap-2, 0x1.ec9b2d0ec8288p-1}, /* 90 */
   {-0x1.8ca4cb94p-27, 0x1.1a2f7b10b6d7p-2, 0x1.ec2cf55d6117cp-1}, /* 91 */
   {0x1.0144524p-27, 0x1.1d3446fd0cd3fp-2, 0x1.ebbd8c1d62f96p-1}, /* 92 */
   {-0x1.abf810cp-28, 0x1.203855b85f89ap-2, 0x1.eb4cf57454132p-1}, /* 93 */
   {0x1.5d4c5d58p-28, 0x1.233bbcca40561p-2, 0x1.eadb2e40746cap-1}, /* 94 */
   {-0x1.a1b0c58p-29, 0x1.263e685b1d714p-2, 0x1.ea68396d87754p-1}, /* 95 */
   {-0x1.77c8dacp-29, 0x1.294061d2eb611p-2, 0x1.e9f41597393c8p-1}, /* 96 */
   {0x1.915540ep-30, 0x1.2c41a580014cfp-2, 0x1.e97ec348fb87fp-1}, /* 97 */
   {-0x1.abb6d9bp-28, 0x1.2f422b2d0990cp-2, 0x1.e90843c55b996p-1}, /* 98 */
   {-0x1.b8ee5d58p-28, 0x1.3241f8cea2836p-2, 0x1.e890962268c49p-1}, /* 99 */
   {-0x1.1cd29828p-28, 0x1.35410a8396266p-2, 0x1.e817baf85c094p-1}, /* 100 */
   {-0x1.e216afp-32, 0x1.383f5e08283e2p-2, 0x1.e79db2a188b0ap-1}, /* 101 */
   {-0x1.24afc3p-31, 0x1.3b3cef6993c0bp-2, 0x1.e7227dbf82004p-1}, /* 102 */
   {-0x1.aa1657cp-31, 0x1.3e39be4767224p-2, 0x1.e6a61c62d5274p-1}, /* 103 */
   {-0x1.c5b65fap-30, 0x1.4135c898485bbp-2, 0x1.e6288ee07fea5p-1}, /* 104 */
   {0x1.23e8978p-32, 0x1.44310de3c284bp-2, 0x1.e5a9d54bbd26cp-1}, /* 105 */
   {-0x1.2b1d77ap-29, 0x1.472b8976d498dp-2, 0x1.e529f06cb187dp-1}, /* 106 */
   {-0x1.daaa348p-31, 0x1.4a253cb97efd1p-2, 0x1.e4a8e007231a2p-1}, /* 107 */
   {-0x1.322f5708p-28, 0x1.4d1e2260c3422p-2, 0x1.e426a500f6e33p-1}, /* 108 */
   {0x1.64758e8p-29, 0x1.50163eca0b337p-2, 0x1.e3a33e996b722p-1}, /* 109 */
   {0x1.12486278p-28, 0x1.530d89a17e007p-2, 0x1.e31eae3fb917bp-1}, /* 110 */
   {-0x1.6c3416ccp-27, 0x1.5603fcf8cd8a3p-2, 0x1.e298f502a579bp-1}, /* 111 */
   {0x1.ab481ffp-29, 0x1.58f9a896aa209p-2, 0x1.e2121016e14fcp-1}, /* 112 */
   {-0x1.6eb838bp-29, 0x1.5bee77aaf890bp-2, 0x1.e18a032eb4df5p-1}, /* 113 */
   {-0x1.d159b8p-32, 0x1.5ee2734efeef5p-2, 0x1.e100ccaa6bd78p-1}, /* 114 */
   {-0x1.a42e4ap-34, 0x1.61d595bedeabcp-2, 0x1.e0766d944915ep-1}, /* 115 */
   {-0x1.43d0dcp-30, 0x1.64c7dd5cc0cd1p-2, 0x1.dfeae63903034p-1}, /* 116 */
   {-0x1.8c7bdb7p-27, 0x1.67b9453ca2122p-2, 0x1.df5e378482eaep-1}, /* 117 */
   {0x1.1c0ead6p-30, 0x1.6aa9d844c980ap-2, 0x1.ded05f6a23a52p-1}, /* 118 */
   {0x1.7d526p-31, 0x1.6d99867e90d92p-2, 0x1.de4160e97b2e2p-1}, /* 119 */
   {0x1.924e0368p-28, 0x1.7088555d3c816p-2, 0x1.ddb13afb14e37p-1}, /* 120 */
   {-0x1.74b7c3ep-30, 0x1.73763c09fba09p-2, 0x1.dd1fef5335416p-1}, /* 121 */
   {-0x1.7943adp-30, 0x1.766340685c982p-2, 0x1.dc8d7ccf2567ap-1}, /* 122 */
   {0x1.79dd614p-29, 0x1.794f5f7522b88p-2, 0x1.dbf9e402aa5c3p-1}, /* 123 */
   {0x1.7b64f32p-30, 0x1.7c3a939c32d81p-2, 0x1.db652607e0db1p-1}, /* 124 */
   {-0x1.2bea5ce8p-28, 0x1.7f24db825141cp-2, 0x1.dacf43268b5bp-1}, /* 125 */
   {0x1.733c024p-30, 0x1.820e3b8bf15ap-2, 0x1.da383a7aed887p-1}, /* 126 */
   {-0x1.eac0fc94p-27, 0x1.84f6a51d077b3p-2, 0x1.d9a00efd84537p-1}, /* 127 */
   {0x1.aca37338p-27, 0x1.87de2f4704f98p-2, 0x1.d906bbf17f4dap-1}, /* 128 */
   {-0x1.910c4fp-30, 0x1.8ac4b7dc0d986p-2, 0x1.d86c4862b5d6ep-1}, /* 129 */
   {-0x1.33bb86p-31, 0x1.8daa52b4dc041p-2, 0x1.d7d0b0374a559p-1}, /* 130 */
   {-0x1.69e1507p-27, 0x1.908ef408ad22p-2, 0x1.d733f5e71c3bcp-1}, /* 131 */
   {0x1.cffacf08p-27, 0x1.9372ab7784d36p-2, 0x1.d696161d786c9p-1}, /* 132 */
   {-0x1.8629d9fp-26, 0x1.965552b0849abp-2, 0x1.d5f7190eeae23p-1}, /* 133 */
   {0x1.415p-30, 0x1.99371687c64f3p-2, 0x1.d556f5155d9ddp-1}, /* 134 */
   {-0x1.bd37aad8p-27, 0x1.9c17cf40715cbp-2, 0x1.d4b5b2caf8386p-1}, /* 135 */
   {0x1.d02cde7p-26, 0x1.9ef79ea4d995dp-2, 0x1.d4134ac5eb246p-1}, /* 136 */
   {-0x1.10547acp-30, 0x1.a1d653d9adf5ep-2, 0x1.d36fc7d291602p-1}, /* 137 */
   {-0x1.01a1a228p-27, 0x1.a4b40f9c0120bp-2, 0x1.d2cb22b45236bp-1}, /* 138 */
   {0x1.3ce2bacp-29, 0x1.a790ce2056b9ap-2, 0x1.d2255c3ae11a5p-1}, /* 139 */
   {-0x1.ccb4a6p-32, 0x1.aa6c828db4ea8p-2, 0x1.d17e774d4e3e2p-1}, /* 140 */
   {0x1.5db4bp-29, 0x1.ad47321f29847p-2, 0x1.d0d672bc0b122p-1}, /* 141 */
   {0x1.32f6a6ep-29, 0x1.b020d7a285e23p-2, 0x1.d02d4fb84d334p-1}, /* 142 */
   {0x1.cf8e39bcp-26, 0x1.b2f97c27f7494p-2, 0x1.cf830c2248c5ep-1}, /* 143 */
   {0x1.8927bbp-30, 0x1.b5d10129a750ap-2, 0x1.ced7af22cb105p-1}, /* 144 */
   {-0x1.3dec3c1p-28, 0x1.b8a77f8d0bbc5p-2, 0x1.ce2b32e50d6cdp-1}, /* 145 */
   {-0x1.26ba536p-28, 0x1.bb7cf08f0290dp-2, 0x1.cd7d98fcf3b1ep-1}, /* 146 */
   {0x1.23c568ep-29, 0x1.be51524e3aa53p-2, 0x1.cccee1da3d56ep-1}, /* 147 */
   {-0x1.f3b3afp-29, 0x1.c1249c1f5f2f6p-2, 0x1.cc1f0f95e1e24p-1}, /* 148 */
   {-0x1.1286a47p-28, 0x1.c3f6d2ef7054bp-2, 0x1.cb6e20ff37e81p-1}, /* 149 */
   {0x1.641214ep-29, 0x1.c6c7f594003d9p-2, 0x1.cabc165bf1b6p-1}, /* 150 */
   {0x1.0cda7c9p-27, 0x1.c997ff2bffccbp-2, 0x1.ca08f0dee434cp-1}, /* 151 */
   {-0x1.5557ac9p-28, 0x1.cc66e7b42e8f1p-2, 0x1.c954b28bca62ep-1}, /* 152 */
   {0x1.555eb62p-28, 0x1.cf34bccc567a1p-2, 0x1.c89f57f6e20f3p-1}, /* 153 */
   {-0x1.4e0e361p-28, 0x1.d2016cbb5e39ap-2, 0x1.c7e8e59999e1fp-1}, /* 154 */
   {0x1.446da1ep-29, 0x1.d4cd039d0ed05p-2, 0x1.c731585f970ebp-1}, /* 155 */
   {0x1.103d328p-29, 0x1.d797767638decp-2, 0x1.c678b3174afe1p-1}, /* 156 */
   {0x1.5814d6p-28, 0x1.da60c7ae9dc22p-2, 0x1.c5bef522be6fbp-1}, /* 157 */
   {-0x1.5e2321ep-29, 0x1.dd28f054cbb3fp-2, 0x1.c5042052c8c42p-1}, /* 158 */
   {-0x1.a259ffep-29, 0x1.dfeff54854631p-2, 0x1.c44833611bc7dp-1}, /* 159 */
   {-0x1.4f28d8p-31, 0x1.e2b5d34665b35p-2, 0x1.c38b2f278ea7ep-1}, /* 160 */
   {-0x1.de571p-36, 0x1.e57a86d137f2p-2, 0x1.c2cd1493d05c2p-1}, /* 161 */
   {0x1.e0d8d14p-29, 0x1.e83e0ffb7bfb4p-2, 0x1.c20de3a08ea07p-1}, /* 162 */
   {-0x1.12a858ep-28, 0x1.eb0067e48baf4p-2, 0x1.c14d9e2bd511ep-1}, /* 163 */
   {0x1.9a17403p-27, 0x1.edc19997a4431p-2, 0x1.c08c413089b2ep-1}, /* 164 */
   {0x1.68c8636p-29, 0x1.f0819163d1bcp-2, 0x1.bfc9d21568f32p-1}, /* 165 */
   {0x1.4cc5eb8p-29, 0x1.f3405a482e11dp-2, 0x1.bf064dd580fc9p-1}, /* 166 */
   {-0x1.fce7cd8p-27, 0x1.f5fde8f3f11d4p-2, 0x1.be41b798f6b97p-1}, /* 167 */
   {-0x1.af8169p-29, 0x1.f8ba4c98a9816p-2, 0x1.bd7c0b1a7f14bp-1}, /* 168 */
   {0x1.6e39e2p-33, 0x1.fb7575d1ea75p-2, 0x1.bcb54cac5dde5p-1}, /* 169 */
   {0x1.30f9256p-28, 0x1.fe2f665dcd168p-2, 0x1.bbed7bd1e17bp-1}, /* 170 */
   {0x1.626de2p-31, 0x1.00740ca0d5fbbp-1, 0x1.bb2499f9fe7a3p-1}, /* 171 */
   {0x1.5cc703p-30, 0x1.01cfc8afeea0ep-1, 0x1.ba5aa650dd495p-1}, /* 172 */
   {-0x1.6191e6p-32, 0x1.032ae54fe4057p-1, 0x1.b98fa2065a5e6p-1}, /* 173 */
   {-0x1.6b1485p-31, 0x1.0485624c328c8p-1, 0x1.b8c38d39737bcp-1}, /* 174 */
   {-0x1.11fbc3ap-29, 0x1.05df3e66a716dp-1, 0x1.b7f668a580fdp-1}, /* 175 */
   {-0x1.0eca7fp-27, 0x1.07387825589ecp-1, 0x1.b728352c44517p-1}, /* 176 */
   {-0x1.8073bc9ep-25, 0x1.089109ef1284dp-1, 0x1.b658f630112edp-1}, /* 177 */
   {-0x1.9dcf0adp-27, 0x1.09e9051603e29p-1, 0x1.b588a13ab750fp-1}, /* 178 */
   {-0x1.06ea9fp-29, 0x1.0b405820e78e7p-1, 0x1.b4b740d3cc07bp-1}, /* 179 */
   {-0x1.36a8d0cp-30, 0x1.0c9704a1ea4e5p-1, 0x1.b3e4d40f5524dp-1}, /* 180 */
   {0x1.63d1f3p-30, 0x1.0ded0bc01a533p-1, 0x1.b3115a3a628afp-1}, /* 181 */
   {0x1.f3181f14p-26, 0x1.0f4270e4787bfp-1, 0x1.b23cd1314c779p-1}, /* 182 */
   {-0x1.f269b78p-29, 0x1.109723e75c5cfp-1, 0x1.b167430cfebdbp-1}, /* 183 */
   {0x1.1d84dc08p-27, 0x1.11eb36bc9db52p-1, 0x1.b090a4915ee88p-1}, /* 184 */
   {-0x1.08e60068p-27, 0x1.133e9ba0061d8p-1, 0x1.afb8fe69a6527p-1}, /* 185 */
   {0x1.cda72abp-27, 0x1.14915d557a7c9p-1, 0x1.aee049bc0aeep-1}, /* 186 */
   {-0x1.f32f95p-30, 0x1.15e36dfb6bb55p-1, 0x1.ae068f6991699p-1}, /* 187 */
   {0x1.138092dp-28, 0x1.1734d6f34d7fp-1, 0x1.ad2bc96c1e1f5p-1}, /* 188 */
   {0x1.6b382dd4p-26, 0x1.188595ae376a5p-1, 0x1.ac4ff962bdb6dp-1}, /* 189 */
   {-0x1.f12fafap-28, 0x1.19d59f592a587p-1, 0x1.ab7326685eb57p-1}, /* 190 */
   {-0x1.2909e5ap-28, 0x1.1b2500aed7ac6p-1, 0x1.aa954823cf815p-1}, /* 191 */
   {-0x1.d66a8978p-25, 0x1.1c73aa0150cf9p-1, 0x1.a9b668fb0503fp-1}, /* 192 */
   {0x1.311ea86p-27, 0x1.1dc1b7db74db1p-1, 0x1.a8d675d9c6cc8p-1}, /* 193 */
   {-0x1.41c02b8p-31, 0x1.1f0f08a1a06a4p-1, 0x1.a7f5853bb4309p-1}, /* 194 */
   {-0x1.ca1f4edp-26, 0x1.205ba57211271p-1, 0x1.a71391146958fp-1}, /* 195 */
   {-0x1.910ce77p-28, 0x1.21a7988f8326bp-1, 0x1.a63092626202fp-1}, /* 196 */
   {0x1.2bfadbeep-25, 0x1.22f2dc71afab6p-1, 0x1.a54c8cd9fd0d9p-1}, /* 197 */
   {-0x1.5f1c02a8p-27, 0x1.243d5df4afb93p-1, 0x1.a4678dbbe5e73p-1}, /* 198 */
   {-0x1.db12b9p-30, 0x1.2587347f493a4p-1, 0x1.a38184db0df23p-1}, /* 199 */
   {-0x1.7b29ep-30, 0x1.26d05490f2f61p-1, 0x1.a29a7a2f40b49p-1}, /* 200 */
   {-0x1.b3ddca4p-29, 0x1.2818be6930629p-1, 0x1.a1b26d8f070d7p-1}, /* 201 */
   {0x1.e112744p-29, 0x1.2960730ff2bcdp-1, 0x1.a0c95e3df5e0ep-1}, /* 202 */
   {-0x1.5269766p-28, 0x1.2aa76dafcbbf4p-1, 0x1.9fdf4fae1df6fp-1}, /* 203 */
   {-0x1.09777e1p-28, 0x1.2bedb1b6b4e15p-1, 0x1.9ef43f6cbe162p-1}, /* 204 */
   {0x1.ae2051fp-28, 0x1.2d333e4617f25p-1, 0x1.9e082e148680ep-1}, /* 205 */
   {-0x1.36f6ced8p-27, 0x1.2e780cb47180ep-1, 0x1.9d1b207f383c3p-1}, /* 206 */
   {-0x1.23fdc6bp-28, 0x1.2fbc23fba2f44p-1, 0x1.9c2d1197130a7p-1}, /* 207 */
   {0x1.bc540ep-33, 0x1.30ff7fd6d967dp-1, 0x1.9b3e0478b961bp-1}, /* 208 */
   {-0x1.cfb4ed7p-28, 0x1.32421da0bf0e9p-1, 0x1.9a4dfb1c89326p-1}, /* 209 */
   {0x1.55802aecp-26, 0x1.3384042a92b1dp-1, 0x1.995cf06920d11p-1}, /* 210 */
   {0x1.60719e4p-28, 0x1.34c52608e3a92p-1, 0x1.986aee6d6837ep-1}, /* 211 */
   {-0x1.cbf2e48p-30, 0x1.36058ac8863b6p-1, 0x1.9777ef832c986p-1}, /* 212 */
   {0x1.9061c32p-27, 0x1.374533ab707dp-1, 0x1.9683f2ad7e2ecp-1}, /* 213 */
   {-0x1.da84dfep-27, 0x1.3884160f9488fp-1, 0x1.958f000fdd50ap-1}, /* 214 */
   {0x1.92e8a74p-29, 0x1.39c23eba6b22ap-1, 0x1.94990dd9cee51p-1}, /* 215 */
   {-0x1.bff5d9ap-29, 0x1.3affa20756bddp-1, 0x1.93a225056084ap-1}, /* 216 */
   {0x1.4c462p-36, 0x1.3c3c4498e98ebp-1, 0x1.92aa41fbb951cp-1}, /* 217 */
   {-0x1.e4613e9p-28, 0x1.3d782261dff62p-1, 0x1.91b167e92d706p-1}, /* 218 */
   {0x1.0eb2964p-30, 0x1.3eb33ed579bbep-1, 0x1.90b794146043cp-1}, /* 219 */
   {-0x1.60abec2p-29, 0x1.3fed94c834d8ap-1, 0x1.8fbcca9583479p-1}, /* 220 */
   {0x1.6954977p-27, 0x1.4127281ddac03p-1, 0x1.8ec1085083553p-1}, /* 221 */
   {0x1.a16fec2p-29, 0x1.425ff1f841235p-1, 0x1.8dc452ca328d3p-1}, /* 222 */
   {-0x1.27bcdd3p-27, 0x1.4397f44aa44f2p-1, 0x1.8cc6a8771e165p-1}, /* 223 */
   {-0x1.60dded4p-28, 0x1.44cf317a563dbp-1, 0x1.8bc8076122736p-1}, /* 224 */
   {-0x1.9a8f405cp-26, 0x1.4605a2b02d705p-1, 0x1.8ac875232f3efp-1}, /* 225 */
   {0x1.32777dcp-27, 0x1.473b532bc5a67p-1, 0x1.89c7e8713120cp-1}, /* 226 */
   {-0x1.1418a7bp-26, 0x1.4870306ca20e2p-1, 0x1.88c670a0ea774p-1}, /* 227 */
   {-0x1.fed182ep-28, 0x1.49a44886b534p-1, 0x1.87c401fdf05e5p-1}, /* 228 */
   {0x1.86144d8p-27, 0x1.4ad796ea1410cp-1, 0x1.86c0a04dbacc5p-1}, /* 229 */
   {0x1.1bc2e6p-33, 0x1.4c0a14640d2afp-1, 0x1.85bc51aa114c2p-1}, /* 230 */
   {-0x1.f53d2fep-28, 0x1.4d3bc5aaa8cd5p-1, 0x1.84b7121b30a13p-1}, /* 231 */
   {-0x1.2e100ap-30, 0x1.4e6cab91556bep-1, 0x1.83b0e0e6b6cccp-1}, /* 232 */
   {-0x1.fa58c62p-29, 0x1.4f9cc1c69fddep-1, 0x1.82a9c1c1ab463p-1}, /* 233 */
   {0x1.bb491ep-33, 0x1.50cc09fdcbd92p-1, 0x1.81a1b3342f858p-1}, /* 234 */
   {0x1.a11541p-28, 0x1.51fa82c3aa029p-1, 0x1.8098b67ea8509p-1}, /* 235 */
   {0x1.ab0a5d3p-27, 0x1.53282b20b96b6p-1, 0x1.7f8ecc791953p-1}, /* 236 */
   {-0x1.cba0438p-28, 0x1.5454fe43a7d7cp-1, 0x1.7e83f96af78ap-1}, /* 237 */
   {-0x1.0dd83a4p-29, 0x1.5581033a81573p-1, 0x1.7d783712e20ecp-1}, /* 238 */
   {-0x1.e9a8299p-28, 0x1.56ac33fbb8253p-1, 0x1.7c6b8acf90fa6p-1}, /* 239 */
   {0x1.225c4aap-29, 0x1.57d6939d4b513p-1, 0x1.7b5df1da18065p-1}, /* 240 */
   {-0x1.82e66ep-27, 0x1.59001b9e64d79p-1, 0x1.7a4f72157cfdfp-1}, /* 241 */
   {0x1.51a6a354p-26, 0x1.5a28d5b36d597p-1, 0x1.794002a7c9023p-1}, /* 242 */
   {0x1.13917f4p-26, 0x1.5b50b4e10bec1p-1, 0x1.782faf6dc7ba2p-1}, /* 243 */
   {0x1.49310ccp-30, 0x1.5c77bc15ab4efp-1, 0x1.771e75c43942ep-1}, /* 244 */
   {0x1.24d493cp-30, 0x1.5d9dee9de49dbp-1, 0x1.760c529bc17bp-1}, /* 245 */
   {-0x1.04638f7p-26, 0x1.5ec347044e0f4p-1, 0x1.74f94b0af972p-1}, /* 246 */
   {-0x1.3f41b28p-29, 0x1.5fe7cb834600cp-1, 0x1.73e55936a516p-1}, /* 247 */
   {-0x1.a5f6f5cp-30, 0x1.610b7515d1562p-1, 0x1.72d083b8214ebp-1}, /* 248 */
   {0x1.19fb2ep-28, 0x1.622e459eafbc1p-1, 0x1.71bac8c7b0592p-1}, /* 249 */
   {-0x1.56d2c2bp-28, 0x1.6350396fe4e62p-1, 0x1.70a42bec51665p-1}, /* 250 */
   {-0x1.3c156c2p-28, 0x1.64715385bed93p-1, 0x1.6f8caa4969708p-1}, /* 251 */
   {-0x1.f23e576p-29, 0x1.659191d2fd57fp-1, 0x1.6e7445d74f711p-1}, /* 252 */
   {0x1.1e4be38p-30, 0x1.66b0f41d484c4p-1, 0x1.6d5afecd4938dp-1}, /* 253 */
   {-0x1.397cc8d8p-27, 0x1.67cf76eac73dfp-1, 0x1.6c40d89625f63p-1}, /* 254 */
   {-0x1.202f686p-28, 0x1.68ed1e0990551p-1, 0x1.6b25cf728c35p-1}, /* 255 */
};

/* The following is a degree-7 polynomial with odd coefficients
   approximating sin2pi(x) for -2^-24 < x < 2^-11+2^-24
   with relative error 2^-77.306.
   Generated with sin_fast.sollya. */
static const double PSfast[] = {
  0x1.921fb54442d18p+2, 0x1.1a62645446203p-52, // degree 1 (h+l)
  -0x1.4abbce625be53p5,                        // degree 3
  0x1.466bc678d8d63p6,                         // degree 5
  -0x1.331554ca19669p6,                        // degree 7
};

/* The following is a degree-6 polynomial with even coefficients
   approximating cos2pi(x) for -2^-24 < x < 2^-11+2^-24
   with relative error 2^-75.188.
   Generated with cos_fast.sollya. */
static const double PCfast[] = {
  0x1p+0, -0x1.923015cp-77,                    // degree 0
  -0x1.3bd3cc9be45dep4,                        // degree 2
  0x1.03c1f080ad892p6,                         // degree 4
  -0x1.55a5c590f9e6ap6,                        // degree 6
};

// Returns (ah + al) * (bh + bl) - (al * bl)
// We can ignore al * bl when assuming al <= ulp(ah) and bl <= ulp(bh)
static inline void d_mul(double *hi, double *lo, double ah, double al,
                         double bh, double bl) {
  double s, t;

  a_mul(hi, &s, ah, bh);
  t = std::fma(al, bh, s);
  *lo = std::fma(ah, bl, t);
}

static inline void
fast_two_sum(double *hi, double *lo, double a, double b)
{
  double e;

  *hi = a + b;
  e = *hi - a; /* exact */
  *lo = b - e; /* exact */
}

/* Put in h+l an approximation of sin2pi(xh+xl),
   for 2^-24 <= xh+xl < 2^-11 + 2^-24,
   and |xl| < 2^-52.36, with absolute error < 2^-77.09
   (see evalPSfast() in sin.sage).
   Assume uh + ul approximates (xh+xl)^2. */
static void
evalPSfast (double *h, double *l, double xh, double xl, double uh, double ul)
{
  double t;
  *h = PSfast[4]; // degree 7
  *h = std::fma (*h, uh, PSfast[3]); // degree 5
  *h = std::fma (*h, uh, PSfast[2]); // degree 3
  s_mul (h, l, *h, uh, ul);
  fast_two_sum (h, &t, PSfast[0], *h);
  *l += PSfast[1] + t;
  // multiply by xh+xl
  d_mul (h, l, *h, *l, xh, xl);
}

/* Put in h+l an approximation of cos2pi(xh+xl),
   for 2^-24 <= xh+xl < 2^-11 + 2^-24,
   and |xl| < 2^-52.36, with relative error < 2^-69.96
   (see evalPCfast() in sin.sage).
   Assume uh + ul approximates (xh+xl)^2. */
static void
evalPCfast (double *h, double *l, double uh, double ul)
{
  double t;
  *h = PCfast[4]; // degree 6
  *h = std::fma (*h, uh, PCfast[3]); // degree 4
  *h = std::fma (*h, uh, PCfast[2]); // degree 2
  s_mul (h, l, *h, uh, ul);
  fast_two_sum (h, &t, PCfast[0], *h);
  *l += PCfast[1] + t;
}

/* h+l <- c1/2^64 + c0/2^128 */
static void
set_dd (double *h, double *l, uint64_t c1, uint64_t c0)
{
  uint64_t e, f, g;
  b64u64_u t;
  if (c1)
    {
      e = __builtin_clzll (c1);
      if (e)
        {
          c1 = (c1 << e) | (c0 >> (64 - e));
          c0 = c0 << e;
        }
      f = 0x3fe - e;
      t.u = (f << 52) | ((c1 << 1) >> 12);
      *h = t.f;
      c0 = (c1 << 53) | (c0 >> 11);
      if (c0)
        {
          g = __builtin_clzll (c0);
          if (g)
            c0 = c0 << g;
          t.u = ((f - 53 - g) << 52) | ((c0 << 1) >> 12);
          *l = t.f;
        }
      else
        *l = 0;
    }
  else if (c0)
    {
      e = __builtin_clzll (c0);
      f = 0x3fe - 64 - e;
      c0 = c0 << (e+1); // most significant bit shifted out
      /* put the upper 52 bits of c0 into h */
      t.u = (f << 52) | (c0 >> 12);
      *h = t.f;
      /* put the lower 12 bits of c0 into l */
      c0 = c0 << 52;
      if (c0)
        {
          g = __builtin_clzll (c0);
          c0 = c0 << (g+1);
          t.u = ((f - 64 - g) << 52) | (c0 >> 12);
          *l = t.f;
        }
      else
        *l = 0;
    }
  else
    *h = *l = 0;
}

/* Assuming 0x1.7137449123ef6p-26 < x < +Inf,
   return i and set h,l such that i/2^11+h+l approximates frac(x/(2pi)).
   Uses portable mul64x64 instead of 128-bit integer types. */
static int
reduce_fast (double *h, double *l, double x, double *err1)
{
  if (x <= 0x1.921fb54442d17p+2) // x < 2*pi
    {
      static const double CH = 0x1.45f306dc9c883p-3;
      static const double CL = -0x1.6b01ec5417056p-57;
      a_mul (h, l, CH, x);            // exact
      *l = std::fma (CL, x, *l);
      *err1 = 0x1.d9p-105 * *h; // error < 2^-104.116 * h
    }
  else // x > 0x1.921fb54442d17p+2
    {
      b64u64_u tv;
      tv.f = x;
      int e = (tv.u >> 52) & 0x7ff; /* 1025 <= e <= 2046 */
      uint64_t m = (1ull << 52) | (tv.u & 0xfffffffffffffull);
      uint64_t c[3];
      // x = m/2^53 * 2^(e-1022)
      if (e <= 1074) // 1025 <= e <= 1074: 2^2 <= x < 2^52
        {
          /* m * T[1] -> 128-bit product */
          uint64_t u_hi, u_lo;
          mul64x64(m, T[1], u_hi, u_lo);
          c[0] = u_lo;
          c[1] = u_hi;
          /* m * T[0] -> 128-bit product */
          mul64x64(m, T[0], u_hi, u_lo);
          uint64_t prev_c1 = c[1];
          c[1] += u_lo;
          c[2] = u_hi + (c[1] < prev_c1);
          e = 1075 - e; // 1 <= e <= 50
        }
      else // 1075 <= e <= 2046, 2^52 <= x < 2^1024
        {
          int i = (e - 1138 + 63) / 64; // i = ceil((e-1138)/64), 0 <= i <= 15
          uint64_t u_hi, u_lo;

          /* m * T[i+2] */
          mul64x64(m, T[i+2], u_hi, u_lo);
          c[0] = u_lo;
          c[1] = u_hi;

          /* m * T[i+1] */
          mul64x64(m, T[i+1], u_hi, u_lo);
          uint64_t prev_c1 = c[1];
          c[1] += u_lo;
          c[2] = u_hi + (c[1] < prev_c1);

          /* m * T[i] */
          mul64x64(m, T[i], u_hi, u_lo);
          c[2] += u_lo;

          e = 1139 + (i<<6) - e; // 1 <= e <= 64
        }
      if (e == 64)
        {
          c[0] = c[1];
          c[1] = c[2];
        }
      else
        {
          c[0] = (c[1] << (64 - e)) | c[0] >> e;
          c[1] = (c[2] << (64 - e)) | c[1] >> e;
        }
      set_dd (h, l, c[1], c[0]);
      *err1 = 0x1.01p-76;
    }

  double i = std::floor (*h * 0x1p11);
  *h = std::fma (i, -0x1p-11, *h);
  return (int)i;
}

/* return the maximal absolute error */
static double
sin_fast (double *h, double *l, double x)
{
  int neg = x < 0, is_sin = 1;
  double absx = neg ? -x : x;

  /* now x > 0x1.7137449123ef6p-26 */
  double err1;
  int i = reduce_fast (h, l, absx, &err1);

  // if i >= 2^10: 1/2 <= frac(x/(2pi)) < 1 thus pi <= x <= 2pi
  // we use sin(pi+x) = -sin(x)
  neg = neg ^ (i >> 10);
  i = i & 0x3ff;

  // now i < 2^10
  // if i >= 2^9: 1/4 <= frac(x/(2pi)) < 1/2 thus pi/2 <= x <= pi
  // we use sin(pi/2+x) = cos(x)
  is_sin = is_sin ^ (i >> 9);
  i = i & 0x1ff;

  // now 0 <= i < 2^9
  // if i >= 2^8: 1/8 <= frac(x/(2pi)) < 1/4
  // we use sin(pi/2-x) = cos(x)
  if (i & 0x100) // case pi/4 <= x_red <= pi/2
    {
      is_sin = !is_sin;
      i = 0x1ff - i;
      *h = 0x1p-11 - *h;
      *l = -*l;
    }

  /* Now 0 <= i < 256 and 0 <= h+l < 2^-11 */
  double sh, sl, ch, cl;
  *h -= SC[i][0];
  // now -2^-24 < h < 2^-11+2^-24
  double uh, ul;
  a_mul (&uh, &ul, *h, *h);
  ul = std::fma (*h + *h, *l, ul);
  // uh+ul approximates (h+l)^2
  evalPSfast (&sh, &sl, *h, *l, uh, ul);
  evalPCfast (&ch, &cl, uh, ul);
  double err;
  static const double sgn[2] = {1.0, -1.0};
  if (is_sin)
    {
      s_mul (&sh, &sl, sgn[neg] * SC[i][2], sh, sl);
      s_mul (&ch, &cl, sgn[neg] * SC[i][1], ch, cl);
      fast_two_sum (h, l, ch, sh);
      *l += sl + cl;
      err = 0x1.55p-69; // 2^-66.588 < 0x1.55p-69
    }
  else
    {
      s_mul (&ch, &cl, sgn[neg] * SC[i][2], ch, cl);
      s_mul (&sh, &sl, sgn[neg] * SC[i][1], sh, sl);
      fast_two_sum (h, l, ch, -sh);
      *l += cl - sl;
      err = 0x1.81p-69; // 2^-68.414 < 0x1.81p-69
    }
  return err + err1;
}

double
cr_sin (double x)
{
  b64u64_u t;
  t.f = x;
  int e = (t.u >> 52) & 0x7ff;

  if (e == 0x7ff) /* NaN, +Inf and -Inf. */
    {
      if ((t.u << 1) != ((uint64_t)0x7ff8 << 49)){
        return 0.0 / 0.0;
      }
      t.u = ~(uint64_t)0;
      return t.f;
    }

  /* now x is a regular number */

  /* For |x| <= 0x1.7137449123ef6p-26, sin(x) rounds to x (to nearest). */
  uint64_t ux = t.u & 0x7fffffffffffffff;
  // 0x3e57137449123ef6 = 0x1.7137449123ef6p-26
  if (ux <= 0x3e57137449123ef6) {
    if (x == 0)
      return x;
    // Taylor expansion of sin(x) is x - x^3/6 around zero
    double res = std::fma (x, -0x1p-54, x);
    return res;
  }

  double h, l, err;
  err = sin_fast (&h, &l, x);
  double left  = h + (l - err), right = h + (l + err);
  /* Fast path succeeds in ~99.998% of cases. When it does not,
     we return the best available fast-path result anyway (accurate
     path has been removed). */
  if (left == right)
    return left;

  /* Accurate path removed: return midpoint of the error interval. */
  return h + l;
}
