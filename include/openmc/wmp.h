#ifndef OPENMC_WMP_H
#define OPENMC_WMP_H

#include "hdf5.h"
#include <iostream>
#include "xtensor/xtensor.hpp"

#include <array>
#include <complex>
#include <string>
#include <tuple>

#include "openmc/vector.h"

namespace openmc {

//========================================================================
// Constants
//========================================================================

// Constants that determine which value to access
constexpr int MP_EA {0}; // Pole
constexpr int MP_RS {1}; // Residue scattering
constexpr int MP_RA {2}; // Residue absorption
constexpr int MP_RF {3}; // Residue fission

// Polynomial fit indices
constexpr int FIT_S {0}; // Scattering
constexpr int FIT_A {1}; // Absorption
constexpr int FIT_F {2}; // Fission

// Multipole HDF5 file version
constexpr std::array<int, 2> WMP_VERSION {1, 1};

//========================================================================
// Windowed multipole data
//========================================================================

class WindowedMultipole {
public:
  // Types
  struct WindowInfo {
    int index_start; // Index of starting pole
    int index_end; // Index of ending pole
    bool broaden_poly; // Whether to broaden polynomial curvefit
  };

  // Constructors, destructors
  WindowedMultipole(hid_t group);

  // Methods

  //! \brief Evaluate the windowed multipole equations for cross sections in the
  //! resolved resonance regions
  //!
  //! \param E Incident neutron energy in [eV]
  //! \param sqrtkT Square root of temperature times Boltzmann constant
  //! \return Tuple of elastic scattering, absorption, and fission cross sections in [b]
  #ifdef OPENMC_OFFLOAD
#pragma omp declare target
#endif
  std::tuple<double, double, double> evaluate(double E, double sqrtkT) const;
  #ifdef OPENMC_OFFLOAD
#pragma omp end declare target
#endif

  //! \brief Evaluates the windowed multipole equations for the derivative of
  //! cross sections in the resolved resonance regions with respect to
  //! temperature.
  //!
  //! \param E Incident neutron energy in [eV]
  //! \param sqrtkT Square root of temperature times Boltzmann constant
  //! \return Tuple of derivatives of elastic scattering, absorption, and
  //!         fission cross sections in [b/K]
  std::tuple<double, double, double> evaluate_deriv(double E, double sqrtkT) const;

  void flatten_wmp_data();

  void copy_to_device();

  void release_from_device();

  #ifdef OPENMC_OFFLOAD
#pragma omp declare target
#endif
  double curvefit(int window, int poly_order, int reaction) const;

  std::complex<double> data(int pole, int res) const;
  #ifdef OPENMC_OFFLOAD
#pragma omp end declare target
#endif

  // Data members
  std::string name_; //!< Name of nuclide
  double E_min_; //!< Minimum energy in [eV]
  double E_max_; //!< Maximum energy in [eV]
  double sqrt_awr_; //!< Square root of atomic weight ratio
  double inv_spacing_; //!< 1 / spacing in sqrt(E) space
  int fit_order_; //!< Order of the fit
  bool fissionable_; //!< Is the nuclide fissionable?
  vector<WindowInfo> window_info_; // Information about a window
  xt::xtensor<double, 3> curvefit_; // Curve fit coefficients (window, poly order, reaction)
  double* device_curvefit_ {nullptr};
  xt::xtensor<std::complex<double>, 2> data_; //!< Poles and residues
  std::complex<double>* device_data_ {nullptr};
  int n_order_;
  int n_reactions_;
  int n_data_size_;

  // Constant data
  #ifdef OPENMC_OFFLOAD
#pragma omp declare target
#endif
  static const int MAX_POLY_COEFFICIENTS; //!< Max order of polynomial fit plus one
  #ifdef OPENMC_OFFLOAD
#pragma omp end declare target
#endif
};

//========================================================================
// Non-member functions
//========================================================================

//! Check to make sure WMP library data version matches
//!
//! \param[in] file  HDF5 file object
void check_wmp_version(hid_t file);

//! \brief Checks for the existence of a multipole library in the directory and
//! loads it
//!
//! \param[in] i_nuclide  Index in global nuclides array
void read_multipole_data(int i_nuclide);


//==============================================================================
//! Doppler broadens the windowed multipole curvefit.
//!
//! The curvefit is a polynomial of the form a/E + b/sqrt(E) + c + d sqrt(E)...
//!
//! \param E       The energy to evaluate the broadening at
//! \param dopp    sqrt(atomic weight ratio / kT) with kT given in eV
//! \param n       The number of components to the polynomial
//! \param factors The output leading coefficient
//==============================================================================

extern "C" void broaden_wmp_polynomials(double E, double dopp, int n, double factors[]);

} // namespace openmc

#endif // OPENMC_WMP_H
