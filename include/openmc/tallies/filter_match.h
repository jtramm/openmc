#ifndef OPENMC_TALLIES_FILTERMATCH_H
#define OPENMC_TALLIES_FILTERMATCH_H

#define FILTERMATCH_BINS_WEIGHTS_SIZE 125

#ifndef DEVICE_PRINTF
#define printf(fmt, ...) (0)
#endif

namespace openmc {

//==============================================================================
//! Stores bins and weights for filtered tally events.
//==============================================================================

class FilterMatch
{
public:
  //std::vector<int> bins_;
  int bins_[FILTERMATCH_BINS_WEIGHTS_SIZE];
  //std::vector<double> weights_;
  double weights_[FILTERMATCH_BINS_WEIGHTS_SIZE];
  int bins_weights_length_ {0};
  int i_bin_;
  bool bins_present_ {false};

  bool is_full() {
    if (bins_weights_length_ >= FILTERMATCH_BINS_WEIGHTS_SIZE) {
      printf("Error: Too many filter matches - tally data will be incorrect. Increase size of FILTERMATCH_BINS_WEIGHTS_SIZE macro.\n");
      return true;
    } else {
      return false;
    }
  }

  void push_back(int bin, double weight) {
    if (!is_full()) {
      bins_[bins_weights_length_] = bin;
      weights_[bins_weights_length_] = weight;
      bins_weights_length_++;
    }
  }
  
  void push_back(int bin) {
    if (!is_full()) {
      bins_[bins_weights_length_] = bin;
      bins_weights_length_++;
    }
  }

      // Equality operator
    bool operator==(const FilterMatch& rhs) const {
        if (bins_weights_length_ != rhs.bins_weights_length_) return false;
        if (i_bin_ != rhs.i_bin_) return false;
        if (bins_present_ != rhs.bins_present_) return false;

        for (int i = 0; i < bins_weights_length_; ++i) {
            if (bins_[i] != rhs.bins_[i]) return false;
            if (fabs(weights_[i] - rhs.weights_[i]) > 1e-6) return false; // Using a small tolerance for floating point comparison
        }

        return true;
    }

    // Inequality operator
    bool operator!=(const FilterMatch& rhs) const {
        return !(*this == rhs);
    }

};

class BigFilterMatch
{
public:
  std::vector<int> bins_;
  //int bins_[FILTERMATCH_BINS_WEIGHTS_SIZE];
  std::vector<double> weights_;
  //double weights_[FILTERMATCH_BINS_WEIGHTS_SIZE];
  //int bins_weights_length_ {0};
  int i_bin_;
  bool bins_present_ {false};
};

} // namespace openmc
#endif // OPENMC_TALLIES_FILTERMATCH_H
