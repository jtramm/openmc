#ifndef OPENMC_RAY_BANK_H
#define OPENMC_RAY_BANK_H

#include "openmc/vector.h"
#include "openmc/random_ray/random_ray.h"
#include "openmc/random_ray/source_region.h"
#include "openmc/random_ray/flat_source_domain.h"
#ifdef OPENMC_MPI
#include <mpi.h>
#endif

namespace openmc {

// // Forward declaration
// class FlatSourceDomain;
// class RandomRay;
// struct RayBufferContainer;

class RayBank {
public:
  //----------------------------------------------------------------------------
  // Constructors
  RayBank();

  //----------------------------------------------------------------------------
  // Methods
  void add_ray_to_bank(RandomRay& ray);
  void buffer_ray_data_to_send(RandomRay& ray, FlatSourceDomain* domain);
  void update(FlatSourceDomain* domain);
  int ray_bank_size();
  void reset_my_ray_list();
  void communicate_rays();
  void communicate_message_metadata();
  void update_my_ray_list(FlatSourceDomain* domain);
  bool is_any_ray_alive(); 

  //----------------------------------------------------------------------------
  // Static data members


  //----------------------------------------------------------------------------
  // Public data members
  vector<RandomRay> my_ray_list_;

  // // Number of ray communications between ranks
  // uint64_t num_comms_total_ {0};
  // uint64_t num_comms_batch_ {0};

private:
  //----------------------------------------------------------------------------
  // Private data members
  int total_sending_rays_;
  int total_receiving_rays_;
  int negroups_;

  // Per-rank send buffers - data is packed directly into these in send-ready format
  // This eliminates intermediate copying and buffering
  struct RankSendBuffers {
    vector<RayExchangeData> ray_data;
    vector<float> angular_flux;
    vector<LocalCoord> coord;
    vector<int> cell_last;
    int count = 0;  // Number of rays buffered for this rank
  };
  std::unordered_map<int, RankSendBuffers> ray_send_buffer_;

  // Vector that contains the number of rays to be received from each rank
  vector<int> num_messages_receiving_;

  // Receiving buffers - allocated once based on total receiving count
  vector<RayExchangeData> received_ray_data_;
  vector<float> received_angular_flux_data_;
  vector<LocalCoord> received_coord_;
  vector<int> received_cell_last_;

}; // class DecompositionMap

} // namespace openmc

#endif // OPENMC_RAY_BANK_H
