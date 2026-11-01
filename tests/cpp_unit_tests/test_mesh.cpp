#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <pugixml.hpp>

#include "openmc/hdf5_interface.h"
#include "openmc/mesh.h"
#include "openmc/xdg.h"

#ifdef OPENMC_XDG_ENABLED
#include "xdg/config.h"
#endif

#ifdef OPENMC_LIBMESH_ENABLED
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <random>

#include "libmesh/elem.h"
#include "libmesh/enum_to_string.h"
#include "libmesh/exodusII_io.h"
#include "libmesh/libmesh.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/mesh_modification.h"
#include "libmesh/mesh_refinement.h"
#include "libmesh/replicated_mesh.h"

#include "openmc/position.h"
#include "openmc/settings.h"
#endif

using namespace openmc;

#ifdef OPENMC_XDG_ENABLED
TEST_CASE("XDG surface bins include every element face")
{
  const auto mesh_dir = std::filesystem::path(__FILE__).parent_path() /
                        "../regression_tests/unstructured_mesh";
  for (const std::string library : {"moab", "libmesh"}) {
    auto backend =
      library == "moab" ? xdg::MeshLibrary::MOAB : xdg::MeshLibrary::LIBMESH;
    if (!xdg::XDGConfig::config().mesh_manager_enabled(backend))
      continue;

    for (auto [filename, expected] : {std::pair {"test_mesh_tets.e", 48000},
           std::pair {"test_mesh_hexes.e", 6000}}) {
      DYNAMIC_SECTION(library << " " << filename)
      {
        pugi::xml_document doc;
        auto node = doc.append_child("mesh");
        node.append_attribute("id") = 1;
        node.append_child("filename").text() =
          (mesh_dir / filename).string().c_str();
        node.append_child("library").text() = library.c_str();
        XDGMesh mesh(node);
        CHECK(mesh.n_surface_bins() == expected);

        // Wrapping an initialized instance must also populate the cache,
        // without loading a file or initializing the mesh manager again.
        XDGMesh external(mesh.xdg_instance());
        CHECK(external.n_surface_bins() == expected);
      }
    }
  }
}
#endif

TEST_CASE("Test mesh hdf5 roundtrip - regular")
{
  // The XML data as a string
  std::string xml_string = R"(
        <mesh id="1">
            <dimension>3 4 5</dimension>
            <lower_left>-2 -3 -5</lower_left>
            <upper_right>2 3 5</upper_right>
       </mesh>
    )";

  // Create a pugixml document object
  pugi::xml_document doc;

  // Load the XML from the string
  pugi::xml_parse_result result = doc.load_string(xml_string.c_str());

  pugi::xml_node root = doc.child("mesh");

  auto mesh = RegularMesh(root);

  hid_t file_id = file_open("mesh.h5", 'w');

  mesh.to_hdf5(file_id);

  file_close(file_id);

  hid_t file_id2 = file_open("mesh.h5", 'r');

  hid_t group = open_group(file_id2, "mesh 1");

  auto mesh2 = RegularMesh(group);

  file_close(file_id2);

  remove("mesh.h5");

  REQUIRE(mesh2.shape_ == mesh.shape_);

  REQUIRE(mesh2.lower_left() == mesh.lower_left());

  REQUIRE(mesh2.upper_right() == mesh.upper_right());
}

TEST_CASE("Test mesh hdf5 roundtrip - rectilinear")
{
  // The XML data as a string
  std::string xml_string = R"(
        <mesh id="1" type="rectilinear">
            <x_grid>0.0 1.0 5.0 10.0</x_grid>
            <y_grid>-10.0 -5.0 0.0</y_grid>
            <z_grid>-100.0 0.0 100.0</z_grid>
        </mesh>
    )";

  // Create a pugixml document object
  pugi::xml_document doc;

  // Load the XML from the string
  pugi::xml_parse_result result = doc.load_string(xml_string.c_str());

  pugi::xml_node root = doc.child("mesh");

  auto mesh = RectilinearMesh(root);

  hid_t file_id = file_open("mesh.h5", 'w');

  mesh.to_hdf5(file_id);

  file_close(file_id);

  hid_t file_id2 = file_open("mesh.h5", 'r');

  hid_t group = open_group(file_id2, "mesh 1");

  auto mesh2 = RectilinearMesh(group);

  file_close(file_id2);

  remove("mesh.h5");

  REQUIRE(mesh2.shape_ == mesh.shape_);

  REQUIRE(mesh2.grid_ == mesh.grid_);
}

TEST_CASE("Test mesh hdf5 roundtrip - cylindrical")
{
  // The XML data as a string
  std::string xml_string = R"(
        <mesh id="1" type="cylindrical">
            <r_grid>0.1 0.2 0.5 1.0</r_grid>
            <phi_grid>0.0 6.283185307179586</phi_grid>
            <z_grid>0.1 0.2 0.4 0.6 1.0</z_grid>
            <origin>0 0 0</origin>
        </mesh>
    )";

  // Create a pugixml document object
  pugi::xml_document doc;

  // Load the XML from the string
  pugi::xml_parse_result result = doc.load_string(xml_string.c_str());

  pugi::xml_node root = doc.child("mesh");

  auto mesh = CylindricalMesh(root);

  hid_t file_id = file_open("mesh.h5", 'w');

  mesh.to_hdf5(file_id);

  file_close(file_id);

  hid_t file_id2 = file_open("mesh.h5", 'r');

  hid_t group = open_group(file_id2, "mesh 1");

  auto mesh2 = CylindricalMesh(group);

  file_close(file_id2);

  remove("mesh.h5");

  REQUIRE(mesh2.shape_ == mesh.shape_);

  REQUIRE(mesh2.grid_ == mesh.grid_);
}

TEST_CASE("Test mesh hdf5 roundtrip - spherical")
{
  // The XML data as a string
  std::string xml_string = R"(
        <mesh id="1" type="spherical">
            <r_grid>0.1 0.2 0.5 1.0</r_grid>
            <theta_grid>0.0 3.141592653589793</theta_grid>
            <phi_grid>0.0 6.283185307179586</phi_grid>
            <origin>0.0 0.0 0.0</origin>
        </mesh>'
    )";

  // Create a pugixml document object
  pugi::xml_document doc;

  // Load the XML from the string
  pugi::xml_parse_result result = doc.load_string(xml_string.c_str());

  pugi::xml_node root = doc.child("mesh");

  auto mesh = SphericalMesh(root);

  hid_t file_id = file_open("mesh.h5", 'w');

  mesh.to_hdf5(file_id);

  file_close(file_id);

  hid_t file_id2 = file_open("mesh.h5", 'r');

  hid_t group = open_group(file_id2, "mesh 1");

  auto mesh2 = SphericalMesh(group);

  file_close(file_id2);

  remove("mesh.h5");

  REQUIRE(mesh2.shape_ == mesh.shape_);

  REQUIRE(mesh2.grid_ == mesh.grid_);
}

TEST_CASE("Test multiple meshes HDF5 roundtrip - spherical")
{
  // The XML data as a string
  std::string xml_string = R"(
        <meshes>
        <mesh id="1" type="spherical">
            <r_grid>0.1 0.2 0.5 1.0</r_grid>
            <theta_grid>0.0 3.141592653589793</theta_grid>
            <phi_grid>0.0 6.283185307179586</phi_grid>
            <origin>0.0 0.0 0.0</origin>
        </mesh>
         <mesh id="2">
            <dimension>3 4 5</dimension>
            <lower_left>-2 -3 -5</lower_left>
            <upper_right>2 3 5</upper_right>
       </mesh>
       </meshes>
    )";

  // Create a pugixml document object
  pugi::xml_document doc;

  // Load the XML from the string
  pugi::xml_parse_result result = doc.load_string(xml_string.c_str());

  pugi::xml_node root = doc.child("meshes");

  read_meshes(root);

  const auto spherical_mesh_xml =
    dynamic_cast<SphericalMesh*>(model::meshes[0].get());
  const auto regular_mesh_xml =
    dynamic_cast<RegularMesh*>(model::meshes[1].get());

  hid_t file_id = file_open("meshes.h5", 'w');

  hid_t root_group = create_group(file_id, "root");

  open_group(file_id, "root");

  meshes_to_hdf5(root_group);

  close_group(root_group);

  file_close(file_id);

  hid_t file_id2 = file_open("meshes.h5", 'r');

  hid_t root_group_read = open_group(file_id2, "root");

  hid_t mesh_group_read = open_group(root_group_read, "meshes");

  read_meshes(mesh_group_read);

  // increment mesh IDs to avoid collision during read
  for (auto& mesh : model::meshes) {
    mesh->set_id(mesh->id() + 10);
  }

  const auto spherical_mesh_hdf5 = dynamic_cast<SphericalMesh*>(
    model::meshes[model::mesh_map[spherical_mesh_xml->id_]].get());
  const auto regular_mesh_hdf5 = dynamic_cast<RegularMesh*>(
    model::meshes[model::mesh_map[regular_mesh_xml->id_]].get());

  remove("meshes.h5");

  REQUIRE(spherical_mesh_hdf5->shape_ == spherical_mesh_xml->shape_);
  REQUIRE(spherical_mesh_hdf5->grid_ == spherical_mesh_xml->grid_);

  REQUIRE(regular_mesh_hdf5->shape_ == regular_mesh_xml->shape_);
  REQUIRE(regular_mesh_hdf5->lower_left() == regular_mesh_xml->lower_left());
  REQUIRE(regular_mesh_hdf5->upper_right() == regular_mesh_xml->upper_right());
}

TEST_CASE("Test get_index_in_direction - regular")
{
  // The XML data as a string
  std::string xml_string = R"(
        <mesh id="1">
            <dimension>2 1 1</dimension>
            <lower_left>-2 -2 -2</lower_left>
            <upper_right>2 2 2</upper_right>
       </mesh>
    )";

  // Create a pugixml document object
  pugi::xml_document doc;

  // Load the XML from the string
  pugi::xml_parse_result result = doc.load_string(xml_string.c_str());

  pugi::xml_node root = doc.child("mesh");

  auto mesh = RegularMesh(root);

  REQUIRE(mesh.get_index_in_direction(-4.1, 0) == 0);
  REQUIRE(mesh.get_index_in_direction(-4.0, 0) == 0);
  REQUIRE(mesh.get_index_in_direction(-3.9, 0) == 0);

  REQUIRE(mesh.get_index_in_direction(-2.1, 0) == 0);
  REQUIRE(mesh.get_index_in_direction(-2.0, 0) == 1); // lower left
  REQUIRE(mesh.get_index_in_direction(-1.9, 0) == 1);

  REQUIRE(mesh.get_index_in_direction(-0.1, 0) == 1);
  REQUIRE(mesh.get_index_in_direction(0.0, 0) == 1);
  REQUIRE(mesh.get_index_in_direction(0.1, 0) == 2);

  REQUIRE(mesh.get_index_in_direction(1.9, 0) == 2);
  REQUIRE(mesh.get_index_in_direction(2.0, 0) == 2); // upper right
  REQUIRE(mesh.get_index_in_direction(2.1, 0) == 3);

  REQUIRE(mesh.get_index_in_direction(3.9, 0) == 3);
  REQUIRE(mesh.get_index_in_direction(4.0, 0) == 3);
  REQUIRE(mesh.get_index_in_direction(4.1, 0) == 3);
}

TEST_CASE("Test get_index_in_direction - rectilinear")
{
  // The XML data as a string
  std::string xml_string = R"(
        <mesh id="1" type="rectilinear">
            <x_grid>-1.0 0.0 2.0</x_grid>
            <y_grid>-1.0 1.0</y_grid>
            <z_grid>-1.0 1.0</z_grid>
        </mesh>
    )";

  // Create a pugixml document object
  pugi::xml_document doc;

  // Load the XML from the string
  pugi::xml_parse_result result = doc.load_string(xml_string.c_str());

  pugi::xml_node root = doc.child("mesh");

  auto mesh = RectilinearMesh(root);

  REQUIRE(mesh.get_index_in_direction(-5.0, 0) == 0);

  REQUIRE(mesh.get_index_in_direction(-1.1, 0) == 0);
  REQUIRE(mesh.get_index_in_direction(-1.0, 0) == 1); // lower left
  REQUIRE(mesh.get_index_in_direction(-0.9, 0) == 1);

  REQUIRE(mesh.get_index_in_direction(1.9, 0) == 2);
  REQUIRE(mesh.get_index_in_direction(2.0, 0) == 2); // upper right
  REQUIRE(mesh.get_index_in_direction(2.1, 0) == 3);

  REQUIRE(mesh.get_index_in_direction(5.0, 0) == 3);
}

TEST_CASE("Test regular mesh ray tracing from outside")
{
  std::string xml_string = R"(
        <mesh id="1">
            <dimension>2 1 1</dimension>
            <lower_left>-2 -2 -2</lower_left>
            <upper_right>2 2 2</upper_right>
       </mesh>
    )";

  pugi::xml_document doc;
  pugi::xml_parse_result result = doc.load_string(xml_string.c_str());
  auto mesh = RegularMesh(doc.child("mesh"));

  vector<int> bins;
  vector<double> lengths;

  mesh.bins_crossed(Position {-6.0, 0.0, 0.0}, Position {6.0, 0.0, 0.0},
    Direction {1.0, 0.0, 0.0}, bins, lengths);

  REQUIRE(bins == vector<int> {0, 1});
  REQUIRE(lengths.size() == 2);
  REQUIRE(lengths[0] == Catch::Approx(1.0 / 6.0));
  REQUIRE(lengths[1] == Catch::Approx(1.0 / 6.0));

  bins.clear();
  lengths.clear();

  mesh.bins_crossed(Position {6.0, 0.0, 0.0}, Position {-6.0, 0.0, 0.0},
    Direction {-1.0, 0.0, 0.0}, bins, lengths);

  REQUIRE(bins == vector<int> {1, 0});
  REQUIRE(lengths.size() == 2);
  REQUIRE(lengths[0] == Catch::Approx(1.0 / 6.0));
  REQUIRE(lengths[1] == Catch::Approx(1.0 / 6.0));
}

#ifdef OPENMC_LIBMESH_ENABLED
namespace {

//! Initialize libMesh once for the test cases that need it, unless something
//! else (such as XDG) already has. Stdout is kept synchronized, or the test
//! report written to it can be lost. Until the guard below goes out of scope,
//! the communicator is OpenMC's libMesh communicator and the solver is random
//! ray, in which a track-length traversal that does not terminate is a fatal
//! error (rather than a warning).
const libMesh::Parallel::Communicator& init_libmesh()
{
  static const char* argv[] = {"test_mesh", "--sync-with-stdio"};
  static std::unique_ptr<libMesh::LibMeshInit> init;
  static libMesh::Parallel::Communicator serial_comm;
  if (!init && !libMesh::initialized()) {
    init = std::make_unique<libMesh::LibMeshInit>(2, argv);
    // Finalize libMesh at exit before the static objects it created during
    // initialization are destroyed (they would be destroyed first otherwise)
    std::atexit([] { init.reset(); });
  }
  const libMesh::Parallel::Communicator& comm =
    init ? init->comm() : serial_comm;
  return comm;
}

struct LibMeshComm {
  LibMeshComm() : comm(init_libmesh()), solver_type(settings::solver_type)
  {
    settings::libmesh_comm = &comm;
    settings::solver_type = SolverType::RANDOM_RAY;
  }
  ~LibMeshComm()
  {
    settings::libmesh_comm = nullptr;
    settings::solver_type = solver_type;
  }
  const libMesh::Parallel::Communicator& comm;
  const SolverType solver_type;
};

//! Distances along the segment from r0 to r1 at which it enters and leaves
//! the box [lo, hi] (equal if it misses the box)
std::pair<double, double> box_crossing(
  Position lo, Position hi, Position r0, Position r1)
{
  const Position d = r1 - r0;
  const double length = d.norm();
  double t_lo = 0.0;
  double t_hi = 1.0;
  for (int i = 0; i < 3; i++) {
    if (d[i] == 0.0) {
      if (r0[i] < lo[i] || r0[i] > hi[i])
        return {0.0, 0.0};
      continue;
    }
    const double ta = (lo[i] - r0[i]) / d[i];
    const double tb = (hi[i] - r0[i]) / d[i];
    t_lo = std::max(t_lo, std::min(ta, tb));
    t_hi = std::min(t_hi, std::max(ta, tb));
  }
  if (t_hi <= t_lo)
    return {0.0, 0.0};
  return {t_lo * length, t_hi * length};
}

//! Total length scored on a mesh for the track from r0 to r1
double scored_length(const Mesh& umesh, Position r0, Position r1)
{
  const double length = (r1 - r0).norm();
  vector<int> bins;
  vector<double> lengths;
  umesh.bins_crossed(r0, r1, (r1 - r0) / length, bins, lengths);
  double total = 0.0;
  for (double f : lengths)
    total += f * length;
  return total;
}

//! For a mesh that tiles the box [lo, hi]: check that the segments of the
//! track from r0 to r1 add up to the part of the track inside the box, and
//! that each segment's midpoint is in the segment's bin
void check_track(const Mesh& umesh, Position lo, Position hi, Position r0,
  Position r1, double margin)
{
  const double length = (r1 - r0).norm();
  const Direction u = (r1 - r0) / length;
  vector<int> bins;
  vector<double> lengths;
  umesh.bins_crossed(r0, r1, u, bins, lengths);

  const auto [t_in, t_out] = box_crossing(lo, hi, r0, r1);
  double t = t_in;
  for (int k = 0; k < bins.size(); k++) {
    const double segment = lengths[k] * length;
    if (segment > 1e-6)
      REQUIRE(umesh.get_bin(r0 + (t + 0.5 * segment) * u) == bins[k]);
    t += segment;
  }
  REQUIRE(t - t_in == Catch::Approx(t_out - t_in).epsilon(0.0).margin(margin));
}

//! Tracks with endpoints uniform in the box [lo, hi] enlarged by a quarter of
//! its width on every side, and tracks crossing the plane x = x0 inside the
//! box at grazing angles (with x components of 1e-3, 1e-5 and 1e-7)
vector<std::pair<Position, Position>> make_tracks(
  Position lo, Position hi, double x0, int n, uint64_t seed)
{
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  const Position width = hi - lo;
  vector<std::pair<Position, Position>> tracks;
  for (int i = 0; i < n; i++) {
    Position r0, r1;
    for (int d = 0; d < 3; d++) {
      r0[d] = lo[d] + (1.5 * uniform(rng) - 0.25) * width[d];
      r1[d] = lo[d] + (1.5 * uniform(rng) - 0.25) * width[d];
    }
    tracks.emplace_back(r0, r1);
  }
  for (double eps : {1e-3, 1e-5, 1e-7}) {
    for (int i = 0; i < n / 10; i++) {
      const double phi = 0.2 * (uniform(rng) - 0.5);
      Position u {eps, std::cos(phi), std::sin(phi)};
      u /= u.norm();
      const Position crossing {x0,
        lo[1] + (0.1 + 0.8 * uniform(rng)) * width[1],
        lo[2] + (0.1 + 0.8 * uniform(rng)) * width[2]};
      const double half_length = 0.6 * width[1];
      tracks.emplace_back(
        crossing - half_length * u, crossing + half_length * u);
    }
  }
  return tracks;
}

//! Tracks from a node of a random active element through another of its
//! nodes, to twice the distance between them: along element edges and side
//! diagonals, and through vertices
vector<std::pair<Position, Position>> node_tracks(
  const libMesh::MeshBase& mesh, int n, uint64_t seed)
{
  std::mt19937_64 rng(seed);
  vector<const libMesh::Elem*> elems;
  for (const auto* elem : mesh.active_element_ptr_range())
    elems.push_back(elem);
  std::uniform_int_distribution<std::size_t> pick_elem(0, elems.size() - 1);
  vector<std::pair<Position, Position>> tracks;
  while (tracks.size() < n) {
    const libMesh::Elem& elem = *elems[pick_elem(rng)];
    std::uniform_int_distribution<unsigned int> pick_node(
      0, elem.n_vertices() - 1);
    const auto& a = elem.point(pick_node(rng));
    const auto& b = elem.point(pick_node(rng));
    const Position r0 {a(0), a(1), a(2)};
    const Position r1 {b(0), b(1), b(2)};
    if ((r1 - r0).norm() > 0.0)
      tracks.emplace_back(r0, r0 + 2.0 * (r1 - r0));
  }
  return tracks;
}

//! Refine the active elements of a mesh whose vertex average has x > x0
void refine_beyond(libMesh::MeshBase& mesh, double x0)
{
  libMesh::MeshRefinement refinement(mesh);
  for (auto* elem : mesh.active_element_ptr_range()) {
    if (elem->vertex_average()(0) > x0)
      elem->set_refinement_flag(libMesh::Elem::REFINE);
  }
  refinement.refine_elements();
}

//! Length of the segment from r0 to r1 inside a tetrahedron
double tet_chord(const libMesh::Elem& tet, Position r0, Position r1)
{
  // Clip the segment against the half-spaces of the four faces
  const Position d = r1 - r0;
  double s_lo = 0.0;
  double s_hi = 1.0;
  for (unsigned int f = 0; f < 4; f++) {
    Position v[3];
    for (unsigned int i = 0; i < 3; i++) {
      const auto& q = tet.point((f + 1 + i) % 4);
      v[i] = {q(0), q(1), q(2)};
    }
    const auto& q = tet.point(f);
    const Position opposite {q(0), q(1), q(2)};
    Position normal = (v[1] - v[0]).cross(v[2] - v[0]);
    if (normal.dot(opposite - v[0]) > 0.0)
      normal *= -1.0;
    // (r0 + s d - v0) . n <= 0
    const double a = (r0 - v[0]).dot(normal);
    const double b = d.dot(normal);
    if (b == 0.0) {
      if (a > 0.0)
        return 0.0;
    } else if (b > 0.0) {
      s_hi = std::min(s_hi, -a / b);
    } else {
      s_lo = std::max(s_lo, -a / b);
    }
  }
  return s_hi > s_lo ? (s_hi - s_lo) * d.norm() : 0.0;
}

} // namespace

TEST_CASE("Track-length traversal gives every tet its exact chord")
{
  // Tetrahedra tiling [-1, 1]^3, whose sides lie in the planes x = 0.5, ...
  LibMeshComm libmesh;
  libMesh::ReplicatedMesh mesh(libmesh.comm, 3);
  libMesh::MeshTools::Generation::build_cube(
    mesh, 4, 4, 4, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, libMesh::TET4);
  LibMesh umesh(mesh);

  // Random tracks, tracks crossing the plane x = 0.5 at small angles (so
  // that they lie within roundoff of sides for long stretches), short tracks,
  // and a track that once had a stretch of its length given to the wrong tet
  const Position lo {-1.0, -1.0, -1.0};
  const Position hi {1.0, 1.0, 1.0};
  auto tracks = make_tracks(lo, hi, 0.5, 500, 8);
  std::mt19937_64 rng(9);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  for (double eps : {1e-8, 1e-6}) {
    for (int i = 0; i < 100; i++) {
      Position u {eps, uniform(rng), uniform(rng)};
      u /= u.norm();
      const Position crossing {0.5, 0.9 * uniform(rng), 0.9 * uniform(rng)};
      tracks.emplace_back(crossing - 1.5 * u, crossing + 1.5 * u);
    }
  }
  for (int i = 0; i < 200; i++) {
    const Position r0 {uniform(rng), uniform(rng), uniform(rng)};
    const Position u {uniform(rng), uniform(rng), uniform(rng)};
    tracks.emplace_back(r0, r0 + 1e-3 * u / u.norm());
  }
  tracks.emplace_back(
    Position {0.49999998499999998, 1.5425527247460553, 0.51059367951670043},
    Position {0.50000001500000002, -1.1317898982833219, -0.84877806310867532});

  for (const auto& [r0, r1] : tracks) {
    const double length = (r1 - r0).norm();
    vector<int> bins;
    vector<double> lengths;
    umesh.bins_crossed(r0, r1, (r1 - r0) / length, bins, lengths);
    std::map<int, double> scored;
    for (int k = 0; k < bins.size(); k++)
      scored[bins[k]] += lengths[k] * length;
    for (int bin = 0; bin < umesh.n_bins(); bin++) {
      const double chord = tet_chord(mesh.elem_ref(bin), r0, r1);
      const double found = scored.count(bin) ? scored[bin] : 0.0;
      REQUIRE(found == Catch::Approx(chord).epsilon(0.0).margin(1e-8));
    }
  }
}

TEST_CASE("Track-length traversal of a refined libMesh mesh with warped sides")
{
  // Hexes tiling [-1, 1]^3, with interior nodes displaced so that the sides
  // between them are warped, refined once for x > 0 and once more for x > 0.5
  // so that coarse elements face refined neighbors across warped sides
  LibMeshComm libmesh;
  libMesh::ReplicatedMesh mesh(libmesh.comm, 3);
  libMesh::MeshTools::Generation::build_cube(
    mesh, 4, 4, 4, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, libMesh::HEX8);
  libMesh::MeshTools::Modification::distort(mesh, 0.1, false);
  refine_beyond(mesh, 0.0);
  refine_beyond(mesh, 0.5);
  AdaptiveLibMesh umesh(mesh);

  // The element volumes add up to the cube's, and every element's vertex
  // average lies in its own bin
  double total_volume = 0.0;
  for (int bin = 0; bin < umesh.n_bins(); bin++) {
    total_volume += umesh.volume(bin);
    REQUIRE(umesh.get_bin(umesh.centroid(bin)) == bin);
  }
  REQUIRE(total_volume == Catch::Approx(8.0).epsilon(1e-12));

  // Random tracks, tracks grazing the refined interface near x = 0, and
  // tracks between nodes
  const Position lo {-1.0, -1.0, -1.0};
  const Position hi {1.0, 1.0, 1.0};
  for (const auto& [r0, r1] : make_tracks(lo, hi, 0.0, 2000, 1))
    check_track(umesh, lo, hi, r0, r1, 1e-7);
  for (const auto& [r0, r1] : node_tracks(mesh, 500, 2))
    check_track(umesh, lo, hi, r0, r1, 1e-7);
}

TEST_CASE("Track-length traversal of refined curved libMesh meshes")
{
  // Second-order elements with interior nodes displaced at random, or
  // smoothly bulged, so that their edges and sides are curved, refined for
  // x > 0: the descendants' vertices include mid-edge (and, for HEX20,
  // interpolated mid-side) nodes, which are off the coarse elements' flat
  // sides and outside of their nodes' bounding box
  for (auto type : {libMesh::HEX20, libMesh::HEX27, libMesh::TET10}) {
    for (std::string shape : {"distorted", "bulged"}) {
      DYNAMIC_SECTION(libMesh::Utility::enum_to_string(type) << " " << shape)
      {
        LibMeshComm libmesh;
        libMesh::ReplicatedMesh mesh(libmesh.comm, 3);
        libMesh::MeshTools::Generation::build_cube(
          mesh, 3, 3, 3, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, type);
        if (shape == "distorted") {
          libMesh::MeshTools::Modification::distort(mesh, 0.15, false);
        } else {
          for (auto* node : mesh.node_ptr_range()) {
            const double x = (*node)(0);
            const double y = (*node)(1);
            const double z = (*node)(2);
            (*node)(0) += 0.3 * (1.0 - x * x) * (1.0 - y * y) * (1.0 - z * z);
          }
        }
        refine_beyond(mesh, 0.0);
        AdaptiveLibMesh umesh(mesh);

        // The element volumes add up to the cube's, with no gaps along the
        // curved edges, every element's vertex average lies in its own bin,
        // and every point of the cube is in one
        double total_volume = 0.0;
        for (int bin = 0; bin < umesh.n_bins(); bin++) {
          total_volume += umesh.volume(bin);
          REQUIRE(umesh.get_bin(umesh.centroid(bin)) == bin);
        }
        REQUIRE(total_volume == Catch::Approx(8.0).epsilon(1e-12));
        std::mt19937_64 rng(12);
        std::uniform_real_distribution<double> uniform(-1.0, 1.0);
        for (int i = 0; i < 50000; i++) {
          const Position r {uniform(rng), uniform(rng), uniform(rng)};
          REQUIRE(umesh.get_bin(r) >= 0);
        }

        const Position lo {-1.0, -1.0, -1.0};
        const Position hi {1.0, 1.0, 1.0};
        for (const auto& [r0, r1] : make_tracks(lo, hi, 1.0 / 3.0, 1000, 3))
          check_track(umesh, lo, hi, r0, r1, 1e-7);
        for (const auto& [r0, r1] : node_tracks(mesh, 500, 7))
          check_track(umesh, lo, hi, r0, r1, 1e-7);
      }
    }
  }
}

TEST_CASE("Track-length traversal of mirrored and folded libMesh meshes")
{
  // Tetrahedra tiling [-1, 1]^3 whose node order is mirrored for all of them,
  // or for those with x < 0, or for none but with an interior node pushed past
  // its neighbors, so that the tetrahedra around it fold over
  for (std::string variant : {"mirrored", "half mirrored", "folded"}) {
    DYNAMIC_SECTION(variant)
    {
      LibMeshComm libmesh;
      libMesh::ReplicatedMesh mesh(libmesh.comm, 3);
      libMesh::MeshTools::Generation::build_cube(
        mesh, 4, 4, 4, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, libMesh::TET4);
      libMesh::MeshTools::Modification::distort(mesh, 0.1, false);
      if (variant == "mirrored") {
        libMesh::MeshTools::Modification::scale(mesh, -1.0, 1.0, 1.0);
      } else if (variant == "half mirrored") {
        for (auto* elem : mesh.active_element_ptr_range()) {
          if (elem->vertex_average()(0) < 0.0) {
            libMesh::Node* node = elem->node_ptr(1);
            elem->set_node(1) = elem->node_ptr(2);
            elem->set_node(2) = node;
          }
        }
        mesh.prepare_for_use();
      } else {
        libMesh::Node* center = nullptr;
        for (auto* node : mesh.node_ptr_range()) {
          if (!center || node->norm() < center->norm())
            center = node;
        }
        (*center)(0) += 0.6;
      }
      int n_inverted = 0;
      for (const auto* elem : mesh.active_element_ptr_range())
        n_inverted += elem->volume() < 0.0;
      REQUIRE(n_inverted > 0);
      LibMesh umesh(mesh);

      // Mirrored elements are reoriented; folded ones have negative volumes
      double total_volume = 0.0;
      int n_negative = 0;
      for (int bin = 0; bin < umesh.n_bins(); bin++) {
        total_volume += umesh.volume(bin);
        n_negative += umesh.volume(bin) < 0.0;
      }
      REQUIRE(total_volume == Catch::Approx(8.0).epsilon(1e-12));
      REQUIRE((n_negative > 0) == (variant == "folded"));

      // Every track is scored in full; segments are in the bin found at
      // their midpoint, except where folded elements overlap their neighbors
      const Position lo {-1.0, -1.0, -1.0};
      const Position hi {1.0, 1.0, 1.0};
      auto tracks = make_tracks(lo, hi, 0.0, 1000, 10);
      for (const auto& track : node_tracks(mesh, 500, 11))
        tracks.push_back(track);
      for (const auto& [r0, r1] : tracks) {
        if (variant == "folded") {
          const auto [t_in, t_out] = box_crossing(lo, hi, r0, r1);
          REQUIRE(scored_length(umesh, r0, r1) ==
                  Catch::Approx(t_out - t_in).epsilon(0.0).margin(1e-7));
        } else {
          check_track(umesh, lo, hi, r0, r1, 1e-7);
        }
      }
    }
  }
}

TEST_CASE("Track-length traversal of an irregular libMesh tet mesh")
{
  // Two patches of a Delaunay tetrahedralization of random points in
  // [-1, 1]^3 (made with scipy): one along the box edge x = 1, y = -1, and
  // one around a node on the side x = -1. A track along the edge, and one
  // along an element edge out through the node, leave the mesh through a
  // vertex at the end of the element grid, where a walk once went back and
  // forth until it reached its bound on element crossings.
  LibMeshComm libmesh;
  libMesh::ReplicatedMesh mesh(libmesh.comm, 3);
  const auto mesh_dir = std::filesystem::path(__FILE__).parent_path() /
                        "../regression_tests/unstructured_mesh";
  libMesh::ExodusII_IO(mesh).read(
    (mesh_dir / "test_mesh_delaunay_patches.e").string());
  mesh.prepare_for_use();
  LibMesh umesh(mesh);

  REQUIRE(scored_length(umesh, {1.0, -1.0, -1.0}, {1.0, -1.0, 5.0}) ==
          Catch::Approx(2.0).epsilon(0.0).margin(1e-9));
  const Position r0 {
    -0.8674586439034242, -0.7078393627538262, -0.90425434128714843};
  const Position r1 {
    -1.2650827121931516, -0.81757977145240668, -0.80931922034290471};
  REQUIRE(scored_length(umesh, r0, r1) ==
          Catch::Approx((r1 - r0).norm() / 3.0).epsilon(0.0).margin(1e-9));

  // Tracks along the elements' edges and through their vertices
  for (const auto& [a, b] : node_tracks(mesh, 2000, 13))
    REQUIRE(scored_length(umesh, a, b) >= 0.0);
}

TEST_CASE("libMesh point lookup past an acute vertex")
{
  // A tetrahedron with a corner of about half a degree at the origin: points
  // past the corner by more than the tolerance (1e-12 of the mesh's size)
  // are outside of it, however far they are from the lines of its edges
  LibMeshComm libmesh;
  libMesh::ReplicatedMesh mesh(libmesh.comm, 3);
  mesh.add_point(libMesh::Point(0.0, 0.0, 0.0), 0);
  mesh.add_point(libMesh::Point(1.0, 0.0, 0.0), 1);
  mesh.add_point(libMesh::Point(1.0, 0.01, 0.0), 2);
  mesh.add_point(libMesh::Point(0.9, 0.005, -0.5), 3);
  auto tet = libMesh::Elem::build(libMesh::TET4);
  for (int i = 0; i < 4; i++)
    tet->set_node(i) = mesh.node_ptr(i);
  tet->set_id(0);
  mesh.add_elem(std::move(tet));
  mesh.prepare_for_use();
  LibMesh umesh(mesh);

  Position bisector {1.0, 0.005, 0.0};
  bisector /= bisector.norm();
  REQUIRE(umesh.get_bin(1e-3 * bisector) == 0);
  REQUIRE(umesh.get_bin(-1e-13 * bisector) == 0);
  REQUIRE(umesh.get_bin(-3e-12 * bisector) == -1);
  REQUIRE(umesh.get_bin(-1e-10 * bisector) == -1);
}

TEST_CASE("libMesh mesh far from the origin")
{
  // Tetrahedra tiling a cube of width 2 centered at (1e4, -2e4, 3e4), with
  // interior nodes displaced
  LibMeshComm libmesh;
  libMesh::ReplicatedMesh mesh(libmesh.comm, 3);
  libMesh::MeshTools::Generation::build_cube(
    mesh, 4, 4, 4, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, libMesh::TET4);
  libMesh::MeshTools::Modification::distort(mesh, 0.3, false);
  const Position center {1e4, -2e4, 3e4};
  libMesh::MeshTools::Modification::translate(
    mesh, center[0], center[1], center[2]);
  LibMesh umesh(mesh);

  // Volumes agree with libMesh's, and every element's vertex average lies in
  // its own bin
  for (int bin = 0; bin < umesh.n_bins(); bin++) {
    REQUIRE(umesh.volume(bin) ==
            Catch::Approx(mesh.elem_ref(bin).volume()).epsilon(1e-9));
    REQUIRE(umesh.get_bin(umesh.centroid(bin)) == bin);
  }

  const Position lo = center - Position {1.0, 1.0, 1.0};
  const Position hi = center + Position {1.0, 1.0, 1.0};
  for (const auto& [r0, r1] : make_tracks(lo, hi, center[0], 1000, 4))
    check_track(umesh, lo, hi, r0, r1, 1e-6);
  for (const auto& [r0, r1] : node_tracks(mesh, 200, 5))
    check_track(umesh, lo, hi, r0, r1, 1e-6);
}

TEST_CASE("Block-restricted libMesh meshes partition the tracks")
{
  // Warped hexes, in subdomain 1 for x < 0 and 0 otherwise, refined for x > 0,
  // with coordinates scaled by 2
  LibMeshComm libmesh;
  libMesh::ReplicatedMesh mesh(libmesh.comm, 3);
  libMesh::MeshTools::Generation::build_cube(
    mesh, 4, 4, 4, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, libMesh::HEX8);
  libMesh::MeshTools::Modification::distort(mesh, 0.1, false);
  for (auto* elem : mesh.active_element_ptr_range())
    elem->subdomain_id() = elem->vertex_average()(0) < 0.0 ? 1 : 0;
  refine_beyond(mesh, 0.0);
  AdaptiveLibMesh low(mesh, 2.0, {1});
  AdaptiveLibMesh high(mesh, 2.0, {0});
  REQUIRE(static_cast<libMesh::dof_id_type>(low.n_bins() + high.n_bins()) ==
          mesh.n_active_elem());

  // The two meshes' scores add up to the part of each track in the cube
  const Position lo {-2.0, -2.0, -2.0};
  const Position hi {2.0, 2.0, 2.0};
  for (const auto& [r0, r1] : make_tracks(lo, hi, 0.0, 1000, 6)) {
    const auto [t_in, t_out] = box_crossing(lo, hi, r0, r1);
    REQUIRE(scored_length(low, r0, r1) + scored_length(high, r0, r1) ==
            Catch::Approx(t_out - t_in).epsilon(0.0).margin(1e-7));
  }
}
#endif
