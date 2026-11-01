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
#include <random>

#include "libmesh/elem.h"
#include "libmesh/libmesh.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/mesh_modification.h"
#include "libmesh/mesh_refinement.h"
#include "libmesh/replicated_mesh.h"

#include "openmc/position.h"
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
//! report written to it can be lost.
const libMesh::Parallel::Communicator& libmesh_comm()
{
  static const char* argv[] = {"test_mesh", "--sync-with-stdio"};
  static std::unique_ptr<libMesh::LibMeshInit> init;
  static libMesh::Parallel::Communicator serial_comm;
  if (!init && !libMesh::initialized())
    init = std::make_unique<libMesh::LibMeshInit>(2, argv);
  const libMesh::Parallel::Communicator& comm =
    init ? init->comm() : serial_comm;
  settings::libmesh_comm = &comm;
  return comm;
}

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
  REQUIRE(t - t_in == Catch::Approx(t_out - t_in).margin(margin));
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

} // namespace

TEST_CASE("Track-length traversal of a refined libMesh mesh with warped sides")
{
  // Hexes tiling [-1, 1]^3, with interior nodes displaced so that the sides
  // between them are warped, refined once for x > 0 and once more for x > 0.5
  // so that coarse elements face refined neighbors across warped sides
  libMesh::ReplicatedMesh mesh(libmesh_comm(), 3);
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

TEST_CASE("Track-length traversal of a refined libMesh mesh of HEX20s")
{
  // Second-order hexes with straight sides, refined for x > 0
  libMesh::ReplicatedMesh mesh(libmesh_comm(), 3);
  libMesh::MeshTools::Generation::build_cube(
    mesh, 3, 3, 3, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, libMesh::HEX20);
  refine_beyond(mesh, 0.0);
  AdaptiveLibMesh umesh(mesh);

  double total_volume = 0.0;
  for (int bin = 0; bin < umesh.n_bins(); bin++)
    total_volume += umesh.volume(bin);
  REQUIRE(total_volume == Catch::Approx(8.0).epsilon(1e-12));

  const Position lo {-1.0, -1.0, -1.0};
  const Position hi {1.0, 1.0, 1.0};
  for (const auto& [r0, r1] : make_tracks(lo, hi, 1.0 / 3.0, 1000, 3))
    check_track(umesh, lo, hi, r0, r1, 1e-7);
  for (const auto& [r0, r1] : node_tracks(mesh, 500, 7))
    check_track(umesh, lo, hi, r0, r1, 1e-7);
}

TEST_CASE("libMesh mesh far from the origin")
{
  // Tetrahedra tiling a cube of width 2 centered at (1e4, -2e4, 3e4), with
  // interior nodes displaced
  libMesh::ReplicatedMesh mesh(libmesh_comm(), 3);
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
  libMesh::ReplicatedMesh mesh(libmesh_comm(), 3);
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
            Catch::Approx(t_out - t_in).margin(1e-7));
  }
}
#endif
