#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
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

namespace {

//! A regular mesh whose traversal skips some of the crossings
class SkippingMesh : public RegularMesh {
public:
  using RegularMesh::RegularMesh;

  void bins_crossed_with_starts(Position r0, Position r1, const Direction& u,
    vector<int>& bins, vector<double>& lengths,
    vector<double>& starts) const override
  {
    vector<int> all_bins;
    vector<double> all_lengths;
    vector<double> all_starts;
    RegularMesh::bins_crossed_with_starts(
      r0, r1, u, all_bins, all_lengths, all_starts);
    bins.clear();
    lengths.clear();
    starts.clear();
    for (int i = 0; i < all_bins.size(); i++) {
      if (std::find(skipped.begin(), skipped.end(), all_bins[i]) ==
          skipped.end()) {
        bins.push_back(all_bins[i]);
        lengths.push_back(all_lengths[i]);
        starts.push_back(all_starts[i]);
      }
    }
  }

  vector<int> skipped;
};

} // namespace

TEST_CASE("Parts of a track that a traversal skips are recovered in order")
{
  // Four bins of width 1 along x
  pugi::xml_document doc;
  auto node = doc.append_child("mesh");
  node.append_attribute("id") = 1;
  node.append_child("dimension").text() = "4 1 1";
  node.append_child("lower_left").text() = "0 0 0";
  node.append_child("upper_right").text() = "4 1 1";
  SkippingMesh mesh(node);
  const Direction u {1.0, 0.0, 0.0};
  vector<int> bins;
  vector<double> lengths;

  SECTION("A part in several bins")
  {
    mesh.skipped = {1, 2};
    mesh.bins_crossed_recovering(
      {0.5, 0.5, 0.5}, {3.5, 0.5, 0.5}, u, bins, lengths);
    REQUIRE(bins == vector<int> {0, 1, 2, 3});
    REQUIRE(lengths[0] == Catch::Approx(1.0 / 6.0));
    REQUIRE(lengths[1] == Catch::Approx(1.0 / 3.0));
    REQUIRE(lengths[2] == Catch::Approx(1.0 / 3.0));
    REQUIRE(lengths[3] == Catch::Approx(1.0 / 6.0));
  }

  SECTION("A part at the start, and one that is partly outside of the mesh")
  {
    mesh.skipped = {0, 3};
    mesh.bins_crossed_recovering(
      {0.5, 0.5, 0.5}, {5.5, 0.5, 0.5}, u, bins, lengths);
    // A skipped part can come back in several pieces, in the order of the track
    REQUIRE(std::is_sorted(bins.begin(), bins.end()));
    std::map<int, double> in_bin;
    for (int k = 0; k < bins.size(); k++)
      in_bin[bins[k]] += lengths[k];
    REQUIRE(in_bin.size() == 4);
    REQUIRE(in_bin[0] == Catch::Approx(0.1));
    REQUIRE(in_bin[1] == Catch::Approx(0.2));
    REQUIRE(in_bin[2] == Catch::Approx(0.2));
    // (the end of the mesh is found by halving, to 1/256 of the skipped part)
    REQUIRE(in_bin[3] == Catch::Approx(0.2).margin(0.5 / 256.0));
  }

  SECTION("Nothing skipped, and the rest of the track outside of the mesh")
  {
    mesh.bins_crossed_recovering(
      {3.5, 0.5, 0.5}, {5.5, 0.5, 0.5}, u, bins, lengths);
    REQUIRE(bins == vector<int> {3});
    REQUIRE(lengths[0] == Catch::Approx(0.25));
  }
}

TEST_CASE("Tracks tangent to a curved mesh surface keep their length")
{
  // Whether roundoff makes the traversal skip part of these tracks depends on
  // the platform; here it skips all of the first and half of the second
  const auto recovered = [](const Mesh& mesh, Position r0, Position r1) {
    vector<int> bins;
    vector<double> lengths;
    mesh.bins_crossed_recovering(
      r0, r1, (r1 - r0) / (r1 - r0).norm(), bins, lengths);
    double total = 0.0;
    for (double f : lengths)
      total += f;
    return total;
  };
  const std::string phi = "0 1.5707963267948966 3.141592653589793 "
                          "4.71238898038469 6.283185307179586";
  auto cylinder = [&](pugi::xml_document& doc, const std::string& r,
                    const std::string& z) {
    auto node = doc.append_child("mesh");
    node.append_attribute("id") = 1;
    node.append_child("r_grid").text() = r.c_str();
    node.append_child("phi_grid").text() = phi.c_str();
    node.append_child("z_grid").text() = z.c_str();
    node.append_child("origin").text() = "0 0 0";
    return node;
  };

  SECTION("A chord starting on the outer surface")
  {
    pugi::xml_document doc;
    CylindricalMesh mesh(cylinder(doc, "0 20 40", "-200 200"));
    REQUIRE(recovered(mesh,
              {37.574539698603807, -13.716922630023351, 19.622947036413315},
              {37.574534904204995, -13.716935763233737, 19.623051940157914}) ==
            Catch::Approx(1.0).margin(1e-6));
  }

  SECTION("A chord tangent to the hole")
  {
    pugi::xml_document doc;
    CylindricalMesh mesh(cylinder(doc, "0.3 0.6 1", "-5 5"));
    REQUIRE(recovered(mesh,
              {0.56548702091727798, -0.82475718754692329, 1.8322049029477787},
              {-0.93576028605670203, 0.35263674833359082,
                -0.95991779578793723}) == Catch::Approx(1.0).margin(1e-6));
  }
}

#ifdef OPENMC_DAGMC_ENABLED
namespace {

//! Length of the part of the segment from r0 to r1 in a tet
double tet_chord(
  const std::array<Position, 4>& corners, Position r0, Position r1)
{
  const Position d = r1 - r0;
  double s_lo = 0.0;
  double s_hi = 1.0;
  for (int f = 0; f < 4; f++) {
    const Position& a = corners[(f + 1) % 4];
    Position normal =
      (corners[(f + 2) % 4] - a).cross(corners[(f + 3) % 4] - a);
    if (normal.dot(corners[f] - a) > 0.0)
      normal *= -1.0;
    const double outside = (r0 - a).dot(normal);
    const double along = d.dot(normal);
    if (along == 0.0) {
      if (outside > 0.0)
        return 0.0;
    } else if (along > 0.0) {
      s_hi = std::min(s_hi, -outside / along);
    } else {
      s_lo = std::max(s_lo, -outside / along);
    }
  }
  return s_hi > s_lo ? (s_hi - s_lo) * d.norm() : 0.0;
}

//! The corners of the tet in each bin of a mesh, each moved by a transformation
template<typename F>
vector<std::array<Position, 4>> tet_corners(const MOABMesh& mesh, F transform)
{
  vector<std::array<Position, 4>> tets(mesh.n_bins());
  for (int bin = 0; bin < mesh.n_bins(); bin++) {
    const auto nodes = mesh.connectivity(bin);
    for (int i = 0; i < 4; i++)
      tets[bin][i] = transform(mesh.vertex(nodes[i]));
  }
  return tets;
}

//! Check the total, and if requested the length in each tet, of a track
void check_moab_track(const MOABMesh& mesh,
  const vector<std::array<Position, 4>>& tets, Position r0, Position r1,
  bool by_tet, double tolerance)
{
  const double length = (r1 - r0).norm();
  vector<int> bins;
  vector<double> lengths;
  mesh.bins_crossed(r0, r1, (r1 - r0) / length, bins, lengths);
  std::map<int, double> scored;
  double total = 0.0;
  for (int k = 0; k < bins.size(); k++) {
    scored[bins[k]] += lengths[k] * length;
    total += lengths[k] * length;
  }
  double total_chords = 0.0;
  for (int bin = 0; bin < mesh.n_bins(); bin++) {
    const double chord = tet_chord(tets[bin], r0, r1);
    total_chords += chord;
    if (by_tet) {
      const double found = scored.count(bin) ? scored[bin] : 0.0;
      REQUIRE(std::abs(found - chord) <= tolerance);
    }
  }
  if (by_tet)
    REQUIRE(std::abs(total - total_chords) <= 24.0 * tolerance);
}

} // namespace

TEST_CASE("MOAB track-length traversal gives every tet its exact chord")
{
  const auto mesh_file = std::filesystem::path(__FILE__).parent_path() /
                         "../regression_tests/unstructured_mesh" /
                         "test_mesh_dagmc_tets.vtk";
  MOABMesh mesh(mesh_file.string());
  mesh.prepare_for_point_location();
  const auto tets = tet_corners(mesh, [](Position p) { return p; });

  std::mt19937_64 rng(31);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  auto direction = [&]() {
    Position v {uniform(rng), uniform(rng), uniform(rng)};
    return v / v.norm();
  };

  SECTION("From inside, nearby and far away (where distances are too coarse "
          "for MOAB to tell the exit from the end of the track)")
  {
    for (double distance : {0.1, 1.0, 1e2, 1e4, 1e5, 1e6}) {
      for (int i = 0; i < 500; i++) {
        const Position center {
          0.5 * uniform(rng), 0.5 * uniform(rng), 0.5 * uniform(rng)};
        const Position v = direction();
        check_moab_track(mesh, tets, center - distance * v, center + 2.0 * v,
          true, 1e-9 * std::max(1.0, distance * 1e-2));
      }
    }
  }

  SECTION("Entering through an edge of the mesh")
  {
    check_moab_track(mesh, tets,
      {-0.9637864349825348, 0.96132752774470487, 0.83244136368245036},
      {0.051831599143884355, -0.049372691906054422, 0.16755863631754964}, true,
      1e-9);
  }

  SECTION("Touching the mesh only at a corner, in the middle of the track")
  {
    check_moab_track(mesh, tets,
      {0.050337025567154969, 1.3485460817031152, -0.7440865636237004},
      {0.94966297443284509, -0.34854608170311507, -0.25591343637629965}, true,
      1e-9);
  }
}

TEST_CASE("MOAB tracks in a mesh whose hull is not its bounding box")
{
  // The mesh turned about an axis, and also mirrored, so that tracks can enter
  // it through edges and corners that are not on faces of its bounding box
  const auto mesh_file = std::filesystem::path(__FILE__).parent_path() /
                         "../regression_tests/unstructured_mesh" /
                         "test_mesh_dagmc_tets.vtk";
  MOABMesh original(mesh_file.string());
  const auto tmp_file =
    std::filesystem::temp_directory_path() / "openmc_test_mesh_turned.vtk";

  for (double mirror : {1.0, -1.0}) {
    auto transform = [mirror](Position p) {
      constexpr double c = 0.6;
      constexpr double s = 0.8;
      return Position {
        mirror * (c * p.x - s * p.y), s * p.x + c * p.y, p.z + 0.1 * p.x};
    };
    {
      std::ofstream out(tmp_file);
      out << "# vtk DataFile Version 2.0\nturned tets\nASCII\n"
             "DATASET UNSTRUCTURED_GRID\n";
      out.precision(17);
      out << "POINTS " << original.n_vertices() << " double\n";
      for (int i = 0; i < original.n_vertices(); i++) {
        const Position p = transform(original.vertex(i));
        out << p.x << " " << p.y << " " << p.z << "\n";
      }
      out << "CELLS " << original.n_bins() << " " << 5 * original.n_bins()
          << "\n";
      for (int bin = 0; bin < original.n_bins(); bin++) {
        const auto nodes = original.connectivity(bin);
        out << "4 " << nodes[0] << " " << nodes[1] << " " << nodes[2] << " "
            << nodes[3] << "\n";
      }
      out << "CELL_TYPES " << original.n_bins() << "\n";
      for (int bin = 0; bin < original.n_bins(); bin++)
        out << "10\n";
    }
    MOABMesh mesh(tmp_file.string());
    mesh.prepare_for_point_location();
    REQUIRE(mesh.n_bins() == original.n_bins());
    const auto tets = tet_corners(mesh, [](Position p) { return p; });

    // Tracks through the nodes (most of which are on the hull), to a point
    // inside and on through the mesh
    std::mt19937_64 rng(37);
    std::uniform_real_distribution<double> uniform(-0.3, 0.3);
    for (int i = 0; i < mesh.n_vertices(); i++) {
      for (int k = 0; k < 20; k++) {
        const Position node = mesh.vertex(i);
        const Position inner =
          transform({uniform(rng), uniform(rng), uniform(rng)});
        const Position r0 = node - 1.3 * (inner - node);
        const Position r1 = node + 3.0 * (inner - node);
        check_moab_track(mesh, tets, r0, r1, true, 1e-9);
      }
    }
  }
  std::filesystem::remove(tmp_file);
}
#endif
