#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <utility>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <pugixml.hpp>

#include "openmc/geometry.h"
#include "openmc/hdf5_interface.h"
#include "openmc/mesh.h"
#include "openmc/random_ray/random_ray.h"
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

//! Slabs in x, each one bin, of which a track gets to know its crossings in
//! order but not where they are
class SlabMesh : public Mesh {
public:
  explicit SlabMesh(vector<std::pair<double, double>> slabs) : slabs_(slabs) {}

  void bins_crossed(Position r0, Position r1, const Direction& u,
    vector<int>& bins, vector<double>& lengths) const override
  {
    for (int i = 0; i < slabs_.size(); i++) {
      const double overlap =
        std::min(r1.x, slabs_[i].second) - std::max(r0.x, slabs_[i].first);
      if (overlap > 0.0) {
        bins.push_back(i);
        lengths.push_back(overlap / (r1.x - r0.x));
      }
    }
  }
  int get_bin(Position r) const override
  {
    for (int i = 0; i < slabs_.size(); i++) {
      if (r.x >= slabs_[i].first && r.x <= slabs_[i].second)
        return i;
    }
    return -1;
  }

  // What is left of the interface is of no use here
  Position sample_element(int32_t, uint64_t*) const override { return {}; }
  void surface_bins_crossed(
    Position, Position, const Direction&, vector<int>&) const override
  {}
  int n_bins() const override { return slabs_.size(); }
  int n_surface_bins() const override { return 0; }
  void to_hdf5_inner(hid_t) const override {}
  std::pair<vector<double>, vector<double>> plot(
    Position, Position) const override
  {
    return {};
  }
  std::string bin_label(int) const override { return ""; }
  double volume(int) const override { return 1.0; }
  std::string get_mesh_type() const override { return "slabs"; }
  Position lower_left() const override
  {
    return {slabs_.front().first, -INFTY, -INFTY};
  }
  Position upper_right() const override
  {
    return {slabs_.back().second, INFTY, INFTY};
  }

private:
  vector<std::pair<double, double>> slabs_;
};

} // namespace

TEST_CASE("Random rays place what is outside of a mesh along the track")
{
  model::n_coord_levels = 1;
  RandomRay ray;
  ray.u() = {0.8, 0.6, 0.0};

  // The pieces of a track along x (in bins, with lengths in x), each with what
  // the mesh does not return between them
  auto check = [&](vector<std::pair<double, double>> slabs, double x0,
                 double x1) {
    SlabMesh mesh(slabs);
    const double length = (x1 - x0) / ray.u().x;
    Position r0 {x0, 0.0, 0.0};
    vector<int> bins;
    vector<double> fractions;
    mesh.bins_crossed(r0, r0 + length * ray.u(), ray.u(), bins, fractions);
    ray.add_mesh_gaps(mesh, r0, length, bins, fractions);

    vector<int> expected_bins;
    vector<double> expected;
    double x = x0;
    for (int i = 0; i < slabs.size() && slabs[i].first < x1; i++) {
      if (slabs[i].first > x) {
        expected_bins.push_back(C_NONE);
        expected.push_back(slabs[i].first - x);
      }
      expected_bins.push_back(i);
      expected.push_back(slabs[i].second - slabs[i].first);
      x = slabs[i].second;
    }
    if (x1 > x) {
      expected_bins.push_back(C_NONE);
      expected.push_back(x1 - x);
    }

    REQUIRE(bins == expected_bins);
    for (int i = 0; i < bins.size(); i++) {
      REQUIRE(
        fractions[i] * (x1 - x0) == Catch::Approx(expected[i]).margin(1e-10));
    }
  };

  SECTION("no gaps, and all gap")
  {
    check({{0.0, 1.0}, {1.0, 2.0}}, 0.0, 2.0);
    check({{5.0, 6.0}}, 0.0, 2.0);
  }
  SECTION("a gap at either end, or at both")
  {
    check({{1.0, 2.0}}, 0.0, 2.0);
    check({{0.0, 1.0}}, 0.0, 2.0);
    check({{1.0, 2.0}}, 0.0, 3.0);
    check({{50.0, 51.0}}, 0.0, 100.0);
  }
  SECTION("gaps between pieces")
  {
    check({{0.0, 1.0}, {1.5, 2.0}}, 0.0, 2.0);
    check({{0.25, 1.0}, {1.5, 1.75}, {3.0, 3.5}}, 0.0, 4.0);
  }
  SECTION("lengths that coincide")
  {
    // What comes before as wide as a piece, and pieces and gaps all of a width
    check({{1.0, 2.0}, {3.0, 4.0}}, 0.0, 5.0);
    check({{0.0, 1.0}, {2.0, 3.0}, {4.0, 5.0}}, 0.0, 6.0);
    check({{1.0, 2.0}, {2.5, 3.5}}, 0.0, 4.5);
  }
  SECTION("what is just long enough to count")
  {
    check({{5.0, 5.0 + 6e-8}}, 0.0, 100.0);
    check({{0.0, 1.0}, {1.0 + 3e-8, 2.0}}, 0.0, 2.0);
  }
  SECTION("what is too short to count")
  {
    auto bins_of = [&](vector<std::pair<double, double>> slabs, double x1) {
      SlabMesh mesh(slabs);
      Position r0 {0.0, 0.0, 0.0};
      const double length = x1 / ray.u().x;
      vector<int> bins;
      vector<double> fractions;
      mesh.bins_crossed(r0, r0 + length * ray.u(), ray.u(), bins, fractions);
      ray.add_mesh_gaps(mesh, r0, length, bins, fractions);
      return bins;
    };

    // Too little in the mesh for it to be placed, and too small a gap to matter
    REQUIRE(bins_of({{5.0, 5.0 + 3e-8}}, 100.0) == vector<int> {C_NONE});
    REQUIRE(
      bins_of({{0.0, 1.0}, {1.0 + 1e-8, 2.0}}, 2.0) == vector<int> {0, 1});
  }
  SECTION("plates at multiples of a pitch")
  {
    std::mt19937_64 rng(1);
    for (int n = 0; n < 2000; n++) {
      vector<std::pair<double, double>> slabs;
      double x = 0.25 * (rng() % 4);
      const int n_slabs = 1 + rng() % 6;
      for (int i = 0; i < n_slabs; i++) {
        const double width = 0.25 * (1 + rng() % 4);
        slabs.push_back({x, x + width});
        x += width + 0.25 * (1 + rng() % 4);
      }
      check(
        slabs, -0.25 * (rng() % 4), slabs.back().second + 0.25 * (rng() % 4));
    }
  }
}

TEST_CASE("Tracks from anywhere find meshes that do not span all angles")
{
  auto check = [](const Mesh& mesh) {
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> position(-1.5, 1.5);
    std::uniform_real_distribution<double> direction(-1.0, 1.0);
    const int n_steps = 2000;
    const double length = 3.0;
    const double step = length / n_steps;
    for (int n = 0; n < 500; n++) {
      Position r {position(rng), position(rng), position(rng)};
      Direction u {direction(rng), direction(rng), direction(rng)};
      u /= u.norm();

      // The length in each bin, from the track and from points along it
      vector<int> bins;
      vector<double> fractions;
      mesh.bins_crossed(r, r + length * u, u, bins, fractions);
      std::map<int, double> traced;
      for (int i = 0; i < bins.size(); i++)
        traced[bins[i]] += fractions[i] * length;
      std::map<int, double> sampled;
      for (int i = 0; i < n_steps; i++) {
        const int bin = mesh.get_bin(r + (i + 0.5) * step * u);
        if (bin >= 0)
          sampled[bin] += step;
      }

      for (const auto& [bin, sampled_length] : sampled)
        REQUIRE(
          traced[bin] == Catch::Approx(sampled_length).margin(4.0 * step));
      for (const auto& [bin, traced_length] : traced)
        REQUIRE(
          sampled[bin] == Catch::Approx(traced_length).margin(4.0 * step));
    }
  };

  auto cylindrical = [&](const char* r, const char* phi) {
    pugi::xml_document doc;
    auto node = doc.append_child("mesh");
    node.append_attribute("id") = 1;
    node.append_child("r_grid").text() = r;
    node.append_child("phi_grid").text() = phi;
    node.append_child("z_grid").text() = "-1 0 1";
    node.append_child("origin").text() = "0 0 0";
    check(CylindricalMesh(node));
  };
  auto spherical = [&](const char* r, const char* theta, const char* phi) {
    pugi::xml_document doc;
    auto node = doc.append_child("mesh");
    node.append_attribute("id") = 1;
    node.append_child("r_grid").text() = r;
    node.append_child("theta_grid").text() = theta;
    node.append_child("phi_grid").text() = phi;
    node.append_child("origin").text() = "0 0 0";
    check(SphericalMesh(node));
  };

  cylindrical("0 0.5 1", "0 1.5 3");
  cylindrical("0 0.5 1", "1 1.8 2.5");
  cylindrical("0.3 0.6 1", "4 4.7 5.5");
  spherical("0 0.5 1", "0 0.4 0.8", "0 6.283185307179586");
  spherical("0.3 0.6 1", "1 1.3 2", "0 6.283185307179586");
  spherical("0 0.5 1", "0 0.7 1.5", "0 0.8 1.5");
  spherical("0.3 0.6 1", "1 1.3 2", "4 4.7 5.5");
}
