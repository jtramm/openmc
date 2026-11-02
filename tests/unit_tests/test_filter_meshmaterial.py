from pathlib import Path

import numpy as np
import openmc
import openmc.lib
import pytest
from pytest import approx


def test_filter_mesh_material(run_in_tmpdir):
    # Create four identical materials
    openmc.reset_auto_ids()
    materials = []
    for i in range(4):
        mat = openmc.Material()
        mat.id = 10*(i+1)
        mat.add_nuclide('Fe56', 1.0)
        materials.append(mat)

    # Create a slab model with four cells
    z_values = [-10., -5., 0., 5., 10.]
    planes = [openmc.ZPlane(z) for z in z_values]
    planes[0].boundary_type = 'vacuum'
    planes[-1].boundary_type = 'vacuum'
    regions = [+left & -right for left, right in zip(planes[:-1], planes[1:])]
    cells = [openmc.Cell(fill=m, region=r) for r, m in zip(regions, materials)]
    model = openmc.Model()
    model.geometry = openmc.Geometry(cells)
    model.settings.particles = 1_000
    model.settings.batches = 5
    model.settings.run_mode = 'fixed source'

    # Create a mesh that does not align with all planar surfaces
    mesh = openmc.RegularMesh()
    mesh.lower_left = (-1., -1., -10.)
    mesh.upper_right = (1., 1., 10.)
    mesh.dimension = (1, 1, 5)

    # Determine material volumes in each mesh element and use result to create a
    # MeshMaterialFilter with corresponding bins
    vols = mesh.material_volumes(model)
    mmf = openmc.MeshMaterialFilter.from_volumes(mesh, vols)
    expected_bins = [(0, 10), (1, 10), (1, 20), (2, 20), (2, 30), (3, 40), (3, 30), (4, 40)]
    np.testing.assert_equal(mmf.bins, expected_bins)

    # Create two tallies, one with a mesh filter and one with mesh-material
    mesh_tally = openmc.Tally()
    mesh_tally.filters = [openmc.MeshFilter(mesh)]
    mesh_tally.scores = ['flux']
    mesh_material_tally = openmc.Tally()
    mesh_material_tally.filters = [mmf]
    mesh_material_tally.scores = ['flux']
    model.tallies = [mesh_tally, mesh_material_tally]

    # Run model to get results on the two tallies
    model.run(apply_tally_results=True)

    # The sum of the flux in each mesh-material combination within a single mesh
    # element should be equal to the flux in that mesh element
    mesh_mean = mesh_tally.mean.ravel()
    meshmat_mean = mesh_material_tally.mean.ravel()
    assert mesh_mean[0] == approx(meshmat_mean[0])
    assert mesh_mean[1] == approx(meshmat_mean[1] + meshmat_mean[2])
    assert mesh_mean[2] == approx(meshmat_mean[3] + meshmat_mean[4])
    assert mesh_mean[3] == approx(meshmat_mean[5] + meshmat_mean[6])
    assert mesh_mean[4] == approx(meshmat_mean[7])
    assert mesh_tally.mean.sum() == approx(mesh_material_tally.mean.sum())

    # Make sure get_pandas_dataframe method works
    mesh_material_tally.get_pandas_dataframe()


@pytest.mark.parametrize('library, feature', [
    ('libmesh', 'libmesh'), ('moab', 'dagmc')])
def test_filter_mesh_material_unstructured(library, feature, run_in_tmpdir):
    """Track-length tallies on an unstructured mesh: the flux in the
    mesh-material combinations of each element adds up to its flux"""
    if not openmc.lib.feature_enabled(feature):
        pytest.skip(f"{library} unstructured meshes are not enabled.")

    # Two materials on either side of a plane through the mesh, which tiles
    # the box [-0.5, 0.5]^3
    openmc.reset_auto_ids()
    mats = [openmc.Material(), openmc.Material()]
    for mat in mats:
        mat.add_nuclide('Fe56', 1.0)
        mat.set_density('g/cm3', 1.0)
    box = openmc.model.RectangularParallelepiped(
        -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, boundary_type='vacuum')
    plane = openmc.XPlane(0.13)
    cells = [openmc.Cell(fill=mats[0], region=-box & -plane),
             openmc.Cell(fill=mats[1], region=-box & +plane)]
    model = openmc.Model()
    model.geometry = openmc.Geometry(cells)
    model.settings.particles = 1_000
    model.settings.batches = 5
    model.settings.run_mode = 'fixed source'
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Box((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)))

    tests_dir = Path(__file__).parents[1]
    mesh_dir = tests_dir / 'regression_tests' / 'unstructured_mesh'
    mesh_file = {'libmesh': 'test_mesh_dagmc_tets.e',
                 'moab': 'test_mesh_dagmc_tets.vtk'}[library]
    mesh = openmc.UnstructuredMesh(mesh_dir / mesh_file, library)
    vols = mesh.material_volumes(model, (100, 100, 100))
    mmf = openmc.MeshMaterialFilter.from_volumes(mesh, vols)

    mesh_tally = openmc.Tally()
    mesh_tally.filters = [openmc.MeshFilter(mesh)]
    mesh_tally.scores = ['flux']
    mesh_tally.estimator = 'tracklength'
    mesh_material_tally = openmc.Tally()
    mesh_material_tally.filters = [mmf]
    mesh_material_tally.scores = ['flux']
    mesh_material_tally.estimator = 'tracklength'
    model.tallies = [mesh_tally, mesh_material_tally]
    with openmc.StatePoint(model.run()) as sp:
        mesh_mean = sp.tallies[mesh_tally.id].mean.ravel()
        meshmat_mean = sp.tallies[mesh_material_tally.id].mean.ravel()

    # Some elements are cut by the plane, so have two materials
    elements = np.array([element for element, _ in mmf.bins])
    assert len(np.unique(elements)) < len(elements)
    summed = np.bincount(elements, weights=meshmat_mean,
                         minlength=mesh_mean.size)
    assert mesh_mean.sum() > 0.0
    assert summed == approx(mesh_mean)
