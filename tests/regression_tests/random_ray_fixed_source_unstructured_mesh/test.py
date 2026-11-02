import os
import subprocess
from itertools import product
from pathlib import Path

import numpy as np
import openmc
import openmc.lib
from openmc.utility_funcs import change_directory
import pytest

from tests.regression_tests import config
from tests.testing_harness import TolerantWeightWindowPyAPITestHarness

# The same 24-tet mesh of the unit cube, as the canonical MOAB (.vtk) and
# libMesh (.e) assets
MESH_FILES = {
    'moab': 'test_mesh_dagmc_tets.vtk',
    'libmesh': 'test_mesh_dagmc_tets.e',
    'xdg-moab': 'test_mesh_dagmc_tets.vtk',
    'xdg-libmesh': 'test_mesh_dagmc_tets.e',
}
FEATURES = {'moab': 'dagmc', 'libmesh': 'libmesh', 'xdg-moab': 'xdg',
            'xdg-libmesh': 'xdg'}


class MGXSTestHarness(TolerantWeightWindowPyAPITestHarness):
    def _cleanup(self):
        super()._cleanup()
        f = 'mgxs.h5'
        if os.path.exists(f):
            os.remove(f)


def make_model(library, mesh_path, case, x_max=0.5, source_shape='flat',
               length_multiplier=None, tally_mesh=None):
    """Random ray model of the box [-0.5, x_max] x [-0.5, 0.5]^2, subdivided
    by the mesh in mesh_path (read by a mesh library, directly or through
    XDG), or by a regular mesh of [-0.5, 0.5]^3 or (in 2D) [-0.5, 0.5]^2, with
    a mesh tally on that mesh or on another one"""

    # One-group material with half scattering
    groups = openmc.mgxs.EnergyGroups([1e-5, 20.0e6])
    xsdata = openmc.XSdata('mat', groups)
    xsdata.order = 0
    xsdata.set_total([1.0])
    xsdata.set_absorption([0.5])
    xsdata.set_scatter_matrix(np.array([[[0.5]]]))
    mgxs_lib = openmc.MGXSLibrary(groups)
    mgxs_lib.add_xsdata(xsdata)
    mgxs_lib.export_to_hdf5('mgxs.h5')

    mat = openmc.Material(name='mat')
    mat.set_density('macro', 1.0)
    mat.add_macroscopic(openmc.Macroscopic('mat'))
    materials = openmc.Materials([mat])
    materials.cross_sections = 'mgxs.h5'

    # Reflective on the low faces and vacuum on the high faces so the flux
    # has a gradient
    box = openmc.model.RectangularParallelepiped(
        -0.5, x_max, -0.5, 0.5, -0.5, 0.5)
    for s in (box.xmin, box.ymin, box.zmin):
        s.boundary_type = 'reflective'
    for s in (box.xmax, box.ymax, box.zmax):
        s.boundary_type = 'vacuum'
    cell = openmc.Cell(name='cube', fill=mat, region=-box)
    root = openmc.Universe(cells=[cell])
    geometry = openmc.Geometry(root)

    # The tet mesh subdivides the single cell into 24 source regions
    if library == 'regular':
        umesh = openmc.RegularMesh()
        umesh.lower_left = (-0.5, -0.5, -0.5)
        umesh.upper_right = (0.5, 0.5, 0.5)
        umesh.dimension = (3, 3, 3)
    elif library == 'regular-2d':
        umesh = openmc.RegularMesh()
        umesh.lower_left = (-0.5, -0.5)
        umesh.upper_right = (0.5, 0.5)
        umesh.dimension = (3, 3)
    elif library.startswith('xdg-'):
        umesh = openmc.UnstructuredMesh(mesh_path, library=library[4:])
        umesh.interface = 'xdg'
    else:
        umesh = openmc.UnstructuredMesh(mesh_path, library=library)
    if length_multiplier is not None:
        umesh.length_multiplier = length_multiplier

    settings = openmc.Settings()
    settings.energy_mode = 'multi-group'
    settings.run_mode = 'fixed source'
    settings.batches = 10
    settings.inactive = 5
    settings.particles = 100
    settings.random_ray['distance_active'] = 10.0
    settings.random_ray['distance_inactive'] = 3.0
    settings.random_ray['ray_source'] = openmc.IndependentSource(
        space=openmc.stats.Box((-0.5, -0.5, -0.5), (x_max, 0.5, 0.5)))
    settings.random_ray['volume_estimator'] = 'hybrid'
    if source_shape != 'flat':
        settings.random_ray['source_shape'] = source_shape
    settings.random_ray['source_region_meshes'] = [(umesh, [root])]
    settings.source = openmc.IndependentSource(
        energy=openmc.stats.Discrete([1.0e6], [1.0]),
        constraints={'domains': [cell]})

    cell_tally = openmc.Tally(name='cell flux')
    cell_tally.filters = [openmc.CellFilter(cell)]
    cell_tally.scores = ['flux']
    cell_tally.estimator = 'tracklength'
    tallies = openmc.Tallies([cell_tally])

    # The mesh is also the tally mesh and the weight window mesh, the tally
    # mesh only, or used for subdivision alone
    if case in ('tally_and_ww', 'tally'):
        mesh_tally = openmc.Tally(name='tet flux')
        mesh_tally.filters = [openmc.MeshFilter(tally_mesh or umesh)]
        mesh_tally.scores = ['flux']
        mesh_tally.estimator = 'tracklength'
        tallies.append(mesh_tally)
    if case == 'tally_and_ww':
        settings.weight_window_generators = openmc.WeightWindowGenerator(
            mesh=tally_mesh or umesh, method='magic', max_realizations=5)

    return openmc.Model(geometry, materials, settings, tallies)


@pytest.mark.parametrize("library,case", product(
    ['moab', 'xdg-libmesh'], ['tally_and_ww', 'subdivision_only']))
def test_random_ray_fixed_source_unstructured_mesh(library, case):
    if not openmc.lib.feature_enabled(FEATURES[library]):
        pytest.skip(f"{library} unstructured meshes are not enabled.")

    with change_directory(f'{library}_{case}'):
        openmc.reset_auto_ids()
        # The mesh path is relative to the case directory
        mesh_path = f'../../unstructured_mesh/{MESH_FILES[library]}'
        model = make_model(library, mesh_path, case)
        harness = MGXSTestHarness('statepoint.10.h5', model)
        harness.main()


@pytest.mark.parametrize("library,mesh_file,x_max", [
    # The cell is twice as long as the mesh
    ('moab', MESH_FILES['moab'], 1.5),
    ('xdg-libmesh', MESH_FILES['libmesh'], 1.5),
    ('regular', None, 1.5),
    ('regular-2d', None, 1.5),
    # The mesh covers the cell: the 24-tet mesh and the regular mesh tile it,
    # and the hex mesh extends beyond it
    ('regular', None, 0.5),
    ('regular-2d', None, 0.5),
    ('moab', MESH_FILES['moab'], 0.5),
    ('xdg-moab', MESH_FILES['moab'], 0.5),
    ('xdg-libmesh', MESH_FILES['libmesh'], 0.5),
    ('xdg-libmesh', 'test_mesh_hexes.e', 0.5),
])
def test_random_ray_unstructured_mesh_coverage(
        library, mesh_file, x_max, tmp_path):
    """A run with a mesh covering only half of the cell it subdivides stops
    with an error, and one with a mesh covering all of it does not, and its
    flux on the mesh adds up to the flux in the cell"""
    if (not library.startswith('regular') and
            not openmc.lib.feature_enabled(FEATURES[library])):
        pytest.skip(f"{library} unstructured meshes are not enabled.")

    openmc.reset_auto_ids()
    mesh_path = (Path(__file__).parents[1] / 'unstructured_mesh' / mesh_file
                 if mesh_file else None)
    covered = x_max <= 0.5
    with change_directory(tmp_path):
        model = make_model(library, mesh_path,
                           'tally' if covered else 'subdivision_only',
                           x_max=x_max)
        model.export_to_model_xml()

    if config['build_inputs']:
        return
    returncode, output = run_openmc(tmp_path)
    if covered:
        assert returncode == 0, output
        check_tally_sums(tmp_path)
    else:
        assert returncode != 0
        assert 'does not cover' in output


@pytest.mark.parametrize("library,option", [
    ('moab', 'linear'),
    ('moab', 'scaled'),
    ('moab', 'tally_mesh'),
    ('xdg-libmesh', 'linear'),
    ('xdg-libmesh', 'scaled'),
    ('xdg-libmesh', 'tally_mesh'),
    # Rays are not followed through a tally mesh, so the native interface will
    # do
    ('libmesh', 'tally_mesh'),
])
def test_random_ray_unstructured_mesh_options(library, option, tmp_path):
    """Runs with linear sources, with the mesh scaled by a length multiplier
    (the 24-tet mesh then tiles [-1, 1]^3), and with a regular subdivision
    mesh and the tally and weight windows on the 24-tet mesh, whose flux on
    the mesh adds up to the flux in the cell"""
    if not openmc.lib.feature_enabled(FEATURES[library]):
        pytest.skip(f"{library} unstructured meshes are not enabled.")

    openmc.reset_auto_ids()
    mesh_path = (Path(__file__).parents[1] / 'unstructured_mesh' /
                 MESH_FILES[library])
    with change_directory(tmp_path):
        if option == 'tally_mesh':
            tally_mesh = openmc.UnstructuredMesh(
                mesh_path, library=library.removeprefix('xdg-'))
            if library.startswith('xdg-'):
                tally_mesh.interface = 'xdg'
            model = make_model('regular', None, 'tally_and_ww',
                               tally_mesh=tally_mesh)
        else:
            model = make_model(
                library, mesh_path, 'tally',
                source_shape='linear' if option == 'linear' else 'flat',
                length_multiplier=2.0 if option == 'scaled' else None)
        model.export_to_model_xml()

    if config['build_inputs']:
        return
    returncode, output = run_openmc(tmp_path)
    assert returncode == 0, output
    check_tally_sums(tmp_path)
    if option == 'tally_mesh':
        # Valid bounds, or -1 for a tet with no source region at its bin's point
        bounds = openmc.WeightWindowsList.from_hdf5(
            tmp_path / 'weight_windows.h5')[0].lower_ww_bounds
        assert np.any(bounds > 0.0)
        assert np.all((bounds > 0.0) | (bounds == -1.0))


def test_random_ray_native_libmesh_subdivision(tmp_path):
    """Rays cannot be followed through a libMesh mesh with the native
    interface, which is reported up front"""
    if not openmc.lib.feature_enabled('libmesh'):
        pytest.skip("libmesh unstructured meshes are not enabled.")

    openmc.reset_auto_ids()
    mesh_path = (Path(__file__).parents[1] / 'unstructured_mesh' /
                 MESH_FILES['libmesh'])
    with change_directory(tmp_path):
        make_model('libmesh', mesh_path, 'subdivision_only').export_to_model_xml()

    if config['build_inputs']:
        return
    returncode, output = run_openmc(tmp_path)
    assert returncode != 0
    assert 'Use the XDG interface' in output


def test_random_ray_mirrored_moab_mesh(tmp_path):
    """A mirrored copy of a MOAB mesh keeps its node order, which turns its
    tets inside out: their volumes must still be positive for weight windows
    to be generated on them"""
    if not openmc.lib.feature_enabled('dagmc'):
        pytest.skip("moab unstructured meshes are not enabled.")

    # x -> -x in the VTK file of the 24-tet mesh of the unit cube
    source = Path(__file__).parents[1] / 'unstructured_mesh' / MESH_FILES['moab']
    lines = source.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('POINTS'))
    for i in range(start + 1, start + 1 + int(lines[start].split()[1])):
        x, y, z = lines[i].split()
        lines[i] = f'{-float(x)} {y} {z}'
    mesh_path = tmp_path / 'mirrored.vtk'
    mesh_path.write_text('\n'.join(lines) + '\n')

    openmc.reset_auto_ids()
    with change_directory(tmp_path):
        make_model('moab', mesh_path, 'tally_and_ww').export_to_model_xml()

    if config['build_inputs']:
        return
    returncode, output = run_openmc(tmp_path)
    assert returncode == 0, output
    check_tally_sums(tmp_path)
    with openmc.StatePoint(tmp_path / 'statepoint.10.h5') as sp:
        volumes = sp.get_tally(name='tet flux').find_filter(
            openmc.MeshFilter).mesh.volumes
    assert np.all(volumes > 0.0)
    assert volumes.sum() == pytest.approx(1.0)
    weight_windows = openmc.WeightWindowsList.from_hdf5(
        tmp_path / 'weight_windows.h5')[0]
    assert np.all(weight_windows.lower_ww_bounds > 0.0)


def run_openmc(path):
    """Run the model in path, returning the return code and the output"""
    args = [config['exe']]
    if config['mpi']:
        args = [config['mpiexec'], '-n', config['mpi_np']] + args
    result = subprocess.run(args, cwd=path, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def check_tally_sums(path):
    """The flux on the mesh of the run in path adds up to the flux in the
    cell"""
    with openmc.StatePoint(path / 'statepoint.10.h5') as sp:
        cell_flux = sp.get_tally(name='cell flux').mean.sum()
        mesh_flux = sp.get_tally(name='tet flux').mean.sum()
    assert cell_flux > 0.0
    assert mesh_flux == pytest.approx(cell_flux, rel=1e-10)
