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
               length_multiplier=None):
    """Random ray model of the box [-0.5, x_max] x [-0.5, 0.5]^2, subdivided
    by the mesh in mesh_path (read by a mesh library, directly or through
    XDG), or by a regular mesh of [-0.5, 0.5]^3"""

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
        mesh_tally.filters = [openmc.MeshFilter(umesh)]
        mesh_tally.scores = ['flux']
        mesh_tally.estimator = 'tracklength'
        tallies.append(mesh_tally)
    if case == 'tally_and_ww':
        settings.weight_window_generators = openmc.WeightWindowGenerator(
            mesh=umesh, method='magic', max_realizations=5)

    return openmc.Model(geometry, materials, settings, tallies)


@pytest.mark.parametrize("library,case", product(
    ['moab', 'libmesh'], ['tally_and_ww', 'subdivision_only']))
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
    ('libmesh', MESH_FILES['libmesh'], 1.5),
    ('xdg-libmesh', MESH_FILES['libmesh'], 1.5),
    ('regular', None, 1.5),
    # The mesh covers the cell: the 24-tet mesh and the regular mesh tile it,
    # the others tile [-1, 1]^3 with other element types
    ('regular', None, 0.5),
    ('moab', MESH_FILES['moab'], 0.5),
    ('libmesh', MESH_FILES['libmesh'], 0.5),
    ('xdg-moab', MESH_FILES['moab'], 0.5),
    ('xdg-libmesh', MESH_FILES['libmesh'], 0.5),
    ('libmesh', 'test_mesh_hexes_warped.e', 0.5),
    ('libmesh', 'test_mesh_hexes_nonconforming.e', 0.5),
    ('libmesh', 'test_mesh_hexes20.e', 0.5),
    ('libmesh', 'test_mesh_tets10.e', 0.5),
    ('libmesh', 'test_mesh_prisms.e', 0.5),
    ('libmesh', 'test_mesh_pyramids.e', 0.5),
])
def test_random_ray_unstructured_mesh_coverage(
        library, mesh_file, x_max, tmp_path):
    """A run with a mesh covering only half of the cell it subdivides stops
    with an error, and one with a mesh covering all of it does not, and its
    flux on the mesh adds up to the flux in the cell"""
    if (library != 'regular' and
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


@pytest.mark.parametrize("library,option", product(
    ['moab', 'libmesh'], ['linear', 'scaled']))
def test_random_ray_unstructured_mesh_options(library, option, tmp_path):
    """Runs with linear sources, and with the mesh scaled by a length
    multiplier (the 24-tet mesh then tiles [-1, 1]^3), whose flux on the mesh
    adds up to the flux in the cell"""
    if not openmc.lib.feature_enabled(FEATURES[library]):
        pytest.skip(f"{library} unstructured meshes are not enabled.")

    openmc.reset_auto_ids()
    mesh_path = (Path(__file__).parents[1] / 'unstructured_mesh' /
                 MESH_FILES[library])
    with change_directory(tmp_path):
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
