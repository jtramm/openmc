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
FEATURES = {'moab': 'dagmc', 'libmesh': 'libmesh'}


class MGXSTestHarness(TolerantWeightWindowPyAPITestHarness):
    def _cleanup(self):
        super()._cleanup()
        f = 'mgxs.h5'
        if os.path.exists(f):
            os.remove(f)


def make_model(library, mesh_path, case, x_max=0.5):
    """Random ray model of the box [-0.5, x_max] x [-0.5, 0.5]^2, subdivided
    by the mesh in mesh_path"""

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
    umesh = openmc.UnstructuredMesh(mesh_path, library=library)

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
    settings.random_ray['source_region_meshes'] = [(umesh, [root])]
    settings.source = openmc.IndependentSource(
        energy=openmc.stats.Discrete([1.0e6], [1.0]),
        constraints={'domains': [cell]})

    cell_tally = openmc.Tally(name='cell flux')
    cell_tally.filters = [openmc.CellFilter(cell)]
    cell_tally.scores = ['flux']
    cell_tally.estimator = 'tracklength'
    tallies = openmc.Tallies([cell_tally])

    # In one case the tet mesh is also the tally mesh and the weight window
    # mesh; in the other it is used for subdivision alone
    if case == 'tally_and_ww':
        mesh_tally = openmc.Tally(name='tet flux')
        mesh_tally.filters = [openmc.MeshFilter(umesh)]
        mesh_tally.scores = ['flux']
        mesh_tally.estimator = 'tracklength'
        tallies.append(mesh_tally)
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
    # The mesh covers the cell: the 24-tet mesh tiles it, the others tile
    # [-1, 1]^3 with other element types
    ('moab', MESH_FILES['moab'], 0.5),
    ('libmesh', MESH_FILES['libmesh'], 0.5),
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
    with an error, and one with a mesh covering all of it does not"""
    if not openmc.lib.feature_enabled(FEATURES[library]):
        pytest.skip(f"{library} unstructured meshes are not enabled.")

    openmc.reset_auto_ids()
    mesh_path = Path(__file__).parents[1] / 'unstructured_mesh' / mesh_file
    with change_directory(tmp_path):
        model = make_model(library, mesh_path, 'subdivision_only', x_max=x_max)
        model.export_to_model_xml()

    if config['build_inputs']:
        return
    args = [config['exe']]
    if config['mpi']:
        args = [config['mpiexec'], '-n', config['mpi_np']] + args
    result = subprocess.run(args, cwd=tmp_path, capture_output=True,
                            text=True)
    output = result.stdout + result.stderr
    if x_max <= 0.5:
        assert result.returncode == 0, output
    else:
        assert result.returncode != 0
        assert 'does not cover' in output
