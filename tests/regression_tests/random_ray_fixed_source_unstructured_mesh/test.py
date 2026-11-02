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
# Features a mesh library needs; regular meshes need none
FEATURES = {'moab': ['dagmc'], 'libmesh': ['libmesh'],
            'xdg-moab': ['xdg', 'dagmc'], 'xdg-libmesh': ['xdg', 'libmesh']}


def enabled(library):
    return all(openmc.lib.feature_enabled(feature)
               for feature in FEATURES.get(library, []))


class MGXSTestHarness(TolerantWeightWindowPyAPITestHarness):
    def _cleanup(self):
        super()._cleanup()
        f = 'mgxs.h5'
        if os.path.exists(f):
            os.remove(f)


def make_model(library, mesh_path, case, x_max=0.5, source_shape='flat',
               length_multiplier=None, tally_mesh=None, ww_method='magic',
               run_mode='fixed source', source_point=None, slab=None,
               slab_fill=True):
    """Random ray model of the box [-0.5, x_max] x [-0.5, 0.5]^2, subdivided
    by the mesh in mesh_path (read by a mesh library, directly or through
    XDG), or by a regular mesh of [-0.5, 0.5]^3 or (in 2D) [-0.5, 0.5]^2, with
    a mesh tally on that mesh or on another one, and optionally a cell (of the
    material or void) in a slab of the given thickness past the box"""

    # One-group material with half scattering
    groups = openmc.mgxs.EnergyGroups([1e-5, 20.0e6])
    xsdata = openmc.XSdata('mat', groups)
    xsdata.order = 0
    xsdata.set_total([1.0])
    xsdata.set_absorption([0.5])
    xsdata.set_scatter_matrix(np.array([[[0.5]]]))
    if run_mode == 'eigenvalue':
        xsdata.set_fission([0.25])
        xsdata.set_nu_fission([0.6])
        xsdata.set_chi([1.0])
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
    if slab is not None:
        box.xmax.boundary_type = 'transmission'
        end = openmc.XPlane(x_max + slab, boundary_type='vacuum')
        root.add_cell(openmc.Cell(
            name='slab', fill=mat if slab_fill else None,
            region=+box.xmax & -end & +box.ymin & -box.ymax & +box.zmin
            & -box.zmax))
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
    settings.run_mode = run_mode
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
    if source_point is not None:
        settings.source = openmc.IndependentSource(
            space=openmc.stats.Point(source_point),
            energy=openmc.stats.Discrete([1.0e6], [1.0]))
    else:
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
            mesh=tally_mesh or umesh, method=ww_method, max_realizations=5)

    return openmc.Model(geometry, materials, settings, tallies)


@pytest.mark.parametrize("library,case", product(
    ['moab', 'xdg-libmesh'], ['tally_and_ww', 'subdivision_only']))
def test_random_ray_fixed_source_unstructured_mesh(library, case):
    if not enabled(library):
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
    # The mesh falls short of the cell by a sliver that is of no consequence,
    # or by enough to matter
    ('regular', None, 0.5 + 1e-9),
    ('moab', MESH_FILES['moab'], 0.5 + 1e-9),
    ('regular', None, 0.5 + 1e-3),
    ('moab', MESH_FILES['moab'], 0.5 + 1e-3),
])
def test_random_ray_unstructured_mesh_coverage(
        library, mesh_file, x_max, tmp_path):
    """A run with a mesh covering only half of the cell it subdivides stops
    with an error, and one with a mesh covering all of it does not, and its
    flux on the mesh adds up to the flux in the cell"""
    if not enabled(library):
        pytest.skip(f"{library} unstructured meshes are not enabled.")

    openmc.reset_auto_ids()
    mesh_path = (Path(__file__).parents[1] / 'unstructured_mesh' / mesh_file
                 if mesh_file else None)
    covered = x_max < 0.5 + 1e-6
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
    # Rays are not followed through a tally mesh, so the native interface will
    # do for libMesh
    ('xdg-libmesh', 'tally_mesh'),
    ('moab', 'fw_cadis'),
    ('xdg-libmesh', 'fw_cadis'),
    ('moab', 'eigenvalue'),
    ('xdg-libmesh', 'eigenvalue'),
])
def test_random_ray_unstructured_mesh_options(library, option, tmp_path):
    """Runs with linear sources, with the mesh scaled by a length multiplier
    (the 24-tet mesh then tiles [-1, 1]^3), with the tally and weight windows
    on the same mesh through the other interface, with FW-CADIS weight windows
    and in eigenvalue mode, whose flux on the mesh adds up to the flux in the
    cell"""
    if not enabled(library) or (option == 'tally_mesh'
                                and not enabled('xdg-moab')):
        pytest.skip(f"{library} unstructured meshes are not enabled.")

    openmc.reset_auto_ids()
    mesh_path = (Path(__file__).parents[1] / 'unstructured_mesh' /
                 MESH_FILES[library])
    with change_directory(tmp_path):
        if option == 'tally_mesh':
            tally_mesh = openmc.UnstructuredMesh(
                mesh_path, library=library.removeprefix('xdg-'))
            tally_mesh.interface = ('native' if library.startswith('xdg-')
                                    else 'xdg')
            model = make_model(library, mesh_path, 'tally_and_ww',
                               tally_mesh=tally_mesh)
        elif option == 'fw_cadis':
            model = make_model(library, mesh_path, 'tally_and_ww',
                               ww_method='fw_cadis')
        elif option == 'eigenvalue':
            model = make_model(library, mesh_path, 'tally',
                               run_mode='eigenvalue')
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
    if option == 'scaled':
        assert mesh_volumes(tmp_path).sum() == pytest.approx(8.0)
    if option == 'tally_mesh':
        # Every tet is a source region, so the mesh is the same bin by bin
        check_weight_windows(tmp_path, all_bins=True)
    elif option == 'fw_cadis':
        check_weight_windows(tmp_path)


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


@pytest.mark.parametrize("library", ['moab', 'xdg-moab'])
def test_random_ray_mirrored_mesh(library, tmp_path):
    """A mirrored copy of a mesh keeps its node order, which turns its tets
    inside out: their volumes must still be positive for weight windows to be
    generated on them. XDG cannot follow rays through such a mesh, but it can
    tally on it."""
    if not enabled(library):
        pytest.skip(f"{library} unstructured meshes are not enabled.")

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
        if library == 'moab':
            model = make_model('moab', mesh_path, 'tally_and_ww')
        else:
            tally_mesh = openmc.UnstructuredMesh(mesh_path, library='moab')
            tally_mesh.interface = 'xdg'
            model = make_model('regular', None, 'tally_and_ww',
                               tally_mesh=tally_mesh)
        model.export_to_model_xml()

    if config['build_inputs']:
        return
    returncode, output = run_openmc(tmp_path)
    assert returncode == 0, output
    check_tally_sums(tmp_path)
    volumes = mesh_volumes(tmp_path)
    assert np.all(volumes > 0.0)
    assert volumes.sum() == pytest.approx(1.0)
    check_weight_windows(tmp_path)


def test_random_ray_small_uncovered_cell(tmp_path):
    """A cell entirely outside of the mesh stops the run, even if it is too
    small to matter next to the path through the other cells on the mesh"""
    openmc.reset_auto_ids()
    with change_directory(tmp_path):
        make_model('regular', None, 'tally', slab=1e-7).export_to_model_xml()

    if config['build_inputs']:
        return
    returncode, output = run_openmc(tmp_path)
    assert returncode != 0
    assert 'does not cover' in output


def test_random_ray_void_cell_outside_of_mesh(tmp_path):
    """Nothing happens to rays in a void cell without a source, so it can
    extend beyond the mesh"""
    openmc.reset_auto_ids()
    with change_directory(tmp_path):
        make_model('regular', None, 'tally', slab=0.5,
                   slab_fill=False).export_to_model_xml()

    if config['build_inputs']:
        return
    returncode, output = run_openmc(tmp_path)
    assert returncode == 0, output
    check_tally_sums(tmp_path)


@pytest.mark.parametrize("library,point", [
    ('moab', (0.1, 0.2, 0.3)),
    ('xdg-libmesh', (0.1, 0.2, 0.3)),
    # On a node and on an edge of the mesh, and on a face of its tets
    ('moab', (0.0, 0.0, 0.0)),
    ('moab', (0.0, 0.0, 0.25)),
    ('moab', (0.1, 0.1, 0.3)),
])
def test_random_ray_point_source(library, point, tmp_path):
    """A point source goes to the mesh bin around it, wherever in the mesh it
    is"""
    if not enabled(library):
        pytest.skip(f"{library} unstructured meshes are not enabled.")

    openmc.reset_auto_ids()
    mesh_path = (Path(__file__).parents[1] / 'unstructured_mesh' /
                 MESH_FILES[library])
    with change_directory(tmp_path):
        make_model(library, mesh_path, 'tally',
                   source_point=point).export_to_model_xml()

    if config['build_inputs']:
        return
    returncode, output = run_openmc(tmp_path)
    assert returncode == 0, output
    check_tally_sums(tmp_path)


def test_random_ray_point_source_outside_of_mesh(tmp_path):
    """No ray would find a point source outside of the mesh, which is reported
    up front"""
    openmc.reset_auto_ids()
    with change_directory(tmp_path):
        make_model('regular', None, 'tally', x_max=1.5,
                   source_point=(1.0, 0.1, 0.1)).export_to_model_xml()

    if config['build_inputs']:
        return
    returncode, output = run_openmc(tmp_path)
    assert returncode != 0
    assert 'outside of the source region mesh' in output


def check_weight_windows(path, all_bins=False):
    """Valid bounds, in all bins or else -1 for a bin without a usable flux"""
    bounds = openmc.WeightWindowsList.from_hdf5(
        path / 'weight_windows.h5')[0].lower_ww_bounds
    assert np.any(bounds > 0.0)
    assert np.all((bounds > 0.0) | (not all_bins and bounds == -1.0))


def mesh_volumes(path):
    """The volumes of the elements of the tally mesh of the run in path"""
    with openmc.StatePoint(path / 'statepoint.10.h5') as sp:
        return sp.get_tally(name='tet flux').find_filter(
            openmc.MeshFilter).mesh.volumes


def run_openmc(path):
    """Run the model in path, returning the return code and the output"""
    args = [config['exe']]
    if config['mpi']:
        args = [config['mpiexec'], '-n', config['mpi_np']] + args
    result = subprocess.run(args, cwd=path, capture_output=True, text=True)
    # (messages are wrapped)
    return result.returncode, ' '.join((result.stdout + result.stderr).split())


def check_tally_sums(path):
    """The flux on the mesh of the run in path adds up to the flux in the
    cell"""
    with openmc.StatePoint(path / 'statepoint.10.h5') as sp:
        cell_flux = sp.get_tally(name='cell flux').mean.sum()
        mesh_flux = sp.get_tally(name='tet flux').mean.sum()
    assert cell_flux > 0.0
    assert mesh_flux == pytest.approx(cell_flux, rel=1e-10)
