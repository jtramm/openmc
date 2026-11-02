import os
import subprocess
import sys
from itertools import product
from pathlib import Path

import h5py
import numpy as np
import openmc
import openmc.lib
from openmc.utility_funcs import change_directory
import pytest
from scipy.io import netcdf_file

from tests.regression_tests import config
from tests.testing_harness import TolerantWeightWindowPyAPITestHarness

# The same 24-tet mesh of the unit cube, as the canonical MOAB (.vtk) and
# libMesh (.e) assets
MESH_FILES = {
    'moab': 'test_mesh_dagmc_tets.vtk',
    'libmesh': 'test_mesh_dagmc_tets.e',
    'xdg-moab': 'test_mesh_dagmc_tets.vtk',
    'xdg-libmesh': 'test_mesh_dagmc_tets.e',
    'xdg-hexes': 'test_mesh_hexes.e',
}
# Features a mesh library needs; regular meshes need none
FEATURES = {'moab': ['dagmc'], 'libmesh': ['libmesh'],
            'xdg-moab': ['xdg', 'dagmc'], 'xdg-libmesh': ['xdg', 'libmesh'],
            'xdg-hexes': ['xdg', 'libmesh']}


def enabled(library):
    return all(openmc.lib.feature_enabled(feature)
               for feature in FEATURES.get(library, []))


class MGXSTestHarness(TolerantWeightWindowPyAPITestHarness):
    def _cleanup(self):
        super()._cleanup()
        f = 'mgxs.h5'
        if os.path.exists(f):
            os.remove(f)


def one_group_materials(run_mode='fixed source', density=1.0):
    """One-group material with half scattering"""
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
    mat.set_density('macro', density)
    mat.add_macroscopic(openmc.Macroscopic('mat'))
    materials = openmc.Materials([mat])
    materials.cross_sections = 'mgxs.h5'
    return materials


def random_ray_settings(lower_left, upper_right, source_shape='flat',
                        run_mode='fixed source'):
    """Settings of a few rays over a box, with enough batches for linear
    sources to be used, which starts after the tenth"""
    settings = openmc.Settings()
    settings.energy_mode = 'multi-group'
    settings.run_mode = run_mode
    settings.batches = 10 if source_shape == 'flat' else 15
    settings.inactive = 5
    settings.particles = 100
    settings.random_ray['distance_active'] = 10.0
    settings.random_ray['distance_inactive'] = 3.0
    settings.random_ray['ray_source'] = openmc.IndependentSource(
        space=openmc.stats.Box(lower_left, upper_right))
    settings.random_ray['volume_estimator'] = 'hybrid'
    if source_shape != 'flat':
        settings.random_ray['source_shape'] = source_shape
    return settings


def flux_tally(name, tally_filter):
    tally = openmc.Tally(name=name)
    tally.filters = [tally_filter]
    tally.scores = ['flux']
    tally.estimator = 'tracklength'
    return tally


def read_tets(path):
    """The nodes of a VTK file, and the nodes of each of its tets"""
    lines = Path(path).read_text().splitlines()
    i_points = next(i for i, line in enumerate(lines)
                    if line.startswith('POINTS'))
    i_cells = next(i for i, line in enumerate(lines)
                   if line.startswith('CELLS'))
    i_types = next(i for i, line in enumerate(lines)
                   if line.startswith('CELL_TYPES'))
    n_points = int(lines[i_points].split()[1])
    points = np.array(
        [line.split() for line in lines[i_points + 1:i_points + 1 + n_points]],
        dtype=float)
    tets = np.array([line.split()[1:] for line in lines[i_cells + 1:i_types]
                     if line.startswith('4 ')], dtype=int)
    return points, tets


def write_tets(path, points, tets):
    """A VTK file of the given tets"""
    lines = ['# vtk DataFile Version 2.0', 'tets', 'ASCII',
             'DATASET UNSTRUCTURED_GRID', f'POINTS {len(points)} double']
    lines += [' '.join(map(str, point)) for point in points]
    lines.append(f'CELLS {len(tets)} {5 * len(tets)}')
    lines += ['4 ' + ' '.join(map(str, tet)) for tet in tets]
    lines.append(f'CELL_TYPES {len(tets)}')
    lines += ['10'] * len(tets)
    Path(path).write_text('\n'.join(lines) + '\n')


def unit_cube_tets():
    """The nodes and tets of the 24-tet mesh of the unit cube"""
    return read_tets(Path(__file__).parents[1] / 'unstructured_mesh' /
                     MESH_FILES['moab'])


def graded_unit_cube_tets():
    """The 24-tet mesh of the unit cube with the nodes in the middle of its
    faces moved along them, for tets with volumes that differ"""
    points, tets = unit_cube_tets()
    return np.where(points == 0.0, 0.4, points), tets


def make_model(library, mesh_path, case, x_max=0.5, source_shape='flat',
               length_multiplier=None, tally_mesh=None, ww_method='magic',
               run_mode='fixed source', source_point=None, slab=None,
               slab_fill=True, subdivide_root=True):
    """Random ray model of the box [-0.5, x_max] x [-0.5, 0.5]^2, subdivided
    by the mesh in mesh_path (through XDG, for the libraries that start with
    it), or by a regular mesh of [-0.5, 0.5]^3 or (in 2D) [-0.5, 0.5]^2, with
    a mesh tally on that mesh or on another one, and optionally a cell (of the
    material or void) in a slab of the given thickness past the box, to which
    the mesh is applied unless subdivide_root is False"""

    materials = one_group_materials(run_mode)
    mat = materials[0]

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

    settings = random_ray_settings((-0.5, -0.5, -0.5),
                                   (x_max + (slab or 0.0), 0.5, 0.5),
                                   source_shape, run_mode)
    settings.random_ray['source_region_meshes'] = [
        (umesh, [root if subdivide_root else cell])]
    if source_point is not None:
        settings.source = openmc.IndependentSource(
            space=openmc.stats.Point(source_point),
            energy=openmc.stats.Discrete([1.0e6], [1.0]))
    else:
        settings.source = openmc.IndependentSource(
            energy=openmc.stats.Discrete([1.0e6], [1.0]),
            constraints={'domains': [cell]})

    tallies = openmc.Tallies(
        [flux_tally('cell flux', openmc.CellFilter(cell))])

    # The mesh is also the tally mesh and the weight window mesh, the tally
    # mesh only, or used for subdivision alone
    if case in ('tally_and_ww', 'tally'):
        tallies.append(flux_tally(
            'tet flux', openmc.MeshFilter(tally_mesh or umesh)))
    if case == 'tally_and_ww':
        settings.weight_window_generators = openmc.WeightWindowGenerator(
            mesh=tally_mesh or umesh, method=ww_method, max_realizations=5)

    return openmc.Model(geometry, materials, settings, tallies)


@pytest.mark.parametrize("library,case", product(
    ['xdg-moab', 'xdg-libmesh'], ['tally_and_ww', 'subdivision_only']))
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
    # The mesh covers the cell: the 24-tet mesh and the regular mesh tile it,
    # and the hex mesh extends beyond it
    ('regular', None, 0.5),
    ('regular-2d', None, 0.5),
    ('xdg-moab', MESH_FILES['moab'], 0.5),
    ('xdg-libmesh', MESH_FILES['libmesh'], 0.5),
    ('xdg-libmesh', 'test_mesh_hexes.e', 0.5),
    # The mesh falls short of the cell by a sliver that is of no consequence,
    # or by enough to count, as with a mesh and a cell faceted separately
    ('regular', None, 0.5 + 1e-9),
    ('regular', None, 0.5 + 1e-3),
    ('xdg-moab', MESH_FILES['moab'], 0.5 + 1e-3),
    ('xdg-libmesh', MESH_FILES['libmesh'], 0.5 + 1e-3),
])
def test_random_ray_unstructured_mesh_coverage(
        library, mesh_file, x_max, tmp_path):
    """The flux on a mesh that covers the cell adds up to the flux in the cell,
    and that on a mesh that falls just short of it to nearly all of it"""
    if not enabled(library):
        pytest.skip(f"{library} unstructured meshes are not enabled.")

    openmc.reset_auto_ids()
    mesh_path = (Path(__file__).parents[1] / 'unstructured_mesh' / mesh_file
                 if mesh_file else None)
    with change_directory(tmp_path):
        make_model(library, mesh_path, 'tally',
                   x_max=x_max).export_to_model_xml()

    if config['build_inputs']:
        return
    returncode, output = run_openmc(tmp_path)
    assert returncode == 0, output
    if x_max < 0.5 + 1e-6:
        check_tally_sums(tmp_path)
    else:
        with openmc.StatePoint(statepoint_path(tmp_path)) as sp:
            cell_flux = sp.get_tally(name='cell flux').mean.sum()
            mesh_flux = sp.get_tally(name='tet flux').mean.sum()
        assert 0.99 * cell_flux < mesh_flux < cell_flux


@pytest.mark.parametrize("library,option", product(
    ['xdg-moab', 'xdg-libmesh'],
    ['linear', 'scaled', 'tally_mesh', 'fw_cadis', 'eigenvalue']))
def test_random_ray_unstructured_mesh_options(library, option, tmp_path):
    """Runs with linear sources, with the mesh scaled by a length multiplier
    (the 24-tet mesh then tiles [-1, 1]^3), with the tally and weight windows
    on the same mesh through the native interface, which only looks up points
    for them, with FW-CADIS weight windows and in eigenvalue mode, whose flux
    on the mesh adds up to the flux in the cell"""
    if not enabled(library):
        pytest.skip(f"{library} unstructured meshes are not enabled.")

    openmc.reset_auto_ids()
    mesh_path = (Path(__file__).parents[1] / 'unstructured_mesh' /
                 MESH_FILES[library])
    with change_directory(tmp_path):
        if option == 'tally_mesh':
            tally_mesh = openmc.UnstructuredMesh(
                mesh_path, library=library.removeprefix('xdg-'))
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


@pytest.mark.parametrize("library", ['moab', 'libmesh'])
def test_random_ray_native_subdivision(library, tmp_path):
    """Rays are followed through unstructured meshes by XDG, and using the
    native interface instead is reported up front"""
    if not enabled(library):
        pytest.skip(f"{library} unstructured meshes are not enabled.")

    openmc.reset_auto_ids()
    mesh_path = (Path(__file__).parents[1] / 'unstructured_mesh' /
                 MESH_FILES[library])
    with change_directory(tmp_path):
        make_model(library, mesh_path,
                   'subdivision_only').export_to_model_xml()

    if config['build_inputs']:
        return
    returncode, output = run_openmc(tmp_path)
    assert returncode != 0
    assert 'Use the XDG interface' in output


@pytest.mark.parametrize("interface,library", [
    ('native', 'moab'), ('xdg', 'xdg-moab')])
def test_random_ray_mirrored_mesh(interface, library, tmp_path):
    """A mirrored copy of a mesh keeps its node order, which turns its tets
    inside out: their volumes must still be positive for weight windows to be
    generated on them. XDG cannot follow rays through such a mesh, but it can
    tally on it, as can the native interface."""
    if not enabled(library):
        pytest.skip(f"{library} unstructured meshes are not enabled.")

    # x -> -x in the 24-tet mesh of the unit cube
    points, tets = unit_cube_tets()
    mesh_path = tmp_path / 'mirrored.vtk'
    write_tets(mesh_path, points * [-1.0, 1.0, 1.0], tets)

    openmc.reset_auto_ids()
    with change_directory(tmp_path):
        tally_mesh = openmc.UnstructuredMesh(mesh_path, library='moab')
        tally_mesh.interface = interface
        make_model('regular', None, 'tally_and_ww',
                   tally_mesh=tally_mesh).export_to_model_xml()

    if config['build_inputs']:
        return
    returncode, output = run_openmc(tmp_path)
    assert returncode == 0, output
    check_tally_sums(tmp_path)
    volumes = mesh_volumes(tmp_path)
    assert np.all(volumes > 0.0)
    assert volumes.sum() == pytest.approx(1.0)
    check_weight_windows(tmp_path)


@pytest.mark.parametrize("slab,slab_fill,rel", [
    (0.5, True, 1e-6),
    (0.5, False, 1e-6),
    # Rays through a mesh lose TINY_BIT at both ends of a cell, which shows
    # here
    (1e-7, True, 0.2),
])
def test_random_ray_cell_outside_of_mesh(slab, slab_fill, rel, tmp_path):
    """A cell outside of a mesh applied to it is not subdivided, so applying
    the mesh to the root universe or only to the cell inside of it is the same,
    whether the other cell is filled or void, and however small"""
    results = []
    for subdivide_root in (True, False):
        path = tmp_path / str(subdivide_root)
        path.mkdir()
        openmc.reset_auto_ids()
        with change_directory(path):
            model = make_model('regular', None, 'tally', slab=slab,
                               slab_fill=slab_fill,
                               subdivide_root=subdivide_root)
            slab_cell = next(
                cell for cell in model.geometry.get_all_cells().values()
                if cell.name == 'slab')
            model.tallies.append(
                flux_tally('slab flux', openmc.CellFilter(slab_cell)))
            model.export_to_model_xml()

        if config['build_inputs']:
            return
        returncode, output = run_openmc(path)
        assert returncode == 0, output
        check_tally_sums(path)
        results.append(fluxes(path, 'cell flux', 'tet flux', 'slab flux'))

    assert results[0]['slab flux'].sum() > 0.0
    for name in ('cell flux', 'tet flux'):
        assert results[0][name] == pytest.approx(results[1][name], rel=1e-6)
    assert results[0]['slab flux'] == pytest.approx(
        results[1]['slab flux'], rel=rel)


def part_of_box_mesh(library, mesh_path, scale=1.0):
    """A mesh inside of the box [-1.5, 1.5]^3 (times a scale, for an
    unstructured mesh), and the region it covers"""
    def cube(x0, x1):
        return -openmc.model.RectangularParallelepiped(
            scale * x0, scale * x1, -0.5 * scale, 0.5 * scale, -0.5 * scale,
            0.5 * scale)

    if library == 'rectilinear':
        mesh = openmc.RectilinearMesh()
        mesh.x_grid = [-0.5, 0.1, 0.5]
        mesh.y_grid = [-0.5, -0.2, 0.0, 0.5]
        mesh.z_grid = [-0.5, 0.4, 0.5]
        return mesh, cube(-0.5, 0.5)

    # Meshes that do not span all angles: a ray can come at them from anywhere
    z_range = +openmc.ZPlane(-0.5) & -openmc.ZPlane(0.5)
    if library == 'half-cylinder':
        mesh = openmc.CylindricalMesh(
            r_grid=[0.0, 0.5, 1.0], z_grid=[-0.5, 0.0, 0.5],
            phi_grid=[0.0, 1.5, np.pi])
        return mesh, -openmc.ZCylinder(r=1.0) & +openmc.YPlane(0.0) & z_range
    if library == 'quarter-cylinder':
        mesh = openmc.CylindricalMesh(
            r_grid=[0.0, 0.5, 1.0], z_grid=[-0.5, 0.0, 0.5],
            phi_grid=[0.5 * np.pi, 2.0, np.pi])
        return mesh, (-openmc.ZCylinder(r=1.0) & -openmc.XPlane(0.0)
                      & +openmc.YPlane(0.0) & z_range)
    if library == 'cone':
        mesh = openmc.SphericalMesh(
            r_grid=[0.0, 0.5, 1.0], theta_grid=[0.0, 0.4, 0.25 * np.pi],
            phi_grid=[0.0, 2.0 * np.pi])
        return mesh, (-openmc.Sphere(r=1.0) & -openmc.ZCone(r2=1.0)
                      & +openmc.ZPlane(0.0))
    if library == 'regular':
        mesh = openmc.RegularMesh()
        mesh.lower_left = (-0.5, -0.5, -0.5)
        mesh.upper_right = (0.5, 0.5, 0.5)
        mesh.dimension = (3, 3, 3)
        return mesh, cube(-0.5, 0.5)
    if library == 'regular-2d':
        mesh = openmc.RegularMesh()
        mesh.lower_left = (-0.5, -0.5)
        mesh.upper_right = (0.5, 0.5)
        mesh.dimension = (3, 3)
        return mesh, (+openmc.XPlane(-0.5) & -openmc.XPlane(0.5)
                      & +openmc.YPlane(-0.5) & -openmc.YPlane(0.5))
    # A hollow cylinder and a hollow sphere: rays can leave and come back
    if library == 'cylindrical':
        mesh = openmc.CylindricalMesh(
            r_grid=[0.4, 0.7, 1.0], z_grid=[-0.5, 0.0, 0.5],
            phi_grid=[0.0, 2.0, 2.0 * np.pi])
        return mesh, (+openmc.ZCylinder(r=0.4) & -openmc.ZCylinder(r=1.0)
                      & +openmc.ZPlane(-0.5) & -openmc.ZPlane(0.5))
    if library == 'spherical':
        mesh = openmc.SphericalMesh(
            r_grid=[0.4, 0.7, 1.0], theta_grid=[0.0, 1.0, np.pi],
            phi_grid=[0.0, 2.0, 2.0 * np.pi])
        return mesh, +openmc.Sphere(r=0.4) & -openmc.Sphere(r=1.0)

    if library == 'xdg-hexes':
        # 1000 hexes of [-10, 10]^3 to start with
        mesh = openmc.UnstructuredMesh(mesh_path, library='libmesh')
        mesh.interface = 'xdg'
        mesh.length_multiplier = 0.05
        return mesh, cube(-0.5, 0.5)

    mesh = openmc.UnstructuredMesh(
        mesh_path, library=library.removeprefix('two-').removeprefix('xdg-'))
    if 'xdg' in library:
        mesh.interface = 'xdg'
    if scale != 1.0:
        mesh.length_multiplier = scale
    if library.startswith('two-'):
        return mesh, cube(-1.25, -0.25) | cube(0.25, 1.25)
    return mesh, cube(-0.5, 0.5)


def make_comparison_model(cells, half_width, mesh, source_shape='flat',
                          run_mode='fixed source', source_point=None,
                          sample_method=None, scale=1.0, ww_method=None,
                          plot=False):
    """Model of cells filling a box of a given half width, all subdivided by
    the mesh, with their fluxes and that on the mesh as tallies. A mesh is also
    what makes runs comparable for linear sources: the position of the start of
    a ray in its active length is not right in a cell without one. All lengths
    are times a scale, including a half width that is not given as such, and
    cross sections divided by it."""
    settings = random_ray_settings(
        (-half_width, -half_width, -half_width),
        (half_width, half_width, half_width), source_shape, run_mode)
    settings.random_ray['distance_active'] *= scale
    settings.random_ray['distance_inactive'] *= scale
    if sample_method is not None:
        settings.random_ray['sample_method'] = sample_method
    settings.random_ray['source_region_meshes'] = [(mesh, cells)]
    if source_point is not None:
        settings.source = openmc.IndependentSource(
            space=openmc.stats.Point(source_point),
            energy=openmc.stats.Discrete([1.0e6], [1.0]))
    else:
        settings.source = openmc.IndependentSource(
            energy=openmc.stats.Discrete([1.0e6], [1.0]),
            constraints={'domains': cells})

    tallies = openmc.Tallies(
        [flux_tally('cell flux', openmc.CellFilter(cells)),
         flux_tally('mesh flux', openmc.MeshFilter(mesh))])
    if ww_method is not None:
        settings.weight_window_generators = openmc.WeightWindowGenerator(
            mesh=mesh, method=ww_method, max_realizations=5)
    plots = None
    if plot:
        voxels = openmc.VoxelPlot()
        voxels.width = (2.0 * half_width,) * 3
        voxels.pixels = (6, 6, 6)
        plots = openmc.Plots([voxels])
    materials = one_group_materials(run_mode, 1.0 / scale)
    for cell in cells:
        cell.fill = materials[0]
    return openmc.Model(openmc.Geometry(cells), materials, settings, tallies,
                        plots)


@pytest.mark.parametrize("library,options", [
    ('regular', {}),
    ('regular', {'source_shape': 'linear'}),
    ('regular', {'run_mode': 'eigenvalue'}),
    # A point source in the mesh, and one outside of it
    ('regular', {'source_point': (0.45, 0.1, 0.1)}),
    ('regular', {'source_point': (1.4, 1.4, 1.4)}),
    ('regular', {'ww_method': 'magic'}),
    ('regular', {'ww_method': 'fw_cadis'}),
    ('regular', {'plot': True}),
    ('regular-2d', {}),
    ('rectilinear', {}),
    ('cylindrical', {}),
    ('cylindrical', {'source_shape': 'linear'}),
    ('spherical', {}),
    ('half-cylinder', {}),
    ('quarter-cylinder', {}),
    ('quarter-cylinder', {'source_shape': 'linear'}),
    ('cone', {}),
    ('xdg-moab', {}),
    ('xdg-moab', {'source_shape': 'linear'}),
    ('xdg-moab', {'source_point': (1.4, 1.4, 1.4)}),
    ('xdg-moab', {'ww_method': 'fw_cadis'}),
    ('xdg-moab', {'plot': True}),
    ('xdg-libmesh', {}),
    ('xdg-libmesh', {'source_shape': 'linear'}),
    ('xdg-libmesh', {'source_point': (1.4, 1.4, 1.4)}),
    ('xdg-libmesh', {'ww_method': 'fw_cadis'}),
    ('xdg-hexes', {}),
    # A mesh in two parts, of the 24-tet mesh moved to either side, and with
    # what comes before them as wide as each, which makes lengths coincide
    ('two-xdg-moab', {}),
    ('two-xdg-moab', {'source_shape': 'linear'}),
    ('two-xdg-moab', {'half_width': 2.25}),
    ('two-xdg-moab', {'half_width': 2.25, 'sample_method': 's2'}),
    ('two-xdg-moab', {'scale': 100.0}),
])
def test_random_ray_mesh_over_part_of_cell(library, options, tmp_path):
    """What is outside of a mesh that covers part of a cell counts as one more
    bin of the mesh. Rays go through it in its place along them, so a run is
    the same as one with that part as a cell of its own, whatever the shape of
    the mesh and however many times a ray enters it."""
    if not enabled(library.removeprefix('two-')):
        pytest.skip(f"{library} unstructured meshes are not enabled.")
    # A ray handed to another rank restarts on a bin boundary. On a radial
    # surface of these meshes, going inward at a grazing angle, it misses its
    # way out again (find_r_crossing takes it to still be on the surface), so
    # that a few rays score to the wrong bin, which is not what is being tested
    # here.
    if config['mpi'] and library in ('cylindrical', 'spherical',
                                     'half-cylinder', 'quarter-cylinder',
                                     'cone'):
        pytest.skip("Radial crossings of restarted rays")

    mesh_path = None
    if library.startswith('two-'):
        mesh_path = tmp_path / 'two_parts.vtk'
        points, tets = unit_cube_tets()
        write_tets(mesh_path,
                   np.vstack([points - [0.75, 0.0, 0.0],
                              points + [0.75, 0.0, 0.0]]),
                   np.vstack([tets, tets + len(points)]))
    elif library in MESH_FILES:
        mesh_path = (Path(__file__).parents[1] / 'unstructured_mesh' /
                     MESH_FILES[library])

    options = dict(options)
    scale = options.get('scale', 1.0)
    half_width = scale * options.pop('half_width', 1.5)
    results = []
    for split in (False, True):
        path = tmp_path / str(split)
        path.mkdir()
        openmc.reset_auto_ids()
        with change_directory(path):
            mesh, covered = part_of_box_mesh(library, mesh_path, scale)
            box = -openmc.model.RectangularParallelepiped(
                -half_width, half_width, -half_width, half_width, -half_width,
                half_width, boundary_type='vacuum')
            cells = ([openmc.Cell(region=box & covered),
                      openmc.Cell(region=box & ~covered)] if split
                     else [openmc.Cell(region=box)])
            make_comparison_model(cells, half_width, mesh,
                                  **options).export_to_model_xml()

        if config['build_inputs']:
            return
        returncode, output = run_openmc(path)
        assert returncode == 0, output
        results.append(fluxes(path, 'cell flux', 'mesh flux'))
        if 'ww_method' in options:
            results[-1]['bounds'] = openmc.WeightWindowsList.from_hdf5(
                path / 'weight_windows.h5')[0].lower_ww_bounds
        if 'plot' in options:
            assert list(path.glob('*.vtk'))

    if 'ww_method' in options:
        assert np.any(results[0]['bounds'] > 0.0)
        assert results[0]['bounds'] == pytest.approx(
            results[1]['bounds'], rel=1e-5)
    for path in (tmp_path / 'False', tmp_path / 'True'):
        assert 'subdivides nothing' not in (path / 'output.txt').read_text()
    inner, outer = results[1]['cell flux']
    assert results[1]['mesh flux'].sum() > 0.0
    assert outer > 0.0
    assert results[0]['cell flux'].sum() == pytest.approx(
        inner + outer, rel=1e-5)
    assert results[0]['mesh flux'] == pytest.approx(
        results[1]['mesh flux'], rel=1e-5)
    # The flux on the mesh is that of the part of the cell that it covers
    assert results[1]['mesh flux'].sum() == pytest.approx(inner, rel=1e-5)


def test_random_ray_mesh_misses_its_cells(tmp_path):
    """A mesh nowhere near the cells it is applied to is reported, as that is
    most likely a mistake"""
    openmc.reset_auto_ids()
    with change_directory(tmp_path):
        box = -openmc.model.RectangularParallelepiped(
            -1.5, 1.5, -1.5, 1.5, -1.5, 1.5, boundary_type='vacuum')
        mesh = openmc.RegularMesh()
        mesh.lower_left = (100.0, 100.0, 100.0)
        mesh.upper_right = (101.0, 101.0, 101.0)
        mesh.dimension = (2, 2, 2)
        make_comparison_model([openmc.Cell(region=box)], 1.5,
                              mesh).export_to_model_xml()

    if config['build_inputs']:
        return
    returncode, output = run_openmc(tmp_path)
    assert returncode == 0, output
    assert 'so it subdivides nothing' in output


def tet_cell(points, tet):
    """A cell of the shape of a tet of the unit cube"""
    nodes = points[tet]
    half_spaces = []
    for i in range(4):
        face = np.delete(nodes, i, axis=0)
        normal = np.cross(face[1] - face[0], face[2] - face[0])
        on_cube = np.any(np.all(np.abs(face) == 0.5, axis=0)
                         & (np.ptp(face, axis=0) == 0.0))
        plane = openmc.Plane(*normal, d=normal @ face[0], boundary_type=(
            'vacuum' if on_cube else 'transmission'))
        half_spaces.append(+plane if normal @ nodes[i] > plane.d else -plane)
    return openmc.Cell(region=half_spaces[0] & half_spaces[1] & half_spaces[2]
                       & half_spaces[3])


@pytest.mark.parametrize("library,graded,source_shape,source_point", [
    ('xdg-moab', False, 'flat', None),
    ('xdg-moab', False, 'linear', None),
    ('xdg-moab', False, 'flat', (0.3, 0.1, 0.05)),
    ('xdg-moab', True, 'flat', None),
    ('xdg-moab', True, 'linear', None),
    ('xdg-moab', True, 'flat', (0.3, 0.1, 0.05)),
    ('xdg-libmesh', False, 'flat', None),
    ('xdg-libmesh', False, 'linear', None),
    ('xdg-libmesh', False, 'flat', (0.3, 0.1, 0.05)),
])
def test_random_ray_tets_as_cells(
        library, graded, source_shape, source_point, tmp_path):
    """With a source region per tet, a run is the same as one with a cell of
    the same shape per tet: bin by bin, the flux on the mesh is that in the
    cell, and the volume of the bin that of the cell"""
    if not enabled(library):
        pytest.skip(f"{library} unstructured meshes are not enabled.")

    points, tets = graded_unit_cube_tets() if graded else unit_cube_tets()
    if graded:
        mesh_path = tmp_path / 'graded.vtk'
        write_tets(mesh_path, points, tets)
    else:
        mesh_path = (Path(__file__).parents[1] / 'unstructured_mesh' /
                     MESH_FILES[library])

    for split in (False, True):
        path = tmp_path / str(split)
        path.mkdir()
        openmc.reset_auto_ids()
        with change_directory(path):
            if split:
                cells = [tet_cell(points, tet) for tet in tets]
                mesh = openmc.RegularMesh()
                mesh.lower_left = (-1.0, -1.0, -1.0)
                mesh.upper_right = (1.0, 1.0, 1.0)
                mesh.dimension = (1, 1, 1)
            else:
                cube = -openmc.model.RectangularParallelepiped(
                    -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, boundary_type='vacuum')
                cells = [openmc.Cell(region=cube)]
                mesh = openmc.UnstructuredMesh(
                    mesh_path, library=library.removeprefix('xdg-'))
                if library.startswith('xdg-'):
                    mesh.interface = 'xdg'
            make_comparison_model(cells, 0.5, mesh, source_shape,
                                  source_point=source_point
                                  ).export_to_model_xml()

        if config['build_inputs']:
            return
        returncode, output = run_openmc(path)
        assert returncode == 0, output

    with openmc.StatePoint(statepoint_path(tmp_path / 'False')) as sp:
        tally = sp.get_tally(name='mesh flux')
        flux = tally.mean.ravel()
        centroids = tally.find_filter(openmc.MeshFilter).mesh.centroids
        volumes = tally.find_filter(openmc.MeshFilter).mesh.volumes
    cell_flux = fluxes(tmp_path / 'True', 'cell flux')['cell flux']

    assert cell_flux.sum() > 0.0
    for i, tet in enumerate(tets):
        nodes = points[tet]
        tet_bin = np.argmin(np.linalg.norm(
            centroids - nodes.mean(axis=0), axis=1))
        assert flux[tet_bin] == pytest.approx(cell_flux[i], rel=1e-5)
        assert volumes[tet_bin] == pytest.approx(
            abs(np.linalg.det(nodes[1:] - nodes[0])) / 6.0)
    if graded:
        assert volumes.max() > 2.0 * volumes.min()


@pytest.mark.parametrize("ww_method", ['magic', 'fw_cadis'])
def test_random_ray_weight_windows_normalized_tallies(ww_method, tmp_path):
    """Whether or not flux tallies are divided by volume, weight windows are
    the same on a mesh whose elements have different volumes"""
    if not enabled('xdg-moab'):
        pytest.skip("xdg-moab unstructured meshes are not enabled.")

    points, tets = graded_unit_cube_tets()
    mesh_path = tmp_path / 'graded.vtk'
    write_tets(mesh_path, points, tets)

    bounds = []
    for normalized in (False, True):
        path = tmp_path / str(normalized)
        path.mkdir()
        openmc.reset_auto_ids()
        with change_directory(path):
            model = make_model('xdg-moab', mesh_path, 'tally_and_ww',
                               ww_method=ww_method)
            model.settings.random_ray[
                'volume_normalized_flux_tallies'] = normalized
            # Enough rays for the sampled volumes of the elements to be close
            model.settings.particles = 2000
            model.export_to_model_xml()

        if config['build_inputs']:
            return
        returncode, output = run_openmc(path)
        assert returncode == 0, output
        bounds.append(openmc.WeightWindowsList.from_hdf5(
            path / 'weight_windows.h5')[0].lower_ww_bounds)

    assert np.all(bounds[0] > 0.0)
    assert bounds[0] == pytest.approx(bounds[1], rel=0.1)


def test_random_ray_exodus_output_normalized_tallies(tmp_path):
    """The flux per volume in the Exodus file of a libMesh tally mesh does not
    depend on whether or not flux tallies are divided by volume"""
    if not enabled('xdg-libmesh'):
        pytest.skip("xdg-libmesh unstructured meshes are not enabled.")

    mesh_path = (Path(__file__).parents[1] / 'unstructured_mesh' /
                 MESH_FILES['libmesh'])
    fluxes_in_file = []
    for normalized in (False, True):
        path = tmp_path / str(normalized)
        path.mkdir()
        openmc.reset_auto_ids()
        with change_directory(path):
            tally_mesh = openmc.UnstructuredMesh(mesh_path, library='libmesh')
            model = make_model('xdg-libmesh', mesh_path, 'tally',
                               tally_mesh=tally_mesh)
            model.settings.random_ray[
                'volume_normalized_flux_tallies'] = normalized
            model.settings.particles = 500
            model.export_to_model_xml()

        if config['build_inputs']:
            return
        returncode, output = run_openmc(path)
        assert returncode == 0, output
        exodus_path = next(path.glob('tally_*.e'))
        if h5py.is_hdf5(exodus_path):
            pytest.skip("Exodus file is NetCDF-4, which scipy cannot read")
        with netcdf_file(exodus_path, mmap=False) as exodus:
            names = [b''.join(name).decode().strip('\x00')
                     for name in exodus.variables['name_nod_var'].data]
            variable = f'vals_nod_var{names.index("flux_total_mean") + 1}'
            fluxes_in_file.append(exodus.variables[variable].data[-1].copy())

    assert np.all(fluxes_in_file[0] > 0.0)
    assert fluxes_in_file[0] == pytest.approx(fluxes_in_file[1], rel=0.1)


@pytest.mark.parametrize("library,point", [
    ('xdg-moab', (0.1, 0.2, 0.3)),
    ('xdg-libmesh', (0.1, 0.2, 0.3)),
    # On a node and on an edge of the mesh, and on a face of its tets
    ('xdg-moab', (0.0, 0.0, 0.0)),
    ('xdg-moab', (0.0, 0.0, 0.25)),
    ('xdg-moab', (0.1, 0.1, 0.3)),
    ('xdg-libmesh', (0.0, 0.0, 0.0)),
    ('xdg-libmesh', (0.0, 0.0, 0.25)),
    ('xdg-libmesh', (0.1, 0.1, 0.3)),
])
def test_random_ray_point_source(library, point, tmp_path):
    """A point source finds a mesh bin, wherever in the mesh it is"""
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


def check_weight_windows(path, all_bins=False):
    """Valid bounds, in all bins or else -1 for a bin without a usable flux"""
    bounds = openmc.WeightWindowsList.from_hdf5(
        path / 'weight_windows.h5')[0].lower_ww_bounds
    assert np.any(bounds > 0.0)
    assert np.all((bounds > 0.0) | (not all_bins and bounds == -1.0))


def statepoint_path(path):
    """The statepoint of the run in path"""
    return next(Path(path).glob('statepoint.*.h5'))


def mesh_volumes(path):
    """The volumes of the elements of the tally mesh of the run in path"""
    with openmc.StatePoint(statepoint_path(path)) as sp:
        return sp.get_tally(name='tet flux').find_filter(
            openmc.MeshFilter).mesh.volumes


def run_openmc(path):
    """Run the model in path, returning the return code and the output"""
    args = [config['exe']]
    if config['mpi']:
        args = [config['mpiexec'], '-n', config['mpi_np']] + args
    # To a file and within a time limit: a mistake here can mean endless output
    with open(path / 'output.txt', 'w+') as output:
        result = subprocess.run(args, cwd=path, stdout=output,
                                stderr=subprocess.STDOUT, timeout=120)
        output.seek(0)
        # (messages are wrapped)
        return result.returncode, ' '.join(output.read().split())


def fluxes(path, *names):
    """The flux of the tallies with the given names of the run in path"""
    with openmc.StatePoint(statepoint_path(path)) as sp:
        return {name: sp.get_tally(name=name).mean.ravel() for name in names}


def check_tally_sums(path):
    """The flux on the mesh of the run in path adds up to the flux in the
    cell"""
    with openmc.StatePoint(statepoint_path(path)) as sp:
        cell_flux = sp.get_tally(name='cell flux').mean.sum()
        mesh_flux = sp.get_tally(name='tet flux').mean.sum()
    assert cell_flux > 0.0
    assert mesh_flux == pytest.approx(cell_flux, rel=1e-10)
