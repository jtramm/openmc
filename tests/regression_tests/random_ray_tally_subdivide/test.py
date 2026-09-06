"""Regression battery for tally meshes that subdivide source regions.

Each variant pins a configuration of the score apportioning machinery
against stored reference results, compared value by value at 1e-6
relative tolerance, so any behavioral drift fails regardless of tally
variance. Numerical soundness itself was demonstrated by convergence
studies in the pull request; run lengths here are short by design.
"""

import os

import numpy as np
import openmc
import openmc.mgxs
from openmc.utility_funcs import change_directory
from openmc.examples import random_ray_three_region_cube
import pytest

from tests.testing_harness import TolerantPyAPITestHarness

L = 10.0


class MGXSTestHarness(TolerantPyAPITestHarness):
    def _cleanup(self):
        super()._cleanup()
        f = 'mgxs.h5'
        if os.path.exists(f):
            os.remove(f)


def build_mgxs(fissile=False, ngroups=1):
    if ngroups == 1:
        groups = openmc.mgxs.EnergyGroups(group_edges=[1e-5, 20.0e6])
        d = openmc.XSdata('mat', groups)
        d.order = 0
        d.set_total([1.0])
        d.set_absorption([0.5])
        d.set_scatter_matrix(np.array([[[0.5]]]))
        if fissile:
            d.set_fission([0.5])
            d.set_nu_fission([0.75])
            d.set_chi([1.0])
    else:
        groups = openmc.mgxs.EnergyGroups(group_edges=[1e-5, 1.0e3, 20.0e6])
        d = openmc.XSdata('mat', groups)
        d.order = 0
        d.set_total([1.0, 1.0])
        d.set_absorption([0.5, 0.7])
        d.set_scatter_matrix(np.array([[[0.3], [0.2]], [[0.0], [0.3]]]))
    lib = openmc.MGXSLibrary(groups)
    lib.add_xsdatas([d])
    lib.export_to_hdf5('mgxs.h5')


def uniform_model(fissile=False, shape='flat', sr_dim=(5, 5, 5), ngroups=1,
                  two_cells=False):
    """Reflective cube of uniform material with 2 cm source regions."""
    openmc.reset_auto_ids()
    model = openmc.Model()
    build_mgxs(fissile, ngroups)

    m = openmc.Material(name='mat')
    m.set_density('macro', 1.0)
    m.add_macroscopic(openmc.Macroscopic('mat'))
    model.materials = openmc.Materials([m])
    model.materials.cross_sections = 'mgxs.h5'

    box = openmc.model.RectangularParallelepiped(
        0, L, 0, L, 0, L, boundary_type='reflective')
    if two_cells:
        plane = openmc.XPlane(5.0)
        cells = [openmc.Cell(fill=m, region=-box & -plane),
                 openmc.Cell(fill=m, region=-box & +plane)]
    else:
        cells = [openmc.Cell(fill=m, region=-box)]
    model.geometry = openmc.Geometry(cells)

    s = model.settings
    s.energy_mode = 'multi-group'
    s.particles = 300
    s.inactive = 5
    s.batches = 15
    s.seed = 1
    if fissile:
        s.run_mode = 'eigenvalue'
    else:
        s.run_mode = 'fixed source'
        s.source = openmc.IndependentSource(
            space=openmc.stats.Box((0, 0, 0), (L, L, L)),
            energy=openmc.stats.Discrete([1.0e6], [1.0]),
            constraints={'domains': cells})

    srmesh = openmc.RegularMesh()
    srmesh.lower_left = (0, 0, 0)
    srmesh.upper_right = (L, L, L)
    srmesh.dimension = sr_dim

    s.random_ray = {
        'distance_inactive': 30.0,
        'distance_active': 200.0,
        'ray_source': openmc.IndependentSource(
            space=openmc.stats.Box((0, 0, 0), (L, L, L))),
        'source_shape': shape,
        'source_region_meshes': [(srmesh, [model.geometry.root_universe])],
    }
    return model, cells, srmesh


def tally_mesh(dim, lo=(0, 0, 0), hi=(L, L, L)):
    mm = openmc.RegularMesh()
    mm.lower_left = lo
    mm.upper_right = hi
    mm.dimension = dim
    return mm


def mesh_flux_tally(mesh, name):
    t = openmc.Tally(name=name)
    t.filters = [openmc.MeshFilter(mesh)]
    t.scores = ['flux']
    return t


def cell_ref_tally(cells):
    t = openmc.Tally(name='cellref')
    t.filters = [openmc.CellFilter(cells)]
    t.scores = ['flux']
    return t


def build_variant(variant):
    if variant == 'hetero':
        # The three-region cube example (with a void region) under a
        # 12x12x12 source region overlay and a subdividing 5x5x5 tally
        openmc.reset_auto_ids()
        model = random_ray_three_region_cube()
        mesh = openmc.RegularMesh()
        mesh.lower_left = (0.0, 0.0, 0.0)
        mesh.upper_right = (30.0, 30.0, 30.0)
        mesh.dimension = (12, 12, 12)
        model.settings.random_ray['source_region_meshes'] = [
            (mesh, [model.geometry.root_universe])]
        tmesh = openmc.RegularMesh()
        tmesh.lower_left = (0.0, 0.0, 0.0)
        tmesh.upper_right = (30.0, 30.0, 30.0)
        tmesh.dimension = (5, 5, 5)
        model.tallies.append(mesh_flux_tally(tmesh, 'subdivided flux'))
        model.settings.inactive = 5
        model.settings.batches = 15
        return model

    if variant == 'shifted':
        # 1D flux gradient with aligned and half-region-shifted bins
        model, cells, _ = uniform_model(sr_dim=(10, 1, 1))
        model.settings.source = openmc.IndependentSource(
            space=openmc.stats.Box((0, 0, 0), (1.0, L, L)),
            energy=openmc.stats.Discrete([1.0e6], [1.0]),
            constraints={'domains': cells})
        model.tallies = openmc.Tallies([
            mesh_flux_tally(tally_mesh((10, 1, 1)), 'aligned'),
            mesh_flux_tally(
                tally_mesh((9, 1, 1), lo=(0.5, 0, 0), hi=(9.5, L, L)),
                'shifted'),
        ])
        return model

    if variant == 'eigenvalue':
        model, cells, _ = uniform_model(fissile=True)
        t = openmc.Tally(name='scores')
        t.filters = [openmc.MeshFilter(tally_mesh((3, 3, 3)))]
        t.scores = ['flux', 'fission', 'nu-fission', 'total']
        model.tallies = openmc.Tallies([t])
        return model

    if variant == 'own_mesh_cellfilter':
        # Tally on the source region mesh itself, restricted by a cell
        # filter excluding half the domain (the point-seeded own-mesh case)
        model, cells, srmesh = uniform_model(two_cells=True)
        t = openmc.Tally(name='half')
        t.filters = [openmc.MeshFilter(srmesh), openmc.CellFilter(cells[0])]
        t.scores = ['flux']
        model.tallies = openmc.Tallies([t, cell_ref_tally(cells)])
        return model

    if variant in ('energy_mesh_first', 'energy_energy_first'):
        model, cells, _ = uniform_model(ngroups=2)
        efilt = openmc.EnergyFilter([1e-5, 1.0e3, 20.0e6])
        mfilt = openmc.MeshFilter(tally_mesh((3, 3, 3)))
        t = openmc.Tally(name='me')
        if variant == 'energy_mesh_first':
            t.filters = [mfilt, efilt]
        else:
            t.filters = [efilt, mfilt]
        t.scores = ['flux']
        model.tallies = openmc.Tallies([t])
        return model

    # Remaining variants share the one-cell uniform model
    shape = 'linear' if variant == 'linear' else 'flat'
    model, cells, _ = uniform_model(shape=shape)
    tallies = []
    if variant in ('two_mesh', 'linear'):
        tallies = [mesh_flux_tally(tally_mesh((3, 3, 3)), 'm3'),
                   mesh_flux_tally(tally_mesh((4, 4, 4)), 'm4')]
    elif variant == 'three_mesh':
        tallies = [mesh_flux_tally(tally_mesh((3, 3, 3)), 'm3'),
                   mesh_flux_tally(tally_mesh((4, 4, 4)), 'm4'),
                   mesh_flux_tally(tally_mesh((7, 7, 7)), 'm7')]
    elif variant == 'edge':
        tallies = [mesh_flux_tally(
            tally_mesh((1, 1, 1), hi=(3.0, L, L)), 'left')]
    elif variant == 'interior':
        tallies = [mesh_flux_tally(
            tally_mesh((1, 1, 1), lo=(4.2, 4.2, 4.2), hi=(5.8, 5.8, 5.8)),
            'inner')]
    elif variant == 'volnorm':
        model.settings.random_ray['volume_normalized_flux_tallies'] = True
        tallies = [mesh_flux_tally(tally_mesh((3, 3, 3)), 'm3')]
    elif variant == 'rotated':
        mf = openmc.MeshFilter(tally_mesh(
            (4, 1, 1), lo=(-10.0, -10.0, -10.0), hi=(10.0, 10.0, 10.0)))
        mf.rotation = (0.0, 0.0, 90.0)
        t = openmc.Tally(name='rot')
        t.filters = [mf]
        t.scores = ['flux']
        tallies = [t]
    elif variant == 'translated':
        mf = openmc.MeshFilter(tally_mesh((1, 1, 1)))
        mf.translation = (5.0, 0.0, 0.0)
        t = openmc.Tally(name='shifted')
        t.filters = [mf]
        t.scores = ['flux']
        tallies = [t]
    elif variant == 'rectilinear':
        mm = openmc.RectilinearMesh()
        mm.x_grid = [0.0, 1.0, 3.5, 10.0]
        mm.y_grid = [0.0, L]
        mm.z_grid = [0.0, L]
        tallies = [mesh_flux_tally(mm, 'rect')]
    elif variant == 'cylindrical':
        mm = openmc.CylindricalMesh(
            r_grid=[0.0, 1.5, 3.5],
            phi_grid=[0.0, 2 * np.pi],
            z_grid=[0.0, 5.0, 10.0],
            origin=(5.0, 5.0, 0.0))
        tallies = [mesh_flux_tally(mm, 'cyl')]
    elif variant == 'spherical':
        mm = openmc.SphericalMesh(
            r_grid=[0.0, 2.0, 4.0], origin=(5.0, 5.0, 5.0))
        tallies = [mesh_flux_tally(mm, 'sph')]
    elif variant == 'adjoint':
        model.settings.random_ray['adjoint'] = True
        tallies = [mesh_flux_tally(tally_mesh((3, 3, 3)), 'm3')]
    elif variant == 'starved':
        # Complementary partial meshes at 8 rays per batch, so straddling
        # regions routinely have their overlap with one of the meshes
        # still unsampled early in the run. Pins that a region's tasks for
        # a mesh are created once its first in-mesh evidence appears,
        # instead of the overlap being left permanently unscored.
        model.settings.particles = 8
        tallies = [
            mesh_flux_tally(tally_mesh((1, 1, 1), hi=(L, 2.5, L)), 'lowy'),
            mesh_flux_tally(tally_mesh((1, 1, 1), lo=(0, 2.5, 0)), 'highy')]
    elif variant == 'starved_transient':
        # The same complementary meshes with no inactive batches, so
        # scoring begins while some straddling regions still lack traced
        # evidence for one of the meshes. Pins that such a region scores
        # nothing for the evidence-less mesh, the limit of the track
        # length weights, rather than scoring whole into both meshes at
        # once and double counting.
        model.settings.particles = 4
        model.settings.inactive = 0
        model.settings.batches = 6
        model.settings.seed = 3
        tallies = [
            mesh_flux_tally(tally_mesh((1, 1, 1), hi=(L, 2.5, L)), 'lowy'),
            mesh_flux_tally(tally_mesh((1, 1, 1), lo=(0, 2.5, 0)), 'highy')]
    model.tallies = openmc.Tallies(tallies + [cell_ref_tally(cells)])
    return model


VARIANTS = ['hetero', 'two_mesh', 'linear', 'three_mesh', 'edge', 'interior',
            'shifted', 'eigenvalue', 'volnorm', 'rotated', 'translated',
            'rectilinear', 'cylindrical', 'spherical', 'energy_mesh_first',
            'energy_energy_first', 'own_mesh_cellfilter', 'adjoint',
            'starved', 'starved_transient']


@pytest.mark.parametrize("variant", VARIANTS)
def test_random_ray_tally_subdivide(variant):
    with change_directory(variant):
        model = build_variant(variant)
        sp_name = f'statepoint.{model.settings.batches}.h5'
        harness = MGXSTestHarness(sp_name, model)
        harness.main()
