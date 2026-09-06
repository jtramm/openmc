"""Regression variants for tally mesh subdivision on DAGMC-enabled builds.

The unstructured variant tallies a tetrahedral MOAB mesh that subdivides
the source regions of a uniform cube, and the dagmc variant cuts curved
DAGMC cells with a regular tally mesh. Reference results are compared
value by value, so any behavioral drift fails regardless of tally
variance.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import openmc
import openmc.lib
import openmc.mgxs
from openmc.utility_funcs import change_directory
import pytest

from tests.testing_harness import TolerantPyAPITestHarness

pytestmark = pytest.mark.skipif(
    not openmc.lib._dagmc_enabled(),
    reason="DAGMC (and its MOAB mesh support) is not enabled.")


class MGXSTestHarness(TolerantPyAPITestHarness):
    def _cleanup(self):
        super()._cleanup()
        for f in ('mgxs.h5', 'tets.h5m', 'dagmc.h5m'):
            if os.path.exists(f):
                os.remove(f)


def build_unstructured():
    h5m_src = Path(__file__).parent.parent / 'external_moab' \
        / 'test_mesh_tets.h5m'
    shutil.copyfile(h5m_src, 'tets.h5m')

    openmc.reset_auto_ids()
    model = openmc.Model()
    groups = openmc.mgxs.EnergyGroups(group_edges=[1e-5, 20.0e6])
    d = openmc.XSdata('mat', groups)
    d.order = 0
    d.set_total([1.0])
    d.set_absorption([0.5])
    d.set_scatter_matrix(np.array([[[0.5]]]))
    lib = openmc.MGXSLibrary(groups)
    lib.add_xsdatas([d])
    lib.export_to_hdf5('mgxs.h5')
    m = openmc.Material(name='mat')
    m.set_density('macro', 1.0)
    m.add_macroscopic(openmc.Macroscopic('mat'))
    model.materials = openmc.Materials([m])
    model.materials.cross_sections = 'mgxs.h5'
    box = openmc.model.RectangularParallelepiped(
        -10, 10, -10, 10, -10, 10, boundary_type='reflective')
    cell = openmc.Cell(fill=m, region=-box)
    model.geometry = openmc.Geometry([cell])
    s = model.settings
    s.energy_mode = 'multi-group'
    s.particles = 120
    s.inactive = 5
    s.batches = 15
    s.seed = 1
    s.run_mode = 'fixed source'
    s.source = openmc.IndependentSource(
        space=openmc.stats.Box((-10,) * 3, (10,) * 3),
        energy=openmc.stats.Discrete([1.0e6], [1.0]),
        constraints={'domains': [cell]})
    srmesh = openmc.RegularMesh()
    srmesh.lower_left = (-10,) * 3
    srmesh.upper_right = (10,) * 3
    srmesh.dimension = (4, 4, 4)
    s.random_ray = {
        'distance_inactive': 40.0,
        'distance_active': 80.0,
        'ray_source': openmc.IndependentSource(
            space=openmc.stats.Box((-10,) * 3, (10,) * 3)),
        'source_shape': 'flat',
        'source_region_meshes': [(srmesh, [model.geometry.root_universe])],
    }
    t = openmc.Tally(name='tets')
    t.filters = [openmc.MeshFilter(
        openmc.UnstructuredMesh('tets.h5m', library='moab'))]
    t.scores = ['flux']
    ref = openmc.Tally(name='cellref')
    ref.filters = [openmc.CellFilter(cell)]
    ref.scores = ['flux']
    model.tallies = openmc.Tallies([t, ref])
    return model


def build_dagmc():
    h5m_src = Path(__file__).parent.parent.parent / 'unit_tests' / 'dagmc' \
        / 'dagmc.h5m'
    shutil.copyfile(h5m_src, 'dagmc.h5m')
    E = 25.0

    openmc.reset_auto_ids()
    model = openmc.Model()
    groups = openmc.mgxs.EnergyGroups(group_edges=[1e-5, 20.0e6])
    lib = openmc.MGXSLibrary(groups)
    for name, st, c in (('fuelxs', 0.5, 0.4), ('waterxs', 1.0, 0.8)):
        d = openmc.XSdata(name, groups)
        d.order = 0
        d.set_total([st])
        d.set_absorption([st * (1 - c)])
        d.set_scatter_matrix(np.array([[[st * c]]]))
        lib.add_xsdatas([d])
    lib.export_to_hdf5('mgxs.h5')

    fuel = openmc.Material(name='no-void fuel')
    fuel.set_density('macro', 1.0)
    fuel.add_macroscopic(openmc.Macroscopic('fuelxs'))
    fuel.id = 40
    water = openmc.Material(name='water')
    water.set_density('macro', 1.0)
    water.add_macroscopic(openmc.Macroscopic('waterxs'))
    water.id = 41
    model.materials = openmc.Materials([fuel, water])
    model.materials.cross_sections = 'mgxs.h5'

    dag = openmc.DAGMCUniverse('dagmc.h5m')
    model.geometry = openmc.Geometry(dag)

    s = model.settings
    s.energy_mode = 'multi-group'
    s.particles = 250
    s.inactive = 5
    s.batches = 15
    s.seed = 1
    s.run_mode = 'fixed source'
    s.source = openmc.IndependentSource(
        space=openmc.stats.Box((-E,) * 3, (E,) * 3),
        energy=openmc.stats.Discrete([1.0e6], [1.0]),
        constraints={'domains': [fuel]})
    s.random_ray = {
        'distance_inactive': 100.0,
        'distance_active': 200.0,
        'ray_source': openmc.IndependentSource(
            space=openmc.stats.Box((-E,) * 3, (E,) * 3)),
        'source_shape': 'flat',
    }

    def rmesh(dim):
        mm = openmc.RegularMesh()
        mm.lower_left = (-E,) * 3
        mm.upper_right = (E,) * 3
        mm.dimension = dim
        return mm

    t7 = openmc.Tally(name='m7')
    t7.filters = [openmc.MeshFilter(rmesh((7, 7, 7)))]
    t7.scores = ['flux']
    t1 = openmc.Tally(name='m1')
    t1.filters = [openmc.MeshFilter(rmesh((1, 1, 1)))]
    t1.scores = ['flux']
    model.tallies = openmc.Tallies([t7, t1])
    return model


@pytest.mark.parametrize("variant", ["unstructured", "dagmc"])
def test_random_ray_tally_subdivide_dagmc(variant):
    with change_directory(variant):
        if variant == 'unstructured':
            model = build_unstructured()
        else:
            model = build_dagmc()
        harness = MGXSTestHarness('statepoint.15.h5', model)
        harness.main()
