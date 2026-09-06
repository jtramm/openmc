"""Contract tests for random ray tally meshes that subdivide source
regions.

The behavioral coverage for this feature lives in the
`random_ray_tally_subdivide` regression battery, which pins each
configuration against stored reference results. The tests here assert
the properties reference results cannot express: the abort contract for
tallies with multiple mesh filters, and run-to-run determinism.
"""

import os

import numpy as np
import pytest

import openmc
import openmc.mgxs

L = 10.0


def uniform_model(tmp_path):
    """Reflective cube of uniform material with 2 cm source regions."""
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
    lib.export_to_hdf5(str(tmp_path / 'mgxs.h5'))

    m = openmc.Material(name='mat')
    m.set_density('macro', 1.0)
    m.add_macroscopic(openmc.Macroscopic('mat'))
    model.materials = openmc.Materials([m])
    model.materials.cross_sections = str(tmp_path / 'mgxs.h5')

    box = openmc.model.RectangularParallelepiped(
        0, L, 0, L, 0, L, boundary_type='reflective')
    cell = openmc.Cell(fill=m, region=-box)
    model.geometry = openmc.Geometry([cell])

    s = model.settings
    s.energy_mode = 'multi-group'
    s.particles = 150
    s.inactive = 5
    s.batches = 15
    s.seed = 1
    s.run_mode = 'fixed source'
    s.source = openmc.IndependentSource(
        space=openmc.stats.Box((0, 0, 0), (L, L, L)),
        energy=openmc.stats.Discrete([1.0e6], [1.0]),
        constraints={'domains': [cell]})

    srmesh = openmc.RegularMesh()
    srmesh.lower_left = (0, 0, 0)
    srmesh.upper_right = (L, L, L)
    srmesh.dimension = (5, 5, 5)

    s.random_ray = {
        'distance_inactive': 30.0,
        'distance_active': 200.0,
        'ray_source': openmc.IndependentSource(
            space=openmc.stats.Box((0, 0, 0), (L, L, L))),
        'source_shape': 'flat',
        'source_region_meshes': [(srmesh, [model.geometry.root_universe])],
    }
    return model


def tally_mesh(dim, lo=(0, 0, 0), hi=(L, L, L)):
    mm = openmc.RegularMesh()
    mm.lower_left = lo
    mm.upper_right = hi
    mm.dimension = dim
    return mm


def test_multiple_mesh_filters_fatal(tmp_path):
    """A tally with two mesh filters over subdivided source regions must
    abort with a clear error rather than misattribute scores."""
    model = uniform_model(tmp_path)
    t = openmc.Tally(name='twomesh')
    t.filters = [openmc.MeshFilter(tally_mesh((3, 3, 3))),
                 openmc.MeshFilter(tally_mesh((4, 4, 4)))]
    t.scores = ['flux']
    model.tallies = openmc.Tallies([t])
    with pytest.raises(RuntimeError, match='multiple mesh filters'):
        model.run(cwd=str(tmp_path))


def test_multiple_mesh_filters_edge_fatal(tmp_path):
    """The two-mesh-filter abort must also fire when one mesh's edge cuts
    source regions and their midpoints fall outside it, rather than
    silently dropping those regions from the tally."""
    model = uniform_model(tmp_path)
    t = openmc.Tally(name='twomesh')
    t.filters = [openmc.MeshFilter(tally_mesh((5, 5, 5))),
                 openmc.MeshFilter(tally_mesh((1, 1, 1), hi=(3.0, L, L)))]
    t.scores = ['flux']
    model.tallies = openmc.Tallies([t])
    with pytest.raises(RuntimeError, match='multiple mesh filters'):
        model.run(cwd=str(tmp_path))


def test_determinism(tmp_path):
    """Repeat runs with a subdividing mesh must be bitwise identical on
    one thread and agree to accumulation-order rounding with threading,
    matching the solver's pre-existing reproducibility contract."""
    for threads, bitwise in ((1, True), (4, False)):
        vals = []
        for rep in (1, 2):
            wd = tmp_path / f't{threads}_{rep}'
            wd.mkdir()
            model = uniform_model(wd)
            t = openmc.Tally(name='m3')
            t.filters = [openmc.MeshFilter(tally_mesh((3, 3, 3)))]
            t.scores = ['flux']
            model.tallies = openmc.Tallies([t])
            os.environ['OMP_NUM_THREADS'] = str(threads)
            try:
                sp = model.run(cwd=str(wd))
            finally:
                os.environ.pop('OMP_NUM_THREADS', None)
            with openmc.StatePoint(sp) as f:
                vals.append(f.get_tally(name='m3').mean.ravel().copy())
        if bitwise:
            assert np.array_equal(vals[0], vals[1])
        else:
            assert np.abs((vals[0] - vals[1]) / vals[0]).max() < 1e-12
