"""Tests for random ray tally meshes that subdivide source regions.

Most tests use a uniform infinite medium (a reflective cube with a uniform
source), where the converged scalar flux is spatially uniform. Every tally
mesh bin must then score in proportion to its volume, no matter how the
mesh cuts across source regions, which gives an exact oracle for the
score apportioning.
"""

import os

import numpy as np
import pytest

import openmc
import openmc.mgxs

L = 10.0


def build_mgxs(path, fissile):
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
    lib = openmc.MGXSLibrary(groups)
    lib.add_xsdatas([d])
    lib.export_to_hdf5(path)


def uniform_model(tmp_path, fissile=False, shape='flat', sr_dim=(5, 5, 5)):
    """Reflective cube of uniform material with 2 cm source regions."""
    openmc.reset_auto_ids()
    model = openmc.Model()
    mgxs_path = str(tmp_path / 'mgxs.h5')
    build_mgxs(mgxs_path, fissile)

    m = openmc.Material(name='mat')
    m.set_density('macro', 1.0)
    m.add_macroscopic(openmc.Macroscopic('mat'))
    model.materials = openmc.Materials([m])
    model.materials.cross_sections = mgxs_path

    box = openmc.model.RectangularParallelepiped(
        0, L, 0, L, 0, L, boundary_type='reflective')
    cell = openmc.Cell(fill=m, region=-box)
    model.geometry = openmc.Geometry([cell])

    s = model.settings
    s.energy_mode = 'multi-group'
    s.particles = 500
    s.inactive = 40
    s.batches = 160
    s.seed = 1
    if fissile:
        s.run_mode = 'eigenvalue'
    else:
        s.run_mode = 'fixed source'
        s.source = openmc.IndependentSource(
            space=openmc.stats.Box((0, 0, 0), (L, L, L)),
            energy=openmc.stats.Discrete([1.0e6], [1.0]),
            constraints={'domains': [cell]})

    srmesh = openmc.RegularMesh()
    srmesh.lower_left = (0, 0, 0)
    srmesh.upper_right = (L, L, L)
    srmesh.dimension = sr_dim

    s.random_ray = {
        'distance_inactive': 30.0,
        'distance_active': 300.0,
        'ray_source': openmc.IndependentSource(
            space=openmc.stats.Box((0, 0, 0), (L, L, L))),
        'source_shape': shape,
        'source_region_meshes': [(srmesh, [model.geometry.root_universe])],
    }
    return model, cell


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


def run_and_read(model, tmp_path, names):
    sp = model.run(cwd=str(tmp_path))
    out = {}
    with openmc.StatePoint(sp) as f:
        for name in names:
            out[name] = f.get_tally(name=name).mean.ravel().copy()
    return out


def assert_uniform(vals, tol):
    worst = np.abs(vals / vals.mean() - 1.0).max()
    assert worst < tol, f'worst bin deviation {100*worst:.2f}%'


@pytest.mark.parametrize('shape', ['flat', 'linear'])
def test_subdividing_meshes_uniform(tmp_path, shape):
    """Two non-conformal tally meshes at once over 2 cm source regions.

    The 3x3x3 mesh has bin planes inside source regions, and the 4x4x4
    mesh does as well, so both meshes subdivide source regions. In the
    uniform medium every bin of each mesh must score equally, and each
    mesh must conserve the total flux reported by a cell tally.
    """
    model, cell = uniform_model(tmp_path, shape=shape)
    model.tallies = openmc.Tallies([
        mesh_flux_tally(tally_mesh((3, 3, 3)), 'm3'),
        mesh_flux_tally(tally_mesh((4, 4, 4)), 'm4'),
    ])
    ref = openmc.Tally(name='cellref')
    ref.filters = [openmc.CellFilter(cell)]
    ref.scores = ['flux']
    model.tallies.append(ref)

    out = run_and_read(model, tmp_path, ['m3', 'm4', 'cellref'])
    assert_uniform(out['m3'], 0.02)
    assert_uniform(out['m4'], 0.02)
    total = out['cellref'][0]
    assert abs(out['m3'].sum() - total) / total < 1e-9
    assert abs(out['m4'].sum() - total) / total < 1e-9


def test_partial_mesh_edge_straddle(tmp_path):
    """A one-bin mesh covering only x < 3 cm.

    Its edge at x = 3 cuts through the source regions spanning x in
    [2, 4], so those regions must contribute only the half of their flux
    lying inside the mesh. The bin must therefore report 30% of the
    domain total.
    """
    model, cell = uniform_model(tmp_path)
    model.tallies = openmc.Tallies([
        mesh_flux_tally(tally_mesh((1, 1, 1), hi=(3.0, L, L)), 'left'),
    ])
    ref = openmc.Tally(name='cellref')
    ref.filters = [openmc.CellFilter(cell)]
    ref.scores = ['flux']
    model.tallies.append(ref)

    out = run_and_read(model, tmp_path, ['left', 'cellref'])
    frac = out['left'][0] / out['cellref'][0]
    assert abs(frac - 0.3) < 0.01


def test_shifted_bins_match_model_prediction(tmp_path):
    """Tally bins shifted half a source region against a flux gradient.

    A 1D slab with the source confined to x < 1 cm produces a decaying
    flux. Source regions are 1 cm slabs. A second tally mesh with 1 cm
    bins shifted by 0.5 cm covers halves of two source regions per bin,
    so under the flat source model each shifted bin must equal the
    average of the two aligned bins it overlaps.
    """
    model, cell = uniform_model(tmp_path, sr_dim=(10, 1, 1))
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Box((0, 0, 0), (1.0, L, L)),
        energy=openmc.stats.Discrete([1.0e6], [1.0]),
        constraints={'domains': [cell]})
    model.tallies = openmc.Tallies([
        mesh_flux_tally(tally_mesh((10, 1, 1)), 'aligned'),
        mesh_flux_tally(
            tally_mesh((9, 1, 1), lo=(0.5, 0, 0), hi=(9.5, L, L)),
            'shifted'),
    ])
    out = run_and_read(model, tmp_path, ['aligned', 'shifted'])
    predicted = 0.5 * (out['aligned'][:-1] + out['aligned'][1:])
    rel = np.abs(out['shifted'] - predicted) / predicted
    assert rel.max() < 0.01


def test_eigenvalue_scores(tmp_path):
    """Eigenvalue mode with a subdividing mesh and reaction scores.

    The uniform fissile medium has an exactly known k of 1.5, and every
    score type must be uniform across the subdividing mesh's bins.
    """
    model, cell = uniform_model(tmp_path, fissile=True)
    t = openmc.Tally(name='m3')
    t.filters = [openmc.MeshFilter(tally_mesh((3, 3, 3)))]
    t.scores = ['flux', 'fission', 'nu-fission', 'total']
    model.tallies = openmc.Tallies([t])

    sp = model.run(cwd=str(tmp_path))
    with openmc.StatePoint(sp) as f:
        assert abs(f.keff.n - 1.5) < 0.005
        tt = f.get_tally(name='m3')
        for sc in ('flux', 'fission', 'nu-fission', 'total'):
            assert_uniform(tt.get_values(scores=[sc]).ravel(), 0.02)


def test_volume_normalized_flux(tmp_path):
    """Volume-normalized flux tallies with a subdividing mesh.

    Every bin of the subdividing mesh must report the same flux density
    as every bin of a conformal mesh.
    """
    model, cell = uniform_model(tmp_path)
    model.settings.random_ray['volume_normalized_flux_tallies'] = True
    model.tallies = openmc.Tallies([
        mesh_flux_tally(tally_mesh((3, 3, 3)), 'm3'),
        mesh_flux_tally(tally_mesh((5, 5, 5)), 'm5'),
    ])
    out = run_and_read(model, tmp_path, ['m3', 'm5'])
    assert_uniform(out['m3'], 1e-6)
    assert_uniform(out['m5'], 1e-6)
    assert abs(out['m3'].mean() / out['m5'].mean() - 1.0) < 1e-6


def test_multiple_mesh_filters_fatal(tmp_path):
    """A tally with two mesh filters over subdivided source regions must
    abort with a clear error rather than misattribute scores."""
    model, cell = uniform_model(tmp_path)
    t = openmc.Tally(name='twomesh')
    t.filters = [openmc.MeshFilter(tally_mesh((3, 3, 3))),
                 openmc.MeshFilter(tally_mesh((4, 4, 4)))]
    t.scores = ['flux']
    model.tallies = openmc.Tallies([t])
    with pytest.raises(RuntimeError, match='multiple mesh filters'):
        model.run(cwd=str(tmp_path))
