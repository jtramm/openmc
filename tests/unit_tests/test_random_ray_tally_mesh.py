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


def build_mgxs(path, fissile, ngroups=1):
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
        assert ngroups == 2 and not fissile
        groups = openmc.mgxs.EnergyGroups(group_edges=[1e-5, 1.0e3, 20.0e6])
        d = openmc.XSdata('mat', groups)
        d.order = 0
        d.set_total([1.0, 1.0])
        d.set_absorption([0.5, 0.7])
        # Within-group scattering plus fast-to-slow downscatter
        d.set_scatter_matrix(
            np.array([[[0.3], [0.2]], [[0.0], [0.3]]]))
    lib = openmc.MGXSLibrary(groups)
    lib.add_xsdatas([d])
    lib.export_to_hdf5(path)


def uniform_model(tmp_path, fissile=False, shape='flat', sr_dim=(5, 5, 5),
                  ngroups=1):
    """Reflective cube of uniform material with 2 cm source regions."""
    openmc.reset_auto_ids()
    model = openmc.Model()
    mgxs_path = str(tmp_path / 'mgxs.h5')
    build_mgxs(mgxs_path, fissile, ngroups)

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


def test_multiple_mesh_filters_edge_fatal(tmp_path):
    """The two-mesh-filter abort must also fire when one mesh's edge cuts
    source regions and their midpoints fall outside it, rather than
    silently dropping those regions from the tally."""
    model, cell = uniform_model(tmp_path)
    t = openmc.Tally(name='twomesh')
    t.filters = [openmc.MeshFilter(tally_mesh((5, 5, 5))),
                 openmc.MeshFilter(tally_mesh((1, 1, 1), hi=(3.0, L, L)))]
    t.scores = ['flux']
    model.tallies = openmc.Tallies([t])
    with pytest.raises(RuntimeError, match='multiple mesh filters'):
        model.run(cwd=str(tmp_path))


def test_rotated_mesh_filter(tmp_path):
    """A rotated mesh filter whose bin planes cut source regions.

    The filter mesh spans [-10, 10] with four bins along its local x axis
    and is rotated 90 degrees about z, so in the lab frame its bin planes
    are y planes, one of which (|y'| = 5) passes through source region
    interiors. The rotated image of the uniform cube covers exactly two
    bins equally, so those two bins must each hold half the total flux and
    the other two must hold none.
    """
    model, cell = uniform_model(tmp_path)
    mf = openmc.MeshFilter(
        tally_mesh((4, 1, 1), lo=(-10.0, -10.0, -10.0), hi=(10.0, 10.0, 10.0)))
    mf.rotation = (0.0, 0.0, 90.0)
    t = openmc.Tally(name='rot')
    t.filters = [mf]
    t.scores = ['flux']
    model.tallies = openmc.Tallies([t])
    ref = openmc.Tally(name='cellref')
    ref.filters = [openmc.CellFilter(cell)]
    ref.scores = ['flux']
    model.tallies.append(ref)

    out = run_and_read(model, tmp_path, ['rot', 'cellref'])
    total = out['cellref'][0]
    vals = np.sort(out['rot'])
    assert abs(out['rot'].sum() - total) / total < 1e-6
    assert vals[0] < 1e-6 * total and vals[1] < 1e-6 * total
    assert abs(vals[2] / total - 0.5) < 0.01
    assert abs(vals[3] / total - 0.5) < 0.01


def test_own_mesh_with_excluding_filter(tmp_path):
    """A tally on the source region mesh itself, restricted by a cell
    filter that excludes part of the domain.

    The excluded regions can never match the tally, which must resolve
    cleanly rather than deferring the tally mapping forever. The included
    half must still score correctly.
    """
    openmc.reset_auto_ids()
    model = openmc.Model()
    build_mgxs(str(tmp_path / 'mgxs.h5'), False)
    m = openmc.Material(name='mat')
    m.set_density('macro', 1.0)
    m.add_macroscopic(openmc.Macroscopic('mat'))
    model.materials = openmc.Materials([m])
    model.materials.cross_sections = str(tmp_path / 'mgxs.h5')

    box = openmc.model.RectangularParallelepiped(
        0, L, 0, L, 0, L, boundary_type='reflective')
    plane = openmc.XPlane(5.0)
    cell_a = openmc.Cell(fill=m, region=-box & -plane)
    cell_b = openmc.Cell(fill=m, region=-box & +plane)
    model.geometry = openmc.Geometry([cell_a, cell_b])

    s = model.settings
    s.energy_mode = 'multi-group'
    s.particles = 500
    s.inactive = 40
    s.batches = 160
    s.seed = 1
    s.run_mode = 'fixed source'
    s.source = openmc.IndependentSource(
        space=openmc.stats.Box((0, 0, 0), (L, L, L)),
        energy=openmc.stats.Discrete([1.0e6], [1.0]),
        constraints={'domains': [cell_a, cell_b]})

    srmesh = openmc.RegularMesh()
    srmesh.lower_left = (0, 0, 0)
    srmesh.upper_right = (L, L, L)
    srmesh.dimension = (5, 5, 5)
    s.random_ray = {
        'distance_inactive': 30.0,
        'distance_active': 300.0,
        'ray_source': openmc.IndependentSource(
            space=openmc.stats.Box((0, 0, 0), (L, L, L))),
        'source_shape': 'flat',
        'source_region_meshes': [(srmesh, [model.geometry.root_universe])],
    }

    t = openmc.Tally(name='half')
    t.filters = [openmc.MeshFilter(srmesh), openmc.CellFilter(cell_a)]
    t.scores = ['flux']
    ref = openmc.Tally(name='aref')
    ref.filters = [openmc.CellFilter(cell_a)]
    ref.scores = ['flux']
    model.tallies = openmc.Tallies([t, ref])

    out = run_and_read(model, tmp_path, ['half', 'aref'])
    total_a = out['aref'][0]
    assert abs(out['half'].sum() - total_a) / total_a < 1e-9


def test_short_inactive_edge_straddle(tmp_path):
    """The partial-coverage edge case with almost no inactive batches.

    With one inactive batch, the piece estimates start from nearly nothing
    and mature during the active phase. Edge-straddling regions must not be
    dropped or grossly misweighted while they do.
    """
    model, cell = uniform_model(tmp_path)
    model.settings.inactive = 1
    model.settings.batches = 121
    model.tallies = openmc.Tallies([
        mesh_flux_tally(tally_mesh((1, 1, 1), hi=(3.0, L, L)), 'left'),
    ])
    ref = openmc.Tally(name='cellref')
    ref.filters = [openmc.CellFilter(cell)]
    ref.scores = ['flux']
    model.tallies.append(ref)

    out = run_and_read(model, tmp_path, ['left', 'cellref'])
    frac = out['left'][0] / out['cellref'][0]
    assert abs(frac - 0.3) < 0.03


def test_three_meshes_at_once(tmp_path):
    """Three non-aligned full-coverage meshes tallied simultaneously.

    Each mesh must independently report uniform bins and conserve the
    total, exercising the per-mesh piece tables side by side.
    """
    model, cell = uniform_model(tmp_path)
    model.tallies = openmc.Tallies([
        mesh_flux_tally(tally_mesh((3, 3, 3)), 'm3'),
        mesh_flux_tally(tally_mesh((4, 4, 4)), 'm4'),
        mesh_flux_tally(tally_mesh((7, 7, 7)), 'm7'),
    ])
    ref = openmc.Tally(name='cellref')
    ref.filters = [openmc.CellFilter(cell)]
    ref.scores = ['flux']
    model.tallies.append(ref)

    out = run_and_read(model, tmp_path, ['m3', 'm4', 'm7', 'cellref'])
    total = out['cellref'][0]
    for name, tol in (('m3', 0.02), ('m4', 0.02), ('m7', 0.04)):
        assert_uniform(out[name], tol)
        assert abs(out[name].sum() - total) / total < 1e-9


def test_adjoint_uniform(tmp_path):
    """Adjoint mode with a subdividing mesh.

    The adjoint workflow runs a forward solve and then an adjoint solve on
    the same domain. Accumulated volumes and moments are regenerated for
    the adjoint phase, while the tally mapping and the piece volume
    fraction estimates carry forward, since both describe static geometry
    shared by the two phases. On the uniform medium the forward flux is
    uniform, so the derived adjoint source and adjoint flux are uniform
    too, and the subdividing mesh's adjoint tally must be uniform and
    conserving.
    """
    model, cell = uniform_model(tmp_path)
    model.settings.random_ray['adjoint'] = True
    model.tallies = openmc.Tallies([
        mesh_flux_tally(tally_mesh((3, 3, 3)), 'm3'),
    ])
    ref = openmc.Tally(name='cellref')
    ref.filters = [openmc.CellFilter(cell)]
    ref.scores = ['flux']
    model.tallies.append(ref)

    out = run_and_read(model, tmp_path, ['m3', 'cellref'])
    assert_uniform(out['m3'], 0.02)
    total = out['cellref'][0]
    assert abs(out['m3'].sum() - total) / total < 1e-9


@pytest.mark.parametrize('mesh_first', [True, False])
def test_mesh_with_energy_filter(tmp_path, mesh_first):
    """A subdividing mesh filter combined with an energy filter.

    With two energy groups the mesh filter's stride in the flattened
    filter index differs from one in one of the two filter orders, so
    this exercises the stride arithmetic that shifts apportioned scores
    to sibling bins. Both groups are spatially uniform (the slow group is
    fed by downscatter), so every mesh bin must be uniform within each
    group and each group must conserve against an energy-filtered cell
    tally.
    """
    model, cell = uniform_model(tmp_path, ngroups=2)
    efilt = openmc.EnergyFilter([1e-5, 1.0e3, 20.0e6])
    mfilt = openmc.MeshFilter(tally_mesh((3, 3, 3)))
    t = openmc.Tally(name='me')
    t.filters = [mfilt, efilt] if mesh_first else [efilt, mfilt]
    t.scores = ['flux']
    ref = openmc.Tally(name='cellref')
    ref.filters = [openmc.CellFilter(cell), openmc.EnergyFilter(
        [1e-5, 1.0e3, 20.0e6])]
    ref.scores = ['flux']
    model.tallies = openmc.Tallies([t, ref])

    out = run_and_read(model, tmp_path, ['me', 'cellref'])
    shape = (27, 2) if mesh_first else (2, 27)
    vals = out['me'].reshape(shape)
    per_group = vals.T if mesh_first else vals
    ref_groups = out['cellref']
    assert ref_groups[0] > 0 and ref_groups[1] > 0
    for g in range(2):
        assert_uniform(per_group[g], 0.02)
        assert abs(per_group[g].sum() - ref_groups[g]) / ref_groups[g] < 1e-9


def test_translated_mesh_filter(tmp_path):
    """A translated mesh filter half covering the domain.

    The mesh spans the cube but the filter translation shifts it by half
    the cube width, so it covers x in [5, 10] in the lab frame, with its
    effective edge cutting through the source regions spanning x in
    [4, 6]. The single bin must report half the domain total.
    """
    model, cell = uniform_model(tmp_path)
    mf = openmc.MeshFilter(tally_mesh((1, 1, 1)))
    mf.translation = (5.0, 0.0, 0.0)
    t = openmc.Tally(name='shifted')
    t.filters = [mf]
    t.scores = ['flux']
    ref = openmc.Tally(name='cellref')
    ref.filters = [openmc.CellFilter(cell)]
    ref.scores = ['flux']
    model.tallies = openmc.Tallies([t, ref])

    out = run_and_read(model, tmp_path, ['shifted', 'cellref'])
    frac = out['shifted'][0] / out['cellref'][0]
    assert abs(frac - 0.5) < 0.01


def test_rectilinear_mesh(tmp_path):
    """A rectilinear tally mesh with unequal bin widths cutting regions.

    Bin planes at x = 1 and x = 3.5 pass through source region
    interiors. In the uniform medium each bin must score in proportion
    to its width.
    """
    model, cell = uniform_model(tmp_path)
    mm = openmc.RectilinearMesh()
    mm.x_grid = [0.0, 1.0, 3.5, 10.0]
    mm.y_grid = [0.0, L]
    mm.z_grid = [0.0, L]
    t = openmc.Tally(name='rect')
    t.filters = [openmc.MeshFilter(mm)]
    t.scores = ['flux']
    model.tallies = openmc.Tallies([t])

    out = run_and_read(model, tmp_path, ['rect'])
    widths = np.array([1.0, 2.5, 6.5])
    assert_uniform(out['rect'] / widths, 0.02)


def test_cylindrical_mesh(tmp_path):
    """A cylindrical tally mesh embedded inside the cube.

    Every mesh surface is curved or interior, so region pieces are cut
    by cylinders rather than planes, and the regions at the outer radius
    straddle the mesh edge. Each (r, z) bin must score in proportion to
    its analytic volume, and the mesh total must match the covered
    fraction of the domain.
    """
    model, cell = uniform_model(tmp_path)
    mm = openmc.CylindricalMesh(
        r_grid=[0.0, 1.5, 3.5],
        phi_grid=[0.0, 2 * np.pi],
        z_grid=[0.0, 5.0, 10.0],
        origin=(5.0, 5.0, 0.0))
    t = openmc.Tally(name='cyl')
    t.filters = [openmc.MeshFilter(mm)]
    t.scores = ['flux']
    ref = openmc.Tally(name='cellref')
    ref.filters = [openmc.CellFilter(cell)]
    ref.scores = ['flux']
    model.tallies = openmc.Tallies([t, ref])

    out = run_and_read(model, tmp_path, ['cyl', 'cellref'])
    r_vols = np.array([np.pi * 1.5**2, np.pi * (3.5**2 - 1.5**2)])
    vols = np.concatenate([r_vols * 5.0, r_vols * 5.0])
    vals = out['cyl']
    density = out['cellref'][0] / L**3
    assert_uniform(vals / vols, 0.03)
    expected_total = density * np.pi * 3.5**2 * 10.0
    assert abs(vals.sum() - expected_total) / expected_total < 0.02


def test_spherical_mesh(tmp_path):
    """A spherical tally mesh embedded inside the cube, with two radial
    shells cutting regions along spheres. Each shell must score in
    proportion to its analytic volume."""
    model, cell = uniform_model(tmp_path)
    mm = openmc.SphericalMesh(r_grid=[0.0, 2.0, 4.0], origin=(5.0, 5.0, 5.0))
    t = openmc.Tally(name='sph')
    t.filters = [openmc.MeshFilter(mm)]
    t.scores = ['flux']
    ref = openmc.Tally(name='cellref')
    ref.filters = [openmc.CellFilter(cell)]
    ref.scores = ['flux']
    model.tallies = openmc.Tallies([t, ref])

    out = run_and_read(model, tmp_path, ['sph', 'cellref'])
    vols = np.array([4 / 3 * np.pi * 2.0**3,
                     4 / 3 * np.pi * (4.0**3 - 2.0**3)])
    vals = out['sph']
    density = out['cellref'][0] / L**3
    assert_uniform(vals / vols, 0.03)
    expected_total = density * 4 / 3 * np.pi * 4.0**3
    assert abs(vals.sum() - expected_total) / expected_total < 0.02


def test_determinism(tmp_path):
    """Repeat runs must be bitwise identical on one thread and agree to
    accumulation-order rounding with threading, matching the solver's
    pre-existing reproducibility contract."""
    for threads, bitwise in ((1, True), (4, False)):
        vals = []
        for rep in (1, 2):
            wd = tmp_path / f't{threads}_{rep}'
            wd.mkdir()
            model, cell = uniform_model(wd)
            model.tallies = openmc.Tallies([
                mesh_flux_tally(tally_mesh((3, 3, 3)), 'm3'),
            ])
            os.environ['OMP_NUM_THREADS'] = str(threads)
            try:
                out = run_and_read(model, wd, ['m3'])
            finally:
                os.environ.pop('OMP_NUM_THREADS', None)
            vals.append(out['m3'])
        if bitwise:
            assert np.array_equal(vals[0], vals[1])
        else:
            assert np.abs((vals[0] - vals[1]) / vals[0]).max() < 1e-12
