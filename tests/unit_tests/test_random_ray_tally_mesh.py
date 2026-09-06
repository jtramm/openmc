"""Contract test for random ray tally meshes that subdivide source
regions.

The behavioral coverage for this feature lives in the
`random_ray_tally_subdivide` regression battery, which pins each
configuration against stored reference results. The abort contract for
tallies with multiple mesh filters is asserted here instead, since that
configuration terminates by design and so cannot have reference
results, while a silent regression of the abort would produce silently
wrong tallies that nothing else could detect.
"""

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


@pytest.mark.parametrize('second_mesh', ['subdividing', 'edge'])
def test_multiple_mesh_filters_fatal(tmp_path, second_mesh):
    """A tally with two mesh filters must abort with a clear error, both
    when a mesh subdivides source regions outright and when one mesh's
    edge cuts regions whose recorded midpoints fall outside it, rather
    than misattributing or silently dropping scores."""
    model = uniform_model(tmp_path)
    if second_mesh == 'subdividing':
        other = tally_mesh((4, 4, 4))
    else:
        other = tally_mesh((1, 1, 1), hi=(3.0, L, L))
    t = openmc.Tally(name='twomesh')
    t.filters = [openmc.MeshFilter(tally_mesh((3, 3, 3))),
                 openmc.MeshFilter(other)]
    t.scores = ['flux']
    model.tallies = openmc.Tallies([t])
    with pytest.raises(RuntimeError, match='multiple mesh filters'):
        model.run(cwd=str(tmp_path))
