import os

import numpy as np
import openmc

from tests.testing_harness import TolerantPyAPITestHarness


class OuterUniverseHarness(TolerantPyAPITestHarness):
    def _get_results(self):
        # Instances as C++ counts them (filter bins), paths as Python finds them
        outstr = super()._get_results()
        with openmc.StatePoint(self._sp_name) as sp:
            tally = sp.get_tally(name='instances')
            distribcell = tally.find_filter(openmc.DistribcellFilter)
            outstr += f'bins: {distribcell.num_bins}\n'
            outstr += '\n'.join(distribcell.paths) + '\n'
        return outstr

    def _cleanup(self):
        super()._cleanup()
        if os.path.exists('mgxs.h5'):
            os.remove('mgxs.h5')


def make_model():
    groups = openmc.mgxs.EnergyGroups([1e-5, 20.0e6])
    library = openmc.MGXSLibrary(groups)
    materials = []
    for name, total, absorption in (('A', 1.0, 0.9), ('B', 0.1, 0.01),
                                    ('C', 0.5, 0.25)):
        xs = openmc.XSdata(name, groups)
        xs.order = 0
        xs.set_total([total])
        xs.set_absorption([absorption])
        xs.set_scatter_matrix(np.array([[[total - absorption]]]))
        library.add_xsdata(xs)
        material = openmc.Material(name=name)
        material.set_density('macro', 1.0)
        material.add_macroscopic(name)
        materials.append(material)
    library.export_to_hdf5('mgxs.h5')
    a, b, c = materials

    # The outer universe is also a tile, and one level deeper than the tiles
    inner = openmc.Cell(fill=c)
    deep = openmc.Cell(fill=a)
    holder = openmc.Cell(fill=openmc.Universe(cells=[deep]))
    tile = openmc.Universe(cells=[holder])
    holder_outer = openmc.Cell(fill=tile)
    lattice = openmc.RectLattice()
    lattice.lower_left = (-1.5, -1.0, -1.0)
    lattice.pitch = (1.0, 2.0, 2.0)
    lattice.universes = [[[openmc.Universe(cells=[inner]), tile]]]
    lattice.outer = openmc.Universe(cells=[holder_outer])

    x1 = openmc.XPlane(1.0)
    box = openmc.model.RectangularParallelepiped(
        -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, boundary_type='reflective')
    left = openmc.Cell(fill=lattice, region=-box & -x1)
    right = openmc.Cell(fill=b, region=-box & +x1)

    settings = openmc.Settings()
    settings.energy_mode = 'multi-group'
    settings.run_mode = 'fixed source'
    settings.batches = 10
    settings.inactive = 5
    settings.particles = 200
    space = openmc.stats.Box((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))
    settings.random_ray['ray_source'] = openmc.IndependentSource(space=space)
    settings.random_ray['distance_inactive'] = 5.0
    settings.random_ray['distance_active'] = 20.0
    energy = openmc.stats.Discrete([1e6], [1.0])
    settings.source = [
        openmc.IndependentSource(energy=energy, constraints={'domains': [cell]})
        for cell in (left, holder_outer)]

    tallies = openmc.Tallies()
    for name, tally_filter in (
            ('cells', openmc.CellFilter([inner, deep, right])),
            ('instances', openmc.DistribcellFilter(deep))):
        tally = openmc.Tally(name=name)
        tally.filters = [tally_filter]
        tally.scores = ['flux', 'total']
        tally.estimator = 'analog'
        tallies.append(tally)

    model = openmc.Model(openmc.Geometry([left, right]),
                         openmc.Materials(materials), settings, tallies)
    model.materials.cross_sections = 'mgxs.h5'
    return model


def test_random_ray_lattice_outer():
    harness = OuterUniverseHarness('statepoint.10.h5', make_model())
    harness.main()
