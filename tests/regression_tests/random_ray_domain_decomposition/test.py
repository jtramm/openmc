import os
import re

import numpy as np
import openmc

from tests.regression_tests import config
from tests.testing_harness import TolerantPyAPITestHarness


class MGXSTestHarness(TolerantPyAPITestHarness):
    def _cleanup(self):
        super()._cleanup()
        f = 'mgxs.h5'
        if os.path.exists(f):
            os.remove(f)


def uneven_load_model():
    """A box with fuel on one side, divided into many source regions, and a
    reflector on the other side that is a single source region."""
    openmc.reset_auto_ids()

    groups = openmc.mgxs.EnergyGroups(group_edges=[1e-5, 20.0e6])

    fuel_data = openmc.XSdata('fuel', groups)
    fuel_data.order = 0
    fuel_data.set_total([0.5])
    fuel_data.set_absorption([0.2])
    fuel_data.set_scatter_matrix(np.array([[[0.3]]]))
    fuel_data.set_fission([0.1])
    fuel_data.set_nu_fission([0.3])
    fuel_data.set_chi([1.0])

    reflector_data = openmc.XSdata('reflector', groups)
    reflector_data.order = 0
    reflector_data.set_total([0.4])
    reflector_data.set_absorption([0.01])
    reflector_data.set_scatter_matrix(np.array([[[0.39]]]))

    library = openmc.MGXSLibrary(groups)
    library.add_xsdatas([fuel_data, reflector_data])
    library.export_to_hdf5('mgxs.h5')

    fuel = openmc.Material(name='fuel')
    fuel.set_density('macro', 1.0)
    fuel.add_macroscopic('fuel')
    reflector = openmc.Material(name='reflector')
    reflector.set_density('macro', 1.0)
    reflector.add_macroscopic('reflector')
    materials = openmc.Materials([fuel, reflector])
    materials.cross_sections = 'mgxs.h5'

    # The fuel is a lattice of 6 x 6 x 6 cells
    n = 6
    fuel_universe = openmc.Universe(cells=[openmc.Cell(fill=fuel)])
    lattice = openmc.RectLattice()
    lattice.lower_left = (0.0, 0.0, 0.0)
    lattice.pitch = (8.0 / n, 10.0 / n, 10.0 / n)
    lattice.universes = [[[fuel_universe] * n] * n] * n

    x_min = openmc.XPlane(0.0, boundary_type='reflective')
    x_mid = openmc.XPlane(8.0)
    x_max = openmc.XPlane(20.0, boundary_type='vacuum')
    y_min = openmc.YPlane(0.0, boundary_type='reflective')
    y_max = openmc.YPlane(10.0, boundary_type='reflective')
    z_min = openmc.ZPlane(0.0, boundary_type='reflective')
    z_max = openmc.ZPlane(10.0, boundary_type='reflective')
    sides = +y_min & -y_max & +z_min & -z_max
    fuel_cell = openmc.Cell(fill=lattice, region=+x_min & -x_mid & sides)
    reflector_cell = openmc.Cell(fill=reflector, region=+x_mid & -x_max & sides)
    geometry = openmc.Geometry([fuel_cell, reflector_cell])

    settings = openmc.Settings()
    settings.energy_mode = 'multi-group'
    settings.particles = 200
    settings.batches = 15
    settings.inactive = 3
    settings.random_ray['distance_inactive'] = 10.0
    settings.random_ray['distance_active'] = 100.0
    settings.random_ray['source_shape'] = 'linear'
    settings.random_ray['ray_source'] = openmc.IndependentSource(
        space=openmc.stats.Box((0.0, 0.0, 0.0), (20.0, 10.0, 10.0)))

    # Three slices of the fuel, each made of whole source regions
    mesh = openmc.RegularMesh()
    mesh.dimension = (3, 1, 1)
    mesh.lower_left = (0.0, 0.0, 0.0)
    mesh.upper_right = (8.0, 10.0, 10.0)
    fuel_tally = openmc.Tally(name='fuel')
    fuel_tally.filters = [openmc.MeshFilter(mesh)]
    fuel_tally.scores = ['flux', 'fission']
    reflector_tally = openmc.Tally(name='reflector')
    reflector_tally.filters = [openmc.MaterialFilter([reflector])]
    reflector_tally.scores = ['flux']
    tallies = openmc.Tallies([fuel_tally, reflector_tally])

    return openmc.Model(geometry, materials, settings, tallies)


def test_random_ray_domain_decomposition(capfd):
    # When run on more than one MPI rank, the random ray solver decomposes the
    # problem, and divides the work evenly by moving source regions between
    # the ranks during the first batches. Dividing this problem evenly takes
    # moving many of them: rays cross a source region every 1.5 cm or so in
    # the fuel, and every 12 cm in the reflector. Each rank gets half of the
    # box to start with, with the boundary in the reflector, which is
    # therefore found by both ranks.
    model = uneven_load_model()
    harness = MGXSTestHarness('statepoint.15.h5', model)
    harness.main()

    if config['mpi'] and int(config['mpi_np']) > 1:
        output = capfd.readouterr().out
        n_moved = re.search(r'SRs Moved Between Ranks\s+= (\d+)', output)
        n_found = re.search(r'SRs Found by Several Ranks\s+= (\d+)', output)
        assert int(n_moved.group(1)) > 0
        assert int(n_found.group(1)) > 0
