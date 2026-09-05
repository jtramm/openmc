import os

import openmc
from openmc.examples import random_ray_three_region_cube

from tests.testing_harness import TolerantPyAPITestHarness


class MGXSTestHarness(TolerantPyAPITestHarness):
    def _cleanup(self):
        super()._cleanup()
        f = 'mgxs.h5'
        if os.path.exists(f):
            os.remove(f)


def test_random_ray_tally_subdivide():
    # A flux tally on a mesh that subdivides the source regions. The
    # source regions come from a 12x12x12 overlay (2.5 cm cells) while the
    # tally mesh is 5x5x5 (6 cm bins), so most tally bin boundaries cut
    # through source region interiors and the scores are apportioned by
    # the traced track length fractions.
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
    tally = openmc.Tally(name='subdivided flux')
    tally.filters = [openmc.MeshFilter(tmesh)]
    tally.scores = ['flux']
    model.tallies.append(tally)

    model.settings.inactive = 5
    model.settings.batches = 15
    harness = MGXSTestHarness('statepoint.15.h5', model)
    harness.main()
