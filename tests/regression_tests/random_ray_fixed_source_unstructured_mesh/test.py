import os

import numpy as np
import openmc
import openmc.lib
import pytest

from tests.testing_harness import TolerantPyAPITestHarness

pytestmark = pytest.mark.skipif(
    not openmc.lib.feature_enabled('dagmc'),
    reason="DAGMC is not enabled.")


class UnstructuredMeshWWTestHarness(TolerantPyAPITestHarness):
    """Compare the statepoint tallies and the generated weight window bounds."""

    def _get_results(self):
        results = super()._get_results()
        ww = openmc.WeightWindowsList.from_hdf5()[0]
        lower = [f'{x:.2e}' for x in ww.lower_ww_bounds.flatten()]
        upper = [f'{x:.2e}' for x in ww.upper_ww_bounds.flatten()]
        return results + '\n'.join(
            [str(ww.mesh), 'Lower Bounds'] + lower + ['Upper Bounds'] + upper)

    def _cleanup(self):
        super()._cleanup()
        for f in ('mgxs.h5', 'weight_windows.h5'):
            if os.path.exists(f):
                os.remove(f)


def test_random_ray_fixed_source_unstructured_mesh():
    openmc.reset_auto_ids()

    # One-group material with half scattering
    groups = openmc.mgxs.EnergyGroups([1e-5, 20.0e6])
    xsdata = openmc.XSdata('mat', groups)
    xsdata.order = 0
    xsdata.set_total([1.0])
    xsdata.set_absorption([0.5])
    xsdata.set_scatter_matrix(np.array([[[0.5]]]))
    mgxs_lib = openmc.MGXSLibrary(groups)
    mgxs_lib.add_xsdata(xsdata)
    mgxs_lib.export_to_hdf5('mgxs.h5')

    mat = openmc.Material(name='mat')
    mat.set_density('macro', 1.0)
    mat.add_macroscopic(openmc.Macroscopic('mat'))
    materials = openmc.Materials([mat])
    materials.cross_sections = 'mgxs.h5'

    # Unit cube matching the canonical 24-tet MOAB mesh asset, reflective on
    # the low faces and vacuum on the high faces so the flux has a gradient
    box = openmc.model.RectangularParallelepiped(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
    for s in (box.xmin, box.ymin, box.zmin):
        s.boundary_type = 'reflective'
    for s in (box.xmax, box.ymax, box.zmax):
        s.boundary_type = 'vacuum'
    cell = openmc.Cell(name='cube', fill=mat, region=-box)
    root = openmc.Universe(cells=[cell])
    geometry = openmc.Geometry(root)

    # The tet mesh subdivides the single cell into 24 source regions and is
    # also the tally mesh and the weight window mesh
    umesh = openmc.UnstructuredMesh(
        '../unstructured_mesh/test_mesh_dagmc_tets.vtk', library='moab')

    settings = openmc.Settings()
    settings.energy_mode = 'multi-group'
    settings.run_mode = 'fixed source'
    settings.batches = 10
    settings.inactive = 5
    settings.particles = 100
    settings.random_ray['distance_active'] = 10.0
    settings.random_ray['distance_inactive'] = 3.0
    settings.random_ray['ray_source'] = openmc.IndependentSource(
        space=openmc.stats.Box((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)))
    settings.random_ray['volume_estimator'] = 'hybrid'
    settings.random_ray['source_region_meshes'] = [(umesh, [root])]
    settings.source = openmc.IndependentSource(
        energy=openmc.stats.Discrete([1.0e6], [1.0]),
        constraints={'domains': [cell]})
    settings.weight_window_generators = openmc.WeightWindowGenerator(
        mesh=umesh, method='magic', max_realizations=5)

    mesh_tally = openmc.Tally(name='tet flux')
    mesh_tally.filters = [openmc.MeshFilter(umesh)]
    mesh_tally.scores = ['flux']
    mesh_tally.estimator = 'tracklength'
    cell_tally = openmc.Tally(name='cell flux')
    cell_tally.filters = [openmc.CellFilter(cell)]
    cell_tally.scores = ['flux']
    cell_tally.estimator = 'tracklength'
    tallies = openmc.Tallies([mesh_tally, cell_tally])

    model = openmc.Model(geometry, materials, settings, tallies)
    harness = UnstructuredMeshWWTestHarness('statepoint.10.h5', model)
    harness.main()
