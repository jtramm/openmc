"""Standing benchmark for random ray volume estimator policies.

Runs every test problem developed during the volume estimator investigation
against one or more policies and prints a consolidated report, so that any
candidate policy is judged on all regimes at once rather than on the problem
that inspired it.

Problems and what they probe:
  lattice     C5G7-style eigenvalue (benign reactor physics). A policy must
              leave this untouched: k must match hybrid, no special treatment
              should activate.
  cube        Three-region cube (the PNE paper problem; external source in a
              near-void). Probes the external-source protections all policies
              share.
  minijet@N   Fusion-like hall with in-scatter-fed thin gas (the mechanism
              that breaks hybrid), run at three ray densities spanning ~2% to
              ~26% cell miss rate at constant total active ray length.
              Scored against a cached multigroup Monte Carlo reference.
  tld         Point-like hot source in a thin-gas room (the JET localized
              FW-CADIS failure mode: streaming-variance negatives). Scored
              against a cached multigroup Monte Carlo reference.

Metrics: negative tally bins (positivity), region-integral errors vs the MC
reference or vs the hybrid result (accuracy/bias), and the policy-activity
diagnostics parsed from the solver output (strong-source fraction, rescues,
demotions). Run time is a few minutes per policy.

Usage:
  python benchmark.py [policy ...]     # default: naive hybrid adaptive
  python benchmark.py adaptive         # judge one candidate

MC references are read from refs/ next to this script and regenerated with
OpenMC multigroup Monte Carlo if missing.
"""
import functools
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

import openmc
import openmc.mgxs

HERE = Path(__file__).parent
REFS = HERE / 'refs'
WORK = Path(os.environ.get('BENCH_WORKDIR', '/tmp/ve_bench'))
print = functools.partial(print, flush=True)

STATS_KEYS = {
    'miss': r'Avg SR Miss Rate per Iteration\s*=\s*([\d.]+)%',
    'naive_total': r'Total\s*=\s*\d+ SRs \(([\d.]+)%\)',
    'strong': r'Strong/External Source\s*=\s*\d+ SRs \(([\d.]+)%\)',
    'demoted': r'Chronic Negative Flux \(demoted\)\s*=\s*(\d+)',
    'small': r'Hit-Starved \(Small\)\s*=\s*\d+ SRs \(([\d.]+)%\)',
}


def one_group_xs(name, total, c, groups):
    d = openmc.XSdata(name, groups)
    d.order = 0
    d.set_total([total])
    d.set_absorption([total * (1.0 - c)])
    d.set_scatter_matrix(np.rollaxis(np.array([[[total * c]]]), 0, 3))
    return d


def build_minijet(policy, rays):
    """Fusion-like hall, two-group, in-scatter-fed slow group."""
    openmc.reset_auto_ids()
    model = openmc.Model()
    ebins = [1e-5, 1.0e3, 20.0e6]
    groups = openmc.mgxs.EnergyGroups(group_edges=ebins)

    def xs(name, tot_fast, s_ff, s_fs, tot_slow, s_ss):
        d = openmc.XSdata(name, groups)
        d.order = 0
        scat = np.array([[s_ff, s_fs], [0.0, s_ss]])
        d.set_total([tot_fast, tot_slow])
        d.set_absorption(np.array([tot_fast, tot_slow]) - scat.sum(axis=1))
        d.set_scatter_matrix(scat[:, :, np.newaxis])
        return d

    lib = openmc.MGXSLibrary(groups)
    lib.add_xsdatas([
        xs('gas', 3.0e-3, 1.7e-3, 1.0e-3, 3.0e-5, 0.9868 * 3.0e-5),
        xs('srcgas', 3.0e-3, 0.55 * 3.0e-3, 0.33 * 3.0e-3,
           1.5e-3, 0.49 * 1.5e-3),
        xs('shield', 0.30, 0.149, 1.0e-3, 0.30, 0.05),
    ])
    mgxs_path = str(WORK / f'mgxs_minijet_{policy}_{rays}.h5')
    lib.export_to_hdf5(mgxs_path)
    mats = {}
    for n in ('gas', 'srcgas', 'shield'):
        m = openmc.Material(name=n)
        m.set_density('macro', 1.0)
        m.add_macroscopic(openmc.Macroscopic(n))
        mats[n] = m
    model.materials = openmc.Materials(mats.values())
    model.materials.cross_sections = mgxs_path

    sphere = openmc.Sphere(r=66.0)
    box = openmc.model.RectangularParallelepiped(
        -70, 70, -70, 70, -70, 70, boundary_type='vacuum')
    cavity = openmc.model.RectangularParallelepiped(-50, -38, -6, 6, -6, 6)
    inner = openmc.Sphere(r=55.0)
    blocks = []
    for (x0, y0, z0) in ((10, -20, -20), (-15, 15, 10), (25, 20, -5),
                         (0, -10, 25), (-30, -25, 5)):
        blocks.append(openmc.model.RectangularParallelepiped(
            x0, x0 + 14, y0, y0 + 14, z0, z0 + 14))
    block_region = -blocks[0]
    for b in blocks[1:]:
        block_region = block_region | -b
    cavity_cell = openmc.Cell(fill=mats['srcgas'], region=-cavity)
    gas_cell = openmc.Cell(fill=mats['gas'],
                           region=-inner & +cavity & ~block_region)
    blocks_cell = openmc.Cell(fill=mats['shield'],
                              region=block_region & -inner)
    shell_cell = openmc.Cell(fill=mats['shield'], region=+inner & -sphere)
    outside = openmc.Cell(region=+sphere & -box)
    model.geometry = openmc.Geometry(
        [cavity_cell, gas_cell, blocks_cell, shell_cell, outside])

    s = model.settings
    s.run_mode = 'fixed source'
    s.energy_mode = 'multi-group'
    s.particles = rays
    s.inactive = 300
    s.batches = 300 + max(1, 300000 // rays)
    s.source = openmc.IndependentSource(
        space=openmc.stats.Box((-50, -6, -6), (-38, 6, 6)),
        energy=openmc.stats.Discrete([2.0e6], [1.0]),
        constraints={'domains': [cavity_cell]})
    mesh = openmc.RegularMesh()
    mesh.lower_left = (-70, -70, -70)
    mesh.upper_right = (70, 70, 70)
    mesh.dimension = (35, 35, 35)
    s.random_ray = {
        'distance_inactive': 400.0,
        'distance_active': 400.0,
        'ray_source': openmc.IndependentSource(
            space=openmc.stats.Box((-70, -70, -70), (70, 70, 70))),
        'source_shape': 'flat',
        'volume_estimator': policy,
        'source_region_meshes': [(mesh, [model.geometry.root_universe])],
    }
    mf = openmc.MeshFilter(mesh)
    ef = openmc.EnergyFilter(ebins)
    t = openmc.Tally(name='flux')
    t.filters = [mf, ef]
    t.scores = ['flux']
    model.tallies = openmc.Tallies([t])
    return model


def build_tld(policy):
    """Point-like hot source cell in a large thin-gas room."""
    openmc.reset_auto_ids()
    model = openmc.Model()
    groups = openmc.mgxs.EnergyGroups(group_edges=[1e-5, 20.0e6])
    lib = openmc.MGXSLibrary(groups)
    lib.add_xsdatas([one_group_xs('air', 3.0e-4, 0.99, groups),
                     one_group_xs('wall', 0.3, 0.5, groups)])
    mgxs_path = str(WORK / f'mgxs_tld_{policy}.h5')
    lib.export_to_hdf5(mgxs_path)
    mats = {}
    for n in ('air', 'wall'):
        m = openmc.Material(name=n)
        m.set_density('macro', 1.0)
        m.add_macroscopic(openmc.Macroscopic(n))
        mats[n] = m
    model.materials = openmc.Materials(mats.values())
    model.materials.cross_sections = mgxs_path

    tld = openmc.model.RectangularParallelepiped(-1, 1, -1, 1, -1, 1)
    room = openmc.model.RectangularParallelepiped(-60, 60, -60, 60, -60, 60)
    box = openmc.model.RectangularParallelepiped(
        -70, 70, -70, 70, -70, 70, boundary_type='vacuum')
    tld_cell = openmc.Cell(fill=mats['air'], region=-tld)
    air_cell = openmc.Cell(fill=mats['air'], region=-room & +tld)
    wall_cell = openmc.Cell(fill=mats['wall'], region=+room & -box)
    model.geometry = openmc.Geometry([tld_cell, air_cell, wall_cell])

    s = model.settings
    s.run_mode = 'fixed source'
    s.energy_mode = 'multi-group'
    s.particles = 2000
    s.batches = 600
    s.inactive = 300
    s.source = openmc.IndependentSource(
        space=openmc.stats.Box((-1, -1, -1), (1, 1, 1)),
        energy=openmc.stats.Discrete([1e6], [1.0]),
        constraints={'domains': [tld_cell]})
    mesh = openmc.RegularMesh()
    mesh.lower_left = (-70, -70, -70)
    mesh.upper_right = (70, 70, 70)
    mesh.dimension = (35, 35, 35)
    s.random_ray = {
        'distance_inactive': 400.0,
        'distance_active': 400.0,
        'ray_source': openmc.IndependentSource(
            space=openmc.stats.Box((-70, -70, -70), (70, 70, 70))),
        'source_shape': 'flat',
        'volume_estimator': policy,
        'source_region_meshes': [(mesh, [model.geometry.root_universe])],
    }
    mf = openmc.MeshFilter(mesh)
    t = openmc.Tally(name='flux')
    t.filters = [mf]
    t.scores = ['flux']
    model.tallies = openmc.Tallies([t])
    return model


def run_model(model, tag):
    """Run in a private directory; return (flux array or None, stats dict)."""
    cwd = WORK / tag
    cwd.mkdir(parents=True, exist_ok=True)
    model.export_to_model_xml(cwd / 'model.xml')
    proc = subprocess.run(['openmc'], cwd=cwd, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    stats = {}
    for key, pat in STATS_KEYS.items():
        m = re.search(pat, out)
        stats[key] = float(m.group(1)) if m else None
    if proc.returncode != 0:
        stats['fatal'] = out.strip().splitlines()[-1][:60] if out else 'fatal'
        return None, stats
    sp_files = sorted(cwd.glob('statepoint.*.h5'))
    with openmc.StatePoint(sp_files[-1]) as f:
        t = f.tallies[next(iter(f.tallies))]
        flux = t.mean
    return flux, stats


def minijet_masks():
    c = -70 + 4.0 * (np.arange(35) + 0.5)
    X, Y, Z = np.meshgrid(c, c, c, indexing='ij')
    R = np.sqrt(X**2 + Y**2 + Z**2)
    cavity = (X >= -50) & (X <= -38) & (np.abs(Y) <= 6) & (np.abs(Z) <= 6)
    blocks = np.zeros_like(cavity)
    for (x0, y0, z0) in ((10, -20, -20), (-15, 15, 10), (25, 20, -5),
                         (0, -10, 25), (-30, -25, 5)):
        blocks |= ((X >= x0) & (X <= x0 + 14) & (Y >= y0) & (Y <= y0 + 14)
                   & (Z >= z0) & (Z <= z0 + 14))
    return {'cavity': cavity.ravel(),
            'gas': ((R < 55) & ~cavity & ~blocks).ravel(),
            'shield': ((R >= 55) & (R < 66)).ravel()}


def tld_masks():
    c = -70 + 4.0 * (np.arange(35) + 0.5)
    X, Y, Z = np.meshgrid(c, c, c, indexing='ij')
    R = np.sqrt(X**2 + Y**2 + Z**2).ravel()
    return {'near': (R < 15), 'mid': (R >= 15) & (R < 40),
            'far': (R >= 40) & (R < 60)}


def ensure_refs():
    REFS.mkdir(exist_ok=True)
    if not (REFS / 'minijet_mc.npy').exists():
        print('generating minijet MC reference (one-time)...')
        model = build_minijet('hybrid', 8000)
        s = model.settings
        s.random_ray = {}
        del model.settings.random_ray
        s.particles, s.batches, s.inactive = 500000, 20, 0
        flux, _ = run_model(model, 'ref_minijet')
        np.save(REFS / 'minijet_mc.npy', flux.reshape(-1, 2))
    if not (REFS / 'tld_mc.npy').exists():
        print('generating tld MC reference (one-time)...')
        model = build_tld('hybrid')
        s = model.settings
        del model.settings.random_ray
        s.particles, s.batches, s.inactive = 500000, 20, 0
        flux, _ = run_model(model, 'ref_tld')
        np.save(REFS / 'tld_mc.npy', flux.ravel())


def fmt_stats(stats):
    out = []
    if stats.get('naive_total') is not None:
        out.append(f"naive {stats['naive_total']:.2f}%")
    if stats.get('strong'):
        out.append(f"strong {stats['strong']:.2f}%")
    if stats.get('demoted'):
        out.append(f"demoted {int(stats['demoted'])}")
    if stats.get('small'):
        out.append(f"small {stats['small']:.2f}%")
    return ', '.join(out) if out else '-'


def main():
    policies = sys.argv[1:] or ['naive', 'hybrid', 'adaptive']
    WORK.mkdir(parents=True, exist_ok=True)
    ensure_refs()
    mj_mc = np.load(REFS / 'minijet_mc.npy')
    tld_mc = np.load(REFS / 'tld_mc.npy')
    mj_m, tld_m = minijet_masks(), tld_masks()

    print(f"\n{'='*98}\nVOLUME ESTIMATOR POLICY BENCHMARK   policies: "
          f"{', '.join(policies)}\n{'='*98}")

    # --- benign lattice (eigenvalue) -------------------------------------
    print("\n[lattice] C5G7-style eigenvalue -- requirement: k identical to "
          "hybrid, no policy activity")
    from openmc.examples import random_ray_lattice
    kref = None
    for pol in policies:
        openmc.reset_auto_ids()
        os.chdir(WORK)  # examples write mgxs.h5 into the current directory
        model = random_ray_lattice()
        model.settings.random_ray['volume_estimator'] = pol
        cwd = WORK / f'lattice_{pol}'
        cwd.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(WORK / 'mgxs.h5', cwd / 'mgxs.h5')
        model.materials.cross_sections = 'mgxs.h5'
        model.export_to_model_xml(cwd / 'model.xml')
        proc = subprocess.run(['openmc'], cwd=cwd, capture_output=True,
                              text=True)
        stats = {k2: (float(mm.group(1)) if (mm := re.search(p, proc.stdout))
                      else None) for k2, p in STATS_KEYS.items()}
        sps = sorted(cwd.glob('statepoint.*.h5'))
        if proc.returncode != 0 or not sps:
            print(f"  {pol:<22} FATAL")
            continue
        with openmc.StatePoint(sps[-1]) as f:
            kval = f.keff
        if pol == 'hybrid':
            kref = kval.nominal_value
        flag = '' if (kref is None or kval.nominal_value == kref) else \
            '  <-- DIFFERS FROM HYBRID'
        print(f"  {pol:<22} k = {kval!s:<22} {fmt_stats(stats)}{flag}")

    # --- three-region cube (external source) -----------------------------
    print("\n[cube] three-region cube (paper problem) -- requirement: match "
          "hybrid, no negatives")
    from openmc.examples import random_ray_three_region_cube
    ref_sum = None
    for pol in policies:
        openmc.reset_auto_ids()
        os.chdir(WORK)
        model = random_ray_three_region_cube()
        model.settings.random_ray['volume_estimator'] = pol
        import shutil
        (WORK / f'cube_{pol}').mkdir(parents=True, exist_ok=True)
        shutil.copy(WORK / 'mgxs.h5', WORK / f'cube_{pol}' / 'mgxs.h5')
        model.materials.cross_sections = 'mgxs.h5'
        flux, stats = run_model(model, f'cube_{pol}')
        if flux is None:
            print(f"  {pol:<22} FATAL: {stats.get('fatal')}")
            continue
        tot = flux.sum()
        if pol == 'hybrid':
            ref_sum = tot
        rel = '' if ref_sum is None else f"vs hybrid {tot/ref_sum-1:+.4%}"
        print(f"  {pol:<22} tally sum {tot:.6e} ({rel})  "
              f"neg {(flux < 0).sum()}  {fmt_stats(stats)}")

    # --- mini-JET at three miss rates ------------------------------------
    for rays in (8000, 2000, 500):
        print(f"\n[minijet @ {rays} rays] in-scatter pathology -- "
              "slow-group region integrals vs MG-MC")
        for pol in policies:
            model = build_minijet(pol, rays)
            flux, stats = run_model(model, f'minijet_{pol}_{rays}')
            if flux is None:
                print(f"  {pol:<22} FATAL: {stats.get('fatal')}")
                continue
            f2 = flux.reshape(-1, 2)
            row = f"  {pol:<22}"
            for rn, m in mj_m.items():
                imc = mj_mc[m, 0].sum()
                row += f" {rn} {(f2[m,0].sum()-imc)/imc:>+9.1%}"
            miss = stats.get('miss')
            row += (f"  neg {(f2[:,0]<0).sum():>5}  miss "
                    f"{miss if miss is not None else '?':>5}%  "
                    f"{fmt_stats(stats)}")
            print(row)

    # --- TLD point source --------------------------------------------------
    print("\n[tld] point-like source, streaming variance -- region integrals "
          "vs MG-MC")
    for pol in policies:
        model = build_tld(pol)
        flux, stats = run_model(model, f'tld_{pol}')
        if flux is None:
            print(f"  {pol:<22} FATAL: {stats.get('fatal')}")
            continue
        f1 = flux.ravel()
        row = f"  {pol:<22}"
        for rn, m in tld_m.items():
            imc = tld_mc[m].sum()
            row += f" {rn} {(f1[m].sum()-imc)/imc:>+9.1%}"
        row += f"  neg {(f1<0).sum():>5}  {fmt_stats(stats)}"
        print(row)

    print(f"\n{'='*98}")


if __name__ == '__main__':
    main()
