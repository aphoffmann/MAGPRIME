"""NEUBAUER: physical multi-sensor gradiometry.

NEUBAUER needs the magnetometer positions and the spacecraft (noise-source)
center.  Exact recovery requires the interference to follow the dipole model
encoded in the geometry, so here we assert the method runs and produces a
sane, finite ambient estimate for a minimal valid two-sensor geometry.
"""

import numpy as np

from magprime.algorithms.interference import NEUBAUER


def test_neubauer_runs_with_geometry(scenario, metrics, report):
    sc = scenario(gains=(1.0, 0.4), amp_factor=5.0)

    NEUBAUER.mag_positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    NEUBAUER.spacecraft_center = np.array([2.0, 0.0, 0.0])
    NEUBAUER.optimize_center = False
    try:
        rec = NEUBAUER.clean(np.copy(sc.B))
    finally:
        NEUBAUER.mag_positions = None
        NEUBAUER.spacecraft_center = None

    assert rec.shape == sc.ambient.shape
    assert np.isfinite(rec).all()
    report(
        rmse_raw=metrics.rmse(sc.noisy, sc.ambient),
        rmse_clean=metrics.rmse(rec, sc.ambient),
    )
