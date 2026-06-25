"""SHEINKER: adaptive two-sensor gradiometer.

Scenario: ambient (Swarm) common to both sensors + one interference source
coupling into each sensor with a different gain.  SHEINKER should recover the
ambient field almost exactly.
"""

import numpy as np

from magprime.algorithms.interference import SHEINKER


def test_sheinker_recovers_ambient(scenario, metrics, report):
    sc = scenario(gains=(1.0, 0.4), amp_factor=5.0)

    rec = SHEINKER.clean(np.copy(sc.B))

    assert rec.shape == sc.ambient.shape
    assert np.isfinite(rec).all()

    err_raw = metrics.rmse(sc.noisy, sc.ambient)
    err_clean = metrics.rmse(rec, sc.ambient)
    report(rmse_raw=err_raw, rmse_clean=err_clean, reduction=err_raw / err_clean)
    # Interference is reduced by well over an order of magnitude.
    assert err_clean < 0.05 * err_raw
    assert metrics.corr(rec, sc.ambient) > 0.99
