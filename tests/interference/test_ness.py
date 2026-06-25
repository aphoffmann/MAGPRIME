"""NESS: gradiometry with a known coupling matrix.

NESS recovers the ambient field exactly when ``aii`` equals the ratio of the
interference gains seen by the two sensors, so this is an (almost) exact
recovery test.
"""

import numpy as np

from magprime.algorithms.interference import NESS


def test_ness_exact_recovery(scenario, metrics, report):
    g_in, g_out = 1.0, 0.4
    sc = scenario(gains=(g_in, g_out), amp_factor=5.0)

    # noise(sensor0) = (g_in / g_out) * noise(sensor1)  ->  aii = g_in / g_out
    NESS.aii = np.array([g_in / g_out] * 3)
    try:
        rec = NESS.clean(np.copy(sc.B))
    finally:
        NESS.aii = None  # reset module-level state for other tests

    assert rec.shape == sc.ambient.shape
    assert np.isfinite(rec).all()

    err_raw = metrics.rmse(sc.noisy, sc.ambient)
    err_clean = metrics.rmse(rec, sc.ambient)
    report(rmse_raw=err_raw, rmse_clean=err_clean)
    # Algebraically exact -> only floating-point error remains.
    assert err_clean < 1e-6 * err_raw
