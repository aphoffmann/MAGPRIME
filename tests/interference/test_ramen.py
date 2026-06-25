"""RAMEN: wavelet-based gradiometry with auto-estimated coupling.

RAMEN estimates the sensor coupling from the data via the wavelet transform,
then solves for the ambient field.  We only require that it reduces the
interference relative to the raw noisy sensor.
"""

import numpy as np

from magprime.algorithms.interference import RAMEN


def test_ramen_reduces_interference(scenario, metrics, report):
    sc = scenario(gains=(1.0, 0.3), amp_factor=5.0)

    RAMEN.aii = None  # force coupling re-estimation; reset shared module state
    RAMEN.fs = 1
    try:
        rec = RAMEN.clean(np.copy(sc.B))
    finally:
        RAMEN.aii = None

    assert rec.shape == sc.ambient.shape
    assert np.isfinite(rec).all()

    err_raw = metrics.rmse(sc.noisy, sc.ambient)
    err_clean = metrics.rmse(rec, sc.ambient)
    report(rmse_raw=err_raw, rmse_clean=err_clean, reduction=err_raw / err_clean)
    assert err_clean < err_raw
