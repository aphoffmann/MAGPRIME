"""REAM: frequency-domain gradiometry for transient interference.

REAM detects intervals where the differenced-field envelope changes sharply
and suppresses the offending spectral peaks there, using the inboard sensor as
the baseline.  We therefore drive it with a transient interference burst.
"""

import numpy as np

from magprime.algorithms.interference import REAM


def test_ream_suppresses_transient(scenario, metrics, report):
    sc = scenario(gains=(1.0, 0.3), amp_factor=10.0, transient=(1500, 2500))

    REAM.delta_B = 5.0  # envelope-change threshold (nT)
    try:
        rec = REAM.clean(np.copy(sc.B))
    finally:
        REAM.delta_B = None  # reset module-level state

    assert rec.shape == sc.ambient.shape
    assert np.isfinite(rec).all()

    # REAM's baseline output is the inboard sensor (B[0]); it should not make
    # the estimate worse, and should improve it over the transient.
    err_raw = metrics.rmse(sc.B[0], sc.ambient)
    err_clean = metrics.rmse(rec, sc.ambient)
    report(rmse_raw=err_raw, rmse_clean=err_clean, reduction=err_raw / err_clean)
    assert err_clean <= err_raw
