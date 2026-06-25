"""PiCoG: principal-component gradiometry.

PiCoG assumes the interference variance dominates the ambient variation along
its polarization direction, so we make the interference large.  It returns the
corrected measurement for the requested sensor (``sens``).
"""

import numpy as np

from magprime.algorithms.interference import PiCoG


def test_picog_reduces_interference(scenario, metrics, report):
    sc = scenario(gains=(1.0, 0.3), amp_factor=8.0)

    rec = PiCoG.clean(np.copy(sc.B), sens=1)

    assert rec.shape == sc.ambient.shape
    assert np.isfinite(rec).all()

    # Baseline is the same sensor PiCoG returns (sensor 1).
    err_raw = metrics.rmse(sc.B[1], sc.ambient)
    err_clean = metrics.rmse(rec, sc.ambient)
    report(rmse_raw=err_raw, rmse_clean=err_clean, reduction=err_raw / err_clean)
    assert err_clean < err_raw
