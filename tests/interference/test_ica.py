"""ICA: blind source separation of the ambient field from interference.

FastICA is stochastic (and has sign/scale ambiguity), so we seed the global
NumPy RNG it relies on and compare via correlation rather than RMSE.
"""

import numpy as np

from magprime.algorithms.interference import ICA


def test_ica_recovers_ambient(scenario, metrics, report):
    np.random.seed(0)  # FastICA uses the global RNG when random_state is None
    sc = scenario(gains=(1.0, 0.3), amp_factor=3.0, n_tones=1)

    rec = ICA.clean(np.copy(sc.B))

    assert rec.shape == sc.ambient.shape
    assert np.isfinite(rec).all()

    # ICA is a blind method with sign/scale ambiguity, so compare via
    # |correlation|.  It substantially improves on the raw noisy sensor
    # (~0.70 vs ~0.43 here) but does not fully recover the ambient field.
    c_raw = abs(metrics.corr(sc.noisy, sc.ambient))
    c_clean = abs(metrics.corr(rec, sc.ambient))
    report(
        rmse_raw=metrics.rmse(sc.noisy, sc.ambient),
        rmse_clean=metrics.rmse(rec, sc.ambient),
        corr_raw=c_raw,
        corr_clean=c_clean,
    )
    assert c_clean > c_raw + 0.1
    assert c_clean > 0.6
