"""AnomalyTracker: per-time-step spectral anomaly detection (smoke test).

AnomalyTracker takes the log of a positive spectrogram column-by-column and
runs RUDE's anomaly detection over frequency, returning a same-shaped image.
"""

import numpy as np

from magprime.algorithms.spectral import AnomalyTracker


def test_anomaly_tracker_smoke():
    rng = np.random.default_rng(0)
    n_freq, n_time = 100, 40

    s = np.abs(rng.standard_normal((n_freq, n_time))) + 1.0  # strictly positive

    out = AnomalyTracker.anomaly_tracker(s, window_length=10, nu=0.1)

    assert out.shape == s.shape
    assert np.isfinite(out).all()
