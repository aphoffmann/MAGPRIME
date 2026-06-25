"""RUDE: window-based anomaly detection (PCA + One-Class SVM).

We build a benign baseline signal and inject a localized high-frequency burst,
then check that RUDE flags samples inside the injected window.
"""

import numpy as np

from magprime.algorithms.anomaly import RUDE


def test_rude_flags_injected_anomaly():
    rng = np.random.default_rng(1)
    fs = 50
    N = 4000
    t = np.arange(N) / fs

    signal = np.sin(2 * np.pi * 1.0 * t) + 0.05 * rng.standard_normal(N)
    lo, hi = 2000, 2400
    signal[lo:hi] += 5.0 * np.sin(2 * np.pi * 12.0 * t[lo:hi])  # anomalous burst

    flag = RUDE.anomaly_detection(
        signal, sampling_rate_hz=fs, window_length_sec=1.0, nu_value=0.05
    )

    assert flag.shape == signal.shape
    assert flag.sum() > 0                 # something was flagged
    assert flag[lo:hi].sum() > 0          # the injected burst region was flagged
