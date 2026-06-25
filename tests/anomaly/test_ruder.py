"""RUDER: recursive-PCA streaming anomaly detection (smoke test).

RUDER consumes a CSV ``DataStream`` row-by-row (header + comma-separated
columns) and writes per-sample anomaly weights to an output file.  The bundled
example file is ~90 MB, so here we generate a small synthetic stream with an
injected anomaly and check RUDER runs end-to-end.
"""

import numpy as np

from magprime.algorithms.anomaly import RUDER


def _write_stream(path, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    values = np.sin(2 * np.pi * 0.01 * t) + 0.05 * rng.standard_normal(n)
    values[600:650] += 8.0  # injected anomaly
    with open(path, "w") as fh:
        fh.write("idx,value\n")  # header (RUDER skips the first row)
        for i, v in enumerate(values):
            fh.write(f"{i},{v}\n")


def test_ruder_runs_on_synthetic_stream(tmp_path):
    data_file = tmp_path / "stream.csv"
    out_file = tmp_path / "ruder_scores.txt"
    _write_stream(data_file)

    stream = RUDER.DataStream(str(data_file))
    weights = RUDER.RUDER(
        window_length=50,
        initialization_length=2,
        data_stream=stream,
        col_n=1,
        filename=str(out_file),
        nu=0.1,
    )

    assert isinstance(weights, np.ndarray)
    assert out_file.exists()
