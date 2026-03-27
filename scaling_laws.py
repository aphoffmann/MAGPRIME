#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import cvxpy as cp
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

EPS = 1e-12


def _unit_columns(A: np.ndarray) -> np.ndarray:
    return A / (np.linalg.norm(A, axis=0, keepdims=True) + EPS)


def synth_dictionary(M: int, n: int, mu_target: float, rng: np.random.Generator) -> np.ndarray:
    """Generate complex sensing matrix with approximate target coherence."""
    A = rng.normal(size=(M, n)) + 1j * rng.normal(size=(M, n))
    A = _unit_columns(A)
    if n >= 2:
        a0 = A[:, 0]
        v = A[:, 1] - (a0.conj().T @ A[:, 1]) * a0
        v /= (np.linalg.norm(v) + EPS)
        alpha = float(np.clip(mu_target, 0.0, 0.999))
        A[:, 1] = alpha * a0 + math.sqrt(max(1.0 - alpha**2, 0.0)) * v
    return _unit_columns(A)


def babel_function(A: np.ndarray) -> float:
    if A.size == 0:
        return 0.0
    C = _unit_columns(A)
    G = np.abs(C.conj().T @ C)
    np.fill_diagonal(G, 0.0)
    return float(np.max(np.sum(G, axis=0)))


def synth_coefficients(
    n: int,
    k_interferers: int,
    IAR: float,
    DR: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[int]]:
    x = np.zeros(n, dtype=complex)
    x[0] = 1.0
    support: List[int] = [0]
    if k_interferers <= 0:
        return x, support

    interferers = rng.choice(np.arange(1, n), size=k_interferers, replace=False)
    mags = np.ones(k_interferers)
    mags[0] = max(DR, 1.0)
    mags = mags / (np.linalg.norm(mags) + EPS) * math.sqrt(IAR)
    phases = np.exp(1j * rng.uniform(0, 2 * np.pi, size=k_interferers))
    x[interferers] = mags * phases
    support.extend(interferers.tolist())
    return x, support


def synth_observation(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    return A @ x


def calculate_delta_s(A: np.ndarray, x: np.ndarray) -> float:
    if np.linalg.norm(x) < EPS:
        return 0.0
    Ax_norm = np.linalg.norm(A @ x)
    x_norm = np.linalg.norm(x)
    ratio = (Ax_norm / (x_norm + EPS)) ** 2
    return float(max(abs(ratio - 1.0), abs(1.0 - ratio)))


def sparsity_exponent(k: float, M: float) -> float:
    return math.log1p(k) / (math.log1p(M) + EPS)


def sparsity_hinge(p: float) -> float:
    return max(0.0, p - 1.0)


@dataclass
class SolverConfig:
    tau: float = 0.01
    cs_iters: int = 3
    ssp_tol_deg: float = 20.0
    verbose: bool = False


def solve_ubss(A: np.ndarray, b: np.ndarray, cfg: SolverConfig) -> np.ndarray:
    n = A.shape[1]
    x_var = cp.Variable(n, complex=True)
    weights = np.ones(n) / max(n, 1)
    w = cp.Parameter(n, complex=False, value=weights, nonneg=True)
    b_param = cp.Parameter(A.shape[0], complex=True)
    b_param.value = b

    constraint = cp.norm(A.conj().T @ (A @ x_var - b_param), "inf") <= cfg.tau
    problem = cp.Problem(cp.Minimize(cp.sum(w.T @ cp.abs(x_var))), [constraint])

    data = b
    b_real = np.real(data)
    b_imag = np.imag(data)
    denom = (np.linalg.norm(b_real) * np.linalg.norm(b_imag) + EPS)
    cos_sim = float(np.dot(b_real, b_imag) / denom)
    threshold = math.cos(math.radians(cfg.ssp_tol_deg))
    ssp = cos_sim >= threshold

    for _ in range(cfg.cs_iters):
        try:
            problem.solve(warm_start=True, solver=cp.SCS, verbose=cfg.verbose)
            if problem.status in ("optimal", "optimal_inaccurate"):
                break
        except Exception:
            pass

        if x_var.value is None:
            init = np.zeros(n, dtype=complex)
            idx = int(np.argmin(np.abs(b_param.value)))
            init[0] = b_param.value[idx]
            x_var.value = init

        if ssp:
            w.value = 1.0 / (np.abs(x_var.value) + 0.01)
        else:
            delta = calculate_delta_s(A, x_var.value)
            if delta < math.sqrt(2.0) - 1.0:
                w.value = 1.0 / (np.abs(x_var.value) + 0.01)
                continue
            else:
                x_hat = np.abs(x_var.value)
                x_ratio = float(np.sum(x_hat[1:]) / (x_hat[0] + 0.01))
                w_val = w.value.copy()
                w_val[0] = np.clip(w_val[0] + 0.1 * (x_ratio - w_val[0]), 0.01, 100.0)
                w.value = w_val

    if x_var.value is None or problem.status not in ("optimal", "optimal_inaccurate"):
        problem.solve(solver=cp.SCS, verbose=cfg.verbose)
    return np.asarray(x_var.value).flatten()


@dataclass
class TrialConfig:
    seed: int
    M: int
    n: int
    k_interferers: int
    IAR: float
    DR: float
    mu_target: float
    tau: float


def run_trial(cfg: TrialConfig, solver_cfg: SolverConfig) -> Dict[str, float]:
    rng = np.random.default_rng(cfg.seed)
    A = synth_dictionary(cfg.M, cfg.n, cfg.mu_target, rng)
    x_true, support = synth_coefficients(cfg.n, cfg.k_interferers, cfg.IAR, cfg.DR, rng)
    b = synth_observation(A, x_true)

    local_cfg = SolverConfig(
        tau=cfg.tau,
        cs_iters=solver_cfg.cs_iters,
        ssp_tol_deg=solver_cfg.ssp_tol_deg,
        verbose=solver_cfg.verbose,
    )
    x_hat = solve_ubss(A, b, local_cfg)

    p_exp = sparsity_exponent(cfg.k_interferers, cfg.M)
    p_hinge = sparsity_hinge(p_exp)

    cond_support = float('inf')
    if len(support) > 0:
        A_sup = A[:, support]
        try:
            cond_support = float(np.linalg.cond(A_sup))
        except np.linalg.LinAlgError:
            cond_support = float('inf')

    a0 = A[:, 0]
    x_naive = (a0.conj().T @ b) / (a0.conj().T @ a0 + EPS)
    snr_in = 10.0 * math.log10((abs(x_true[0]) ** 2 + EPS) / (abs(x_true[0] - x_naive) ** 2 + EPS))
    snr_out = 10.0 * math.log10((abs(x_true[0]) ** 2 + EPS) / (abs(x_true[0] - x_hat[0]) ** 2 + EPS))
    delta_snr = snr_out - snr_in

    error_ratio = abs(x_true[0] - x_hat[0]) ** 2 / (abs(x_true[0]) ** 2 + EPS)

    return {
        "M": cfg.M,
        "n": cfg.n,
        "k": cfg.k_interferers,
        "rho_k": cfg.k_interferers / max(cfg.M, 1),
        "p_exponent": p_exp,
        "p_hinge": p_hinge,
        "IAR": cfg.IAR,
        "DR": cfg.DR,
        "mu_target": cfg.mu_target,
        "mu_A": babel_function(A),
        "tau": cfg.tau,
        "cond_support": cond_support,
        "snr_in": snr_in,
        "snr_out": snr_out,
        "delta_snr": delta_snr,
        "error_ratio": float(error_ratio.real),
    }


@dataclass
class SweepConfig:
    M_choices: Tuple[int, ...] = (3, 4)
    n_choices: Tuple[int, ...] = (6, 8)
    k_values: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    IAR_values: Tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
    DR_values: Tuple[float, ...] = (1.0, 3.0, 10.0, 30.0, 100.0)
    mu_values: Tuple[float, ...] = (0.0, 0.3, 0.6, 0.9)
    tau_values: Tuple[float, ...] = (0.005, 0.01, 0.02, 0.05)
    repeats: int = 5
    seed: int = 0


def _sample_trial_configs(sc: SweepConfig) -> List[TrialConfig]:
    rng = np.random.default_rng(sc.seed)
    configs: List[TrialConfig] = []
    for M in sc.M_choices:
        for n in sc.n_choices:
            for mu in sc.mu_values:
                for tau in sc.tau_values:
                    for IAR in sc.IAR_values:
                        for DR in sc.DR_values:
                            for k in sc.k_values:
                                if k >= n:
                                    continue
                                for _ in range(sc.repeats):
                                    seed = int(rng.integers(0, 2**32 - 1))
                                    configs.append(TrialConfig(seed, M, n, k, IAR, DR, mu, tau))
    return configs


def run_sweep(sc: SweepConfig, solver_cfg: SolverConfig, max_workers: Optional[int] = None) -> pd.DataFrame:
    configs = _sample_trial_configs(sc)
    rows: List[Dict[str, float]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_trial, cfg, solver_cfg) for cfg in configs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="simulating"):
            rows.append(fut.result())
    return pd.DataFrame(rows)


def fit_delta_snr(
    df: pd.DataFrame,
    mask: Optional[pd.Series] = None,
    label: str = "overall",
) -> Dict[str, float]:
    required = ["delta_snr", "IAR", "DR", "mu_A", "p_exponent", "p_hinge", "tau", "cond_support"]
    d = df[required].replace([np.inf, -np.inf], np.nan)
    if mask is not None:
        d = d[mask.reindex(df.index, fill_value=False)]
    d = d.dropna()
    if len(d) < 10:
        return {"status": "insufficient_data", "samples": int(len(d)), "label": label}

    y = d["delta_snr"].to_numpy()
    log_tau = np.log(d["tau"].to_numpy() + EPS)
    cond = d["cond_support"].to_numpy()
    cond = np.clip(cond, EPS, 1e12)
    log_cond = np.log(cond)
    X = np.column_stack([
        np.ones(len(d)),
        np.log(d["IAR"].to_numpy() + EPS),
        np.log(d["DR"].to_numpy() + EPS),
        np.log(1.0 / np.maximum(1.0 - d["mu_A"].to_numpy(), 1e-6)),
        d["p_exponent"].to_numpy(),
        d["p_hinge"].to_numpy(),
        log_tau,
        log_cond,
    ])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    preds = X @ beta
    resid = y - preds
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum(resid ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    dof = max(len(d) - X.shape[1], 1)
    sigma2 = float(resid @ resid / dof)
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    se = np.sqrt(np.clip(np.diag(XtX_inv) * sigma2, a_min=0.0, a_max=None))

    return {
        "status": "ok",
        "samples": int(len(d)),
        "label": label,
        "c0": float(beta[0]),
        "alpha": float(beta[1]),
        "beta": float(beta[2]),
        "gamma": float(beta[3]),
        "eta": float(beta[4]),
        "eta_h": float(beta[5]),
        "theta_tau": float(beta[6]),
        "theta_cond": float(beta[7]),
        "c0_se": float(se[0]),
        "alpha_se": float(se[1]),
        "beta_se": float(se[2]),
        "gamma_se": float(se[3]),
        "eta_se": float(se[4]),
        "eta_h_se": float(se[5]),
        "theta_tau_se": float(se[6]),
        "theta_cond_se": float(se[7]),
        "r2": float(r2),
    }


def _binned_mean_heatmap(x: np.ndarray, y: np.ndarray, z: np.ndarray, bins: int = 40):
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[mask]
    y = y[mask]
    z = z[mask]
    if x.size == 0:
        return None, None, None
    sum_hist, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=z)
    count_hist, _, _ = np.histogram2d(x, y, bins=bins)
    with np.errstate(invalid='ignore'):
        mean_hist = np.divide(sum_hist, count_hist, out=np.full_like(sum_hist, np.nan), where=count_hist > 0)
    return mean_hist, xedges, yedges



def plot_heatmaps(df: pd.DataFrame, filename: str) -> None:
    if df.empty or 'delta_snr' not in df:
        return
    x_log_iar = np.log10(df['IAR'].to_numpy() + EPS)
    y_log_dr = np.log10(df['DR'].to_numpy() + EPS)
    mu = df['mu_A'].to_numpy()
    p_exp = df['p_exponent'].to_numpy()
    log_tau = np.log10(df['tau'].to_numpy() + EPS)
    log_cond = np.log10(np.clip(df['cond_support'].to_numpy(), EPS, 1e12))
    delta = df['delta_snr'].to_numpy()

    pairs = [
        (x_log_iar, y_log_dr, 'log10 IAR', 'log10 DR'),
        (mu, p_exp, 'Babel sum', 'p exponent'),
        (log_tau, log_cond, 'log10 tau', 'log10 cond(A_S)'),
        (x_log_iar, mu, 'log10 IAR', 'Babel sum'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (x_data, y_data, xlabel, ylabel) in zip(axes.ravel(), pairs):
        mean_hist, xedges, yedges = _binned_mean_heatmap(x_data, y_data, delta)
        if mean_hist is None:
            ax.set_visible(False)
            continue
        im = ax.imshow(
            mean_hist.T,
            origin='lower',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect='auto',
            cmap='viridis',
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title('?SNR mean')
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)




def main(max_workers: Optional[int] = None) -> None:
    solver_cfg = SolverConfig(tau=0.01, cs_iters=3, ssp_tol_deg=20.0, verbose=False)
    sweep_cfg = SweepConfig(seed=1234)
    df = run_sweep(sweep_cfg, solver_cfg, max_workers=max_workers)

    mask_sparse = df["p_exponent"] <= 1.0
    mask_dense = df["p_exponent"] > 1.0

    fit_all = fit_delta_snr(df, label="overall")
    fit_sparse = fit_delta_snr(df, mask_sparse, label="p_leq_1")
    fit_dense = fit_delta_snr(df, mask_dense, label="p_gt_1")

    print("\nDelta-SNR fit (overall):", fit_all)
    print("Delta-SNR fit (p <= 1):", fit_sparse)
    print("Delta-SNR fit (p > 1):", fit_dense)

    plot_heatmaps(df, "cvx_microbench_heatmaps.png")

    df.to_csv("cvx_microbench_trials.csv", index=False)


if __name__ == "__main__":
    main()
