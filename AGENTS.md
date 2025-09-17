Here is a self-contained spec you can hand to an agent and get useful work back.

# Project premise

We want scaling laws that predict how well an Underdetermined Blind Source Separation (UBSS) method removes spacecraft magnetic interference from magnetometer data. UBSS operates in a time–frequency (TF) domain using a Non-Stationary Gabor Transform (NSGT). The algorithm clusters TF samples to estimate a local mixing matrix, then solves a weighted L1 problem per sample to separate the ambient field from interferers.

# What we are measuring

We will run Monte Carlo simulations that generate multi-sensor time series with a known ambient signal and several interference sources. We will transform to TF, run UBSS, and measure performance per bin and per scene. We will sweep three independent factors.

1. Magnetometer to active-source ratio per TF bin
   Let M be the number of sensors. Let k\_active be the number of sources that are nonzero in a bin. Define ρ\_k = k\_active / M.

2. Interference magnitude
   Define IAR = total interference energy per bin divided by ambient energy per bin. Also define DR = dynamic range among interferers in a bin, equal to max source magnitude divided by min nonzero source magnitude.

3. Placement geometry
   Interferers are magnetic dipoles. The mixing matrix columns depend on sensor locations, source locations, and dipole orientations. Geometry affects mutual coherence μ(A) and the condition number of A^H A, which influence identifiability.

# What we are fitting

Primary goal
Scaling laws for separation performance as a function of IAR, DR, μ(A), and ρ\_k. We will fit exponents in log space.

Secondary goal
Saturation behavior. At high IAR and low μ(A) the gains flatten. We will fit a simple saturating curve that captures the knee.

# Outputs that must be produced

1. A tidy dataframe per simulation seed with one row per TF bin that includes

   * Bin index
   * Complex ambient truth a\_bin and estimate â\_bin
   * Complex mixture vector b\_bin in C^M
   * Cluster label used in the bin
   * Local A used by the solver in that bin, or an identifier that maps to it
   * Masks for magnitude filter and SSP test
   * Diagnostics: μ(A), cond(A^H A), δ\_s surrogate from the code
   * Derived per-bin metrics: SNR\_bin, magnitude relative error, phase error
   * Derived per-bin IAR and DR if oracle source contributions are available. If not available, store proxies such as the ratio between the two largest centroid magnitudes in the bin

2. A scene-level dataframe with one row per simulation that includes

   * Energy-weighted SNR\_out for the ambient estimate
   * ΔSNR = SNR\_out − SNR\_in
   * nRMSE = ||a − â||2 / ||a||2
   * Median and percentile summaries of per-bin phase error
   * Fraction of SSP bins that passed filtering
   * Scene-level IAR and DR summaries
   * Aggregates of μ(A), cond, and δ\_s such as medians and percentiles

3. Fitted models with coefficients and confidence intervals

   * Log–log linear model for nRMSE or SNR\_out
   * Optional saturating model for ΔSNR

4. Plots

   * Log–log scatter with fit lines for nRMSE vs IAR, DR, μ(A), and ρ\_k
   * ΔSNR vs IAR with a saturating fit
   * Error vs μ(A) with separate curves for low and high ρ\_k
   * Distribution plots that compare SSP bins and non-SSP bins

# Metrics and formulas

Per-bin metrics

* SNR\_bin = 10 log10( |a|^2 / (|a − â|^2 + ε) )
* MagRE\_bin = | |â| − |a| | / (|a| + ε)
* Phase error Δφ\_bin = wrap( arg(â) − arg(a) ) in \[−π, π]

Scene-level metrics

* Energy-weighted SNR\_out = Σ |a|^2 SNR\_bin / Σ |a|^2
* nRMSE = ||a − â||2 / ||a||2, computed on time series after inverse NSGT
* ΔSNR = SNR\_out − SNR\_in

Fitting targets

* Preferred fit target for exponents: log nRMSE or log SNR\_out
* Communication target: ΔSNR with a saturating link

Scaling ansatz for exponents

* log nRMSE ≈ c0 − α log IAR + β log DR + γ log μ + η log(ρ\_k + ε)
  where c0 is the intercept. Expect α positive in magnitude, β positive, γ positive, and η positive.

Saturating model for ΔSNR

* ΔSNR ≈ ΔSNR\_max \[ 1 − (1 + (IAR / I0)^p )^−1 ] − γ log μ − η log(ρ\_k + ε) + ε0

# Required code hooks inside UBSS

Add an analysis mode that exposes TF products without changing solver behavior.

1. NSGT forward and stacking
   Return the NSGT object, stacked complex TF matrix B\_stack in C^{M × Nbins}, and a vector of subband lengths for inverse stitching.

2. Filtering and clustering artifacts
   Return magnitude mask, SSP mask, labels per kept bin, centroids, and the complex mixing matrix columns A for those centroids.

3. Reconstruction with logs
   When solving for x in each kept bin, record

* A used by that bin
* μ(A), cond(A^H A), δ\_s from calculate\_delta\_s(A, x)
* The selected cluster index

4. Per-bin metric computation
   Given access to ambient truth TF coefficients a\_bin, compute SNR\_bin, MagRE\_bin, and Δφ\_bin. If oracle per-source TF contributions are available, compute IAR and DR per bin. Otherwise compute scene-level IAR and DR from time-series energies and store a per-bin proxy, such as the ratio between the largest and second largest centroid magnitudes.

# Simulation design

Signal model

* Ambient a(t) from a real dataset or a controlled stochastic process
* K interference sources. Create TF-sparse activations with random phases. Include narrowband tones, chirps, and on–off keyed signals. Control DR by rescaling per-source amplitudes
* Geometry from magnetic dipoles. Sensor positions fixed or swept. Sources at varying distances and orientations. Use the standard dipole field model to compute gains

Sweeps

* ρ\_k: set activation probabilities so that expected active sources per bin run from 1 to M+2
* IAR: sweep median per-bin IAR over values such as 0.1, 0.3, 1, 3, 10, 30
* DR: sweep from 1 to 100
* Geometry: sweep array aperture to source distance ratio d/r, and randomize dipole orientations. Log μ and cond rather than trying to fit directly to d/r

Design of experiments

* Use a Latin hypercube in the space of IAR, DR, and a geometry index to cover broadly with fewer runs. For each design point, run multiple seeds

# Analysis workflow

1. Run simulation script to produce Parquet or CSV files with per-bin and per-scene tables
2. Compute scene-level summaries with energy weights
3. Fit log–log linear models for exponents. Use bootstrap resampling for confidence intervals
4. Fit the saturating ΔSNR model and extract ΔSNR\_max and I0 per geometry cohort
5. Produce plots and a brief table of exponents for two regimes

   * SSP-rich bins only
   * Non-SSP bins only

# File structure

* `ubss_mc.py`
  Generates simulations. Saves per-bin and per-scene tables. Supports CLI args for number of seeds and sweep ranges

* `ubss_tf_hooks.py`
  Contains NSGT forward, filtering, clustering artifacts, reconstruction with logs, and metric helpers. Imports your UBSS solver but does not change default behavior

* `ubss_scaling.ipynb`
  Loads tables. Computes summaries. Fits models. Generates figures

* `data/`
  Stores Parquet or CSV outputs. Use one file per seed for per-bin tables. Use one file per sweep for per-scene summaries

# Acceptance criteria

* Reproducible runs that regenerate the same summary stats when provided the same seed set
* Per-bin table includes all fields listed above and passes a basic sanity check
* Scene-level fits yield stable exponents with narrow bootstrap intervals over at least one order of magnitude in IAR
* Plots clearly show increasing error with increasing μ and ρ\_k, and a knee in ΔSNR vs IAR

# Pseudocode outline

```
for each design_point in DOE:
    for seed in seeds:
        synthesize ambient a(t) and K sources with target IAR and DR
        compute dipole mixing for chosen geometry to get A_true(t,f)
        mix to sensor channels
        run UBSS analysis mode on each axis:
            get B_stack, masks, labels, A_centroids
            for kept bins:
                solve weighted L1, log x_hat, μ, cond, δ_s, label
                compute per-bin SNR, MagRE, phase error using ambient TF truth
                compute IAR_bin and DR_bin if available
        inverse NSGT to time domain to get â(t)
        compute scene metrics: SNR_out, ΔSNR, nRMSE
        write per-bin rows and one scene row to disk
load all scene rows
fit log–log models, fit saturating ΔSNR, save figures and model summaries
```

# Notes for the agent

* Use complex Hermitian inner products for TF operations.
* Use energy weighting when aggregating bins so loud ambient regions are not underrepresented.
* Cache NSGT plan objects by length and fs to avoid repeated allocations.
* Start with a small grid of design points and confirm instrumentation, then scale up.
* See simulation_A.py and simulation_B.py for example simulations