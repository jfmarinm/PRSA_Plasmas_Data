# Python reproduction of manuscript Figures 1–5

**Manuscript:** *Fundamental-derivative criticality of ion-acoustic solitons in fluid-Poisson plasmas with general equations of state*

Open-source replacement for the author's MATLAB figure programs. The scripts reproduce the exact stationary fluid–Poisson calculations, reduced Gardner comparisons, and the figure styling as closely as Matplotlib permits. They produce editable vector PDF output and high-resolution PNG previews. No MATLAB or commercial toolbox is required.

## Installation

Use Python 3.10 or newer. From this repository directory:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# On Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
```

The only Python dependencies are **NumPy, SciPy and Matplotlib**. No system LaTeX installation is required. The scripts use the open-source Matplotlib STIX mathematical fonts, with Times/serif styling where available.

## One command for each figure

```bash
python fig1.py                      # FULL Gaussian fP simulation + Fig. 1
python fig2.py                      # EOS criticality map and amplitudes
python fig3.py                      # Focusing/defocusing regime branches
python fig4.py                      # Exact equilibria + Gardner phase portraits
python fig5.py                      # DL boundaries, flat-top law, localization
```

For a convenience launcher without the expensive production Figure 1 run:

```bash
python run_all.py
python run_all.py --quick-fig1     # includes the Figure 1 smoke test
python run_all.py --full-fig1      # includes the full-resolution Figure 1 simulation
```

Each script is executable independently from any working directory as long as the supplied shared `.py` modules remain beside it. By default outputs are in **`outputs/`** beside the scripts. Scripts 2–5 also regenerate numerical CSV tables. A common output switch is `--output-dir /your/destination`.

### Figure 1: expensive full simulation and data provenance

`python fig1.py` reproduces the author's **full** numerical setup through `fission_solver.py`:

- Spectral grid `N=8192`, domain length `L=2400`, Gaussian centre `x0=-900`, amplitude `0.20`, width `45`.
- EOS: `sigma=0.2`, `p1=1`, `p2=0.15`, `p3=0.02`.
- Conservative Fourier pseudospectral discretization (2/3 dealiasing), classical RK4, nonlinear Poisson equation by damped Newton and preconditioned GMRES.
- CFL `0.05`, final time `1200`, 600 requested output frames and right-going spectral initialization.


For a fast solver smoke test:

```bash
python fig1.py --quick
```

This writes `outputs/Fig_01_QUICK.pdf`, `Fig_01_QUICK.png`, and `gaussian_fission_QUICK.npz`. It uses reduced resolution, time and amplitude.

The full run saves `outputs/gaussian_fission.npz`, enabling subsequent rendering without recomputation:

```bash
python fig1.py --input outputs/gaussian_fission.npz
```

The right-panel open circles default to an **independent, linear-mode RK4 frequency measurement**, matching the standalone fallback in the author's MATLAB figure script. They are **not** measured frequencies from the full nonlinear fP simulation. To show independently measured frequencies instead, supply `--measurements-csv linear_dispersion_measurements.csv`, with columns `k,omega`.

The original MATLAB `parula` colormap is represented by Matplotlib `viridis`. This changes only the colour rendering, not the data.

### Figures 2 and 3: archival full-PDE markers

Figures 2 and 3 recompute the exact stationary Sagdeev branches and Gardner/mKdV predictions from the governing equations. Their six full-fluid–Poisson post-propagation amplitude markers in each figure are written to `_markers.csv` for transparency.

These **historical markers are not independently generated** by `fig2.py` or `fig3.py`. The stationary branch calculations and marker plotting require no additional files.

### Figure 4: reduced and exact descriptions

Top panels show spatial equilibrium branches from the reduced Gardner model and markers from the exact Bernoulli stationary relation. Bottom panels show **reduced Gardner Hamiltonian level sets and homoclinic separatrices**.

### Figure 5: exact stationary numerical methods

All data are recomputed within `fig5.py` by cancellation-resistant evaluations of `H=B(n;M)-ln(n)`, Gauss–Legendre quadratures, scalar root-finding and `solve_ivp`. The script verifies the exact rarefactive double-layer threshold and derives a numerical plateau width without loading CSVs. Only panel (d) uses logarithmic axes on *both* axes. Panel (c) plots width against the already-logarithmic coordinate `ln[1/(p3-p3DL)]` on linear axes.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests check numerical benchmarks for all five scripts. Figure 1 is tested with a short solver run, and Figures 2–5 use stationary numerical checks.

## Repository contents

```text
fig1.py ... fig5.py         Figure-level entry points
fission_solver.py           Full fP fission model needed by fig1.py
run_all.py                  Optional one-command launcher
plasma_common.py            Sagdeev / Bernoulli routines (Figs. 2–4)
figure_style.py             Shared visual style and exports
requirements.txt            Python package dependencies
tests/                      Physics and numerical regression tests
outputs/                    Generated files (PDF, PNG, CSV)
```
