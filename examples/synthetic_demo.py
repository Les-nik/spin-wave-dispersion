"""Check the whole pipeline on data with a known answer.

The script builds a fake MuMax3 output folder: a 30 nm permalloy-like film
in the Damon-Eshbach geometry, where m_z(t, x) is a sum of travelling waves
whose frequencies follow the Kalinikos-Slavin formula. It writes one OVF file
per time step, reads them back with ``load_series``, computes the dispersion
and draws the analytical curve on top. If the bright line and the dashed
curve agree, the reader, the time step, the FFT and the k sign are all right.

Run from the repository root:

    python examples/synthetic_demo.py
"""

from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np

from swdispersion import compute_dispersion, kalinikos_slavin, load_series, write_ovf
from swdispersion.plotting import plot_dispersion

# Film and field
MS, AEX, THICKNESS, B0 = 800e3, 13e-12, 30e-9, 0.05

# Grid and sampling, similar to a small MuMax3 run
NX, DX = 512, 10e-9
NT, DT = 400, 12.5e-12
TAU = 3e-9  # decay time, mimics damping


def make_run(directory: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    x = np.arange(NX) * DX
    k_grid = 2 * np.pi * np.fft.fftfreq(NX, DX)
    k_grid = k_grid[(np.abs(k_grid) > 0) & (np.abs(k_grid) < 150e6)]
    f_grid = kalinikos_slavin(k_grid, B0, MS, AEX, THICKNESS)
    phases = rng.uniform(0, 2 * np.pi, k_grid.size)
    # Damon-Eshbach waves are non-reciprocal in amplitude: waves moving one
    # way are localised at the top surface, the other way at the bottom.
    # A simple 2:1 ratio stands in for that here.
    weights = np.where(k_grid > 0, 1.0, 0.5) / np.sqrt(1 + (k_grid / 50e6) ** 2)

    for i in range(NT):
        t = i * DT
        mz = np.sum(
            weights[:, None]
            * np.cos(k_grid[:, None] * x[None, :] - 2 * np.pi * f_grid[:, None] * t + phases[:, None]),
            axis=0,
        )
        mz = 1e-3 * mz * np.exp(-t / TAU) / weights.size**0.5
        mz += 2e-6 * rng.normal(size=NX)  # numerical noise floor
        m = np.zeros((1, 1, NX, 3))
        m[0, 0, :, 2] = mz
        m[0, 0, :, 1] = np.sqrt(1 - mz**2)  # static magnetization along y
        write_ovf(directory / f"m{i:06d}.ovf", m, cellsize=(DX, 10e-9, THICKNESS), time=t)


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "synthetic.out"
        run.mkdir()
        make_run(run)
        series = load_series(run, component="z")

    print(f"time series {series.data.shape}, dt = {series.dt:.3e} s, dx = {series.dx:.1e} m")
    disp = compute_dispersion(series.data, series.dt, series.dx)

    fig, ax = plt.subplots(figsize=(5, 3.6))
    plot_dispersion(disp, ax=ax, fmax=25e9, kmax=120e6)
    k = np.linspace(-120e6, 120e6, 400)
    ax.plot(k * 1e-6, kalinikos_slavin(k, B0, MS, AEX, THICKNESS) * 1e-9,
            "--", color="#4dd0e1", lw=0.9, label="Kalinikos-Slavin")
    ax.legend(loc="upper center", frameon=False, labelcolor="#4dd0e1")
    ax.set_title("Synthetic Damon-Eshbach film, $m_z$")
    fig.tight_layout()
    fig.savefig(out / "synthetic_dispersion.png", dpi=200)
    print(f"saved {out / 'synthetic_dispersion.png'}")


if __name__ == "__main__":
    main()
