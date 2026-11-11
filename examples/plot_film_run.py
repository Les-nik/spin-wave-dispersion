"""Figure for the README from a run of ``film_dispersion.mx3``.

    python examples/plot_film_run.py path/to/film_dispersion.out

Draws the simulated dispersion with two analytical curves: the continuous
Kalinikos-Slavin formula, and the same formula with the finite-difference
exchange term that MuMax3 uses on 10 nm cells.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from swdispersion import compute_dispersion, kalinikos_slavin, load_series
from swdispersion.plotting import plot_dispersion

MS, AEX, THICKNESS, B0 = 800e3, 13e-12, 30e-9, 0.05  # as in film_dispersion.mx3


def main():
    run = Path(sys.argv[1])
    # The first snapshots are taken before the pulse; skip them.
    series = load_series(run, component="z", start=5)
    disp = compute_dispersion(series.data, series.dt, series.dx)

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    plot_dispersion(disp, ax=ax, fmax=30e9, kmax=150e6)
    k = np.linspace(-150e6, 150e6, 600)
    ax.plot(k * 1e-6, kalinikos_slavin(k, B0, MS, AEX, THICKNESS) * 1e-9,
            "--", color="#4dd0e1", lw=0.9, label="Kalinikos-Slavin")
    ax.plot(k * 1e-6, kalinikos_slavin(k, B0, MS, AEX, THICKNESS, cell_size=series.dx) * 1e-9,
            ":", color="white", lw=1.1, label="with discrete exchange (10 nm)")
    ax.legend(loc="lower center", frameon=False, labelcolor="white", fontsize=8)
    ax.set_title(f"MuMax3, 30 nm film, $m_z$, {len(series.t)} snapshots")
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "docs" / "figures" / "film_dispersion.png"
    fig.savefig(out, dpi=200)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
