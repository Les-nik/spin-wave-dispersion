"""Dispersion and one mode profile from a real MuMax3 run, using the Python API.

    python examples/analyse_run.py path/to/film_dispersion.out --frequency 12

The steps are the same as in ``python -m swdispersion``, written out so they
can be changed: which component, which layers, which frequency to look at.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from swdispersion import compute_dispersion, load_series, mode_profile
from swdispersion.plotting import plot_dispersion, plot_mode_profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="MuMax3 .out folder")
    parser.add_argument("--frequency", type=float, default=12.0, help="mode frequency in GHz")
    parser.add_argument("--component", default="z")
    args = parser.parse_args()

    # 1. Read every snapshot, keep one component, average over y.
    #    start=1 skips the first file, which is taken before the pulse arrives.
    series = load_series(args.run, component=args.component, start=1)
    print(f"{len(series.t)} snapshots, dt = {series.dt:.3e} s, data {series.data.shape}")

    # 2. Dispersion: 2D FFT over (t, x), amplitudes averaged over z.
    disp = compute_dispersion(series.data, series.dt, series.dx)

    # 3. Mode profile at one frequency: amplitude and phase in every (z, x) cell.
    f0 = args.frequency * 1e9
    profile = mode_profile(series.data, series.dt, f0)  # shape (nz, nx)

    fig = plt.figure(figsize=(10, 4))
    ax_d = fig.add_subplot(1, 2, 1)
    plot_dispersion(disp, ax=ax_d)
    ax_d.axhline(args.frequency, color="w", lw=0.8, ls=":")
    axes_p = [fig.add_subplot(2, 2, 2), fig.add_subplot(2, 2, 4)]
    plot_mode_profile(profile, series.dx, series.dz, axes=axes_p)
    fig.tight_layout()
    fig.savefig(args.run.with_suffix(".png"), dpi=200)
    print(f"saved {args.run.with_suffix('.png')}")


if __name__ == "__main__":
    main()
