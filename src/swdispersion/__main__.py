"""Command line: reduce a MuMax3 output folder and plot its dispersion.

    python -m swdispersion path/to/run.out --component z --save run.npz

The .npz file holds the reduced time series (one component, averaged over y),
usually a few MB instead of the GB of the original OVF files. It can be
copied off a cluster and analysed on a laptop with ``TimeSeries.load``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .dispersion import compute_dispersion
from .plotting import plot_dispersion
from .series import TimeSeries, load_series


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m swdispersion", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help=".out folder or a saved .npz file")
    parser.add_argument("--component", default="z", choices="xyz")
    parser.add_argument("--start", type=int, default=0, help="first snapshot to use")
    parser.add_argument("--fmax", type=float, default=None, help="upper frequency limit in GHz")
    parser.add_argument("--kmax", type=float, default=None, help="k limit in rad/um")
    parser.add_argument("--save", type=Path, default=None, help="write the reduced series to this .npz")
    parser.add_argument("--figure", type=Path, default=None, help="write the plot to this file")
    args = parser.parse_args(argv)

    if args.source.suffix == ".npz":
        series = TimeSeries.load(args.source)
    else:
        series = load_series(args.source, component=args.component, start=args.start)
        print(f"read {len(series.t)} snapshots, data shape {series.data.shape}")
    if args.save:
        series.save(args.save)
        print(f"saved {args.save}")

    disp = compute_dispersion(series.data, series.dt, series.dx)
    ax, _ = plot_dispersion(
        disp,
        fmax=None if args.fmax is None else args.fmax * 1e9,
        kmax=None if args.kmax is None else args.kmax * 1e6,
    )
    ax.set_title(f"$m_{series.component}$")
    plt.tight_layout()
    if args.figure:
        plt.savefig(args.figure, dpi=200)
        print(f"saved {args.figure}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
