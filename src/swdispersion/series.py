"""Load a MuMax3 ``autosave`` output folder as one time series.

A dynamic MuMax3 run with ``autosave(m, t_sampl)`` writes one file per time
step: ``m000000.ovf``, ``m000001.ovf``, ... A full 4D array of all of them
(time, z, y, x, component) is often several GB. Most postprocessing needs
much less: one component, and the y direction averaged out when the system is
uniform (or periodic) along y. ``load_series`` therefore reduces every file
as soon as it is read, so memory use is set by the reduced array.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ovf import OVFHeader, read_ovf

_COMPONENTS = {"x": 0, "y": 1, "z": 2}


@dataclass
class TimeSeries:
    """One magnetization component sampled in time.

    Attributes
    ----------
    t : ndarray, shape (nt,)
        Time of each snapshot in seconds.
    data : ndarray, shape (nt, nz, nx), or (nt, nz, ny, nx) without y-averaging
    dx, dy, dz : float
        Cell size in metres.
    component : str
        Which component was loaded ("x", "y" or "z").
    """

    t: np.ndarray
    data: np.ndarray
    dx: float
    dy: float
    dz: float
    component: str

    @property
    def dt(self) -> float:
        """Time step from a straight-line fit to the snapshot times.

        MuMax3 saves a snapshot at the first solver step after each
        ``t_sampl`` interval, so the spacing jitters slightly. The slope of a
        linear fit is a better estimate than ``t[1] - t[0]``.
        """
        slope, _ = np.polyfit(np.arange(len(self.t)), self.t, 1)
        return float(slope)

    def save(self, path: str | Path) -> None:
        """Save to a compressed .npz file (much smaller than the OVF folder)."""
        np.savez_compressed(
            path, t=self.t, data=self.data, dx=self.dx, dy=self.dy, dz=self.dz,
            component=self.component,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TimeSeries":
        f = np.load(path)
        return cls(
            t=f["t"], data=f["data"], dx=float(f["dx"]), dy=float(f["dy"]),
            dz=float(f["dz"]), component=str(f["component"]),
        )


def list_snapshots(directory: str | Path, prefix: str = "m") -> list[Path]:
    """Return the files ``<prefix>000000.ovf``, ``<prefix>000001.ovf``, ... in order.

    Files such as ``m_stab.ovf`` or ``m_final.ovf`` do not match the pattern
    and are skipped.
    """
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.ovf$")
    files = [p for p in Path(directory).iterdir() if pattern.match(p.name)]
    return sorted(files, key=lambda p: int(pattern.match(p.name).group(1)))


def load_series(
    directory: str | Path,
    component: str = "z",
    prefix: str = "m",
    average_y: bool = True,
    z: int | slice | None = None,
    start: int = 0,
    stop: int | None = None,
) -> TimeSeries:
    """Read an autosave folder into a :class:`TimeSeries`.

    Parameters
    ----------
    directory
        The ``.out`` folder written by MuMax3.
    component
        "x", "y" or "z".
    prefix
        Name of the saved quantity (``m`` for ``autosave(m, ...)``).
    average_y
        Average over y. Use it when the system is uniform along y; it removes
        one dimension from the data.
    z
        Keep only these layers (an index or a slice). ``None`` keeps all.
    start, stop
        Range of snapshots to read, e.g. ``start=1`` to skip the snapshot
        taken before the excitation.
    """
    files = list_snapshots(directory, prefix)[start:stop]
    if not files:
        raise FileNotFoundError(f"no {prefix}NNNNNN.ovf files in {directory}")

    ci = _COMPONENTS[component]
    zsel = slice(None) if z is None else (slice(z, z + 1) if isinstance(z, int) else z)

    times, frames = [], []
    header: OVFHeader | None = None
    for path in files:
        data, header = read_ovf(path)
        frame = data[zsel, :, :, ci]  # (nz, ny, nx)
        if average_y:
            frame = frame.mean(axis=1)  # (nz, nx)
        frames.append(frame)
        times.append(header.time if header.time is not None else np.nan)

    t = np.asarray(times)
    if np.isnan(t).any():
        warnings.warn("some files have no time in the header; using index as time")
        t = np.arange(len(files), dtype=float)

    series = TimeSeries(
        t=t, data=np.stack(frames), dx=header.xstepsize, dy=header.ystepsize,
        dz=header.zstepsize, component=component,
    )
    _check_sampling(series)
    return series


def _check_sampling(series: TimeSeries, tolerance: float = 0.05) -> None:
    """Warn when the snapshots are far from evenly spaced in time.

    The FFT assumes equal spacing. A large deviation usually means a missing
    file or two runs mixed in one folder.
    """
    if len(series.t) < 3:
        return
    steps = np.diff(series.t)
    deviation = np.max(np.abs(steps - series.dt)) / series.dt
    if deviation > tolerance:
        warnings.warn(
            f"time steps vary by up to {deviation:.0%} of dt. Either snapshots "
            "are missing, or the solver step is not much smaller than the "
            "sampling interval (MuMax3 saves at the first step after each "
            "interval); MaxDt = t_sampl / 10 in the .mx3 script avoids the latter"
        )
