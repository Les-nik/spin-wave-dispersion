"""Plotting helpers. Kept separate so the numerical code does not need matplotlib."""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from .dispersion import Dispersion


def plot_dispersion(
    disp: Dispersion,
    ax=None,
    fmax: float | None = None,
    kmax: float | None = None,
    log: bool = True,
    cmap: str = "magma",
    vmin_ratio: float = 1e-2,
):
    """Show the dispersion as an image: k in rad/um on x, f in GHz on y.

    ``log=True`` uses a logarithmic colour scale, which shows weak higher
    modes next to the strong fundamental one. ``vmin_ratio`` sets the lowest
    colour relative to the maximum. With the default 1e-2, everything more
    than 100 times weaker than the strongest mode is drawn black, which hides
    the numerical background.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    k = disp.k * 1e-6  # rad/m -> rad/um
    f = disp.f * 1e-9  # Hz -> GHz
    amp = disp.amplitude

    vmax = amp.max()
    norm = mcolors.LogNorm(vmin=vmax * vmin_ratio, vmax=vmax) if log else None
    # pcolormesh with "nearest" centres each pixel on its (k, f) value.
    mesh = ax.pcolormesh(
        k, f, np.clip(amp, vmax * vmin_ratio, None), shading="nearest",
        cmap=cmap, norm=norm, rasterized=True,
    )
    ax.set_xlabel(r"$k$ (rad/$\mu$m)")
    ax.set_ylabel(r"$f$ (GHz)")
    if fmax is not None:
        ax.set_ylim(0, fmax * 1e-9)
    if kmax is not None:
        ax.set_xlim(-kmax * 1e-6, kmax * 1e-6)
    return ax, mesh


def plot_mode_profile(profile: np.ndarray, dx: float, dz: float, axes=None):
    """Amplitude and phase of a (z, x) mode profile, one above the other."""
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
    nz, nx = profile.shape
    extent = [0, nx * dx * 1e9, 0, nz * dz * 1e9]  # nm

    im0 = axes[0].imshow(np.abs(profile), origin="lower", extent=extent,
                         aspect="auto", cmap="viridis")
    axes[0].set_ylabel("z (nm)")
    axes[0].set_title("amplitude")
    im1 = axes[1].imshow(np.angle(profile), origin="lower", extent=extent,
                         aspect="auto", cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[1].set_ylabel("z (nm)")
    axes[1].set_xlabel("x (nm)")
    axes[1].set_title("phase")
    return axes, (im0, im1)
