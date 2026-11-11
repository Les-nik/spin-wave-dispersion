"""Spin-wave dispersion f(k) and mode profiles from a magnetization time series.

Sign convention
---------------
A wave travelling in +x is written as ``m(t, x) = cos(k x - 2 pi f t)``.
``numpy.fft.fft`` uses ``exp(-i ...)`` on both axes, so a plain ``fft2`` over
(t, x) puts this wave at (-f, +k). Keeping only the positive frequencies would
then show it at negative k, i.e. the k axis would be mirrored. That does not
matter for reciprocal systems, but it does for non-reciprocal ones (DMI,
Damon-Eshbach surface modes). Here the time axis is transformed with
``exp(+i 2 pi f t)`` (``ifft`` times ``nt``), so waves moving in +x appear at
+k with f > 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Dispersion:
    """Spectral amplitude on a (frequency, wavenumber) grid.

    Attributes
    ----------
    f : ndarray, shape (nf,)
        Frequencies in Hz, from 0 up to the Nyquist frequency 1 / (2 dt).
    k : ndarray, shape (nk,)
        Wavenumbers in rad/m, ascending, centred on k = 0.
    amplitude : ndarray, shape (nf, nk)
        Mean absolute value of the Fourier transform.
    """

    f: np.ndarray
    k: np.ndarray
    amplitude: np.ndarray

    def peak_frequency(self, fmin: float = 0.0, fmax: float = np.inf) -> np.ndarray:
        """For each k, the frequency of the strongest signal in [fmin, fmax].

        Useful for comparing with an analytical curve or for fitting.
        """
        band = (self.f >= fmin) & (self.f <= fmax)
        idx = np.argmax(self.amplitude[band], axis=0)
        return self.f[band][idx]


def _time_window(nt: int, window: str | None) -> np.ndarray:
    if window is None:
        return np.ones(nt)
    if window == "hann":
        # A Hann window brings the signal smoothly to zero at both ends.
        # Without it, the abrupt cut at the end of the run spreads every
        # mode over many frequency bins ("spectral leakage").
        return np.hanning(nt)
    raise ValueError(f"unknown window '{window}'")


def compute_dispersion(
    data: np.ndarray,
    dt: float,
    dx: float,
    window: str | None = "hann",
    subtract_mean: bool = True,
) -> Dispersion:
    """2D Fourier transform over time (axis 0) and x (last axis).

    Parameters
    ----------
    data
        Array with time on the first axis and x on the last, e.g. the
        ``TimeSeries.data`` array of shape (nt, nz, nx). All axes in between
        (z, y) are transformed separately and their amplitudes averaged.
    dt, dx
        Time step (s) and cell size along x (m).
    window
        "hann" (default) or None.
    subtract_mean
        Remove the time average of every cell first. The static
        magnetization is then not counted as a huge f = 0 peak.
    """
    data = np.asarray(data, dtype=float)
    nt, nx = data.shape[0], data.shape[-1]

    if subtract_mean:
        data = data - data.mean(axis=0, keepdims=True)

    w = _time_window(nt, window).reshape((nt,) + (1,) * (data.ndim - 1))

    # exp(+i 2 pi f t) in time (see module docstring), exp(-i k x) in space.
    spectrum = np.fft.ifft(data * w, axis=0) * nt
    spectrum = np.fft.fft(spectrum, axis=-1)

    # Average |F| over all middle axes (z, y), keeping (f, k).
    amplitude = np.abs(spectrum)
    if amplitude.ndim > 2:
        amplitude = amplitude.mean(axis=tuple(range(1, amplitude.ndim - 1)))

    n_pos = nt // 2
    f = np.fft.fftfreq(nt, dt)[:n_pos]
    k = np.fft.fftshift(np.fft.fftfreq(nx, dx)) * 2 * np.pi
    amplitude = np.fft.fftshift(amplitude[:n_pos], axes=-1)
    return Dispersion(f=f, k=k, amplitude=amplitude)


def mode_profile(
    data: np.ndarray,
    dt: float,
    frequency: float,
    window: str | None = "hann",
) -> np.ndarray:
    """Complex amplitude of the oscillation at one frequency, in every cell.

    This is a single-frequency Fourier transform along time:
    ``sum_t w(t) m(t) exp(+i 2 pi f t)``. ``abs()`` of the result is the
    precession amplitude and ``angle()`` its phase, so a standing wave shows
    a constant phase with jumps of pi, and a travelling wave a phase that
    grows linearly along x.

    Parameters
    ----------
    data
        Time on axis 0, any spatial shape after it, e.g. (nt, nz, nx).
    frequency
        In Hz. Usually read off the dispersion plot.
    """
    data = np.asarray(data, dtype=float)
    nt = data.shape[0]
    data = data - data.mean(axis=0, keepdims=True)
    t = np.arange(nt) * dt
    weights = _time_window(nt, window) * np.exp(2j * np.pi * frequency * t)
    # tensordot sums over the time axis for every cell at once.
    return np.tensordot(weights, data, axes=(0, 0))
