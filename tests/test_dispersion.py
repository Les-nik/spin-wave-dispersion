import numpy as np
import pytest

from swdispersion.dispersion import compute_dispersion, mode_profile
from swdispersion.theory import GAMMA_E, MU0, kalinikos_slavin

NT, NX = 256, 128
DT, DX = 1e-11, 5e-9


def _travelling_wave(f_index, k_index, direction=+1):
    """cos(k x - 2 pi f t) moving in +x (direction=+1) or -x (-1), exactly on the FFT grid."""
    f0 = f_index / (NT * DT)
    k0 = 2 * np.pi * k_index / (NX * DX)
    t = np.arange(NT)[:, None] * DT
    x = np.arange(NX)[None, :] * DX
    return np.cos(direction * k0 * x - 2 * np.pi * f0 * t), f0, k0


@pytest.mark.parametrize("direction", [+1, -1])
def test_peak_position_and_sign(direction):
    data, f0, k0 = _travelling_wave(f_index=40, k_index=10, direction=direction)
    disp = compute_dispersion(data, DT, DX)
    i_f, i_k = np.unravel_index(np.argmax(disp.amplitude), disp.amplitude.shape)
    assert disp.f[i_f] == pytest.approx(f0)
    # A wave moving in +x must appear at +k, one moving in -x at -k.
    assert disp.k[i_k] == pytest.approx(direction * k0)


def test_extra_axes_are_averaged():
    data, _, _ = _travelling_wave(20, 5)
    stacked = np.stack([data, 2 * data, 3 * data], axis=1)  # (t, z, x)
    disp = compute_dispersion(stacked, DT, DX)
    assert disp.amplitude.shape == (NT // 2, NX)


def test_static_offset_removed():
    data, _, _ = _travelling_wave(20, 5)
    disp = compute_dispersion(data + 0.9, DT, DX)
    i_f, _ = np.unravel_index(np.argmax(disp.amplitude), disp.amplitude.shape)
    assert disp.f[i_f] > 0


def test_mode_profile_standing_wave():
    """A standing wave sin(kx) cos(wt): amplitude follows |sin(kx)|, phase jumps by pi."""
    f0 = 40 / (NT * DT)
    k0 = 2 * np.pi * 4 / (NX * DX)
    t = np.arange(NT)[:, None] * DT
    x = np.arange(NX)[None, :] * DX
    data = np.sin(k0 * x) * np.cos(2 * np.pi * f0 * t)

    prof = mode_profile(data, DT, f0)
    amp = np.abs(prof) / np.abs(prof).max()
    np.testing.assert_allclose(amp, np.abs(np.sin(k0 * x[0])), atol=1e-6)

    phase = np.angle(prof)
    positive = np.sin(k0 * x[0]) > 0.1
    negative = np.sin(k0 * x[0]) < -0.1
    jump = np.angle(np.exp(1j * (phase[positive].mean() - phase[negative].mean())))
    assert abs(abs(jump) - np.pi) < 1e-3


def test_theory_kittel_limit():
    """At k -> 0 an in-plane film follows f = gamma/2pi sqrt(B (B + mu0 Ms))."""
    B, Ms = 0.05, 800e3
    f = kalinikos_slavin(np.array([1.0]), B, Ms, 13e-12, 30e-9)
    kittel = GAMMA_E / (2 * np.pi) * np.sqrt(B * (B + MU0 * Ms))
    assert f[0] == pytest.approx(kittel, rel=1e-4)


def test_theory_geometries():
    """Damon-Eshbach waves go up in frequency with k, backward-volume waves first go down."""
    k = np.array([1.0, 5e6])
    args = (0.05, 800e3, 13e-12, 30e-9)
    de = kalinikos_slavin(k, *args, phi=np.pi / 2)
    bv = kalinikos_slavin(k, *args, phi=0.0)
    assert de[1] > de[0]
    assert bv[1] < bv[0]
