"""Analytical dispersion of the lowest spin-wave mode in a thin film.

Kalinikos-Slavin approximation for the uniform-thickness mode (n = 0) with
unpinned surface spins [B. A. Kalinikos and A. N. Slavin, J. Phys. C 19,
7013 (1986)]:

    omega^2 = omega_0 (omega_0 + omega_M F)

    omega_0 = gamma (B_eff + 2 A k^2 / M_s)
    omega_M = gamma mu_0 M_s
    F = P + sin^2(theta) [1 - P (1 + cos^2 phi) + (omega_M / omega_0) P (1 - P) sin^2 phi]
    P = 1 - (1 - exp(-k d)) / (k d)

theta is the angle between the static magnetization and the film normal
(pi/2 for in-plane magnetization), phi the angle between k and the in-plane
part of the magnetization. Two common cases:

* Damon-Eshbach geometry (k perpendicular to in-plane M): theta = phi = pi/2
* Backward-volume geometry (k parallel to in-plane M): theta = pi/2, phi = 0

B_eff is the internal field along M: the applied field plus anisotropy and
demagnetizing contributions that are not already in F. For a film with
perpendicular anisotropy K_u and in-plane M, B_eff = B_0 - 2 K_u / M_s is a
first approximation.

MuMax3 computes the exchange field with finite differences on cells of size
a, which replaces k^2 in the exchange term by (2 sin(k a / 2) / a)^2. Pass
``cell_size`` to use the same discrete form; the difference reaches about
1 GHz at k a = 1 for permalloy.
"""

from __future__ import annotations

import numpy as np

MU0 = 4e-7 * np.pi
GAMMA_E = 1.760859e11  # electron gyromagnetic ratio, rad / (s T)


def kalinikos_slavin(
    k: np.ndarray,
    B_eff: float,
    Ms: float,
    Aex: float,
    thickness: float,
    theta: float = np.pi / 2,
    phi: float = np.pi / 2,
    gamma: float = GAMMA_E,
    cell_size: float | None = None,
) -> np.ndarray:
    """Frequency in Hz of the lowest thickness mode for wavenumbers ``k`` (rad/m).

    Units: B_eff in T, Ms in A/m, Aex in J/m, thickness in m.
    Negative k gives the same frequency as +k (the formula is reciprocal).
    ``cell_size`` (m) switches to the discrete exchange term used by MuMax3.
    """
    k = np.abs(np.asarray(k, dtype=float))
    # Avoid 0/0 at k = 0; P -> 0 there.
    kd = np.where(k == 0, 1e-12, k * thickness)
    P = 1 - (1 - np.exp(-kd)) / kd

    k2_exchange = k**2 if cell_size is None else (2 * np.sin(k * cell_size / 2) / cell_size) ** 2
    omega_0 = gamma * (B_eff + 2 * Aex * k2_exchange / Ms)
    omega_M = gamma * MU0 * Ms
    F = P + np.sin(theta) ** 2 * (
        1 - P * (1 + np.cos(phi) ** 2)
        + omega_M / omega_0 * P * (1 - P) * np.sin(phi) ** 2
    )
    return np.sqrt(omega_0 * (omega_0 + omega_M * F)) / (2 * np.pi)
