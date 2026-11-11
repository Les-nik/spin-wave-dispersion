"""Spin-wave dispersion and mode profiles from MuMax3 simulations."""

from .dispersion import Dispersion, compute_dispersion, mode_profile
from .ovf import OVFHeader, read_ovf, write_ovf
from .series import TimeSeries, list_snapshots, load_series
from .theory import kalinikos_slavin

__all__ = [
    "Dispersion",
    "OVFHeader",
    "TimeSeries",
    "compute_dispersion",
    "kalinikos_slavin",
    "list_snapshots",
    "load_series",
    "mode_profile",
    "read_ovf",
    "write_ovf",
]

__version__ = "0.1.0"
