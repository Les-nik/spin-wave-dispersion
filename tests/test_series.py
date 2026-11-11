import numpy as np
import pytest

from swdispersion.ovf import write_ovf
from swdispersion.series import TimeSeries, list_snapshots, load_series


def _write_run(directory, nt=12, dt=1e-11, shape=(2, 3, 8), jitter=0.0):
    rng = np.random.default_rng(1)
    nz, ny, nx = shape
    for i in range(nt):
        m = np.zeros((nz, ny, nx, 3))
        m[..., 1] = 1.0
        m[..., 2] = i  # m_z equals the snapshot index, easy to check
        t = i * dt + jitter * dt * rng.uniform(-1, 1)
        write_ovf(directory / f"m{i:06d}.ovf", m, cellsize=(2e-9, 5e-9, 1e-9), time=t)
    # Files that must be ignored by list_snapshots:
    write_ovf(directory / "m_stab.ovf", np.zeros((nz, ny, nx, 3)), cellsize=(1, 1, 1))
    write_ovf(directory / "regions.ovf", np.zeros((nz, ny, nx, 1)), cellsize=(1, 1, 1))


def test_list_snapshots_sorted_numerically(tmp_path):
    _write_run(tmp_path, nt=12)
    names = [p.name for p in list_snapshots(tmp_path)]
    assert names[0] == "m000000.ovf"
    assert names[-1] == "m000011.ovf"
    assert len(names) == 12


def test_load_series_shapes(tmp_path):
    _write_run(tmp_path)
    s = load_series(tmp_path, component="z")
    assert s.data.shape == (12, 2, 8)  # y averaged out
    np.testing.assert_allclose(s.data[:, 0, 0], np.arange(12))

    s_full = load_series(tmp_path, average_y=False, z=1)
    assert s_full.data.shape == (12, 1, 3, 8)


def test_dt_from_fit_with_jitter(tmp_path):
    _write_run(tmp_path, nt=40, dt=1e-11, jitter=0.02)
    s = load_series(tmp_path)
    assert s.dt == pytest.approx(1e-11, rel=1e-3)


def test_save_and_load(tmp_path):
    _write_run(tmp_path)
    s = load_series(tmp_path, start=2)
    s.save(tmp_path / "run.npz")
    s2 = TimeSeries.load(tmp_path / "run.npz")
    np.testing.assert_array_equal(s.data, s2.data)
    assert s2.dx == s.dx and s2.component == "z"
    assert len(s2.t) == 10
