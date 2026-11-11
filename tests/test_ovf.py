import numpy as np
import pytest

from swdispersion.ovf import read_ovf, write_ovf


@pytest.mark.parametrize("binary", [4, 8, 0])
def test_roundtrip(tmp_path, binary):
    rng = np.random.default_rng(0)
    data = rng.normal(size=(3, 4, 5, 3))  # (z, y, x, components)
    path = tmp_path / "m000000.ovf"
    write_ovf(path, data, cellsize=(2e-9, 5e-9, 1e-9), time=1.5e-10, binary=binary)

    read, header = read_ovf(path)

    assert read.shape == (3, 4, 5, 3)
    assert header.xnodes == 5 and header.ynodes == 4 and header.znodes == 3
    assert header.xstepsize == pytest.approx(2e-9)
    assert header.time == pytest.approx(1.5e-10)
    tol = 1e-6 if binary == 4 else 1e-8
    np.testing.assert_allclose(read, data, rtol=tol, atol=tol)


def test_axis_order(tmp_path):
    """x must change fastest in the file, so data[z, y, x] keeps its meaning."""
    data = np.zeros((2, 3, 4, 1))
    data[1, 2, 3, 0] = 7.0
    path = tmp_path / "q.ovf"
    write_ovf(path, data, cellsize=(1, 1, 1))
    read, _ = read_ovf(path)
    assert read[1, 2, 3, 0] == 7.0
    assert read.sum() == 7.0


def test_bad_control_number(tmp_path):
    path = tmp_path / "bad.ovf"
    write_ovf(path, np.ones((1, 1, 2, 3)), cellsize=(1, 1, 1))
    content = bytearray(path.read_bytes())
    pos = content.index(b"Binary 4\n") + len(b"Binary 4\n")
    content[pos : pos + 4] = np.array([1.0], dtype="<f4").tobytes()
    path.write_bytes(bytes(content))
    with pytest.raises(ValueError, match="control number"):
        read_ovf(path)


def test_rejects_ovf1(tmp_path):
    path = tmp_path / "old.ovf"
    path.write_bytes(b"# OOMMF: rectangular mesh v1.0\n")
    with pytest.raises(ValueError, match="OVF 2.0"):
        read_ovf(path)
