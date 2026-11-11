"""Read and write OVF 2.0 files, the format MuMax3 uses for saved fields.

An OVF 2.0 file has two parts:

1. A text header. Every line starts with ``#`` and holds ``key: value``
   pairs, e.g. ``# xnodes: 1430``. MuMax3 also writes the simulation time
   into a ``# Desc: Total simulation time:  1.2e-10  s`` line.
2. A data block, introduced by ``# Begin: Data Binary 4`` (float32),
   ``# Begin: Data Binary 8`` (float64) or ``# Begin: Data Text``.

In the binary formats the first number is a fixed "control number"
(1234567.0 for float32, 123456789012345678.0 for float64). The reader checks
it to make sure the byte order is right. After it come the values, with x
changing fastest, then y, then z, and the vector components interleaved:
``mx(0,0,0), my(0,0,0), mz(0,0,0), mx(1,0,0), ...``. That ordering is why a
plain ``reshape(z, y, x, valuedim)`` gives the right array.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Control numbers defined by the OVF 2.0 specification.
_CONTROL = {4: 1234567.0, 8: 123456789012345678.0}
# OVF 2.0 binary data is little-endian ("<").
_DTYPE = {4: np.dtype("<f4"), 8: np.dtype("<f8")}

_TIME_RE = re.compile(r"Total simulation time:\s*([-+0-9.eE]+)")


@dataclass
class OVFHeader:
    """The header fields needed for postprocessing."""

    xnodes: int
    ynodes: int
    znodes: int
    xstepsize: float
    ystepsize: float
    zstepsize: float
    valuedim: int
    time: float | None = None
    title: str = ""
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Shape of the data array: (z, y, x, components)."""
        return (self.znodes, self.ynodes, self.xnodes, self.valuedim)


def _parse_header(text: str) -> OVFHeader:
    raw: dict[str, str] = {}
    time = None
    for line in text.splitlines():
        line = line.lstrip("#").strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key, value = key.strip().lower(), value.strip()
        if key == "desc":
            match = _TIME_RE.search(value)
            if match:
                time = float(match.group(1))
            continue
        raw[key] = value

    try:
        return OVFHeader(
            xnodes=int(raw["xnodes"]),
            ynodes=int(raw["ynodes"]),
            znodes=int(raw["znodes"]),
            xstepsize=float(raw["xstepsize"]),
            ystepsize=float(raw["ystepsize"]),
            zstepsize=float(raw["zstepsize"]),
            valuedim=int(raw.get("valuedim", 1)),
            time=time,
            title=raw.get("title", ""),
            raw=raw,
        )
    except KeyError as missing:
        raise ValueError(f"OVF header is missing the field {missing}") from None


def read_ovf(path: str | Path) -> tuple[np.ndarray, OVFHeader]:
    """Read one OVF 2.0 file.

    Returns
    -------
    data : ndarray, shape (z, y, x, valuedim)
        The field values, as float32 for "Binary 4" files and float64 otherwise.
    header : OVFHeader
        Grid size, cell size, number of components and simulation time.
    """
    content = Path(path).read_bytes()

    if not content.startswith(b"# OOMMF OVF 2.0"):
        raise ValueError(f"{path}: only OVF 2.0 files are supported")

    marker = b"# Begin: Data "
    start = content.find(marker)
    if start < 0:
        raise ValueError(f"{path}: no data block found")

    header = _parse_header(content[:start].decode("ascii", errors="replace"))

    # The rest of the marker line says which format follows ("Binary 4", ...).
    line_end = content.index(b"\n", start)
    data_format = content[start + len(marker) : line_end].decode().strip().lower()
    n_values = int(np.prod(header.shape))

    if data_format.startswith("binary"):
        nbytes = int(data_format.split()[1])
        dtype = _DTYPE[nbytes]
        offset = line_end + 1
        control = np.frombuffer(content, dtype=dtype, count=1, offset=offset)[0]
        if control != _CONTROL[nbytes]:
            raise ValueError(f"{path}: wrong control number {control}")
        values = np.frombuffer(
            content, dtype=dtype, count=n_values, offset=offset + nbytes
        )
    elif data_format.startswith("text"):
        end = content.find(b"# End: Data", line_end)
        values = np.array(content[line_end + 1 : end].split(), dtype=float)
    else:
        raise ValueError(f"{path}: unknown data format '{data_format}'")

    # frombuffer gives a read-only view of the file bytes; copy() makes it
    # a normal array that the caller can modify.
    return values.reshape(header.shape).copy(), header


def write_ovf(
    path: str | Path,
    data: np.ndarray,
    cellsize: tuple[float, float, float],
    time: float = 0.0,
    title: str = "m",
    binary: int = 4,
) -> None:
    """Write an array of shape (z, y, x, valuedim) as an OVF 2.0 file.

    Used by the tests and the synthetic demo. ``binary`` is 4 (float32),
    8 (float64) or 0 (text).
    """
    data = np.asarray(data)
    if data.ndim == 3:
        data = data[..., np.newaxis]
    nz, ny, nx, dim = data.shape
    dx, dy, dz = cellsize
    labels = " ".join(f"{title}_{c}" for c in "xyz") if dim == 3 else title

    lines = [
        "# OOMMF OVF 2.0",
        "# Segment count: 1",
        "# Begin: Segment",
        "# Begin: Header",
        f"# Title: {title}",
        "# meshtype: rectangular",
        "# meshunit: m",
        "# xmin: 0",
        "# ymin: 0",
        "# zmin: 0",
        f"# xmax: {nx * dx}",
        f"# ymax: {ny * dy}",
        f"# zmax: {nz * dz}",
        f"# valuedim: {dim}",
        f"# valuelabels: {labels}",
        "# valueunits: " + " ".join(["1"] * dim),
        f"# Desc: Total simulation time:  {time}  s",
        f"# xbase: {dx / 2}",
        f"# ybase: {dy / 2}",
        f"# zbase: {dz / 2}",
        f"# xnodes: {nx}",
        f"# ynodes: {ny}",
        f"# znodes: {nz}",
        f"# xstepsize: {dx}",
        f"# ystepsize: {dy}",
        f"# zstepsize: {dz}",
        "# End: Header",
    ]

    with open(path, "wb") as f:
        if binary in (4, 8):
            dtype = _DTYPE[binary]
            lines.append(f"# Begin: Data Binary {binary}")
            f.write(("\n".join(lines) + "\n").encode("ascii"))
            f.write(np.array([_CONTROL[binary]], dtype=dtype).tobytes())
            f.write(data.astype(dtype).tobytes())
            f.write(f"\n# End: Data Binary {binary}\n".encode("ascii"))
        elif binary == 0:
            lines.append("# Begin: Data Text")
            f.write(("\n".join(lines) + "\n").encode("ascii"))
            np.savetxt(f, data.reshape(-1, dim), fmt="%.9g")
            f.write(b"# End: Data Text\n")
        else:
            raise ValueError("binary must be 4, 8 or 0 (text)")
        f.write(b"# End: Segment\n")
