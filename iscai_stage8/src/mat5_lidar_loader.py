from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

MI_INT8 = 1
MI_UINT8 = 2
MI_INT16 = 3
MI_UINT16 = 4
MI_INT32 = 5
MI_UINT32 = 6
MI_SINGLE = 7
MI_DOUBLE = 9
MI_INT64 = 12
MI_UINT64 = 13
MI_MATRIX = 14
MI_COMPRESSED = 15

class Mat5Error(ValueError):
    pass

def _pad8(n: int) -> int:
    return (n + 7) & ~7

def _read_element(buf: bytes, pos: int, endian: str):
    if pos + 8 > len(buf):
        raise Mat5Error("truncated MAT element")
    first = struct.unpack_from(endian + "I", buf, pos)[0]
    if (first >> 16) != 0:
        dtype = first & 0xFFFF
        nbytes = first >> 16
        data = buf[pos + 4:pos + 4 + nbytes]
        return dtype, data, pos + 8
    dtype, nbytes = struct.unpack_from(endian + "II", buf, pos)
    start = pos + 8
    end = start + nbytes
    if end > len(buf):
        raise Mat5Error("truncated MAT payload")
    return dtype, buf[start:end], start + _pad8(nbytes)

def _matrix_to_array(data: bytes, endian: str, expected_name: str):
    pos = 0
    _, flags, pos = _read_element(data, pos, endian)
    if len(flags) < 4:
        raise Mat5Error("invalid array flags")

    _, dims_raw, pos = _read_element(data, pos, endian)
    if len(dims_raw) % 4 != 0:
        raise Mat5Error("invalid dimensions")
    dims = tuple(struct.unpack(endian + "i" * (len(dims_raw)//4), dims_raw))

    _, name_raw, pos = _read_element(data, pos, endian)
    name = name_raw.decode("utf-8", errors="strict").rstrip("\x00")
    if name != expected_name:
        return None

    dtype, numeric_raw, pos = _read_element(data, pos, endian)
    if dtype != MI_DOUBLE:
        raise Mat5Error(f"{expected_name} must be MATLAB double, dtype tag={dtype}")

    dt = np.dtype(endian + "f8")
    arr = np.frombuffer(numeric_raw, dtype=dt)
    expected = 1
    for d in dims:
        expected *= int(d)
    if arr.size != expected:
        raise Mat5Error(f"numeric size mismatch: {arr.size} vs {expected}")

    # MATLAB stores arrays column-major.
    arr = arr.reshape(dims, order="F")
    return np.array(arr, dtype=np.float64, copy=True)

def _scan(buf: bytes, endian: str, expected_name: str):
    pos = 0
    while pos + 8 <= len(buf):
        dtype, payload, next_pos = _read_element(buf, pos, endian)
        pos = next_pos
        if dtype == MI_COMPRESSED:
            out = _scan(zlib.decompress(payload), endian, expected_name)
            if out is not None:
                return out
        elif dtype == MI_MATRIX:
            out = _matrix_to_array(payload, endian, expected_name)
            if out is not None:
                return out
    return None

def load_mat5_double_matrix(path, expected_name="data", expected_shape=None):
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 128:
        raise Mat5Error("MAT file too small")
    tag = raw[126:128]
    if tag == b"IM":
        endian = "<"
    elif tag == b"MI":
        endian = ">"
    else:
        raise Mat5Error(f"unsupported MATLAB endian tag: {tag!r}")

    arr = _scan(raw[128:], endian, expected_name)
    if arr is None:
        raise Mat5Error(f"variable {expected_name!r} not found")
    if expected_shape is not None and tuple(arr.shape) != tuple(expected_shape):
        raise Mat5Error(f"shape mismatch: {arr.shape} vs {tuple(expected_shape)}")
    if not np.isfinite(arr).all():
        raise Mat5Error("non-finite LiDAR values")
    return arr
