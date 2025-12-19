import numpy as np
import threading
from typing import Optional

try:
    from pynq import Overlay, allocate
except Exception:
    Overlay = None
    allocate = None


class HWState:
    overlay = None
    ip = None
    dim = 256
    in1_i32 = None
    in2_i32 = None
    out_i32 = None
    in1_i8 = None
    in2_i8 = None
    i8_supported = False
    ip_fast = None
    ip_slow = None
    fast_in1_i32 = None
    fast_in2_i32 = None
    fast_out_i32 = None
    slow_in1_i32 = None
    slow_in2_i32 = None
    slow_out_i32 = None
    lock_fast = None
    lock_slow = None
    lock_single = None

_state = HWState()


def init(bitfile_path: str = "overlay/matrix_mult_dual.bit", ip_name: str = "matrix_mult_0", dim: int = 256, ip_name_slow: Optional[str] = "matrix_mult_fabric_0") -> bool:
    _state.dim = int(dim)
    if Overlay is None or allocate is None:
        _state.overlay = None
        _state.ip = None
        _state.in1_i32 = None
        _state.in2_i32 = None
        _state.out_i32 = None
        _state.in1_i8 = None
        _state.in2_i8 = None
        _state.i8_supported = False
        _state.ip_fast = None
        _state.ip_slow = None
        _state.fast_in1_i32 = None
        _state.fast_in2_i32 = None
        _state.fast_out_i32 = None
        _state.slow_in1_i32 = None
        _state.slow_in2_i32 = None
        _state.slow_out_i32 = None
        return False
    try:
        ov = Overlay(bitfile_path)
        ip_fast = getattr(ov, ip_name)
        ip_slow = None
        if ip_name_slow is not None:
            try:
                ip_slow = getattr(ov, ip_name_slow)
            except Exception:
                ip_slow = None
        _state.overlay = ov
        _state.ip = ip_fast
        _state.ip_fast = ip_fast
        _state.ip_slow = ip_slow
        _state.lock_fast = threading.Lock()
        _state.lock_slow = threading.Lock()
        _state.lock_single = threading.Lock()
        d = _state.dim
        try:
            _state.in1_i8 = allocate(shape=(d * d,), dtype="i1")
            _state.in2_i8 = allocate(shape=(d * d,), dtype="i1")
            _state.i8_supported = True
        except Exception:
            _state.in1_i8 = None
            _state.in2_i8 = None
            _state.i8_supported = False
        _state.in1_i32 = allocate(shape=(d * d,), dtype="i4")
        _state.in2_i32 = allocate(shape=(d * d,), dtype="i4")
        _state.out_i32 = allocate(shape=(d * d,), dtype="i4")
        ip_fast.write(ip_fast.register_map.in1_1.address, _state.in1_i32.physical_address)
        ip_fast.write(ip_fast.register_map.in2_1.address, _state.in2_i32.physical_address)
        ip_fast.write(ip_fast.register_map.out_r_1.address, _state.out_i32.physical_address)
        if ip_slow is not None:
            _state.fast_in1_i32 = _state.in1_i32
            _state.fast_in2_i32 = _state.in2_i32
            _state.fast_out_i32 = _state.out_i32
            _state.slow_in1_i32 = allocate(shape=(d * d,), dtype="i4")
            _state.slow_in2_i32 = allocate(shape=(d * d,), dtype="i4")
            _state.slow_out_i32 = allocate(shape=(d * d,), dtype="i4")
            ip_slow.write(ip_slow.register_map.in1_1.address, _state.slow_in1_i32.physical_address)
            ip_slow.write(ip_slow.register_map.in2_1.address, _state.slow_in2_i32.physical_address)
            ip_slow.write(ip_slow.register_map.out_r_1.address, _state.slow_out_i32.physical_address)
        return True
    except Exception:
        _state.overlay = None
        _state.ip = None
        _state.in1_i32 = None
        _state.in2_i32 = None
        _state.out_i32 = None
        _state.in1_i8 = None
        _state.in2_i8 = None
        _state.i8_supported = False
        _state.ip_fast = None
        _state.ip_slow = None
        _state.fast_in1_i32 = None
        _state.fast_in2_i32 = None
        _state.fast_out_i32 = None
        _state.slow_in1_i32 = None
        _state.slow_in2_i32 = None
        _state.slow_out_i32 = None
        return False


def deinit():
    try:
        if _state.in1_i32 is not None:
            _state.in1_i32.freebuffer()
    except Exception:
        pass
    try:
        if _state.in2_i32 is not None:
            _state.in2_i32.freebuffer()
    except Exception:
        pass
    try:
        if _state.out_i32 is not None:
            _state.out_i32.freebuffer()
    except Exception:
        pass
    try:
        if _state.in1_i8 is not None:
            _state.in1_i8.freebuffer()
    except Exception:
        pass
    try:
        if _state.in2_i8 is not None:
            _state.in2_i8.freebuffer()
    except Exception:
        pass
    _state.in1_i32 = None
    _state.in2_i32 = None
    _state.out_i32 = None
    _state.in1_i8 = None
    _state.in2_i8 = None
    _state.i8_supported = False
    try:
        if _state.slow_in1_i32 is not None:
            _state.slow_in1_i32.freebuffer()
    except Exception:
        pass
    try:
        if _state.slow_in2_i32 is not None:
            _state.slow_in2_i32.freebuffer()
    except Exception:
        pass
    try:
        if _state.slow_out_i32 is not None:
            _state.slow_out_i32.freebuffer()
    except Exception:
        pass
    _state.fast_in1_i32 = None
    _state.fast_in2_i32 = None
    _state.fast_out_i32 = None
    _state.slow_in1_i32 = None
    _state.slow_in2_i32 = None
    _state.slow_out_i32 = None
    _state.ip_fast = None
    _state.ip_slow = None
    _state.lock_fast = None
    _state.lock_slow = None
    _state.lock_single = None
    _state.ip = None
    _state.overlay = None


def is_hardware_available() -> bool:
    return ((_state.ip_fast is not None) or (_state.ip is not None)) and (Overlay is not None) and (allocate is not None)


def hw_mmult_tile(a_tile: np.ndarray, b_tile: np.ndarray) -> np.ndarray:
    tile = a_tile.shape[0]
    if _state.ip is None or allocate is None:
        return a_tile.astype(np.int64) @ b_tile.astype(np.int64)
    in1 = _state.in1_i32
    in2 = _state.in2_i32
    out = _state.out_i32
    np.copyto(in1, a_tile.reshape(tile * tile).astype(np.int32, copy=False))
    np.copyto(in2, b_tile.reshape(tile * tile).astype(np.int32, copy=False))
    ip = _state.ip
    try:
        ip.write(0x00, 0x01)
        while True:
            reg = ip.read(0x00)
            if reg != 1:
                break
        out_np = np.array(out, copy=True).reshape(tile, tile)
        return out_np
    except Exception:
        return a_tile.astype(np.int64) @ b_tile.astype(np.int64)


def hw_mmult_tile_i8(a_tile: np.ndarray, b_tile: np.ndarray) -> np.ndarray:
    tile = a_tile.shape[0]
    if (_state.ip_fast is None and _state.ip is None) or allocate is None:
        return a_tile.astype(np.int64) @ b_tile.astype(np.int64)
    ip = _state.ip_fast if _state.ip_fast is not None else _state.ip
    out = _state.fast_out_i32 if _state.fast_out_i32 is not None else _state.out_i32
    in1 = _state.fast_in1_i32 if _state.fast_in1_i32 is not None else _state.in1_i32
    in2 = _state.fast_in2_i32 if _state.fast_in2_i32 is not None else _state.in2_i32
    np.copyto(in1, a_tile.reshape(tile * tile).astype(np.int32, copy=False))
    np.copyto(in2, b_tile.reshape(tile * tile).astype(np.int32, copy=False))
    lock = _state.lock_fast if _state.lock_fast is not None else _state.lock_single
    try:
        with lock:
            ip.write(0x00, 0x01)
            while True:
                reg = ip.read(0x00)
                if reg != 1:
                    break
            out_np = np.array(out, copy=True).reshape(tile, tile)
            return out_np.astype(np.int32, copy=False)
    except Exception:
        return a_tile.astype(np.int64) @ b_tile.astype(np.int64)

def hw_mmult_tile_i8_pipe(a_tile: np.ndarray, b_tile: np.ndarray, pipe_idx: int = 0) -> np.ndarray:
    tile = a_tile.shape[0]
    if ((_state.ip_fast is None and _state.ip is None) or allocate is None):
        return a_tile.astype(np.int64) @ b_tile.astype(np.int64)
    use_fast = True if pipe_idx == 0 else False
    ip = _state.ip_fast if (use_fast and _state.ip_fast is not None) else _state.ip_slow
    if ip is None:
        ip = _state.ip_fast if _state.ip_fast is not None else _state.ip
    out = (
        _state.fast_out_i32 if (use_fast and _state.fast_out_i32 is not None) else _state.slow_out_i32
    )
    in1 = (
        _state.fast_in1_i32 if (use_fast and _state.fast_in1_i32 is not None) else _state.slow_in1_i32
    )
    in2 = (
        _state.fast_in2_i32 if (use_fast and _state.fast_in2_i32 is not None) else _state.slow_in2_i32
    )
    if out is None or in1 is None or in2 is None:
        out = _state.out_i32
        in1 = _state.in1_i32
        in2 = _state.in2_i32
    np.copyto(in1, a_tile.reshape(tile * tile).astype(np.int32, copy=False))
    np.copyto(in2, b_tile.reshape(tile * tile).astype(np.int32, copy=False))
    lock = _state.lock_fast if (use_fast and _state.lock_fast is not None) else _state.lock_slow
    if lock is None:
        lock = _state.lock_single
    try:
        with lock:
            ip.write(0x00, 0x01)
            while True:
                reg = ip.read(0x00)
                if reg != 1:
                    break
            out_np = np.array(out, copy=True).reshape(tile, tile)
            return out_np.astype(np.int32, copy=False)
    except Exception:
        return a_tile.astype(np.int64) @ b_tile.astype(np.int64)

def available_pipelines_i8() -> int:
    c = 0
    if _state.ip_fast is not None:
        c += 1
    if _state.ip_slow is not None:
        c += 1
    if c == 0 and _state.ip is not None:
        c = 1
    return c
