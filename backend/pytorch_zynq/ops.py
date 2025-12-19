import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from .hardware import hw_mmult_tile, hw_mmult_tile_i8, hw_mmult_tile_i8_pipe, available_pipelines_i8, is_hardware_available, _state


def _pad_tile(src: np.ndarray, tile: int) -> np.ndarray:
    h, w = src.shape
    if h == tile and w == tile:
        return src
    out = np.zeros((tile, tile), dtype=np.int32)
    out[:h, :w] = src
    return out

def _pad_tile_i8(src: np.ndarray, tile: int) -> np.ndarray:
    h, w = src.shape
    if h == tile and w == tile:
        return src
    out = np.zeros((tile, tile), dtype=np.int8)
    out[:h, :w] = src
    return out


def _quantize_to_int8(t: torch.Tensor):
    if t.numel() == 0:
        return torch.zeros_like(t, dtype=torch.int8), 1.0
    if t.dtype == torch.int8:
        return t, 1.0
    if hasattr(t, "is_quantized") and t.is_quantized:
        try:
            if t.dtype == torch.qint8:
                z = int(getattr(t, "q_zero_point", lambda: 0)())
                s = float(getattr(t, "q_scale", lambda: 1.0)())
                ir = t.int_repr().to(torch.int16)
                if z != 0:
                    ir = ir - z
                return ir.clamp_(-128, 127).to(torch.int8), float(s)
        except Exception:
            pass
    x = t.detach().to(torch.float32)
    s = x.abs().max().item()
    scale = 1.0 if s == 0 else (s / 127.0)
    q = torch.clamp(torch.round(x / scale), -128, 127).to(torch.int8)
    return q, float(scale)

def _dequantize_from_int32(c_int32: torch.Tensor, scale_a: float, scale_b: float, dtype_out: torch.dtype):
    if dtype_out in (torch.float32, torch.float16):
        out = c_int32.to(torch.float32) * (scale_a * scale_b)
        if dtype_out == torch.float16:
            out = out.to(torch.float16)
        return out
    if dtype_out == torch.int32:
        return c_int32
    if dtype_out == torch.int16:
        x = torch.clamp(c_int32, -32768, 32767)
        return x.to(torch.int16)
    return c_int32

def mmult(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.dim() != 2 or b.dim() != 2:
        raise RuntimeError("mmult supports only 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise RuntimeError("shapes not aligned for matmul")

    # always use accelerated kernel (with hardware or CPU fallback inside hw_* helpers)

    if a.dtype == torch.int8 and b.dtype == torch.int8:
        a_np = a.detach().cpu().contiguous().numpy()
        b_np = b.detach().cpu().contiguous().numpy()
        tile = int(_state.dim)
        N, M = a_np.shape
        _, P = b_np.shape
        out = np.zeros((N, P), dtype=np.int32)
        pipes = max(1, available_pipelines_i8())
        for i in range(0, N, tile):
            i_end = min(i + tile, N)
            for j in range(0, P, tile):
                j_end = min(j + tile, P)
                acc = np.zeros((i_end - i, j_end - j), dtype=np.int64)
                tasks = []
                with ThreadPoolExecutor(max_workers=pipes) as ex:
                    pipes_list = [0] if pipes == 1 else list(range(pipes))
                    si = 0
                    for k in range(0, M, tile):
                        k_end = min(k + tile, M)
                        A_block = a_np[i:i_end, k:k_end].astype(np.int8, copy=False)
                        B_block = b_np[k:k_end, j:j_end].astype(np.int8, copy=False)
                        A_p = _pad_tile_i8(A_block, tile)
                        B_p = _pad_tile_i8(B_block, tile)
                        tasks.append(ex.submit(hw_mmult_tile_i8_pipe, A_p, B_p, pipes_list[si % len(pipes_list)]))
                        si += 1
                    for fut in as_completed(tasks):
                        C_full = fut.result()
                        valid = C_full[: (i_end - i), : (j_end - j)]
                        acc += valid.astype(np.int64)
                out[i:i_end, j:j_end] = acc.astype(np.int32)
        return torch.from_numpy(out)

    if a.dtype in (torch.float16, torch.float32) or b.dtype in (torch.float16, torch.float32):
        qa, sa = _quantize_to_int8(a)
        qb, sb = _quantize_to_int8(b)
        A = qa.detach().cpu().contiguous().numpy()
        B = qb.detach().cpu().contiguous().numpy()
        tile = int(_state.dim)
        N, M = A.shape
        _, P = B.shape
        out = np.zeros((N, P), dtype=np.int32)
        pipes = max(1, available_pipelines_i8())
        for i in range(0, N, tile):
            i_end = min(i + tile, N)
            for j in range(0, P, tile):
                j_end = min(j + tile, P)
                acc = np.zeros((i_end - i, j_end - j), dtype=np.int64)
                tasks = []
                with ThreadPoolExecutor(max_workers=pipes) as ex:
                    pipes_list = [0] if pipes == 1 else list(range(pipes))
                    si = 0
                    for k in range(0, M, tile):
                        k_end = min(k + tile, M)
                        A_block = A[i:i_end, k:k_end].astype(np.int8, copy=False)
                        B_block = B[k:k_end, j:j_end].astype(np.int8, copy=False)
                        A_p = _pad_tile_i8(A_block, tile)
                        B_p = _pad_tile_i8(B_block, tile)
                        tasks.append(ex.submit(hw_mmult_tile_i8_pipe, A_p, B_p, pipes_list[si % len(pipes_list)]))
                        si += 1
                    for fut in as_completed(tasks):
                        C_full = fut.result()
                        valid = C_full[: (i_end - i), : (j_end - j)]
                        acc += valid.astype(np.int64)
                out[i:i_end, j:j_end] = acc.astype(np.int32)
        c32 = torch.from_numpy(out)
        dtype_out = torch.result_type(a, b)
        return _dequantize_from_int32(c32, sa, sb, dtype_out)

    a32 = (a if a.dtype == torch.int32 else a.to(torch.int32)).detach().cpu().contiguous()
    b32 = (b if b.dtype == torch.int32 else b.to(torch.int32)).detach().cpu().contiguous()
    if not is_hardware_available():
        return torch.matmul(a32, b32)
    a_np = a32.numpy()
    b_np = b32.numpy()
    tile = int(_state.dim)
    N, M = a_np.shape
    _, P = b_np.shape
    out = np.zeros((N, P), dtype=np.int32)
    for i in range(0, N, tile):
        i_end = min(i + tile, N)
        for j in range(0, P, tile):
            j_end = min(j + tile, P)
            acc = np.zeros((i_end - i, j_end - j), dtype=np.int64)
            for k in range(0, M, tile):
                k_end = min(k + tile, M)
                A_block = a_np[i:i_end, k:k_end].astype(np.int32, copy=False)
                B_block = b_np[k:k_end, j:j_end].astype(np.int32, copy=False)
                A_p = _pad_tile(A_block, tile)
                B_p = _pad_tile(B_block, tile)
                C_full = hw_mmult_tile(A_p, B_p)
                valid = C_full[: (i_end - i), : (j_end - j)]
                acc += valid.astype(np.int64)
            out[i:i_end, j:j_end] = acc.astype(np.int32)
    return torch.from_numpy(out)


def register_aten_impls():
    lib = torch.library.Library("aten", "IMPL", "PrivateUse1")
    try:
        lib.impl("matmul", mmult)
        lib.impl("mm", mmult)
    except Exception:
        pass
