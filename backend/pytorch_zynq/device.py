import torch
import numpy as np
from .hardware import _state, is_hardware_available, hw_mmult_tile_i8, available_pipelines_i8
from typing import Set

_registered = False

def register_zynq_device(name: str = "zynq"):
    global _registered
    try:
        if hasattr(torch.utils, "rename_privateuse1_backend"):
            torch.utils.rename_privateuse1_backend(name)
        elif hasattr(torch._C, "_set_privateuse1_backend_name"):
            torch._C._set_privateuse1_backend_name(name)
        else:
            raise RuntimeError("PyTorch does not expose PrivateUse1 backend rename on this build")
        try:
            backend_name = name
            try:
                if hasattr(torch._C, "_get_privateuse1_backend_name"):
                    backend_name = torch._C._get_privateuse1_backend_name()
            except Exception:
                backend_name = name
            if hasattr(torch, "_register_device_module"):
                try:
                    torch._register_device_module(backend_name, "pytorch_zynq.device")
                except Exception:
                    pass
                try:
                    torch._register_device_module("zynq", "pytorch_zynq.device")
                except Exception:
                    pass
        except Exception:
            pass
        try:
            enable_full_device()
            register_privateuse1_impls()
        except Exception:
            pass
        _registered = True
    except Exception:
        _registered = False
        raise

def is_registered() -> bool:
    return _registered

_orig_matmul = torch.matmul
_orig_mm = torch.mm
_orig_bmm = torch.bmm if hasattr(torch, "bmm") else None
_orig_addmm = torch.addmm
_orig_tensor_to = torch.Tensor.to
_orig_module_to = torch.nn.Module.to
import torch.nn.functional as F
_orig_linear_func = F.linear
_orig_conv2d_func = F.conv2d

_zynq_ids: Set[int] = set()

def _is_zynq_device_arg(args, kwargs):
    if len(args) > 0 and isinstance(args[0], torch.device):
        return args[0].type in ("zynq", "privateuseone")
    if len(args) > 0 and isinstance(args[0], str):
        return args[0] in ("zynq", "privateuseone")
    dev = kwargs.get("device", None)
    if isinstance(dev, torch.device):
        return dev.type in ("zynq", "privateuseone")
    if isinstance(dev, str):
        return dev in ("zynq", "privateuseone")
    return False

def _is_zynq_tensor(t: torch.Tensor) -> bool:
    try:
        return id(t) in _zynq_ids
    except Exception:
        return False

def _mark_zynq(t):
    try:
        _zynq_ids.add(id(t))
    except Exception:
        pass
    return t

def _tensor_to(self, *args, **kwargs):
    if _is_zynq_device_arg(args, kwargs):
        out = _orig_tensor_to(self, device="cpu")
        return _mark_zynq(out)
    dev = kwargs.get("device", None)
    if isinstance(dev, (str, torch.device)):
        if (isinstance(dev, str) and dev == "cpu") or (isinstance(dev, torch.device) and dev.type == "cpu"):
            out = _orig_tensor_to(self, device="cpu")
            try:
                if _is_zynq_tensor(out):
                    try:
                        _zynq_ids.discard(id(out))
                    except Exception:
                        pass
            except Exception:
                pass
            return out
    if len(args) > 0:
        a0 = args[0]
        if isinstance(a0, str) and a0 == "cpu":
            out = _orig_tensor_to(self, device="cpu")
            try:
                if _is_zynq_tensor(out):
                    try:
                        _zynq_ids.discard(id(out))
                    except Exception:
                        pass
            except Exception:
                pass
            return out
        if isinstance(a0, torch.device) and a0.type == "cpu":
            out = _orig_tensor_to(self, device="cpu")
            try:
                if _is_zynq_tensor(out):
                    try:
                        _zynq_ids.discard(id(out))
                    except Exception:
                        pass
            except Exception:
                pass
            return out
    return _orig_tensor_to(self, *args, **kwargs)

def _module_to(self, *args, **kwargs):
    if _is_zynq_device_arg(args, kwargs):
        out = _orig_module_to(self, device="cpu")
        for p in out.parameters():
            _mark_zynq(p)
        for _, b in out.named_buffers():
            _mark_zynq(b)
        setattr(out, "_zynq_enabled", True)
        return out
    dev = kwargs.get("device", None)
    if isinstance(dev, (str, torch.device)):
        if (isinstance(dev, str) and dev == "cpu") or (isinstance(dev, torch.device) and dev.type == "cpu"):
            out = _orig_module_to(self, device="cpu")
            for p in out.parameters():
                try:
                    if _is_zynq_tensor(p):
                        _zynq_ids.discard(id(p))
                except Exception:
                    pass
            for _, b in out.named_buffers():
                try:
                    if _is_zynq_tensor(b):
                        _zynq_ids.discard(id(b))
                except Exception:
                    pass
            try:
                if hasattr(out, "_zynq_enabled"):
                    delattr(out, "_zynq_enabled")
            except Exception:
                pass
            return out
    if len(args) > 0:
        a0 = args[0]
        if isinstance(a0, str) and a0 == "cpu":
            out = _orig_module_to(self, device="cpu")
            for p in out.parameters():
                try:
                    if _is_zynq_tensor(p):
                        _zynq_ids.discard(id(p))
                except Exception:
                    pass
            for _, b in out.named_buffers():
                try:
                    if _is_zynq_tensor(b):
                        _zynq_ids.discard(id(b))
                except Exception:
                    pass
            try:
                if hasattr(out, "_zynq_enabled"):
                    delattr(out, "_zynq_enabled")
            except Exception:
                pass
            return out
        if isinstance(a0, torch.device) and a0.type == "cpu":
            out = _orig_module_to(self, device="cpu")
            for p in out.parameters():
                try:
                    if _is_zynq_tensor(p):
                        _zynq_ids.discard(id(p))
                except Exception:
                    pass
            for _, b in out.named_buffers():
                try:
                    if _is_zynq_tensor(b):
                        _zynq_ids.discard(id(b))
                except Exception:
                    pass
            try:
                if hasattr(out, "_zynq_enabled"):
                    delattr(out, "_zynq_enabled")
            except Exception:
                pass
            return out
    return _orig_module_to(self, *args, **kwargs)

def _tile_pad_i8(src: np.ndarray, tile: int) -> np.ndarray:
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
    if s == 0:
        scale = 1.0
    else:
        scale = s / 127.0
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

def _zynq_matmul(a: torch.Tensor, b: torch.Tensor):
    try:
        if not (_is_zynq_tensor(a) and _is_zynq_tensor(b)):
            return _orig_matmul(a, b)
        if torch.is_grad_enabled():
            return _orig_matmul(a, b)
        if not is_hardware_available():
            return _orig_matmul(a, b)
        tile = int(_state.dim)
        da = a.dtype
        db = b.dtype
        if da in (torch.float16, torch.float32) or db in (torch.float16, torch.float32):
            qa, sa = _quantize_to_int8(a)
            qb, sb = _quantize_to_int8(b)
            from .ops import mmult as _mm
            c32 = _mm(qa, qb)
            dtype_out = torch.result_type(da, db)
            return _dequantize_from_int32(c32, sa, sb, dtype_out)
        else:
            from .ops import mmult as _mm
            if da == torch.int8 and db == torch.int8:
                c32 = _mm(a, b)
                dtype_out = torch.int32
                return _dequantize_from_int32(c32, 1.0, 1.0, dtype_out)
            a32 = a.to(torch.int32)
            b32 = b.to(torch.int32)
            c32 = _mm(a32, b32)
            dtype_out = torch.result_type(da, db)
            return _dequantize_from_int32(c32, 1.0, 1.0, dtype_out)
    except Exception:
        return _orig_matmul(a, b)

def _zynq_mm(a: torch.Tensor, b: torch.Tensor):
    return _zynq_matmul(a, b)

def _zynq_bmm(a: torch.Tensor, b: torch.Tensor):
    try:
        if not (_is_zynq_tensor(a) and _is_zynq_tensor(b)):
            return _orig_bmm(a, b)
        if torch.is_grad_enabled():
            return _orig_bmm(a, b)
        if not is_hardware_available():
            return _orig_bmm(a, b)
        Bn, N, M = a.shape
        _, M2, P = b.shape
        if M != M2:
            return _orig_bmm(a, b)
        outs = []
        for idx in range(Bn):
            outs.append(_zynq_matmul(a[idx], b[idx]))
        return torch.stack(outs, 0)
    except Exception:
        return _orig_bmm(a, b)

def _zynq_addmm(input: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor, beta=1, alpha=1):
    try:
        if not (_is_zynq_tensor(mat1) and _is_zynq_tensor(mat2)):
            return _orig_addmm(input, mat1, mat2, beta=beta, alpha=alpha)
        if torch.is_grad_enabled():
            return _orig_addmm(input, mat1, mat2, beta=beta, alpha=alpha)
        prod = _zynq_matmul(mat1, mat2)
        base = input if input is not None else torch.zeros_like(prod)
        return base.mul(int(beta)) + prod.mul(int(alpha))
    except Exception:
        return _orig_addmm(input, mat1, mat2, beta=beta, alpha=alpha)

def _zynq_linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None):
    try:
        if torch.is_grad_enabled():
            return _orig_linear_func(input, weight, bias)
        if not (_is_zynq_tensor(input) and _is_zynq_tensor(weight)):
            return _orig_linear_func(input, weight, bias)
        if not is_hardware_available():
            return _orig_linear_func(input, weight, bias)
        out = _zynq_matmul(input, weight.t())
        if bias is not None:
            out = out + bias.to(out.dtype)
        return out
    except Exception:
        return _orig_linear_func(input, weight, bias)


def _zynq_conv2d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None, stride=1, padding=0, dilation=1, groups=1):
    try:
        use_orig = False
        if torch.is_grad_enabled():
            use_orig = True
        else:
            if not (_is_zynq_tensor(input) and _is_zynq_tensor(weight)):
                use_orig = True
        if groups < 1:
            use_orig = True
        if use_orig:
            return _orig_conv2d_func(input, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        N, C, H, W = input.shape
        OC, Cw, KH, KW = weight.shape
        if Cw != C:
            return _orig_conv2d_func(input, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        if isinstance(stride, int):
            sh, sw = stride, stride
        else:
            sh, sw = stride
        if isinstance(padding, int):
            ph, pw = padding, padding
        else:
            ph, pw = padding
        if isinstance(dilation, int):
            dh, dw = dilation, dilation
        else:
            dh, dw = dilation
        OH = (H + 2 * ph - dh * (KH - 1) - 1) // sh + 1
        OW = (W + 2 * pw - dw * (KW - 1) - 1) // sw + 1
        inp32 = input.to(torch.float32)
        if groups == 1:
            cols = F.unfold(inp32, (KH, KW), dilation=dilation, padding=padding, stride=stride)
            cols_t = cols.transpose(1, 2)
            A = cols_t
            W2 = weight.to(torch.float32).view(OC, C * KH * KW).t()
            outs = []
            from concurrent.futures import ThreadPoolExecutor
            pipes = max(1, available_pipelines_i8())
            with ThreadPoolExecutor(max_workers=pipes) as ex:
                tasks = []
                for idx in range(N):
                    tasks.append(ex.submit(_zynq_matmul, A[idx], W2))
                for fut in tasks:
                    outs.append(fut.result())
            prod = torch.stack(outs, 0)
            out = prod.transpose(1, 2)
        else:
            OCg = OC // groups
            Cg = C // groups
            outs_g = []
            for g in range(groups):
                inp_g = inp32[:, g * Cg : (g + 1) * Cg, :, :]
                w_g = weight[g * OCg : (g + 1) * OCg, :, :, :]
                cols_g = F.unfold(inp_g, (KH, KW), dilation=dilation, padding=padding, stride=stride)
                A = cols_g.transpose(1, 2)
                W2 = w_g.to(torch.float32).view(OCg, Cg * KH * KW).t()
                outs = []
                from concurrent.futures import ThreadPoolExecutor
                pipes = max(1, available_pipelines_i8())
                with ThreadPoolExecutor(max_workers=pipes) as ex:
                    tasks = []
                    for idx in range(N):
                        tasks.append(ex.submit(_zynq_matmul, A[idx], W2))
                    for fut in tasks:
                        outs.append(fut.result())
                prod = torch.stack(outs, 0)
                out_g = prod.transpose(1, 2)
                outs_g.append(out_g)
            out = torch.cat(outs_g, dim=1)
        if bias is not None:
            if groups == 1:
                out = out + bias.view(1, OC, 1).to(out.dtype)
            else:
                OCg = OC // groups
                outs_g = []
                for g in range(groups):
                    b = bias[g * OCg : (g + 1) * OCg]
                    outs_g.append(out[:, g * OCg : (g + 1) * OCg, :].add(b.view(1, OCg, 1).to(out.dtype)))
                out = torch.cat(outs_g, dim=1)
        return out.view(N, OC, OH, OW)
    except Exception:
        return _orig_conv2d_func(input, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)


def enable_full_device():
    torch.Tensor.to = _tensor_to
    torch.nn.Module.to = _module_to
    torch.matmul = _zynq_matmul
    torch.mm = _zynq_mm
    if _orig_bmm is not None:
        torch.bmm = _zynq_bmm
    torch.addmm = _zynq_addmm
    F.linear = _zynq_linear
    F.conv2d = _zynq_conv2d

def disable_full_device():
    torch.Tensor.to = _orig_tensor_to
    torch.nn.Module.to = _orig_module_to
    torch.matmul = _orig_matmul
    torch.mm = _orig_mm
    if _orig_bmm is not None:
        torch.bmm = _orig_bmm
    torch.addmm = _orig_addmm
    F.linear = _orig_linear_func
    F.conv2d = _orig_conv2d_func

def enable_implicit_accel():
    torch.matmul = _zynq_matmul
    torch.mm = _zynq_mm
    if _orig_bmm is not None:
        torch.bmm = _zynq_bmm
    torch.addmm = _zynq_addmm

def disable_implicit_accel():
    torch.matmul = _orig_matmul
    torch.mm = _orig_mm
    if _orig_bmm is not None:
        torch.bmm = _orig_bmm
    torch.addmm = _orig_addmm

# removed model-level global accel helpers to ensure acceleration only
# when tensors or modules are explicitly moved to "zynq" via .to("zynq")

def register_privateuse1_impls():
    lib = torch.library.Library("aten", "IMPL", "PrivateUse1")
    try:
        lib.impl("matmul", _zynq_matmul)
        lib.impl("mm", _zynq_mm)
        if _orig_bmm is not None:
            lib.impl("bmm", _zynq_bmm)
        lib.impl("addmm", _zynq_addmm)
        lib.impl("linear", _zynq_linear)
        try:
            lib.impl("convolution", _zynq_conv2d)
        except Exception:
            pass
    except Exception:
        pass
