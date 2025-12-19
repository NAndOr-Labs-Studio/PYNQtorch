import torch
import torch.nn as nn
from .ops import mmult

class ZynqLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.int32), requires_grad=False)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = mmult(x, self.weight.t())
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out
