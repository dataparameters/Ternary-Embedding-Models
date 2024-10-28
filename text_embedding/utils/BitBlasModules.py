import os

os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

import bitblas
import torch
import torch.nn as nn

# bitblas.set_log_level("Debug")
torch.manual_seed(0)

class bitlinear(bitblas.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = False,
            A_dtype: str = "float16",
            W_dtype: str = "int2",
            accum_dtype: str = "float16",
            out_dtype: str = "float16",
            group_size: int = -1,
            with_scaling: bool = False,
            with_zeros: bool = False,
            zeros_mode: str = None,
            opt_M: list = [1, 16, 32, 64, 128, 256, 512],
            fast_decoding: bool = True,
            alpha: torch.dtype = torch.float16,
            b:torch.Tensor=None
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            A_dtype=A_dtype,
            W_dtype=W_dtype,
            accum_dtype=accum_dtype,
            out_dtype=out_dtype,
            group_size=group_size,
            with_scaling=with_scaling,
            with_zeros=with_zeros,
            zeros_mode=zeros_mode,
            opt_M=opt_M,
            fast_decoding=fast_decoding,
        )
        self.alpha = nn.Parameter(alpha,requires_grad=False)
        self.b = nn.Parameter(b,requires_grad=False)

    def forward(self, A: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        out = super().forward(A, out)
        out *= self.alpha
        if self.b is not None:
            out += self.b.view(1, -1).expand_as(out)
        return out.to(torch.float32)


def Ternarize(W):
    with torch.no_grad():
        m = W.abs().mean()
        m *= 2
        W = torch.clamp(torch.round(W / m), min=-1, max=1)
        return W, m


def convert_to_bitlinear(layer):
    w, a = Ternarize(layer.weight.data)
    bitlayer = bitlinear(
        in_features=layer.in_features,
        out_features=layer.out_features,
        bias=False,
        A_dtype="float16",  # activation A dtype
        W_dtype="int2",  # weight W dtype
        accum_dtype="float16",  # accumulation dtype
        out_dtype="float16",  # output dtype
        # configs for weight only quantization
        group_size=-1,  # setting for grouped quantization
        with_scaling=False,  # setting for scaling factor
        with_zeros=False,  # setting for zeros
        zeros_mode=None,  # setting for how to calculating zeros
        # Target optimization var for dynamic symbolic.
        # For detailed information please checkout docs/PythonAPI.md
        # By default, the optimization var is [1, 16, 32, 64, 128, 256, 512]
        opt_M=[1, 16, 32, 64, 128, 256, 512],
        fast_decoding=True,
        alpha=a.to(torch.float16),
        b = layer.bias.data.to(torch.float16)
    )
    bitlayer.load_and_transform_weight(w.to(torch.int8))
    return bitlayer


def replace_linear2bitblas(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            new_layer = convert_to_bitlinear(module)
            setattr(model, name, new_layer)
        elif len(list(module.children())) > 0:
            replace_linear2bitblas(module)
