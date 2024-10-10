import os

os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_LAUNCH_BLOCKING"]="1"


import ctypes
from functools import reduce
import operator
import bitblas
import torch
import torch.nn as nn
from GPTQ.gptq import Quantizer
from GPTQ.modelutils import *

# bitblas.set_log_level("Debug")

class bitlinear(bitblas.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = False,
            A_dtype: str = "float16",
            W_dtype: str = "int4",
            accum_dtype: str = "float16",
            out_dtype: str = "float16",
            group_size: int = -1,
            with_scaling: bool = False,
            with_zeros: bool = False,
            zeros_mode: str = None,
            opt_M: list = [1, 16, 32, 64, 128, 256, 512],
            fast_decoding: bool = True,
            alpha: torch.float16 = 1.
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
    
    def groupmul(self,A,w):
        w=[ctypes.c_void_p(w.data_ptr()),None,None,None]
        C = torch.empty(
            A.shape[:-1] + (self.scales.shape[0],), dtype=A.dtype, device=A.device
        )
        A_void = ctypes.c_void_p(A.data_ptr())
        # m is the product of the last n - 1 dimensions of A
        m = ctypes.c_int32(reduce(operator.mul, A.shape[:-1], 1))
        self.bitblas_matmul.call_lib(
            A_void , *w, ctypes.c_void_p(C.data_ptr()), m
        )
        return C


    def forward(self, A: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        out = super().forward(A, out)
        out = self.alpha.mT*out
        if self.bias is not None:
            self.bias=self.bias.to('cuda:0')
            out += self.bias.view(1, -1).expand_as(out)
        return out.to(torch.float32)


def convert_to_bitlinear(layer,scale):
    bitlayer = bitlinear(
        in_features=layer.in_features,
        out_features=layer.out_features,
        bias=False,
        A_dtype="float16",  # activation A dtype
        W_dtype="int4",  # weight W dtype
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
        alpha=scale.to(torch.float16)
    )
    w=layer.weight.data.to(torch.int8)
    bitlayer.load_and_transform_weight(w)
    if layer.bias is not None:
        bitlayer.bias = layer.bias.data.to(torch.float16)
    return bitlayer

def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale,q-zero

def replace_linear2bitblas(model,args):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            quantizer = Quantizer()
            quantizer.configure(
                args.wbits, perchannel=True, sym=False, mse=False
            )
            W = module.weight.data
            if args.groupsize==-1:
                quantizer.find_params(W, weight=True)
                scale,module.weight.data = quantize(
                    W, quantizer.scale, quantizer.zero, quantizer.maxq
                )
            else:
                quantizer.find_params(W[:, 0: args.groupsize], weight=True)
                scale,module.weight.data[:, 0:args.groupsize] = quantize(
                    W[:, 0:args.groupsize], quantizer.scale, quantizer.zero, quantizer.maxq
                )
                for j in range(args.groupsize, W.shape[1], args.groupsize):
                    quantizer.find_params(W[:, j:(j + args.groupsize)], weight=True)
                    s,module.weight.data[:, j:(j + args.groupsize)] = quantize(
                        W[:, j:(j + args.groupsize)], quantizer.scale, quantizer.zero, quantizer.maxq
                    )
                    scale=torch.cat((scale,s),1)
            new_layer = convert_to_bitlinear(module,scale)
            setattr(model, name, new_layer)
        elif len(list(module.children())) > 0:
            replace_linear2bitblas(module,args)