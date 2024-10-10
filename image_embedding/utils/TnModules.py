import torch
import torch.nn as nn


def Ternarize(W):
    with torch.no_grad():
        m = W.abs().mean()
        m *= 2
        W = torch.clamp(torch.round(W / m), min=-1, max=1)
        return W * m


class TnLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(TnLinear, self).__init__(*args, **kwargs)

    def forward(self, x):
        w = self.weight
        w_tn = w + (Ternarize(w.data) - w).detach()
        output = nn.functional.linear(x, w_tn, self.bias)
        return output


def replace_linear(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            new_linear = TnLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=(module.bias is not None)
            )
            new_linear.weight = module.weight
            if module.bias is not None:
                new_linear.bias = module.bias

            #new_layers = new_linear
            setattr(model, name, new_linear)
        elif len(list(module.children())) > 0:
            replace_linear(module)
