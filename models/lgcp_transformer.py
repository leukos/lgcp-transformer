import torch
import torch.nn as nn

from model.set_transformer import ISAM, SAB, PMA
from model.set_transformer2 import ISAB as ISAB2, SAB as SAB2, PMA as PMA2
from models.utils import MySequential, MyLinear, MyReLU, MyFlatten

class LGCPTransformer2(nn.Module):
    def __init__(self, n_inputs=2, n_outputs=1, n_enc_layers=2,
                 dim_hidden=128, norm="none", sample_size=1000):
        super(LGCPTransformer, self).__init__()

        num_heads = 8
        #num_inds = 32

        layers = [MyLinear(n_inputs, dim_hidden)]
        for i in range(n_enc_layers):
            #layers.append(ISAB(dim_hidden, dim_hidden, num_heads, num_inds, norm=norm, sample_size=sample_size))
            layers.append(SAB(dim_hidden, dim_hidden, num_heads, norm=norm, sample_size=sample_size))
        if norm != "none":
            layers.append(get_norm(norm, sample_size=sample_size, dim_V=dim_hidden))
        self.enc = MySequential(*layers)
        self.dec = MySequential(
            PMA(dim_hidden, num_heads, 1),
            MyFlatten(),
            MyLinear(4*dim_hidden, 256),
            MyReLU(),
            MyLinear(256, n_outputs)
        )

    def forward(self, x, lengths):
        x, lengths = reshape_x_and_lengths(x, lengths, x.device)
        out, _= self.enc(x, lengths)
        out = self.dec(out, lengths)[0].squeeze()
        return out

