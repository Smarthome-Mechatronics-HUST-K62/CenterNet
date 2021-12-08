import numpy as np
import torch.nn as nn
from model.dla_components import DLA_BASE,DLAUp


class DLA(nn.Module):
    def __init__(self,levels,channels,down_ratio=4):
        super(DLA,self).__init__()
        assert down_ratio in [2,4,8,16], "assert should be 2 / 4 / 8 / 16"
        self.first_level = int(np.log2(down_ratio))
        self.dla_base = DLA_BASE(levels,channels)
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(channels[self.first_level:],scales=scales)
        self.intermediate_channel = channels[self.first_level]
    
    def forward(self,x):
        out_bases = self.dla_base(x)
        out_ups = self.dla_up(out_bases[self.first_level:])
        
        return out_ups

def get_dla_34(down_ratio=4):
    levels = [1,1,1,2,2,1]
    channels = [16,32,64,128,256,512]
    return DLA(levels,channels,down_ratio=down_ratio)

if __name__ == '__main__':
    import torch
    x = torch.randn(1,3,512,512)
    DLA = get_dla_34()
    print(DLA(x).shape)