from model.components import*
import numpy as np

class DLA_BASE(nn.Module):
    def __init__(self,levels,channels,residual_root=False):
        super(DLA_BASE,self).__init__()
        self.base_layer = ConvBlock(in_c=3,out_c=channels[0],k=7,s=1,p=3,use_bn=True,use_relu=True)
        
        #non-tree levels
        self.level_0 = self.make_conv_level(channels[0],channels[0],num_repeats=levels[0])
        self.level_1 = self.make_conv_level(channels[0],channels[1],num_repeats=levels[1],stride=2) # 1/2 downsize
        
        #tree levels
        self.level_2 = Tree(levels[2],channels[1],channels[2],stride=2,level_root=False)
        self.level_3 = Tree(levels[3],channels[2],channels[3],stride=2,level_root=True)
        self.level_4 = Tree(levels[4],channels[3],channels[4],stride=2,level_root=True)
        self.level_5 = Tree(levels[5],channels[4],channels[5],stride=2,level_root=True)
    
    def make_conv_level(self,inplanes,planes,num_repeats,stride=1):
        modules = nn.ModuleList()
        for i in range(num_repeats):
            modules += [ConvBlock(in_c=inplanes,out_c=planes,k=3,s=stride if i==0 else 1,p=1,use_bn=True,use_relu=True)]
            inplanes = planes
        
        return nn.Sequential(*modules)
    
    def forward(self,x):
        y = []
        x = self.base_layer(x)
        
        x = self.level_0(x)
        y.append(x)
        
        x = self.level_1(x)
        y.append(x)
        
        x = self.level_2(x)
        y.append(x)
        
        x = self.level_3(x)
        y.append(x)
        
        x = self.level_4(x)
        y.append(x)
        
        x = self.level_5(x)
        y.append(x)
        
        return y


class IDAUp(nn.Module):
    def __init__(self,out_channel,in_channels,scales):
        """
        This class implements iterative deep aggregation module with inputs are layers of dla base
        out_channel (int)
        in_channels (list of int)
        scales (list of int)
        len(in_channels) = len(scales)
        """
        super(IDAUp,self).__init__()
        assert(len(in_channels) == len(scales)), "Length of in_channels should be equal to length of scales"
        self.in_channels = in_channels
        
        self.projs = []
        self.ups = []
        for i, in_channel in enumerate(in_channels):
            if in_channel == out_channel:
                self.projs += [Identity()]
            
            else:
                self.projs += [ConvBlock(in_channel,out_channel,k=1,s=1,p=0,use_bn=True,use_relu=True)]
            
            scale = int(scales[i])
            if scale == 1:
                self.ups += [Identity()]
            
            else:
                self.ups += [nn.ConvTranspose2d(out_channel,out_channel,kernel_size=2*scale,stride=scale,padding=scale//2,output_padding=0,groups=out_channel,bias=False)]

        #aggregation nodes
        self.nodes = []
        for i in range(1,len(in_channels)):
            self.nodes += [ConvBlock(out_channel*2,out_channel,k=3,s=1,p=1,use_bn=True,use_relu=True)]
    
    def forward(self,layers):
        assert(len(layers) == len(self.in_channels)), "Number of layer inputs should be equal to number or input channel"
        
        #pass proj and up each input to get outputs respectively
        for i in range(len(layers)):
            layers[i] = self.ups[i](self.projs[i](layers[i]))
        
        
        #aggregate nodes
        agg_node = layers[0]
        nodes = []
        for i in range(1,len(layers)):
            agg_node = self.nodes[i-1](torch.cat([agg_node,layers[i]],dim=1))
            nodes += [agg_node]
        
        return agg_node, nodes


class DLAUp(nn.Module):
    def __init__(self,out_channels,scales,in_channels=None):
        super(DLAUp,self).__init__()
        assert(len(out_channels) == len(scales)), "Length of output channel list should be equal to length of scale list"
        if in_channels != None:
            assert(len(out_channels) == len(in_channels)),"Length of output channel list shoudl be equal to length of input channel list"
        else:
            in_channels = out_channels
        
        scales = np.array(scales,dtype=np.int32)
        self.idas = []
        for i in range(len(out_channels)-1):
            j = -i - 2
            self.idas += [IDAUp(out_channels[j], in_channels[j:], scales[j:]//scales[j])]
            scales[j+1:] = scales[j]
            in_channels[j+1:] = [out_channels[j] for _ in out_channels[j+1:]]
    
    def forward(self,layers):
        for i in range(len(layers)-1):
            x,y = self.idas[i](layers[-i-2:])
            layers[-i-1:] = y
        
        return x
                
# if __name__ == '__main__':
#     import torch
#     import numpy as np
#     x = torch.randn(1,3,512,512)
#     levels = [1,1,1,2,2,1]
#     channels = [16,32,64,128,256,512]
#     dla = DLA_BASE(levels,channels)
#     ys = dla(x)
#     for i in range(len(ys)):
#         print(ys[i].shape)
    
#     down_ratio = 4
#     first_level = int(np.log2(down_ratio))
#     print("first level : ",first_level)
#     scales = [2 ** i for i in range(len(channels[first_level:]))]
#     print("scales : ",scales)
#     dla_up = DLAUp(channels[first_level:],scales=scales)
#     print(dla_up(ys[2:]).shape)
    