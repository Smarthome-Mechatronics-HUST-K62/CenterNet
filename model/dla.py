from components import*


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

# if __name__ == '__main__':
#     import torch
#     x = torch.randn(1,3,512,512)
#     levels = [1,1,1,2,2,1]
#     channels = [16,32,64,128,256,512]
#     dla = DLA_BASE(levels,channels)
#     ys = dla(x)
#     for i in range(len(ys)):
#         print(ys[i].shape)
    