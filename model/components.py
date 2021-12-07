import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self,in_c,out_c,k,s,p,use_bn=True,use_relu=True):
        """[Conv -> Bn -> ReLU]

        Args:
            in_c ([int]): [in channels]
            out_c ([int]): [out channels]
            k ([int]): [kernel_size]
            s ([int]): [stride]
            p ([int]): [padding]
            use_bn (bool, optional): [True if use batchnorm after conv, else not use]. Defaults to True.
            use_relu (bool, optional): [True if use relu after batchnorm / conv , else not use]. Defaults to True.
        """
        super(ConvBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels=in_c,out_channels=out_c,kernel_size=k,stride=s,padding=p,bias=False)
        self.batchnorm = nn.BatchNorm2d(out_c) if use_bn else None
        self.relu = nn.ReLU(inplace=True) if use_relu else None
    
    def forward(self,x):
        out = self.conv(x)
        out = self.batchnorm(out) if self.batchnorm else out
        out = self.relu(out) if self.relu else out
        return out


class BasicBlock(nn.Module):
    def __init__(self,inplanes,planes,stride=1):
        """[
            main_branch:  ConvBlock (3x3 stride padding=1) -> ConvBlock (3x3 stride=1 padding=1)
            residual branch (if needed): ConvBlock (1x1 stride padding=0)
        ]

        Args:
            inplanes ([int]): [in_channels]
            planes ([int]): [out_channels]
            stride (int, optional): [stride to decide if downsample or not]. Defaults to 1.
        """
        super(BasicBlock,self).__init__()
        assert stride == 1 or stride == 2, "Stride should be equal to only 1 or 2"
        self.conv_block_1 = ConvBlock(in_c=inplanes,out_c=planes,k=3,s=stride,p=1,use_bn=True,use_relu=True)
        self.conv_block_2 = ConvBlock(in_c=planes,out_c=planes,k=3,s=1,p=1,use_bn=True,use_relu=False)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self,x,residual=None):
        if residual == None:
            residual = x
            
        out = self.conv_block_1(x)
        out = self.conv_block_2(out)
        
        out += residual
        out = self.relu(out)
        
        return out


class Root(nn.Module):
    def __init__(self,in_c,out_c):
        """[Conv -> Bn -> ReLU
            kernel size = 1
            stride = 1
            padding = 0
            Input (list of tensor 4d)
        ]
        
        Args:
            in_channels ([int]): [in_channels]
            out_channels ([int]): [out_channels]
        """
        super(Root,self).__init__()
        self.conv_block = ConvBlock(in_c,out_c,k=1,s=1,p=0,use_bn=True,use_relu=True)
    
    def forward(self,*x):
        inp = torch.cat(x,1)
        out = self.conv_block(inp)
        return out


class Tree(nn.Module):
    def __init__(self, levels, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                  root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = BasicBlock(in_channels, out_channels, stride,
                               )
            self.tree2 = BasicBlock(out_channels, out_channels, 1,
                               )
        else:
            self.tree1 = Tree(levels - 1, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              root_residual=root_residual)
            self.tree2 = Tree(levels - 1, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity,self).__init__()
    
    def forward(self,x):
        return x

# if __name__ == '__main__':
#     x = torch.randn(1,64,256,256)
#     tree = Tree(levels=2,in_channels=64,out_channels=128,stride=2,level_root=True)
#     print("output tree shape : ",tree(x).shape)
    