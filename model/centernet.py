import torch.nn as nn

from model.components import ConvBlock
from model.dla import DLA, get_dla_34

# Define head for human pose estimation tasks (include human detection)
class CenterHead(nn.Module):
    def __init__(self,intermediate_channel):
        super(CenterHead,self).__init__()
        
        self.hm_head = nn.Sequential(
            ConvBlock(intermediate_channel,256,k=3,s=1,p=1,use_bn=False,use_relu=True),
            nn.Conv2d(256,1,kernel_size=1,stride=1,padding=0)
        )
        
        self.wh_head = nn.Sequential(
            ConvBlock(intermediate_channel,256,k=3,s=1,p=1,use_bn=False,use_relu=True),
            nn.Conv2d(256,2,kernel_size=1,stride=1,padding=0)
        )
        
        self.reg_head = nn.Sequential(
            ConvBlock(intermediate_channel,256,k=3,s=1,p=1,use_bn=False,use_relu=True),
            nn.Conv2d(256,2,kernel_size=1,stride=1,padding=0)
        )
        
        self.pose_hm_head = nn.Sequential(
            ConvBlock(intermediate_channel,256,k=3,s=1,p=1,use_bn=False,use_relu=True),
            nn.Conv2d(256,17,kernel_size=1,stride=1,padding=0)
        )
        
        self.pose_offset_head = nn.Sequential(
            ConvBlock(intermediate_channel,256,k=3,s=1,p=1,use_bn=False,use_relu=True),
            nn.Conv2d(256,2,kernel_size=1,stride=1,padding=0)
        )
        
        self.pose_kps_head =nn.Sequential(
            ConvBlock(intermediate_channel,256,k=3,s=1,p=1,use_bn=False,use_relu=True),
            nn.Conv2d(256,34,kernel_size=1,stride=1,padding=0)
        )
    
    def forward(self,x):
        return [
            self.hm_head(x),
            self.wh_head(x),
            self.reg_head(x),
            self.pose_hm_head(x),
            self.pose_offset_head(x),
            self.pose_kps_head(x)
        ]
    

class CenterNet(nn.Module):
    def __init__(self,down_ratio=4):
        super(CenterNet,self).__init__()
        self.backbone = get_dla_34()
        self.head = CenterHead(self.backbone.intermediate_channel)
    
    def forward(self,x):
        return self.head(self.backbone(x))

# if __name__ == '__main__':
#     import torch
#     x = torch.randn(1,3,512,512)
#     net = CenterNet()
#     y = net(x)
#     for out_head in y:
#         print(out_head.shape)