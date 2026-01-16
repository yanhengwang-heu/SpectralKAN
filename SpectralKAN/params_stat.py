import torch
from efficient_kan import KAN,KANLinear
from thop import profile
from torchstat import stat
import torch.nn as nn
from einops import rearrange, repeat

# model1=KANLinear(154*5*5,256)
# for name,params in model1.named_parameters():
#     print(name,':',params.size())
# input = torch.randn(1,154*5*5)
# flops,params = profile(model1,inputs=(input,))

# print(flops,params)
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
# class SKan(nn.Module):
#     def __init__(self,num_class):
#         super(SKan,self).__init__()
#         self.layer1 = KAN([25,16,1])
#         self.layer2 = KAN([198,16,num_class])
#     def forward(self,x):
#         x = rearrange(x,'b c h -> (b c) h')
#         x = self.layer1(x)
#         x = rearrange(x,'(b c) d -> b (c d)',c=198)
#         x = self.layer2(x)
#         return x
class SKan(nn.Module):
    def __init__(self,num_class,band,patch_size):
        super(SKan,self).__init__()
        self.band=band
        self.layer1 = KAN([patch_size**2,16,1])
        self.layer2 = KAN([band,num_class])
    def forward(self,x):
        x = rearrange(x,'b c h -> (b c) h')
        x = self.layer1(x)
        x = rearrange(x,'(b c) d -> b (c d)',c=self.band)
        x = self.layer2(x)
        return x
model = SKan(2,224,5)
# model = KAN([154*5*5,16,2])
total=get_parameter_number(model)
print(total)
# stat(model,(1,225,5*5))




# from sstvit import SSTViT
# model2 = SSTViT(
#     image_size = 5,
#     near_band = 1,
#     num_patches = 154,
#     num_classes = 2,
#     dim = 32,
#     depth = 2,
#     heads = 4,
#     dim_head=16,
#     mlp_dim = 8,
#     b_dim = 512,
#     b_depth = 3,
#     b_heads = 8,
#     b_dim_head= 32,
#     b_mlp_head = 8,
#     dropout = 0.2,
#     emb_dropout = 0.1,
# )

# total=get_parameter_number(model2)
# print(total)
# input = torch.randn(1,154,25)
# flops,params = profile(model2,inputs=(input,input))
# print(flops,params)


