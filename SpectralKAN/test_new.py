import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.io import loadmat
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# 假设输入的高光谱图像 (H, W, C) 和模型
barbara_T1 = loadmat("../change_detection_data/Barbara/barbara_2013.mat")['HypeRvieW']
# barbara_T1 = (barbara_T1-barbara_T1.min())/(barbara_T1.max()-barbara_T1.min())
barbara_T2 = loadmat("../change_detection_data/Barbara/barbara_2014.mat")['HypeRvieW']
# barbara_T2 = (barbara_T2-barbara_T2.min())/(barbara_T2.max()-barbara_T2.min())
barbara_gt = loadmat("../change_detection_data/Barbara/barbara_gtChanges.mat")['HypeRvieW']
print('image_size=',barbara_T1.shape)

image_t1 = np.zeros(barbara_T1.shape)
image_t2 = np.zeros(barbara_T1.shape)
for i in range(barbara_T1.shape[2]):
    input_max = max(np.max(barbara_T1[:,:,i]),np.max(barbara_T2[:,:,i]))
    input_min = min(np.min(barbara_T1[:,:,i]),np.min(barbara_T2[:,:,i]))
    image_t1[:,:,i] = (barbara_T1[:,:,i]-input_min)/(input_max-input_min)
    image_t2[:,:,i] = (barbara_T2[:,:,i]-input_min)/(input_max-input_min)

# bayArea_T1 = loadmat("../change_detection_data/BayArea/Bay_Area_2013.mat")['HypeRvieW']
# # bayArea_T1 = (bayArea_T1-bayArea_T1.min())/(bayArea_T1.max()-bayArea_T1.min())
# bayArea_T2 = loadmat("../change_detection_data/BayArea/Bay_Area_2015.mat")['HypeRvieW']
# # bayArea_T2 = (bayArea_T2-bayArea_T2.min())/(bayArea_T2.max()-bayArea_T2.min())
# bayArea_gt = loadmat("../change_detection_data/BayArea/bayArea_gtChanges.mat")['HypeRvieW']
# print('image_size=',bayArea_T1.shape)


# image_t1 = np.zeros(bayArea_T1.shape)
# image_t2 = np.zeros(bayArea_T2.shape)
# for i in range(bayArea_T1.shape[2]):
#     input_max = max(np.max(bayArea_T1[:,:,i]),np.max(bayArea_T2[:,:,i]))
#     input_min = min(np.min(bayArea_T1[:,:,i]),np.min(bayArea_T2[:,:,i]))
#     image_t1[:,:,i] = (bayArea_T1[:,:,i]-input_min)/(input_max-input_min)
#     image_t2[:,:,i] = (bayArea_T2[:,:,i]-input_min)/(input_max-input_min)



patch_size = 5  # 定义patch的大小
batch_size = 1000  # 每次处理1000个patch
band=224

from efficient_kan import WKAN

def compute_cosine_similarity(querys, means):
    """
    计算 querys 与 means 中每个均值的余弦相似度
    
    参数:
    - querys: 形状为 (N, D) 的张量，表示 N 个样本的特征，每个样本有 D 维特征
    - means: 形状为 (2, D) 的张量，表示标签为0和1的特征均值
    
    返回:
    - similarities: 形状为 (N, 2) 的张量，表示每个样本与 means[0] 和 means[1] 的余弦相似度
    """
    # 计算 querys 与 means 的余弦相似度，使用广播机制
    similarities = F.cosine_similarity(querys.unsqueeze(1), means.unsqueeze(0), dim=-1)
    
    return similarities

class SpectralKAN(nn.Module):
    def __init__(self,num_class,band,patch_size):
        super(SpectralKAN,self).__init__()
        self.band=band
        self.layer1 = WKAN([patch_size**2,16,1])
        self.layer2 = WKAN([band,num_class])
    def forward(self,x):
        x = rearrange(x,'b c h -> (b c) h')
        x = self.layer1(x)
        x = rearrange(x,'(b c) d -> b (c d)',c=self.band)
        x = self.layer2(x)
        return x

model = SpectralKAN(2,band,patch_size)

#____________________changde detection___________________________

# model=ViT(image_size=9, num_patches=224, num_classes=2, dim=128, depth=8, heads=8, mlp_dim=128)
checkpoint = torch.load('log/SpectralKAN_farmland.pth', map_location='cpu')

model.load_state_dict(checkpoint)
print('-----------model加载完毕-------')

# 自定义数据集，按patch提取
class HyperspectralPatchDataset(Dataset):
    def __init__(self, image_t1, image_t2, patch_size):
        self.image_t1 = image_t1
        self.image_t2 = image_t2
        self.patch_size = patch_size
        self.H, self.W, self.C = image_t1.shape
        
        # 对两个时相的图像进行pad
        self.pad_image_t1 = np.pad(image_t1, ((patch_size//2, patch_size//2),
                                              (patch_size//2, patch_size//2),
                                              (0, 0)), 'reflect')
        self.pad_image_t2 = np.pad(image_t2, ((patch_size//2, patch_size//2),
                                              (patch_size//2, patch_size//2),
                                              (0, 0)), 'reflect')
    
    def __len__(self):
        return self.H * self.W  # 每个像素点对应一个patch
    
    def __getitem__(self, idx):
        row = idx // self.W
        col = idx % self.W
        
        # 从两个时相的pad后的图像中提取对应的patch
        patch_t1 = self.pad_image_t1[row:row + self.patch_size, col:col + self.patch_size, :]
        patch_t2 = self.pad_image_t2[row:row + self.patch_size, col:col + self.patch_size, :]
        
        # 返回两个时相的patch和对应像素的索引
        return (patch_t1, patch_t2), (row, col)

# 定义数据集
dataset = HyperspectralPatchDataset(image_t1, image_t2, patch_size)

# 定义DataLoader，batch_size设为1000
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 初始化结果图
output_image = np.zeros((image_t1.shape[0], image_t1.shape[1]))

# 模型测试，逐批处理patch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)

model.eval()


with torch.no_grad():
    for batch_data in data_loader:
        patches, indices = batch_data
        patches_t1, patches_t2 = patches  # 分别获取两个时相的patch
        
        # 转换为模型需要的格式 (N, C, H, W)
        patches_t1 = patches_t1.permute(0, 3, 1, 2).float()
        patches_t2 = patches_t2.permute(0, 3, 1, 2).float()
        patches_t1 = rearrange(patches_t1,'b c w h -> b c (w h)')
        patches_t2 = rearrange(patches_t2,'b c w h -> b c (w h)')
        # import ipdb;ipdb.set_trace()
        # 模型前向传播
        predictions = model(patches_t1.to(device)-patches_t2.to(device))  # 输出的形状为 (batch_size, ...)
        # import ipdb;ipdb.set_trace()
        # 处理模型的输出，这里假设输出是单个值，可以根据你的任务进行调整
        _, predictions = predictions.topk(1, 1, True, True)
        # import ipdb;ipdb.set_trace()
        # 将结果写回到 output_image 中
        for i, (row, col) in enumerate(zip(indices[0],indices[1])):
            output_image[row, col] = predictions[i].item()

output_image=1-output_image
# 最终的预测结果图已经保存在output_image中
print("Prediction complete, output shape:", output_image.shape)
from PIL import Image
output_image_normalized = (output_image - output_image.min()) / (output_image.max() - output_image.min()) * 255
output_image_normalized = output_image_normalized.astype(np.uint8)  # 转换为 uint8 类型

# 将 NumPy 数组转换为 PIL 图像对象
img = Image.fromarray(output_image_normalized)

# 显示图像
img.show()

# 保存图像
img.save("result/Barbara.png")
