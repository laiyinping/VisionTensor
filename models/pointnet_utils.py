import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)  # [b,3,N] -> [b,64,N]
        self.conv2 = torch.nn.Conv1d(64, 128, 1)      # [b,64,N] -> [b,128,N]
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)    # [b,128,N] -> [b,1024,N]
        self.fc1 = nn.Linear(1024, 512)   # [b,1024,1] -> [b,512,1]
        self.fc2 = nn.Linear(512, 256)    # [b,512,1] -> [b,256,1]
        self.fc3 = nn.Linear(256, 9)      # [b,256,1] -> [b,9,1]
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]               #获得batchsize
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]  #得到每个特征上的最大值
        x = x.view(-1, 1024)                  #全连接之前，进行展平

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)                     ##tensor不能反向传播，variable可以反向传播
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()             # 输入的X shape为【b,d,n】,即【批量数，特征维度，点数】
        trans = self.stn(x)
        x = x.transpose(2, 1)          # 交换1、2维
        if D > 3:                      # 如果特征维度大于3，即除了XYZ之外，还有别的维度特征，比如颜色啊、法线之类的
            feature = x[:, :, 3:]      # 将除了XYZ的特征维度拿出来，因为空间坐标变换时这些不需要改变
            x = x[:, :, :3]            # 仅获取XYZ信息 ->【b,N,3】
        x = torch.bmm(x, trans)        # 将坐标位置与3*3的T-Net相乘
        if D > 3:
            x = torch.cat([x, feature], dim=2)       #将转换后的XYZ与之前的特征连接在一起
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:     # 如果需要特征空间的变换。因为前面的卷积已经将点云转换成了64维的特征
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]   # 在每个特征维度上取最大值的值
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat       # 返回全局特征、3*3变换以及特征变换
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)    #如果是分割，那就需要加上前面的特征空间，再返回
            return torch.cat([x, pointfeat], 1), trans, trans_feat

# 对齐特征的时候，由于特征空间维度更高，优化难度大，所以加了一项正则项，
# 让求解出来的仿射变换矩阵接近于正交
def feature_transform_reguliarzer(trans):
    d = trans.size()[1]                    # 维度
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

if __name__ == '__main__':
    input = torch.randn(4,3,10000)
    chanel = input.size()[1]
    net = STN3d(3)
    output = net(input)

