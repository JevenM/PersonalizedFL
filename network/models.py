import torch
import torch.nn as nn
import torch.nn.functional as func
from collections import OrderedDict
import math
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
                    ("bn1", nn.BatchNorm2d(64)),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("maxpool1", nn.MaxPool2d(kernel_size=3, stride=2)),
                    ("conv2", nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                    ("bn2", nn.BatchNorm2d(192)),
                    ("relu2", nn.ReLU(inplace=True)),
                    ("maxpool2", nn.MaxPool2d(kernel_size=3, stride=2)),
                    ("conv3", nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                    ("bn3", nn.BatchNorm2d(384)),
                    ("relu3", nn.ReLU(inplace=True)),
                    ("conv4", nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                    ("bn4", nn.BatchNorm2d(256)),
                    ("relu4", nn.ReLU(inplace=True)),
                    ("conv5", nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                    ("bn5", nn.BatchNorm2d(256)),
                    ("relu5", nn.ReLU(inplace=True)),
                    ("maxpool5", nn.MaxPool2d(kernel_size=3, stride=2)),
                ]
            )
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(256 * 6 * 6, 4096)),
                    ("bn6", nn.BatchNorm1d(4096)),
                    ("relu6", nn.ReLU(inplace=True)),
                    ("fc2", nn.Linear(4096, 4096)),
                    ("bn7", nn.BatchNorm1d(4096)),
                    ("relu7", nn.ReLU(inplace=True)),
                    ("fc3", nn.Linear(4096, num_classes)),
                ]
            )
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def getallfea(self, x):
        fealist = []
        for i in range(len(self.features)):
            if i in [1, 5, 9, 12, 15]:
                fealist.append(x.clone().detach())
            x = self.features[i](x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        for i in range(len(self.classifier)):
            if i in [1, 4]:
                fealist.append(x.clone().detach())
            x = self.classifier[i](x)
        return fealist

    def getfinalfea(self, x):
        for i in range(len(self.features)):
            x = self.features[i](x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        for i in range(len(self.classifier)):
            if i == 6:
                return [x]
            x = self.classifier[i](x)
        return x

    def get_sel_fea(self, x, plan=0):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if plan == 0:
            y = x
        elif plan == 1:
            y = self.classifier[5](
                self.classifier[4](
                    self.classifier[3](
                        self.classifier[2](self.classifier[1](self.classifier[0](x)))
                    )
                )
            )
        else:
            y = []
            y.append(x)
            x = self.classifier[2](self.classifier[1](self.classifier[0](x)))
            y.append(x)
            x = self.classifier[5](self.classifier[4](self.classifier[3](x)))
            y.append(x)
            y = torch.cat(y, dim=1)
        return y


class PamapModel(nn.Module):
    def __init__(self, n_feature=64, out_dim=10):
        super(PamapModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=27, out_channels=16, kernel_size=(1, 9))
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 9))
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        self.fc1 = nn.Linear(in_features=32 * 44, out_features=n_feature)
        self.fc1_relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=n_feature, out_features=out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(self.relu1(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool2(self.relu2(self.bn2(x)))
        x = x.reshape(-1, 32 * 44)
        feature = self.fc1_relu(self.fc1(x))
        out = self.fc2(feature)
        return out

    def getallfea(self, x):
        fealist = []
        x = self.conv1(x)
        fealist.append(x.clone().detach())
        x = self.pool1(self.relu1(self.bn1(x)))
        x = self.conv2(x)
        fealist.append(x.clone().detach())
        return fealist

    def getfinalfea(self, x):
        x = self.conv1(x)
        x = self.pool1(self.relu1(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool2(self.relu2(self.bn2(x)))
        x = x.reshape(-1, 32 * 44)
        feature = self.fc1_relu(self.fc1(x))
        return [feature]

    def get_sel_fea(self, x, plan=0):
        if plan == 0:
            x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            x = x.reshape(-1, 32 * 44)
            fealist = x
        elif plan == 1:
            x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            x = x.reshape(-1, 32 * 44)
            feature = self.fc1_relu(self.fc1(x))
            fealist = feature
        else:
            fealist = []
            x = self.conv1(x)
            x = self.pool1(self.relu1(self.bn1(x)))
            fealist.append(x.view(x.shape[0], -1))
            x = self.conv2(x)
            x = self.pool2(self.relu2(self.bn2(x)))
            fealist.append(x.view(x.shape[0], -1))
            x = x.reshape(-1, 32 * 44)
            feature = self.fc1_relu(self.fc1(x))
            fealist.append(feature)
            fealist = torch.cat(fealist, dim=1)
        return fealist


class lenet5v(nn.Module):
    def __init__(self, algs_name, prompt_dim):
        super(lenet5v, self).__init__()
        self.algs_name = algs_name
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.embedd_dim = 256
        self.fc1 = nn.Linear(self.embedd_dim, 11)
        # self.fc1 = nn.Linear(self.embedd_dim, 120)
        # self.relu3 = nn.ReLU()
        # self.fc2 = nn.Linear(120, 84)
        # self.relu4 = nn.ReLU()
        # self.fc3 = nn.Linear(84, 11)

        self.prompt = nn.Parameter(torch.randn(prompt_dim, self.embedd_dim))
        self.prompt_W_k = nn.Parameter(torch.randn(self.embedd_dim, self.embedd_dim))
        self.prompt_W_v = nn.Parameter(torch.randn(self.embedd_dim, self.embedd_dim))

        # self.Wei_z = nn.Parameter(torch.randn(512, prompt_dim))
        # self.Wei_r = nn.Parameter(torch.randn(512, prompt_dim))
        # self.Wei = nn.Parameter(torch.randn(512, 256))

        self.relu5 = nn.ReLU()
        self.scale = math.sqrt(self.embedd_dim)  # 缩放因子
        # self.fc4 = nn.Linear((prompt_dim+1)*256, 256)
        self.fc4 = nn.Linear(2 * self.embedd_dim, self.embedd_dim)
        self.fc5 = nn.Linear((prompt_dim) * self.embedd_dim, self.embedd_dim)
        # 定义一个可学习的缩放因子
        self.hyp = nn.Parameter(torch.randn(1))

    def forward(self, x, flag=0, server_prompt=None):
        if server_prompt is not None:
            server_prompt.require_grad = False
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)

        if self.algs_name == "fedlp" and flag != 0:
            # z_t = torch.sigmoid(torch.cat((server_prompt, self.prompt), dim=1) @ self.Wei_z) # mxm
            # z_r = torch.sigmoid(torch.cat((server_prompt, self.prompt), dim=1) @ self.Wei_r) # mxm
            # tilde_prompt = torch.relu(torch.cat((server_prompt, z_r @ self.prompt), dim=1) @ self.Wei)
            # prompt = (1-z_t) @ self.prompt + z_t @ tilde_prompt

            W_K = server_prompt @ self.prompt_W_k
            W_V = server_prompt @ self.prompt_W_v

            scores = (self.prompt @ W_K.t()) / self.scale
            # scores = (prompt @ W_K.t()) / self.scale

            attn = F.softmax(scores, dim=-1)
            output = attn @ W_V
            # 1. 融合
            # 使用 sigmoid 将 self.hyp 限制在 [0, 1] 范围
            h = torch.sigmoid(self.hyp)
            output = output * (1 - h) + self.prompt * h
            # 2. 拼接
            # output = torch.cat((output, self.prompt * h), dim=1)
            # output = torch.cat((output, prompt * h), dim=1)
            output = output.reshape(-1)  # [prompt_dim*self.embedd_dim]
            output = self.fc5(output)  # [self.embedd_dim]
            # 下面这行打开注释会导致收敛加快，但是性能降低
            # output = self.relu5(output)
            # 增加一个批次维度，并复制 batch_size 份
            output_repeated = output.unsqueeze(0).repeat(
                x.size(0), 1
            )  # 形状: [batch_size, self.embedd_dim]
            y = torch.cat(
                (y, output_repeated), dim=1
            )  # 在最后一个维度上拼接 [batch_size, 2*self.embedd_dim]
            y = self.fc4(y)
            # 不能注释下面这行
            y = self.relu5(y)
            # 直接融合不行
            # y = 0.5*y + 0.5*output_repeated
            # y = self.relu5(y)

        y = self.fc1(y)
        # y = self.relu3(y)
        # y = self.fc2(y)
        # y = self.relu4(y)
        # y = self.fc3(y)
        return y

    def getallfea(self, x):
        fealist = []
        y = self.conv1(x)
        fealist.append(y.clone().detach())
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        fealist.append(y.clone().detach())
        return fealist

    def getfinalfea(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        return [y]

    def get_sel_fea(self, x, plan=0):
        if plan == 0:
            x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            x = x.view(x.shape[0], -1)
            fealist = x
        elif plan == 1:
            x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            x = x.view(x.shape[0], -1)
            x = self.relu3(self.fc1(x))
            x = self.relu4(self.fc2(x))
            fealist = x
        else:
            fealist = []
            x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            x = x.view(x.shape[0], -1)
            fealist.append(x)
            x = self.relu3(self.fc1(x))
            fealist.append(x)
            x = self.relu4(self.fc2(x))
            fealist.append(x)
            fealist = torch.cat(fealist, dim=1)
        return fealist
