# from typing_extensions import override
from torchvision.models import resnet101
from torchvision.models.resnet import ResNet, model_urls, BasicBlock, Bottleneck
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor, tensor, randn
import torch
import torch.nn as nn
import pdb


class GAB(nn.Module):
    def __init__(self, in_planes):
        super(GAB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes, in_planes, 1, padding='same'),
                                   nn.ReLU(),
                                   nn.Conv2d(in_planes, in_planes, 1, padding='same'))
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):      # torch.Size([2, 2048, 7, 7])
        x = self.avg_pool(inputs)   # torch.Size([2, 2048, 1, 1])
        x = self.conv2(x)           # torch.Size([2, 2048, 1, 1])
        x = self.sigmoid(x)         # torch.Size([2, 2048, 1, 1])
        C_A = x * inputs            # torch.Size([2, 2048, 7, 7])

        x = torch.mean(C_A, dim=1, keepdim=True)    # torch.Size([2, 1, 7, 7])
        x = self.sigmoid(x)         # torch.Size([2, 1, 7, 7])
        S_A = x * C_A               # torch.Size([2, 2048, 7, 7])
        return S_A


class CAB(nn.Module):
    def __init__(self, in_planes, classes, k):
        super(CAB, self).__init__()
        self.classes = classes
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, k*classes, 1, padding='same'),
                                   nn.BatchNorm2d(k*classes),
                                   nn.ReLU())
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):          # torch.Size([2, 2048, 7, 7])
        F1 = self.conv1(inputs)         # torch.Size([2, 30, 7, 7])

        F2 = F1.clone().detach().requires_grad_(True)
        GMP = self.max_pool(F2)         # torch.Size([2, 30, 1, 1])

        x = GMP.reshape([inputs.size(0), self.k, self.classes, GMP.size(
            2), GMP.size(3)])   # torch.Size([2, 6, 5, 1, 1])
        S = torch.mean(x, dim=1, keepdim=False)  # torch.Size([2, 5, 1, 1])

        x = F1.reshape([inputs.size(0), self.k, self.classes, inputs.size(
            2), inputs.size(3)])  # torch.Size([2, 6, 5, 7, 7])
        # x = F1
        x = torch.mean(x, dim=1, keepdim=False)  # torch.Size([2, 5, 7, 7])

        x = S * x                               # torch.Size([2, 5, 7, 7])
        M = torch.mean(x, dim=1, keepdim=True)  # torch.Size([2, 1, 7, 7])

        semantic = inputs * M                   # torch.Size([2, 2048, 7, 7])
        return semantic


class AttnCABfc(nn.Module):
    def __init__(self, in_planes, n_class, k=5):
        super(AttnCABfc, self).__init__()
        self.gab_ = GAB(in_planes)
        self.cab_ = CAB(in_planes, n_class, k)
        self.avg_pool_ = nn.AdaptiveAvgPool2d(1)
        self.fc_ = nn.Sequential(
            nn.Linear(in_planes, in_planes),
            nn.BatchNorm1d(in_planes),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_planes, in_planes),
            nn.BatchNorm1d(in_planes),
            nn.ReLU(),
            nn.Linear(in_planes, n_class),
            nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.gab_(x)        # torch.Size([2, 2048, 7, 7])
        x = self.cab_(x)        # torch.Size([2, 2048, 7, 7])
        x = self.avg_pool_(x)    # torch.Size([2, 2048, 1, 1])
        x = torch.flatten(x, 1)  # torch.Size([2, 2048])
        x = self.fc_(x)          # torch.Size([2, 5])

        return x


def _attn(input_feats, classes, k=5):
    module = AttnCABfc(input_feats.size(1), classes, k)
    return module


class ResNetFeats(ResNet):

    # @override
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x


def _resnet(arch: str, block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            pretrained: bool,
            progress: bool,
            **kwargs: Any,
            ) -> ResNetFeats:
    model = ResNetFeats(block, layers, **kwargs)
    if pretrained:
        pass
    return model


def resNet101(n_class):

    model = _resnet('resnet101', Bottleneck, [3, 4, 23, 3], False, False)

    out = model(randn(2, 3, 224, 224))
    CAB_net = _attn(out, 5, 6)
    print(CAB_net)
    print(out.size())

    Fin = CAB_net(out)
    print(Fin.size())
    pdb.set_trace()

    model = resnet101(pretrained=True, progress=True)
    n_inputs = model.fc.in_features

    sequential_layers = nn.Sequential(
        nn.Linear(n_inputs, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(2048, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        nn.Linear(2048, n_class),
        nn.LogSoftmax(dim=1)
    )

    model.fc = sequential_layers
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


if __name__ == '__main__':
    resNet101(5)
