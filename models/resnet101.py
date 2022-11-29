from typing_extensions import override
from torchvision.models import resnet101
from models.attentionblocks import AttnCABfc
from torchvision.models.resnet import ResNet, model_urls, BasicBlock, Bottleneck
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor, tensor, randn
import torch.nn as nn
import pdb

def resNet101ABTest(n_class):

    model = _resnet('resnet101', Bottleneck, [3, 4, 23, 3], False, False)
    model = ResNet101AB()
    out = model(randn(2, 3, 224, 224))
    # CAB_net = _attn(out, 5, 6)
    # print(CAB_net)
    print(out.size())

    # Fin = CAB_net(out)
    #print(Fin.size())
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


class ResNet101AB(nn.Module):
    def __init__(self, in_planes = 2048, classes = 5, k = 5):
        super(ResNet101AB, self).__init__()
        self.backbone = _resnet('resnet101', Bottleneck, [3, 4, 23, 3], False, False)
        self.attnblocks = AttnCABfc(in_planes, classes, k)

    def forward(self, x):
        x = self.backbone(x)
        x = self.attnblocks(x)

        return x

def resNet101Legacy(n_class):
    model = resnet101(pretrained= True, progress= True)

    n_inputs = model.fc.in_features

    sequential_layers = nn.Sequential(
        nn.Linear(n_inputs, n_class),
        nn.LogSoftmax(dim=1)
    )
    
    model.fc = sequential_layers

    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def resNet101Custom(n_class):
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
    resNet101Legacy(5)