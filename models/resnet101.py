from typing_extensions import override
from torchvision.models import resnet101
from torchvision.models.resnet import ResNet, model_urls, BasicBlock, Bottleneck, load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
import torch.nn as nn
import pdb


class ResNetCAB(ResNet):

    @override
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def _resnet(arch: str, block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            pretrained: bool,
            progress: bool,
            **kwargs: Any,
            ) -> ResNetCAB:
    model = ResNetCAB(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resNet101(n_class):

    model = _resnet('resnet101', Bottleneck, [3, 4, 23, 3], False, False)
    print(model)
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
