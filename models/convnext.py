from torchvision.models import convnext_small
from torchvision.models.convnext import LayerNorm2d
import torch.nn as nn


def convNextSmallCustom(n_class):

    model = convnext_small(pretrained=True, progress=True)
    model.named_children()
    n_inputs = None
    for name, child in model.named_children():
        if name == 'classifier':
            for sub_name, sub_child in child.named_children():
                if sub_name == '2':
                    n_inputs = sub_child.in_features
    sequential_layers = nn.Sequential(
        LayerNorm2d((768,), eps=1e-06, elementwise_affine=True),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(n_inputs, 2048, bias=True),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(2048, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        nn.Linear(2048, n_class),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = sequential_layers

    return model

def convNextSmallegacy(n_class):

    model = convnext_small(pretrained=True, progress=True)
    print(model)

    model.named_children()
    n_inputs = None
    for name, child in model.named_children():
        if name == 'classifier':
            for sub_name, sub_child in child.named_children():
                if sub_name == '2':
                    n_inputs = sub_child.in_features
    sequential_layers = nn.Sequential(
        LayerNorm2d((768,), eps=1e-06, elementwise_affine=True),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(n_inputs, n_class, bias=True),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = sequential_layers

    return model

if __name__ == '__main__':
    convNextSmallegacy(5)