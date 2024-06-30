import torch
import torch.nn as nn
from torchvision.models import resnet34
from timm.models.vision_transformer import VisionTransformer


class CNNModel(nn.Module):
    def __init__(self, num_classes=100):
        super(CNNModel, self).__init__()
        self.model = resnet34(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class TransformerModel(nn.Module):
    def __init__(self, num_classes=100, pretrained=True):
        super(TransformerModel, self).__init__()
        self.model = VisionTransformer(
            img_size=224,
            patch_size=16,
            num_classes=num_classes,
            embed_dim=768,
            depth=3,
            num_heads=8,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=nn.LayerNorm
        )
        if pretrained:
            self.load_pretrained_weights()
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def load_pretrained_weights(self):
        state_dict = torch.hub.load(
            'rwightman/pytorch-image-models',
            'vit_base_patch16_224',
            pretrained=False
        ).state_dict()

        del state_dict['head.weight']
        del state_dict['head.bias']

        self.model.load_state_dict(state_dict, strict=False)



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':

    model1 = CNNModel()
    model2 = TransformerModel()
    print(count_parameters(model1))
    print(count_parameters(model2))

    for name, module in model1.named_children():
        print(name,module)
        print('\n')

    for name, module in model2.named_children():
        print(name,module)
