from torchvision.models import ResNet101_Weights
import torchvision
import torch.nn as nn
import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ViTImageEncoder(nn.Module):
    def __init__(self, finetuned=True):
        super(ViTImageEncoder, self).__init__()
        self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        # self.model = self.model.encoder
        # finetune?
        if not finetuned:
            for param in self.model.parameters():
                param.requires_grad = False
        
    def forward(self, images):
        x = self.model._process_input(images)
        n = x.shape[0]

        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        out = self.model.encoder(x)
        return out


class ImageEncoder(nn.Module):
    def __init__(self, finetuned=True):
        super(ImageEncoder, self).__init__()
        model = torchvision.models.resnet101(weights=ResNet101_Weights.DEFAULT)
        # ResNet-101网格表示提取器
        self.grid_rep_extractor = nn.Sequential(*(list(model.children())[:-2]))
        for param in self.grid_rep_extractor.parameters():
            param.requires_grad = finetuned
        
    def forward(self, images):
        out = self.grid_rep_extractor(images) 
        return out


class EntireImageEncoder(nn.Module):
    def __init__(self, finetuned=True):
        super(EntireImageEncoder, self).__init__()
        model = torchvision.models.resnet101(weights=ResNet101_Weights.DEFAULT)
        self.grid_rep_extractor = nn.Sequential(*(list(model.children())[:-1]))
        for param in self.grid_rep_extractor.parameters():
            param.requires_grad = finetuned

    def forward(self, images):
        out = self.grid_rep_extractor(images).reshape(-1, 2048)
        return out