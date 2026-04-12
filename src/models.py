import torch.nn as nn
from torchvision import models

class SimpleCnn(nn.Module):
    """Lightweight 5-block CNN for image classification.

    Architecture: 5 × (Conv2d → BatchNorm → ReLU → MaxPool) → Dropout → Linear.
    Expects input tensors of shape (B, 3, 224, 224).
    """

    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = self._block(3, 8)
        self.conv2 = self._block(8, 16)
        self.conv3 = self._block(16, 32)
        self.conv4 = self._block(32, 64)
        self.conv5 = self._block(64, 96)

        #self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(96 * 5 * 5, n_classes)

    @staticmethod
    def _block(in_ch, out_ch, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(x.size(0), -1)
        #x = self.dropout(x)
        logits = self.out(x)
        return logits

class SimpsonResNet(nn.Module):
    """ResNet-based classifier fine-tuned for 42 Simpson characters.
    
    Uses pretrained weights and replaces the final FC layer.
    Input shape: (B, 3, 224, 224).
    """

    def __init__(self, n_classes=42, model_name='resnet50', pretrained=True):
        super().__init__()
        
        model_dict = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152,
        }
        
        if model_name not in model_dict:
            raise ValueError(f"Unsupported model: {model_name}. Choose from {list(model_dict.keys())}")
        
        self.backbone = model_dict[model_name](weights='DEFAULT' if pretrained else None)
        num_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Linear(num_features, n_classes)

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        """Замораживает все слои кроме классификатора."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        """Размораживает все слои (для второго этапа fine-tuning)."""
        for param in self.backbone.parameters():
            param.requires_grad = True