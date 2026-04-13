"""
models.py — архитектуры классификаторов и фабричная функция build_model.

Доступные модели:
    "simple_cnn"          — лёгкая свёрточная сеть
    "resnet{18,34,50,101,152}"  — семейство ResNet
    "efficientnet-b{0..7}"     — семейство EfficientNet
"""

import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet


class SimpleCnn(nn.Module):
    """Lightweight 5-block CNN for image classification.

    Architecture: 5 × (Conv2d → BatchNorm → ReLU → MaxPool) → Linear.
    Expects input tensors of shape (B, 3, 224, 224).
    """

    def __init__(self, n_classes: int):
        super().__init__()
        self.conv1 = self._block(3, 8)
        self.conv2 = self._block(8, 16)
        self.conv3 = self._block(16, 32)
        self.conv4 = self._block(32, 64)
        self.conv5 = self._block(64, 96)
        self.out = nn.Linear(96 * 5 * 5, n_classes)

    @staticmethod
    def _block(in_ch: int, out_ch: int, kernel_size: int = 3) -> nn.Sequential:
        """Single conv block: Conv2d → BatchNorm2d → ReLU → MaxPool2d."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        x = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))
        return self.out(x.view(x.size(0), -1))

    def freeze_backbone(self):
        """No-op: SimpleCnn has no pretrained backbone."""
        pass

    def unfreeze_backbone(self):
        """No-op: SimpleCnn has no pretrained backbone."""
        pass


class SimpsonResNet(nn.Module):
    """ResNet-based classifier with a replaceable final FC layer.

    Args:
        n_classes:   Number of output classes.
        model_name:  ResNet variant — "resnet18" / "resnet34" / "resnet50" /
                     "resnet101" / "resnet152".
        pretrained:  Load ImageNet weights if True.
    """

    _VARIANTS = {
        "resnet18":  models.resnet18,
        "resnet34":  models.resnet34,
        "resnet50":  models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152,
    }

    def __init__(self, n_classes: int = 42, model_name: str = "resnet50", pretrained: bool = True):
        super().__init__()
        if model_name not in self._VARIANTS:
            raise ValueError(
                f"Unsupported ResNet variant: '{model_name}'. "
                f"Choose from: {list(self._VARIANTS)}"
            )
        self.backbone = self._VARIANTS[model_name](weights="DEFAULT" if pretrained else None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, n_classes)

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze all layers except the final FC classifier."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class SimpsonEfficientNet(nn.Module):
    """EfficientNet-based classifier with a replaceable final FC layer.

    Args:
        n_classes:   Number of output classes.
        model_name:  EfficientNet variant, e.g. "efficientnet-b4".
        pretrained:  Load ImageNet weights via EfficientNet.from_pretrained if True.
    """

    def __init__(
        self,
        n_classes: int = 42,
        model_name: str = "efficientnet-b4",
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone = (
            EfficientNet.from_pretrained(model_name)
            if pretrained
            else EfficientNet.from_name(model_name)
        )
        self.backbone._fc = nn.Linear(self.backbone._fc.in_features, n_classes)

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze all layers except the final _fc classifier."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone._fc.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_RESNET_VARIANTS = set(SimpsonResNet._VARIANTS)


def build_model(model_name: str, n_classes: int, pretrained: bool = True) -> nn.Module:
    """Instantiate a model by name.

    Args:
        model_name: One of "simple_cnn", "resnet*", or "efficientnet-b*".
        n_classes:  Number of output classes.
        pretrained: Load pretrained ImageNet weights when available.

    Returns:
        Configured nn.Module ready for training.

    Raises:
        ValueError: If model_name is not recognised.
    """
    if model_name == "simple_cnn":
        return SimpleCnn(n_classes=n_classes)
    if model_name in _RESNET_VARIANTS:
        return SimpsonResNet(n_classes=n_classes, model_name=model_name, pretrained=pretrained)
    if model_name.startswith("efficientnet-"):
        return SimpsonEfficientNet(n_classes=n_classes, model_name=model_name, pretrained=pretrained)
    raise ValueError(
        f"Unknown model '{model_name}'. "
        f"Supported: simple_cnn, {sorted(_RESNET_VARIANTS)}, efficientnet-b0 … efficientnet-b7"
    )