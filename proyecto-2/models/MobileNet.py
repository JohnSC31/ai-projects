import torch
import torch.nn as nn
import torchvision.models as models

class MobileNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()

        # Cargar MobileNetV2 sin pesos preentrenados
        self.backbone = models.mobilenet_v2(weights=None)

        # Cambiar la primera capa para 1 canal (espectrogramas en escala de grises)
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=1,  # espectrograma = 1 canal
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )

        # Cambiar la capa final de clasificación
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
