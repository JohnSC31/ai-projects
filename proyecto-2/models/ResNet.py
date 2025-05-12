import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper function for 3x3 convolution
conv_k_3 = lambda channel1, channel2, stride: nn.Conv2d(channel1, channel2, stride=stride, kernel_size=3, padding=1)

class residual_block(nn.Module):
    '''
    Bloque residual básico utilizado en ResNet18/34.
    Este bloque consiste en dos capas convolucionales 3x3 con Batch Normalization y ReLU.
    La conexión de atajo (shortcut) se aplica si 'change_size' es True,
    lo que implica un cambio en las dimensiones espaciales o en el número de canales.
    '''
    def __init__(self, in_channel, out_channel, stride=1, change_size=False):
        super().__init__()
        self.conv1 = conv_k_3(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = conv_k_3(out_channel, out_channel, 1) # La segunda conv en el bloque siempre tiene stride 1
        self.bn2 = nn.BatchNorm2d(out_channel)
        
        self.change_size = change_size # Indica si se necesita una proyección en el atajo
        if self.change_size:
            # Si el tamaño cambia (por stride > 1 o in_channel != out_channel),
            # se utiliza una convolución 1x1 en el atajo para ajustar dimensiones.
            self.residual = nn.Sequential(nn.Conv2d(in_channel, 
                                                    out_channel, 
                                                    kernel_size=1,
                                                    stride=stride),
                                         nn.BatchNorm2d(out_channel)
                                         )      
    def forward(self, x):
        identity = x
        if self.change_size:
            identity = self.residual(x)
            
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y += identity # Suma la salida del bloque con el atajo
        return F.relu(y)

class ResNet18(nn.Module):
    '''
    Implementación de un modelo ResNet18 adaptado para entradas de 1 canal
    (como los espectrogramas) y con la configuración de canales típica de CIFAR-10
    (16, 32, 64 canales en las etapas residuales).
    '''
    def __init__(self, num_classes=10, in_channels=1): # in_channels=1 para espectrogramas (monocanal)
        super().__init__()
        # Convolución inicial: adaptada para 1 canal de entrada
        # y usando stride=1 y kernel_size=3 como en la adaptación de ResNet56 para CIFAR10.
        self.conv1 = conv_k_3(in_channels, 16, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        n_blocks = 2 # Para ResNet18, se utilizan 2 bloques residuales básicos por etapa.

        # Etapa 1: 2 bloques residuales. Canales de entrada/salida 16. Stride 1.
        # No se necesita proyección en el atajo para el primer bloque ya que ni los canales ni el stride cambian.
        self.block1 = self.create_block(n=n_blocks, in_channel=16, 
                                        out_channel=16, stride=1, 
                                        change_size=False)
        
        # Etapa 2: 2 bloques residuales. Entrada 16, Salida 32. Stride 2.
        # Se necesita proyección en el atajo para el primer bloque debido al cambio de canales y al stride.
        self.block2 = self.create_block(n=n_blocks, in_channel=16, 
                                        out_channel=32, stride=2,
                                        change_size=True) 
        
        # Etapa 3: 2 bloques residuales. Entrada 32, Salida 64. Stride 2.
        # Se necesita proyección en el atajo para el primer bloque.
        self.block3 = self.create_block(n=n_blocks, in_channel=32, 
                                        out_channel=64, stride=2,
                                        change_size=True)
        
        # Capa Fully Connected final
        self.fc = nn.Linear(64, num_classes)

    def create_block(self, n, in_channel, out_channel, stride, change_size=True):
        '''
        Crea una secuencia de 'n' bloques residuales para una etapa.
        El primer bloque de la secuencia puede cambiar el tamaño/canales (si 'change_size' es True).
        Los bloques subsiguientes dentro de la misma etapa mantienen el tamaño y usan stride=1.
        '''
        block = [residual_block(in_channel, out_channel, stride, change_size=change_size)]
        for i in range(n - 1):
            # Los bloques subsiguientes usan conexión de atajo de identidad (no necesitan proyección)
            block.append(residual_block(out_channel, out_channel, stride=1, change_size=False)) 
        return nn.Sequential(*block)   
        
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.block3(self.block2(self.block1(y))) # Aplicación secuencial de las etapas de bloques
        y = F.adaptive_avg_pool2d(y, 1) # Global Average Pooling para reducir las dimensiones espaciales a 1x1
        return self.fc(y.view(y.size(0), -1)) # Aplanar y pasar por la capa completamente conectada