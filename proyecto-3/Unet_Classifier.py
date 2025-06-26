import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from Unet import UNet

class UNet_Classifier(pl.LightningModule):
    def __init__(self, 
                 lr=1e-3, 
                 in_channels=3, 
                 out_channels=1, 
                 num_classes=10, 
                 ae_w=1.0, 
                 cl_w=1.0,
                 unet=None):
        super().__init__()
        self.save_hyperparameters()
        if unet is None:
            self.unet = UNet(lr=lr, in_channels=in_channels, out_channels=out_channels)
        else:
            self.unet = unet
        self.ae_w = ae_w  # Weight for reconstruction loss
        self.cl_w = cl_w  # Weight for classification loss
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # [B, C, H, W] -> [B, C, 1, 1]
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x_hat, b = self.unet(x)
        pooled = self.global_pool(b).view(b.size(0), -1)
        logits = self.classifier(pooled)
        return x_hat, logits

    def training_step(self, batch, batch_idx):
        x, y_cls = batch  # y_seg: segmentation/reconstruction target, y_cls: class index
        x_hat, logits = self.forward(x)
        loss_recon = F.mse_loss(x_hat, x)
        loss_cls = F.cross_entropy(logits, y_cls)
        loss = self.ae_w * loss_recon + self.cl_w * loss_cls
        acc = (logits.argmax(dim=1) == y_cls).float().mean()
        self.log("train_loss", loss)
        self.log("train_recon_loss", loss_recon)
        self.log("train_cls_loss", loss_cls)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_seg, y_cls = batch  # y_seg: segmentation/reconstruction target, y_cls: class index
        x_hat, logits = self.forward(x)
        loss_recon = F.mse_loss(x_hat, y_seg)
        loss_cls = F.cross_entropy(logits, y_cls)
        loss = self.ae_w * loss_recon + self.cl_w * loss_cls
        acc = (logits.argmax(dim=1) == y_cls).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

    def test_step(self, batch, batch_idx):
        x, y_seg, y_cls = batch  # y_seg: segmentation/reconstruction target, y_cls: class index
        x_hat, logits = self.forward(x)
        loss_recon = F.mse_loss(x_hat, y_seg)
        loss_cls = F.cross_entropy(logits, y_cls)
        loss = self.ae_w * loss_recon + self.cl_w * loss_cls
        acc = (logits.argmax(dim=1) == y_cls).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
    def freeze_unet(self):
        for param in self.unet.parameters():
            param.requires_grad = False