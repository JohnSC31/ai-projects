from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv2d => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class BernoulliNoise(nn.Module):
    """
    Applies salt-and-pepper (Bernoulli) noise to the input tensor.
    Each pixel has a probability `p` of being flipped (0->1 or 1->0).
    """
    def __init__(self, p=0.05):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        # Generate mask: 1 means flip, 0 means keep
        mask = torch.bernoulli(torch.full_like(x, self.p))
        flipped = 1 - x
        return mask * flipped + (1 - mask) * x

class VAE(LightningModule):
    def __init__(
                self, 
                in_channels=1, 
                out_channels=1, 
                hidden_dim=128, 
                latent_dim=64, 
                num_classes=30,
                noise_p=0.01, 
                lr=1e-3,
                kld_w=0.00025,
                recon_w=0.5,
                cl_w=0.5):
        """
        Variational Autoencoder (VAE) with Bernoulli noise layer.
        Args:
            in_channels (int): Entry channel of the inputs
            out_channels (int): Output channels
            hidden_dim (int): Dimension of the hidden layers.
            latent_dim (int): Dimension of the latent space.
            noise_p (float): Probability of flipping pixels in Bernoulli noise.
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.kld_w = kld_w
        self.recon_w = recon_w
        self.cls_w = cl_w

        # Add Bernoulli noise layer
        self.bernoulli_noise = BernoulliNoise(p=noise_p)
        # Encoder / Lower dimensionality
        self.encoder1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)

        # Encoder / Mean and log variance
        self.flatten_bottleneck = nn.Flatten()
        self.linear1 = nn.Linear(512 * 28 * 28, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_mean = nn.Linear(hidden_dim, latent_dim)
        self.linear_log_var = nn.Linear(hidden_dim, latent_dim)

        # Classifier
        self.classifier = nn.Linear(latent_dim, num_classes)


        # Decoder / Lineal layers
        self.backlinear1 = nn.Linear(latent_dim, hidden_dim)
        self.backlinear2 = nn.Linear(hidden_dim, hidden_dim)
        self.backlinear3 = nn.Linear(hidden_dim, 512 * 28 * 28)
        self.unflatten_back = nn.Unflatten(1, (512, 28, 28))
        # Decoder / Higher dimensionality
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)


    def encode(self, x):
        n_x = self.bernoulli_noise(x)
        e1 = self.encoder1(n_x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))
        b_flat = self.flatten_bottleneck(b)
        h = F.relu(self.linear1(b_flat))
        h = F.relu(self.linear2(h))
        # Mean and log variance
        mean = F.relu(self.linear_mean(h))
        log_var = F.relu(self.linear_log_var(h))

        return mean, log_var, e1, e2, e3

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
        
    def decode(self, z, e1, e2, e3):
        bh1 = F.relu(self.backlinear1(z))
        bh2 = F.relu(self.backlinear2(bh1))
        bh3 = F.relu(self.backlinear3(bh2))
        unflatten = self.unflatten_back(bh3)
        d3 = self.upconv3(unflatten)
        d3 = self.decoder3(torch.cat([d3, e3], dim=1))

        d2 = self.upconv2(d3)
        d2 = self.decoder2(torch.cat([d2, e2], dim=1))

        d1 = self.upconv1(d2)
        d1 = self.decoder1(torch.cat([d1, e1], dim=1))
        return self.final_conv(d1)

    # Extrae solo el vector latente z sin decodificar
    def get_latent_representation(self, x):
        mean, log_var, _, _, _ = self.encode(x)
        return self.reparameterize(mean, log_var)

    def forward(self, x, return_latent=False):
        mean, log_var, e1, e2, e3 = self.encode(x)
        z = self.reparameterize(mean, log_var)
        if return_latent:
            return z
        y = self.classifier(z)
        return self.decode(z, e1, e2, e3), y, mean, log_var, z

    def training_step(self, batch, batch_idx):
        x, y_cls = batch
        recon_x, logits, mean, log_var, z = self(x)
        recon_loss = F.mse_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        cls_loss = F.cross_entropy(logits, y_cls)
        loss = recon_loss*self.recon_w+cls_loss*self.cls_w+KLD*self.kld_w
        acc = (logits.argmax(dim=1) == y_cls).float().mean()
        self.log('train/recon_loss', recon_loss)
        self.log('train/cls_loss', cls_loss)
        self.log('train/KLD_loss', KLD)
        self.log('train/loss', loss)
        self.log('train/acc', acc)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y_cls = batch
        recon_x, logits, mean, log_var, z = self(x)
        recon_loss = F.mse_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        cls_loss = F.cross_entropy(logits, y_cls)
        loss = recon_loss*self.recon_w+cls_loss*self.cls_w+KLD*self.kld_w
        acc = (logits.argmax(dim=1) == y_cls).float().mean()
        self.log('val/recon_loss', recon_loss)
        self.log('val/cls_loss', cls_loss)
        self.log('val/KLD_loss', KLD)
        self.log('val/loss', loss)
        self.log('val/acc', acc)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y_cls = batch
        recon_x, logits, mean, log_var, z = self(x)
        recon_loss = F.mse_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        cls_loss = F.cross_entropy(logits, y_cls)
        loss = recon_loss*self.recon_w+cls_loss*self.cls_w+KLD*self.kld_w
        acc = (logits.argmax(dim=1) == y_cls).float().mean()
        self.log('test/recon_loss', recon_loss)
        self.log('test/cls_loss', cls_loss)
        self.log('test/KLD_loss', KLD)
        self.log('test/loss', loss)
        self.log('test/acc', acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer