import torch
import random
from torchvision import transforms
from PIL import Image

class SaltAndPepperNoise:
    """Implementación del filtro Salt and Pepper para Denoising Autoencoder."""
    
    def __init__(self, prob=0.05):
        self.prob = prob
    
    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = transforms.functional.to_tensor(img)
        
        noisy_img = img.clone()
        _, h, w = noisy_img.shape
        
        salt_mask = torch.rand(h, w) < (self.prob / 2)
        pepper_mask = torch.rand(h, w) < (self.prob / 2)
        
        noisy_img[:, salt_mask] = 1.0
        noisy_img[:, pepper_mask] = 0.0
        
        return noisy_img

def get_denoising_transform(image_size=(128, 128), noise_prob=0.05):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        SaltAndPepperNoise(prob=noise_prob),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])