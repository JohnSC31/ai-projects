import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl
from collections import defaultdict

class ButterflyDataset(Dataset):
    """Dataset para imágenes de mariposas según estructura especificada."""
    def __init__(self, samples, transform=None, classes=None):
        self.samples = samples
        self.transform = transform
        self.classes = classes
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label
    
def ButterflyDatasetFactory(split_dir, transform=None, labeled_percentage=0.3):
        labeled_samples = []
        unlabeled_samples = []
        classes = sorted([d for d in os.listdir(split_dir) 
                             if os.path.isdir(os.path.join(split_dir, d))])
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        counter = {cls: i for i, cls in enumerate(classes)}

        for species in classes:
            species_dir = os.path.join(split_dir, species)
            total = len(os.listdir(species_dir))
            labeled_count = int(total * labeled_percentage)
            num_labeled_samples = 0
            for img_name in os.listdir(species_dir):
                if img_name.lower().endswith('.jpg'):
                    img_path = os.path.join(species_dir, img_name)
                    if num_labeled_samples < labeled_count:
                        labeled_samples.append((img_path, class_to_idx[species]))
                        num_labeled_samples += 1
                    else:
                        unlabeled_samples.append((img_path, None))
        
        return ButterflyDataset(labeled_samples, transform, classes), ButterflyDataset(unlabeled_samples, transform, classes), classes
        

class ButterflyDataModule(pl.LightningDataModule):
    """DataModule para el proyecto de clasificación de mariposas."""
    
    def __init__(self, data_root="dataset", batch_size=32, image_size=(128, 128), labeled_percentage=0.3):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.image_size = image_size
        self.labeled_percentage = labeled_percentage

        self.train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_test_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def setup(self, stage=None):
        self.train_dataset_labeled, self.train_dataset_unlabeled, classes = ButterflyDatasetFactory(
            os.path.join(self.data_root, "train"),
            transform=self.train_transform
        )

        self.val_dataset_labeled, self.val_dataset_unlabeled, _ = ButterflyDatasetFactory(
            os.path.join(self.data_root, "valid"),
            transform=self.val_test_transform
        )

        self.test_dataset_labeled, self.test_dataset_unlabeled, _ = ButterflyDatasetFactory(
            os.path.join(self.data_root, "test"),
            transform=self.val_test_transform
        )
        self.classes = classes

    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset_labeled,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    def train_dataloader_unlabeled(self):
        return DataLoader(
            self.train_dataset_unlabeled,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    def val_dataloader_unlabeled(self):
        return DataLoader(
            self.val_dataset_unlabeled,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset_labeled,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    def test_dataloader_unlabeled(self):
        return DataLoader(
            self.test_dataset_unlabeled,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset_labeled,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )