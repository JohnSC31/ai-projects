
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class DatasetSplitter:
    """
    A utility class to split a PyTorch Dataset into training, validation, and test sets
    and create DataLoaders for each split.
    """
    def __init__(self, dataset: Dataset, split_ratios: tuple = (0.8, 0.1, 0.1), batch_size: int = 32, seed: int = 42):
        """
        Initializes the DatasetSplitter.

        Args:
            dataset (Dataset): The full PyTorch Dataset to split.
            split_ratios (tuple): A tuple of three floats representing the
                                  proportion for train, validation, and test sets.
                                  Must sum to approximately 1.0.
                                  Defaults to (0.8, 0.1, 0.1).
            batch_size (int): The batch size for the DataLoaders. Defaults to 32.
            seed (int): The random seed for reproducible splitting. Defaults to 42.
        """
        if not isinstance(dataset, Dataset):
            raise TypeError("Dataset invalido.")
        if not (isinstance(split_ratios, tuple) and len(split_ratios) == 3 and sum(split_ratios) > 0.99 and sum(split_ratios) < 1.01):
             raise ValueError("split_ratios debe ser una tupla de 3 que sume aproximadamente 1")
        if not isinstance(batch_size, int) or batch_size <= 0:
             raise ValueError("batch_size debe ser entero positivo")
        if not isinstance(seed, int):
             raise ValueError("seed debe ser un entero.")


        self.full_dataset = dataset
        self.split_ratios = split_ratios
        self.batch_size = batch_size
        self.seed = seed

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None

        # Perform the split upon initialization
        self._perform_split()

    def _perform_split(self):
        """Realiza el split del dataset completo dado."""
        dataset_size = len(self.full_dataset)
        train_size = int(self.split_ratios[0] * dataset_size)
        val_size = int(self.split_ratios[1] * dataset_size)
        test_size = dataset_size - train_size - val_size # Ensure all samples are included

        print(f"Splitting dataset (Total: {dataset_size}) into:")
        print(f"  Train: {train_size} samples ({self.split_ratios[0]*100:.1f}%)")
        print(f"  Validation: {val_size} samples ({self.split_ratios[1]*100:.1f}%)")
        print(f"  Test: {test_size} samples ({self.split_ratios[2]*100:.1f}%)")


        generator = torch.Generator().manual_seed(self.seed)

        self._train_dataset, self._val_dataset, self._test_dataset = random_split(
            self.full_dataset, [train_size, val_size, test_size], generator=generator
        )

        print("Split completo.")


    def configure_splits(self, bilateral=False, augment=False):
        """
        Configuracion para aplicacion del filtro y augmentation para todos los datasets
        """
        if self._train_dataset and self._val_dataset and self._test_dataset:
            self._train_dataset.dataset.datasetConfig(bilateral=bilateral, augment=augment)
            self._train_dataset.dataset.datasetConfig(bilateral=bilateral, augment=augment)
            self._train_dataset.dataset.datasetConfig(bilateral=bilateral, augment=augment)

        print("Configuration complete.")


    @property
    def train_dataset(self) -> Dataset:
        """Returns the training dataset (Subset object)."""
        return self._train_dataset

    @property
    def val_dataset(self) -> Dataset:
        """Returns the validation dataset (Subset object)."""
        return self._val_dataset

    @property
    def test_dataset(self) -> Dataset:
        """Returns the test dataset (Subset object)."""
        return self._test_dataset

    @property
    def train_dataloader(self) -> DataLoader:
        """Returns the training DataLoader."""
        if self._train_dataloader is None:
            self._train_dataloader = DataLoader(self._train_dataset, batch_size=self.batch_size, shuffle=True)
            print(f"\nCreated Train DataLoader with batch size {self.batch_size}. Number of batches: {len(self._train_dataloader)}")
        return self._train_dataloader

    @property
    def val_dataloader(self) -> DataLoader:
        """Returns the validation DataLoader."""
        if self._val_dataloader is None:
            self._val_dataloader = DataLoader(self._val_dataset, batch_size=self.batch_size, shuffle=False)
            print(f"Created Validation DataLoader with batch size {self.batch_size}. Number of batches: {len(self._val_dataloader)}")
        return self._val_dataloader

    @property
    def test_dataloader(self) -> DataLoader:
        """Returns the test DataLoader."""
        if self._test_dataloader is None:
            self._test_dataloader = DataLoader(self._test_dataset, batch_size=self.batch_size, shuffle=False)
            print(f"Created Test DataLoader with batch size {self.batch_size}. Number of batches: {len(self._test_dataloader)}")
        return self._test_dataloader
