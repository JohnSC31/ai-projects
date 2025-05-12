from torch.utils.data import Dataset
import os
import librosa
import torch
import numpy as np
import cv2
import random

class SpokenDigitDataset(Dataset):

    def __init__(self, root_dir, sr=16000, n_mels=128, resize=(224, 224), bilateral=False, augment=False):
        self.root_dir = root_dir
        self.sr = sr
        self.n_mels = n_mels
        self.filepaths = []
        self.labels = []
        self.resize = resize
        self.bilateral = bilateral
        self.augment = augment

        for speaker_id in sorted(os.listdir(root_dir)):
            speaker_path = os.path.join(root_dir, speaker_id)
            if not os.path.isdir(speaker_path):
                continue
            for fname in os.listdir(speaker_path):
                if fname.endswith(".wav"):
                    path = os.path.join(speaker_path, fname)
                    digit = int(fname.split('_')[0])  # etiqueta
                    self.filepaths.append(path)
                    self.labels.append(digit)

    """
        Para configurar el dataset necesario, con filtro bilatera y/o augmentation
    """
    def datasetConfig(self, bilateral, augment, sr=16000, n_mels=128):
        self.bilateral = bilateral
        self.augment = augment
        self.sr = sr
        self.n_mels = n_mels


    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label = self.labels[idx]

        y, sr = librosa.load(filepath, sr=self.sr)

        # se aplicar ruido
        # if self.augment:
            # noise = np.random.normal(0, 0.005, size=y.shape)
            # y = y + noise

        # Extraer log-Mel espectrograma
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # 4. Aplicar filtro bilateral (si se solicita)
        if self.bilateral:
            log_mel_spec = log_mel_spec.astype(np.float32)
            log_mel_spec = cv2.normalize(log_mel_spec, None, 0, 255, cv2.NORM_MINMAX)
            log_mel_spec = cv2.bilateralFilter(log_mel_spec, d=9, sigmaColor=75, sigmaSpace=75)
            log_mel_spec = cv2.normalize(log_mel_spec, None, 0, 1, cv2.NORM_MINMAX)

        if self.augment:
            log_mel_spec = self.__apply_spec_augment(log_mel_spec)

        # 5. Redimensionar segun la configuracion
        resized_log_mel = cv2.resize(log_mel_spec, self.resize, interpolation=cv2.INTER_LINEAR)

        # Convertir a tensor de forma (1, n_mels, tiempo)
        log_mel_tensor = torch.tensor(resized_log_mel, dtype=torch.float32).unsqueeze(0)

        return log_mel_tensor, label

    def __apply_spec_augment(self, spec, freq_mask_param=20, time_mask_param=4, num_freq_masks=1, num_time_masks=1):

        spec = spec.copy()

        num_mels, num_frames = spec.shape

        # Frequency masking
        for _ in range(num_freq_masks):
            f = random.randint(0, freq_mask_param)
            f0 = random.randint(0, max(1, num_mels - f))
            spec[f0:f0+f, :] = 0

        # Time masking
        for _ in range(num_time_masks):
            t = random.randint(0, time_mask_param)
            t0 = random.randint(0, max(1, num_frames - t))
            spec[:, t0:t0+t] = 0

        return spec
