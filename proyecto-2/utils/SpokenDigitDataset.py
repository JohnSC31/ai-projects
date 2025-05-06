from torch.utils.data import Dataset
import os
import librosa
import torch
import numpy as np

class SpokenDigitDataset(Dataset):
    def __init__(self, root_dir, sr=16000, n_mels=128):
        self.root_dir = root_dir
        self.sr = sr
        self.n_mels = n_mels
        self.filepaths = []
        self.labels = []

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

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label = self.labels[idx]

        y, sr = librosa.load(filepath, sr=self.sr)

        # Extraer log-Mel espectrograma
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Convertir a tensor de forma (1, n_mels, tiempo)
        log_mel_tensor = torch.tensor(log_mel_spec, dtype=torch.float32).unsqueeze(0)

        return log_mel_tensor, label
