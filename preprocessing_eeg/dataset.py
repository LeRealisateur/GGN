import os

import mne
import numpy as np
import torch
from mne import read_epochs
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

mne.set_log_level("ERROR")


def generate_distance_topology(raw):
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    pos = np.array([
        ch['loc'][:3] for ch in raw.info['chs']
        if ch['kind'] == mne.io.constants.FIFF.FIFFV_EEG_CH and np.any(ch['loc'][:3])
    ])

    diff = pos[:, None, :] - pos[None, :, :]

    squared_diff = diff ** 2
    squared_distances = squared_diff.sum(axis=2)
    conn_matrix = np.sqrt(squared_distances)

    return torch.tensor(conn_matrix)


def generate_correlation_topology(epoch_data):
    """
    Generates a correlation-based connectivity matrix based on EEG channel signals.
    """
    # Change to pearson
    correlation_matrix = np.corrcoef(epoch_data)
    return torch.tensor(correlation_matrix, dtype=torch.float32)


def combine_topologies(distance_topology, correlation_topology, alpha=0.5):
    combined_topology = alpha * distance_topology + (1 - alpha) * correlation_topology
    return combined_topology


def normalize_epochs_data(X):
    scaler = StandardScaler()
    for i in range(X.shape[0]):
        X[i, :] = scaler.fit_transform(X[i, :].reshape(-1, 1)).flatten()
    return X


class EEGDataset(Dataset):
    def __init__(self, base_dir, subjects=None, tasks=None, task_dict=None):
        self.files = []
        self.labels = []

        for subject in subjects:
            for task in tasks:
                task_dir = os.path.join(base_dir, subject, task)
                task_files = sorted([
                    os.path.join(task_dir, file)
                    for file in os.listdir(task_dir)
                    if file.endswith("-epo.fif")
                ])
                self.files.extend(task_files)

                task_label = task_dict[task][0]
                self.labels.extend([task_label] * len(task_files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        epoch_file = self.files[idx]
        epoch = read_epochs(epoch_file, preload=True)
        epoch_data = epoch.get_data()[0]

        epoch_data = normalize_epochs_data(epoch_data)

        epoch_tensor = torch.tensor(epoch_data, dtype=torch.float32)
        epoch_tensor = self.apply_fft_transform(epoch_tensor)

        label = self.labels[idx]
        label_tensor = torch.tensor(label, dtype=torch.long)

        distance_topology = generate_distance_topology(epoch)
        correlation_topology = generate_correlation_topology(epoch_data)
        epoch_topology = combine_topologies(distance_topology, correlation_topology)

        return epoch_tensor, epoch_topology, label_tensor

    def apply_fft_transform(self, X, n_fft=128):
        """
        Applies FFT to each sample in the dataset.

        Parameters:
        - X: Tensor of shape [samples, channels, time_steps].

        Returns:
        - Tensor of shape [samples, channels, freq_bins] in the frequency domain.
        """
        X_fft = [self.fft_transform(sample.clone().detach(), n_fft=n_fft) for sample in X]
        return torch.stack(X_fft)

    def fft_transform(self, data, n_fft=128):
        """
        Applies FFT to each channel in the EEG data.

        Parameters:
        - data: Tensor of shape [channels, time_steps] (for one sample).
        - n_fft: Number of FFT points.

        Returns:
        - Tensor of shape [channels, freq_bins] representing frequency-domain data.
        """
        # Apply FFT along the time dimension (last dimension)
        data_fft = torch.fft.rfft(data, n=n_fft, dim=-1)
        data_mag = torch.abs(data_fft)
        return data_mag


def create_dataloader(config):
    subjects = config['data']['subjects']
    tasks = config['data']['tasks']
    batch_size = 50

    task_dict = {
        "audioactive": [0, "audio"],
        "audiopassive": [0, "audio"],
        "thermalactive": [1, "pain"],
        "thermalpassive": [1, "pain"],
    }

    train_dataset = EEGDataset(os.path.join('./data', "train"), subjects, tasks, task_dict)
    val_dataset = EEGDataset(os.path.join('./data', "val"), subjects, tasks, task_dict)
    test_dataset = EEGDataset(os.path.join('./data', "test"), subjects, tasks, task_dict)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
