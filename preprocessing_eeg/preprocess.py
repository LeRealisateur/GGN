import mne
import numpy as np
import os.path as path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import joblib


class EEGDataset(Dataset):
    def __init__(self, data, label):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


def load_raw_data(bids_root, subjects_id, task_dict, epochs_length=4):
    raws = []
    for subject_id in subjects_id:
        for task in task_dict.keys():
            raw_data = mne.io.read_raw_brainvision(
                path.join(bids_root, subject_id, "eeg", f"{subject_id}_task-{task}_eeg.vhdr"),
                eog=["VEOG", "HEOG"],
                misc=["rating", "temp", "stim"],
                preload=True,
            )
            if 'Iz' in raw_data.ch_names:
                raw_data.drop_channels(["Iz"])

            events = mne.make_fixed_length_events(raw_data, id=task_dict[task][0], duration=epochs_length)

            annotations = mne.annotations_from_events(
                events,
                sfreq=raw_data.info["sfreq"],
                event_desc={task_dict[task][0]: task_dict[task][1]},
            )

            raw_data.set_annotations(annotations)
            raws.append(raw_data.copy())

    return raws


def preprocess_raw_data(raws):
    raw = mne.concatenate_raws(raws).load_data()
    raw = raw.pick(['eeg'])
    raw.set_eeg_reference(ref_channels="average", verbose=True)
    raw.notch_filter(np.arange(60, 241, 60), verbose=True)
    return raw


def create_epochs(raw, t_min=0, t_max=4):
    events, event_id = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, event_id, t_min, t_max, baseline=None)
    return epochs


def normalize_epochs_data(X):
    scaler = StandardScaler()
    for i in range(X.shape[0]):
        X[i, :, :] = scaler.fit_transform(X[i, :, :])
    return X


def get_labels(epochs, mapping):
    y = epochs.events[:, 2]
    y = pd.Series(y).map(mapping)
    return y.values.astype(int)


def split_data(X, y, test_size=0.2, validation_size=0.1, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_dataloaders(X_temporal_train, X_temporal_val, X_temporal_test,
                       X_topology_train, X_topology_val, X_topology_test,
                       y_train, y_val, y_test, batch_size=10):
    X_temporal_train_tensor = X_temporal_train.clone().detach()
    X_topology_train_tensor = X_topology_train.clone().detach()
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_temporal_val_tensor = X_temporal_val.clone().detach()
    X_topology_val_tensor = X_topology_val.clone().detach()
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    X_temporal_test_tensor = X_temporal_test.clone().detach()
    X_topology_test_tensor = X_topology_test.clone().detach()
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Define TensorDatasets and DataLoaders
    train_data = TensorDataset(X_temporal_train_tensor, X_topology_train_tensor, y_train_tensor)
    val_data = TensorDataset(X_temporal_val_tensor, X_topology_val_tensor, y_val_tensor)
    test_data = TensorDataset(X_temporal_test_tensor, X_topology_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def apply_fft_transform(X, n_fft=128):
    """
    Applies FFT to each sample in the dataset.

    Parameters:
    - X: Tensor of shape [samples, channels, time_steps].

    Returns:
    - Tensor of shape [samples, channels, freq_bins] in the frequency domain.
    """
    X_fft = [fft_transform(torch.tensor(sample), n_fft=n_fft) for sample in X]
    return torch.stack(X_fft)


def fft_transform(data, n_fft=128):
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


def generate_distance_topology(raw):
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    pos = np.array([ch['loc'][:3] for ch in raw.info['chs'] if ch['kind'] == 2])

    diff = pos[:, None, :] - pos[None, :, :]

    squared_diff = diff ** 2
    squared_distances = squared_diff.sum(axis=2)
    conn_matrix = np.sqrt(squared_distances)

    return torch.tensor(conn_matrix)


def generate_correlation_topology(raw):
    """
    Generates a correlation-based connectivity matrix based on EEG channel signals.
    """
    data = raw.get_data(picks='eeg')[:64]
    # Change to pearson
    correlation_matrix = np.corrcoef(data)
    return torch.tensor(correlation_matrix, dtype=torch.float32)


def combine_topologies(distance_topology, correlation_topology, alpha=0.5):
    combined_topology = alpha * distance_topology + (1 - alpha) * correlation_topology
    return combined_topology


def check_label_distribution(loader, label_names=None):
    if label_names is None:
        label_names = {0: "audio", 1: "pain"}
    label_counts = {label: 0 for label in label_names.keys()}

    for _, _, labels in loader:
        for label in labels:
            label_counts[label.item()] += 1

    print("Label distribution in loader:")
    for label, count in label_counts.items():
        print(f"{label_names[label]}: {count} samples")

    return label_counts


def load_data(config):
    bids_root = config['data']['path']
    subjects_id = config['data']['subjects']

    task_dict = {
        "audioactive": [1, "audio"],
        "audiopassive": [1, "audio"],
        "thermalactive": [2, "pain"],
        "thermalpassive": [2, "pain"],
    }

    # Load raw data
    raws = load_raw_data(bids_root, subjects_id, task_dict)

    # Preprocess raw data
    raw = preprocess_raw_data(raws)

    # Create epochs
    epochs = create_epochs(raw)

    # Normalize data
    X = normalize_epochs_data(epochs.get_data())

    distance_topology = generate_distance_topology(raw)
    correlation_topology = generate_correlation_topology(raw)

    # Combine topologies
    alpha = 0.5
    X_topology = combine_topologies(distance_topology, correlation_topology, alpha=alpha)

    X_temporal = apply_fft_transform(X).float()

    # Get labels
    mapping = {10001: 0, 10002: 1}
    y = get_labels(epochs, mapping)

    # Split data
    test_size = config['data']['train_test_split']['test_size']
    random_state = config['data']['train_test_split']['random_state']

    X_temporal_train, X_temporal_val, X_temporal_test, y_train, y_val, y_test = split_data(X_temporal, y, test_size=test_size,
                                                                                           random_state=random_state)

    X_topology_train = X_topology.repeat(len(y_train), 1, 1)
    X_topology_val = X_topology.repeat(len(y_val), 1, 1)
    X_topology_test = X_topology.repeat(len(y_test), 1, 1)

    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(X_temporal_train, X_temporal_val, X_temporal_test,
                                                               X_topology_train, X_topology_val, X_topology_test,
                                                               y_train, y_val, y_test)

    print("Training Data:")
    train_label_counts = check_label_distribution(train_loader)

    print("\nValidation Data:")
    val_label_counts = check_label_distribution(val_loader)

    print("\nTest Data:")
    test_label_counts = check_label_distribution(test_loader)

    return train_loader, val_loader, test_loader
