import os
import shutil

import mne
import numpy as np
import os.path as path
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import joblib


def save_epochs_by_subject_task(output_dir, subject_id, task, epochs):
    """
    Save each epoch for a specific subject and task into separate files.

    Parameters:
    - output_dir: Base directory where epochs will be saved.
    - subject_id: ID of the subject.
    - task: Task name.
    - epochs: MNE Epochs object containing data.

    Returns:
    - None
    """
    # Create directories for subject and task
    task_dir = os.path.join(output_dir, subject_id, task)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)

    # Iterate through epochs and save each one
    for i in range(len(epochs.events)):
        single_epoch = epochs[i:i + 1]
        epoch_file_path = os.path.join(task_dir, f"{i + 1}-epo.fif")
        single_epoch.save(epoch_file_path, overwrite=True)
        print(f"Saved epoch {i + 1} for {subject_id} task {task} to {epoch_file_path}")


def preprocess_raw_data_and_save_epochs(bids_root, subjects_id, task_dict, output_dir, epochs_length=4):
    for subject_id in subjects_id:
        for task in task_dict.keys():
            raw_data = mne.io.read_raw_brainvision(
                path.join('../.', bids_root, subject_id, "eeg", f"{subject_id}_task-{task}_eeg.vhdr"),
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

            raw_data = preprocess_raw_data(raw_data)

            # Create epochs
            epochs = create_epochs(raw_data)

            save_epochs_by_subject_task(output_dir, subject_id, task, epochs)


def preprocess_raw_data(raw):
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


def split_data(X, test_size=0.2, validation_size=0.1, random_state=42):
    X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
    X_train, X_val = train_test_split(X_train, test_size=validation_size, random_state=random_state)

    return X_train, X_val, X_test


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


def train_test_split_files(preprocessed_data_path, output_path, test_size=0.2, random_state=42):
    """
    Split files into train, validation, and test sets while maintaining task structure.

    Parameters:
    - preprocessed_data_path: Path to the preprocessed data directory.
    - output_path: Path to save the split data.
    - test_size: Proportion of data for testing.
    - val_size: Proportion of training data for validation.
    - random_state: Random seed for reproducibility.

    Returns:
    - None
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for subject_id in os.listdir(preprocessed_data_path):
        subject_dir = os.path.join(preprocessed_data_path, subject_id)
        if not os.path.isdir(subject_dir):
            continue

        print(f"Processing subject: {subject_id}")

        for task in os.listdir(subject_dir):
            task_dir = os.path.join(subject_dir, task)
            if not os.path.isdir(task_dir):
                continue

            print(f"Processing task: {task}")

            # Collect all epoch files for the task
            all_files = [
                os.path.join(task_dir, file)
                for file in sorted(os.listdir(task_dir))
                if file.endswith("-epo.fif")
            ]

            if not all_files:
                print(f"No epoch files found for {subject_id} task {task}")
                continue

            # Split files into train, validation, and test
            train_files, val_files, test_files = split_data(all_files, test_size=test_size, random_state=random_state)

            # Save files into train, val, and test folders
            for split, files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
                split_dir = os.path.join(output_path, split, subject_id, task)
                os.makedirs(split_dir, exist_ok=True)
                for file in files:
                    shutil.copy(file, split_dir)

            print(f"Saved splits for {subject_id} task {task} in {output_path}")


def preprocess_data(config):
    bids_root = config['data']['path']
    subjects_id = config['data']['subjects']
    preprocessed_data_path = '.././data/processed'

    task_dict = {
        "audioactive": [1, "audio"],
        "audiopassive": [1, "audio"],
        "thermalactive": [2, "pain"],
        "thermalpassive": [2, "pain"],
    }

    # Load raw data
    preprocess_raw_data_and_save_epochs(bids_root, subjects_id, task_dict, preprocessed_data_path)

    test_size = config['data']['train_test_split']['test_size']
    random_state = config['data']['train_test_split']['random_state']

    train_test_split_files(preprocessed_data_path, '.././data', test_size, random_state)


if __name__ == '__main__':
    with open(r"../config.yml", "r") as file:
        config = yaml.safe_load(file)

    preprocess_data(config)
