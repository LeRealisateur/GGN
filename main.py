# Manipulation de données
import pandas as pd
import numpy as np

# Traitement du signal
from scipy import signal
import mne

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch

# Visualisation
import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px

# Gestion de Notebooks
import ipywidgets as widgets

# Utilitaires
import joblib
import yaml
import pickle
import os
import sys
import importlib

# Importation code local
sys.path.append('preprocessing')
import preprocess
import dataset
import torchcam
import models.GGN.ggn_model as GGN
import models.GGN.train as train

importlib.reload(preprocess)
importlib.reload(GGN)
importlib.reload(train)
importlib.reload(dataset)

test_losses = []
accuracies = []
recalls = []
precisions = []
f1_scores = []
auc_rocs = []

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

running_model = config['project_config']['running_model']

subjects_id = config['data']['subjects']

bids_root = config['data']['path']

save_path = config['output']['results_save_path']

if not subjects_id:
    subjects_id = [
        d for d in os.listdir(bids_root)
        if os.path.isdir(os.path.join(bids_root, d)) and d.startswith("sub-")
    ]
    print(f"Aucun ID de sujet spécifié. Tous les sujets détectés : {subjects_id}")

for subject in subjects_id:
    if running_model == "GGN":
        raw_data = mne.io.read_raw_brainvision(
            os.path.join(bids_root, subject, "eeg", f"{subject}_task-{config['data']['tasks'][0]}_eeg.vhdr"),
            eog=["VEOG", "HEOG"],
            misc=["rating", "temp", "stim"],
            preload=True,
        )
        if 'Iz' in raw_data.ch_names:
            raw_data.drop_channels(["Iz"])

        montage = mne.channels.make_standard_montage('standard_1020')
        raw_data.set_montage(montage)
        coords = np.array([
            ch['loc'][:3] for ch in raw_data.info['chs']
            if ch['kind'] == mne.io.constants.FIFF.FIFFV_EEG_CH and np.any(ch['loc'][:3])
        ])
        info = raw_data.info

        train_loader, val_loader, test_loader = dataset.create_dataloader([subject], config)

        model = GGN.GGN(**config['models']['GGN']['parameters'], subject_id=subject, coords=coords, info=info,
                        save_path=save_path)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 1

        # Train and validate the model
        train.train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

        # Test the model
        avg_test_loss, accuracy, recall, precision, f1, auc_roc = train.test(model, test_loader, criterion, device)
        test_losses.append(avg_test_loss)
        accuracies.append(accuracy)
        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1)
        auc_rocs.append(auc_roc)

        model.explain_temporal_cnn(test_loader, device)


    elif running_model == "SVM":
        train_loader, val_loader, test_loader = dataset.create_dataloader([subject], config)

        # Convert data loaders to numpy arrays
        X_train, y_train = preprocess.dataloader_without_topology_to_numpy(train_loader)
        X_val, y_val = preprocess.dataloader_without_topology_to_numpy(val_loader)
        X_test, y_test = preprocess.dataloader_without_topology_to_numpy(test_loader)

        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
        print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

        # Reshape data
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        # Train the SVM
        svm_params = config['models']['SVM']['parameters']
        svm = SVC(**svm_params)
        svm.fit(X_train, y_train)

        # Evaluate on validation set
        y_val_pred = svm.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)

        # Test the model
        y_test_pred = svm.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        accuracies.append(test_acc)

# Calculate mean test accuracy across all subjects
mean_test_loss = np.mean(test_losses)
mean_accuracy = np.mean(accuracies)
mean_recall = np.mean(recalls)
mean_precision = np.mean(precisions)
mean_f1 = np.mean(f1_scores)
mean_auc_roc = np.mean(auc_rocs)

print(f"Mean Test Loss across all subjects: {mean_test_loss:.4f}")
print(f"Mean Test Accuracy across all subjects: {mean_accuracy:.2f}%")
print(f"Mean Recall (Sensitivity) across all subjects: {mean_recall:.2f}")
print(f"Mean Precision across all subjects: {mean_precision:.2f}")
print(f"Mean F1 Score across all subjects: {mean_f1:.2f}")
print(f"Mean AUC-ROC across all subjects: {mean_auc_roc:.2f}")
