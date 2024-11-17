# Manipulation de données
import numpy as np

# Traitement du signal
import mne

# Machine Learning et Deep Learning
#from sklearn.model_selection import train_test_split, GridSearchCV
#from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch

# Visualisation
import matplotlib.pyplot as plt
#import seaborn as sns
#import plotly.express as px

# Gestion de Notebooks
#import papermill as pm
#import ipywidgets as widgets

# Utilitaires
import yaml
import pickle
import importlib
import models.GGN.ggn_model as GGN
import train
import preprocessing_eeg.preprocess as preprocess

if __name__ == '__main__':
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)

    train_loader, val_loader, test_loader = preprocess.load_data(config)

    ### 1. Choisir le modèle
    in_channels = 65
    hidden_channels = 128
    out_channels = 2


    model = GGN.GGN(in_channels, hidden_channels, out_channels)

    ### 3. Entraîner le modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    train.train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
    train.test(model, test_loader, criterion, device)
