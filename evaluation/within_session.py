import pandas as pd
import numpy as np
import mne
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
import matplotlib.pyplot as plt
import os
import sys
import importlib
sys.path.append('preprocessing')
import preprocess
import dataset
import torchcam
import models.GGN.ggn_model as GGN
import models.GGN.train as train
import models.EEGNet
import models.EEGDeformer
importlib.reload(preprocess)
importlib.reload(GGN)
importlib.reload(train)
importlib.reload(dataset)



def train_model_on_subject(train_loader, val_loader, model, criterion, optimizer, num_epochs=20, verbose=True):
    """
    Entraîne le modèle pour un seul sujet.

    Arguments:
    - train_loader : DataLoader pour l'entraînement
    - val_loader : DataLoader pour la validation
    - model : Modèle PyTorch
    - criterion : Fonction de perte
    - optimizer : Optimiseur
    - num_epochs : Nombre d'époques d'entraînement
    - verbose : Si True, affiche les logs d'entraînement et de validation

    Retour :
    - model : Modèle entraîné
    - history : Dictionnaire contenant les historiques de perte et d'accuracy

    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}


    for epoch in range(1, num_epochs + 1):
        # Phase d'entraînement
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            
            #On passe sur le GPU 
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            inputs, labels = inputs.float(), labels.long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = correct_train / total_train
        train_loss = running_loss / len(train_loader)

        # Phase de validation
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                
                #On passe sur le GPU
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                inputs, labels = inputs.float(), labels.long()
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                running_val_loss += val_loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = correct_val / total_val
        val_loss = running_val_loss / len(val_loader)
        
        # Enregistrer les historiques
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_accuracy)

        # Affichage des logs
        if verbose:
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
            print(f"  Val Loss: {val_loss:.4f}   | Val Accuracy: {val_accuracy:.4f}")

    return model, history

def plot_and_save_history(history, subject_id, output_dir="Plots"):
    """
    Trace et sauvegarde les courbes de perte et d'accuracy.

    Arguments:
    - history : Dictionnaire contenant les historiques de perte et d'accuracy
    - subject_id : ID du sujet
    - output_dir : Dossier où sauvegarder les plots
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Tracer les courbes
    plt.figure(figsize=(12, 6))
    epochs = range(1, len(history['train_loss']) + 1)

    # Perte
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title(f'Loss for Subject {subject_id}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
    plt.title(f'Accuracy for Subject {subject_id}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Sauvegarder l'image
    plot_path = os.path.join(output_dir, f"subject_{subject_id}_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot sauvegardé pour le sujet {subject_id} : {plot_path}")
    

def evaluate_model(test_loader, model, device):
    # Evaluating the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            
            #On passe sur le GPU 
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            inputs, labels = inputs.float(), labels.long()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

def within_session_evaluation(config, model, criterion, optimizer):
    """
    Effectue l'évaluation within-session pour chaque sujet.
    ------------------------ V2---------------------------
    """
    subjects_id = config['data']['subjects']
    bids_root = config['data']['path']
    
    # Détecter automatiquement les sujets si aucun n'est spécifié
    if not subjects_id:
        subjects_id = [
            d for d in os.listdir(bids_root)
            if os.path.isdir(os.path.join(bids_root, d)) and d.startswith("sub-")
        ]
        print(f"Aucun ID de sujet spécifié. Tous les sujets détectés : {subjects_id}")

    accuracies = []

    for subject in subjects_id:
        print(f"Traitement du sujet : {subject}")

        # Charger les DataLoaders pour le sujet
        train_loader, val_loader, test_loader = create_dataloader([subject], config, use_topology=False)

        # Convertir les DataLoaders en tableaux NumPy
        X_train, y_train = dataloader_to_numpy(train_loader)
        X_val, y_val = dataloader_to_numpy(val_loader)
        X_test, y_test = dataloader_to_numpy(test_loader)

        # Vérifier les dimensions des données
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
        print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        # Entraîner le modèle
        model, history = train_model_on_subject(train_loader, val_loader, model, criterion, optimizer)

        # Évaluer le modèle sur l'ensemble de test
        accuracy = evaluate_model(test_loader, model)
        accuracies.append(accuracy)

        # Sauvegarder le modèle pour le sujet
        model_path = os.path.join(config['output']['model_save_path'], f"subject_{subject}_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Modèle sauvegardé pour le sujet {subject} : {model_path}")

        # Sauvegarder les graphiques de l'historique d'entraînement
        plot_and_save_history(history, subject)

    # Calculer et afficher l'accuracy moyenne sur tous les sujets
    mean_accuracy = np.mean(accuracies)
    print(f"Accuracy moyenne sur tous les sujets : {mean_accuracy:.2f}")
    return mean_accuracy