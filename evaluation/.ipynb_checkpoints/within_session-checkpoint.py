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

def train_model_on_subject(train_loader, val_loader, model, criterion, optimizer, device, num_epochs=20, verbose=True):
    """
    Trains the model for a single subject.

    Arguments:
    - train_loader : DataLoader for training
    - val_loader : DataLoader for validation
    - model : PyTorch model
    - criterion : Loss function
    - optimizer : Optimizer
    - device : Device to run the model ('cuda' or 'cpu')
    - num_epochs : Number of training epochs
    - verbose : If True, print training and validation logs

    Returns:
    - model : Trained model
    - history : Dictionary containing loss and accuracy history
    """
    # Move model and criterion to the specified device
    model = model.to(device)
    criterion = criterion.to(device)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, _, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

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

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, _, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                running_val_loss += val_loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = correct_val / total_val
        val_loss = running_val_loss / len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_accuracy)

        if verbose:
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
            print(f"  Val Loss: {val_loss:.4f}   | Val Accuracy: {val_accuracy:.4f}")

    return model, history

def evaluate_model(test_loader, model, device):
    """
    Evaluates the model on a test set.

    Arguments:
    - test_loader : DataLoader for test data
    - model : Trained PyTorch model
    - device : Device to run the model ('cuda' or 'cpu')

    Returns:
    - accuracy : Accuracy of the model on the test data
    """
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, _, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

def within_session_evaluation(config, model, criterion, optimizer, device):
    """
    Performs within-session evaluation for each subject.

    Arguments:
    - config : Configuration dictionary
    - model : PyTorch model
    - criterion : Loss function
    - optimizer : Optimizer
    - device : Device to run the model ('cuda' or 'cpu')

    Returns:
    - mean_accuracy : Mean accuracy across all subjects
    """
    subjects_id = config['data']['subjects']
    bids_root = config['data']['path']
    
    # Automatically detect subjects if none are specified
    if not subjects_id:
        subjects_id = [
            d for d in os.listdir(bids_root)
            if os.path.isdir(os.path.join(bids_root, d)) and d.startswith("sub-")
        ]
        print(f"No subject ID specified. Detected subjects: {subjects_id}")

    accuracies = []

    for subject in subjects_id:
        print(f"Processing subject: {subject}")

        # Load DataLoaders for the subject
        train_loader, val_loader, test_loader = dataset.create_dataloader([subject], config)

        # Train the model
        model, history = train_model_on_subject(train_loader, val_loader, model, criterion, optimizer, device)

        # Evaluate the model on the test set
        accuracy = evaluate_model(test_loader, model, device)
        accuracies.append(accuracy)

        # Save the model for the subject
        model_path = os.path.join(config['output']['model_save_path'], f"subject_{subject}_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved for subject {subject}: {model_path}")

        # Save training history plots
        plot_and_save_history(history, subject)

    # Calculate mean accuracy across all subjects
    mean_accuracy = np.mean(accuracies)
    print(f"Mean accuracy across all subjects: {mean_accuracy:.2f}")
    return mean_accuracy

def plot_and_save_history(history, subject_id, output_dir="Plots"):
    """
    Plots and saves loss and accuracy curves.

    Arguments:
    - history : Dictionary containing loss and accuracy history
    - subject_id : ID of the subject
    - output_dir : Directory to save the plots
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(12, 6))
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
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

    # Save the plot
    plot_path = os.path.join(output_dir, f"subject_{subject_id}_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved for subject {subject_id}: {plot_path}")
