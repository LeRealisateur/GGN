import os
import numpy as np
import mne

import numpy as np

def add_noise_to_data(data, noise_level=0.001, min_std=0.00001):
    """
    Ajoute un bruit gaussien à un signal tout en limitant son impact sur les canaux à faible variance.
    
    Arguments :
    - data : ndarray, les données EEG (n_epochs, n_channels, n_times).
    - noise_level : float, intensité globale du bruit.
    - min_std : float, seuil minimal pour l'écart-type afin de stabiliser le bruit.

    Retourne :
    - data_noisy : ndarray, les données avec du bruit ajouté.
    """
    # Calculer l'écart-type par canal
    std_per_channel = np.std(data, axis=2, keepdims=True)
    
    # Appliquer un seuil minimum à l'écart-type
    std_per_channel = np.maximum(std_per_channel, min_std)
    
    # Générer le bruit gaussien en utilisant le nouvel écart-type
    noise = np.random.normal(0, noise_level * std_per_channel, size=data.shape)
    
    return data + noise


def process_and_save_noisy_data(base_dir, output_dir, subjects_id, noise_level=0.001, min_std=0.00001):
    """
    Parcourt les fichiers .fif pour les sujets spécifiés, ajoute du bruit et sauvegarde les fichiers bruités.
    Arguments :
    - base_dir : str, répertoire contenant les fichiers originaux.
    - output_dir : str, répertoire où sauvegarder les fichiers bruités.
    - subjects_id : list, liste des IDs des sujets à traiter.
    - noise_level : float, intensité du bruit.
    """
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Le chemin spécifié n'existe pas : {base_dir}")
    
    print(f"Chemin de base : {base_dir}")
    print(f"Subjects à traiter : {subjects_id}")

    for subject in subjects_id:
        subject_dir = os.path.join(base_dir, subject)
        if not os.path.exists(subject_dir):
            print(f"Répertoire manquant pour le sujet : {subject}")
            continue
        
        print(f"Traitement des fichiers pour le sujet : {subject}")
        for root, dirs, files in os.walk(subject_dir):
            for file in files:
                if file.endswith("-epo.fif"):
                    print(f"Traitement du fichier : {file}")
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(root, base_dir)
                    noisy_path = os.path.join(output_dir, relative_path)  # Conserve la structure relative
                    os.makedirs(noisy_path, exist_ok=True)

                    try:
                        # Charger les données
                        epochs = mne.read_epochs(file_path, preload=True)
                        data = epochs.get_data()
                        print(f"Shape des données : {data.shape}")
                        
                        # Ajouter du bruit
                        data_noisy = add_noise_to_data(data, noise_level, min_std)
                        epochs._data = data_noisy
                        
                        # Sauvegarder les données bruitées dans le dossier parent
                        noisy_file_path = os.path.join(noisy_path, file)
                        epochs.save(noisy_file_path, overwrite=True)
                        print(f"Fichier bruité sauvegardé : {noisy_file_path}")
                    except Exception as e:
                        print(f"Erreur lors du traitement du fichier {file}: {e}")
