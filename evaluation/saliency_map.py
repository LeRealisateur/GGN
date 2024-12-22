import os
from os import path

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.viz import create_3d_figure, set_3d_view
from mne_connectivity.viz import plot_connectivity_circle

import torch

channel_names = [
    'Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1',
    'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz',
    'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz',
    'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3',
    'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8',
    'F6', 'AF8', 'AF4', 'F2', 'FCz'
]


def visualize_saliency_with_mne(
        attention_tensor, edge_indices,epoch=None,
        threshold=0.1, save_path=None
):
    """
    Visualizes saliency using MNE's connectivity circle and optionally saves the plot to a specified folder.

    Parameters:
    - attention_tensor: Attention weights.
    - edge_indices: Edge indices for connectivity.
    - channel_names: List of channel names.
    - subject_id: Identifier for the subject (optional).
    - epoch: Epoch number (optional).
    - title: Title of the plot.
    - threshold: Threshold for attention weights to include in the plot.
    - save_path: File path to save the plot (optional).
    """
    num_nodes = len(channel_names)

    title = f"Connection_map_at_Epoch_{epoch}"

    # Initialize connectivity matrix
    connectivity_matrix = np.zeros((num_nodes, num_nodes))
    edge_indices_single = edge_indices[:, :2016]
    attention_tensor = attention_tensor.flatten()

    # Populate connectivity matrix with attention weights
    for i in range(attention_tensor.shape[0]):
        src = edge_indices_single[0, i]
        tgt = edge_indices_single[1, i]
        if attention_tensor[i] >= threshold:
            connectivity_matrix[src, tgt] = attention_tensor[i]
            connectivity_matrix[tgt, src] = attention_tensor[i]

    # Normalize the connectivity matrix
    if np.max(connectivity_matrix) > 0:
        connectivity_matrix /= np.max(connectivity_matrix)

    fig = plt.figure(figsize=(8, 8), facecolor='black')
    ax = fig.add_subplot(111, polar=True)  # Explicitly create polar axes

    # Plot the connectivity circle
    plot_connectivity_circle(
        con=connectivity_matrix,
        node_names=channel_names,
        n_lines=300,
        ax=ax,
        title=title,
        show=False
    )

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        image_path = os.path.join(save_path, f'{title}.png')
        plt.savefig(image_path, format='png', dpi=300)
        plt.close()


def aggregate_connection_to_channel_attention(attention_connection, edge_indices, num_channels):
    """
    Aggregates connection-wise attention scores to channel-wise scores by assigning each
    connection's score to both involved electrodes.

    Parameters:
    - attention_connection: 1D array or torch.Tensor with attention scores per connection (num_connections).
    - edge_indices: 2D array with shape (2, num_connections), representing connections between channels.
    - num_channels: Total number of channels.

    Returns:
    - channel_attention: 1D NumPy array with aggregated attention scores per channel.
    """
    attention_connection = attention_connection.cpu().detach().numpy()
    edge_indices = edge_indices.cpu().detach().numpy()
    edge_indices_single = edge_indices[:, :2016]
    channel_attention = np.zeros(num_channels)

    # Iterate over all connections and assign scores
    for i in range(attention_connection.shape[0]):
        src = edge_indices_single[0, i]
        tgt = edge_indices_single[1, i]
        score = attention_connection[i]
        channel_attention[src] += score
        channel_attention[tgt] += score

    return channel_attention


def visualize_saliency_topomap(
        attention_tensor,
        edge_indices,
        subject_id=None,
        epoch=None,
        threshold=0.4,
        save_path=None,
        montage_name='standard_1020'
):
    """
    Visualize saliency scores on a topographic map using MNE.

    Parameters:
    - attention_tensor: 1D array or torch.Tensor with attention scores.
    - edge_indices: 2D array representing connections between channels.
    - subject_id: Identifier for the subject (optional).
    - epoch: Epoch number (optional).
    - threshold: Threshold for attention scores (default: 0.4).
    - save_path: Path to save the topomap image (optional).
    - montage_name: Name of the EEG montage (default: 'standard_1020').
    """
    # Dynamic title
    title = f"Saliency_Topomap_at_Epoch_{epoch}"

    # Aggregate attention to channels
    channel_attention = aggregate_connection_to_channel_attention(attention_tensor, edge_indices, 64)

    # Apply threshold
    attention_thresh = np.copy(channel_attention)
    attention_thresh[attention_thresh < threshold] = 0

    # Normalize attention scores
    attention_norm = attention_thresh / np.max(attention_thresh) if np.max(attention_thresh) > 0 else attention_thresh

    # Create MNE info and montage
    info = mne.create_info(ch_names=channel_names, sfreq=1000., ch_types='eeg')
    montage = mne.channels.make_standard_montage(montage_name)
    info.set_montage(montage)

    # Create EvokedArray for topomap
    evoked = mne.EvokedArray(attention_norm[:, np.newaxis], info)

    # Determine vlim dynamically
    min_val = evoked.data[:, 0].min()
    max_val = evoked.data[:, 0].max()
    abs_max = max(abs(min_val), abs(max_val))

    # Create figure and plot topomap
    fig, ax = plt.subplots(figsize=(6, 6))
    im, _ = mne.viz.plot_topomap(
        evoked.data[:, 0],
        evoked.info,
        axes=ax,
        cmap='RdBu_r',
        vlim=(-abs_max, abs_max),
        sensors=True,
        names=channel_names,
        contours=8,
        extrapolate="head",
        show=False
    )

    # Add title
    fig.suptitle(title, color='black', fontsize=16, y=0.98)

    # Save or display the figure
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        image_path = os.path.join(save_path, f'{title}.png')
        im.figure.savefig(image_path, format='png', dpi=300, bbox_inches='tight')
    else:
        plt.show()

    # Close figure to free memory
    plt.close(fig)

