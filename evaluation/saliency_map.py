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
        attention_tensor, edge_indices,
        subject_id=None, epoch=None, title="Saliency Connectivity",
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

    # Construct the dynamic title
    dynamic_title = title
    if subject_id is not None:
        dynamic_title += f" subject {subject_id}"
    if epoch is not None:
        dynamic_title += f" epoch {epoch + 1}"

    fig = plt.figure(figsize=(8, 8), facecolor='black')
    ax = fig.add_subplot(111, polar=True)  # Explicitly create polar axes

    # Plot the connectivity circle
    plot_connectivity_circle(
        con=connectivity_matrix,
        node_names=channel_names,
        n_lines=300,
        ax=ax,
        title=dynamic_title,
        show=False
    )

    if save_path is not None:
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, f'{dynamic_title}.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='png', dpi=300)
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
    title = f"Saliency_Topomap_for_subject_{subject_id}_at_epoch_{epoch}.png"

    channel_attention = aggregate_connection_to_channel_attention(attention_tensor, edge_indices, 64)

    attention_thresh = np.copy(channel_attention)
    attention_thresh[attention_thresh < threshold] = 0

    attention_norm = attention_thresh / np.max(attention_thresh)

    info = mne.create_info(ch_names=channel_names, sfreq=1000.,
                           ch_types='eeg')  # Adjust sfreq and ch_types as needed

    # Set montage for sensor locations
    montage = mne.channels.make_standard_montage(montage_name)
    info.set_montage(montage)

    evoked = mne.EvokedArray(attention_norm[:, np.newaxis], info)

    fig, ax = plt.subplots(figsize=(6, 6))
    im, cm = mne.viz.plot_topomap(
        evoked.data[:, 0],
        evoked.info,
        axes=ax,
        show=False,
        cmap='inferno',
        vlim=(0, 1),
        sensors=False,
        contours=0
    )
    ax.set_title(title, color='black', fontsize=14)

    pos = mne.channels.find_layout(info).pos[:, :2]
    pos[:, 1] -= 0.02
    for i, (x, y) in enumerate(pos):
        ch_name = info['ch_names'][i]  # Channel name
        ax.text(x, y, ch_name, fontsize=6, ha='center', va='center', color='black')

    mne.viz.plot_sensors(info, axes=ax, kind='topomap', show_names=True, pointsize=0.8)

    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('Normalized Attention Weight', color='black')

    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    if save_path is not None:
        if os.path.isdir(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_path = os.path.join(save_path, title)
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

