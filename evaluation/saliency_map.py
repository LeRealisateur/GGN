import os
from os import path

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.viz import create_3d_figure, set_3d_view
from mne_connectivity.viz import plot_connectivity_circle
from mne.viz.backends.renderer import _get_renderer
import pyvista as pv
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


def visualize_saliency_3d_with_topology(attention_tensor, edge_indices, coords, info, subject_id, epoch, save_path,
                                        threshold=0.1):
    fig = create_3d_figure(size=(800, 800))

    title = f"3D map for subject {subject_id} at epoch {epoch}.png"

    sphere = mne.make_sphere_model(r0="auto", head_radius="auto", info=info)
    mne.viz.plot_alignment(
        # Plot options
        show_axes=True,
        dig="fiducials",
        surfaces="head",
        trans=mne.Transform("head", "mri", trans=np.eye(4)),
        bem=sphere,
        info=info,
        fig = fig,
    )

    # Determine connectivity matrix
    if edge_indices is not None:
        attention_tensor = attention_tensor.flatten()
        num_channels = coords.shape[0]
        connectivity_matrix = np.zeros((num_channels, num_channels))
        for i in range(attention_tensor.shape[0]):
            src = edge_indices[0, i]
            tgt = edge_indices[1, i]
            val = attention_tensor[i]
            if val >= threshold:
                connectivity_matrix[src, tgt] = val
                connectivity_matrix[tgt, src] = val
    else:
        # attention_tensor should be a matrix of shape (n_channels, n_channels)
        connectivity_matrix = attention_tensor.copy()
        connectivity_matrix[connectivity_matrix < threshold] = 0
        num_channels = connectivity_matrix.shape[0]

    # Normalize for better visual contrast
    max_val = np.max(connectivity_matrix)
    if max_val > 0:
        connectivity_matrix /= max_val

    plotter = fig.plotter

    # Draw edges
    max_width = 5.0
    for i in range(num_channels):
        for j in range(i + 1, num_channels):
            val = connectivity_matrix[i, j]
            if val > 0:
                lw = 1 + val * (max_width - 1)
                x = [coords[i, 0], coords[j, 0]]
                y = [coords[i, 1], coords[j, 1]]
                z = [coords[i, 2], coords[j, 2]]

                # Create a line segment using PyVista
                line = pv.Line((x[0], y[0], z[0]), (x[1], y[1], z[1]))

                # Add the line to the plotter
                # Use opacity or color to represent 'val' as desired.
                plotter.add_mesh(
                    line,
                    color=(0, 0, 1),
                    opacity=val,  # use attention value as transparency
                    line_width=lw
                )

    # Adjust the 3D view
    set_3d_view(
        fig,
        azimuth=135, elevation=80,
        focalpoint=(0.0, 0.0, 0.0),
    )
    save_path = os.path.join(save_path, title)
    fig.plotter.screenshot(save_path)
    plotter.close()

