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


def visualize_saliency_with_mne(attention_tensor, edge_indices, title="Saliency Connectivity", threshold=0.1):
    """
    Visualizes saliency using MNE's connectivity circle.
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

    # Plot the connectivity circle
    plot_connectivity_circle(
        con=connectivity_matrix,
        node_names=channel_names,
        title=title,
        vmin=0, vmax=1,
        show=True
    )


def visualize_saliency_3d_with_topology(attention_tensor, edge_indices, coords, info, threshold=0.1):
    # fig = create_3d_figure(size=(800, 800))

    sphere = mne.make_sphere_model(r0="auto", head_radius="auto", info=info)
    fig = mne.viz.plot_alignment(
        # Plot options
        show_axes=True,
        dig="fiducials",
        surfaces="head",
        trans=mne.Transform("head", "mri", trans=np.eye(4)),  # identity
        bem=sphere,
        info=info,
    )

    # Determine connectivity matrix
    if edge_indices is not None:
        # Flatten if needed
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
        azimuth=40,
        elevation=80,
        distance=0.6,
        focalpoint=(0.0, 0.0, 0.0),
    )

    fig.plotter.screenshot('test.png')


def aggregate_node_attention(edge_index, attention_weights, num_nodes):
    """
    Aggregates attention weights to node-level scores.

    Args:
        edge_index (Tensor): Edge index of shape [2, num_edges].
        attention_weights (Tensor): Attention weights of shape [num_edges, num_heads].
        num_nodes (int): Total number of nodes.

    Returns:
        Tensor: Node-level attention scores of shape [num_nodes].
    """
    node_attention = torch.zeros(num_nodes, device=attention_weights.device)
    for i in range(attention_weights.shape[0]):
        pass
        # node_attention.scatter_add()
