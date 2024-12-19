from torchcam.methods import GradCAM
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn

from evaluation.saliency_map import visualize_saliency_with_mne, visualize_saliency_3d_with_topology, \
    visualize_saliency_topomap
from .connectivity_graph_generator import ConnectivityGraphGenerator
from .spatial_decoder import SpatialDecoder
from .temporal_cnn import TemporalCNN
from .temporal_encoder import TemporalEncoder
from .classifier import GGNClassifier
from contextlib import contextmanager


class GGN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, subject_id, coords, info, save_path):
        super(GGN, self).__init__()
        self.in_channels = in_channels
        self.connectivity_graph = ConnectivityGraphGenerator(129, hidden_channels, out_channels)
        self.temporal_encoder = TemporalEncoder(65, 128, 2)
        self.temporal_cnn = TemporalCNN(in_channels=128, hidden_channels=64)

        self.spatial_decoder = SpatialDecoder(hidden_channels, hidden_channels, out_channels, 4)

        self.subject_id = subject_id
        self.coords = coords
        self.info = info
        self.save_path = save_path

        self.classifier = GGNClassifier(65, out_channels)
        self.sampled_edge_indices = None

    def forward(self, x_temporal, x_topology, epoch):
        batch_size = x_topology.shape[0]
        num_nodes = x_topology.shape[1]
        batch = torch.arange(batch_size).repeat_interleave(num_nodes).to(x_topology.device)

        # Step 2: Obtain node features from Temporal Encoder
        temporal_features = self.temporal_encoder(x_temporal)

        temporal_features = temporal_features.unsqueeze(1).repeat(1, num_nodes, 1)

        # Step 1: Generate adjacency matrices and edge indices
        # Ajout de temporal features dans para learner
        sampled_edge_indices = self.connectivity_graph(x_topology, x_temporal)
        self.sampled_edge_indices = sampled_edge_indices

        spatial_features = self.spatial_decoder(sampled_edge_indices, temporal_features, batch)

        if not self.training and epoch != 'test':
            attentions = self.spatial_decoder.cached_attention
            cat_attentions = torch.cat(attentions, dim=1)
            mean_attentions = cat_attentions.mean(dim=1)
            visualize_saliency_with_mne(mean_attentions, sampled_edge_indices, subject_id=self.subject_id, epoch=epoch,
                                        save_path=self.save_path)
            visualize_saliency_topomap(mean_attentions, sampled_edge_indices, self.subject_id, epoch, save_path=self.save_path)

        spatial_features = spatial_features.mean(dim=-1)

        temporal_features = self.temporal_cnn(temporal_features)

        spatial_features = spatial_features.unsqueeze(-1)
        cat_features = torch.cat((temporal_features, spatial_features), 1)  # temporal_features[50,64]

        output = self.classifier(cat_features)

        return output

    def explain_temporal_cnn(self, test_loader, device):
        """
        Generate Grad-CAM and visualize the results for all epochs with improved readability.

        Parameters:
        - test_loader: Data loader to retrieve batches of input.
        - device: Device (CPU/GPU).

        Returns:
        - Grad-CAM visualized on the temporal data for all batches.
        """

        self.temporal_cnn.to(device)

        with self.temporal_cnn.evaluation_mode():
            with torch.set_grad_enabled(True):
                for batch_idx, (x_temporal_batch, _, targets) in enumerate(test_loader):
                    x_temporal = x_temporal_batch.to(device).requires_grad_()
                    targets = targets.to(device)

                    # Pass through temporal CNN
                    temporal_features = self.temporal_encoder(x_temporal).unsqueeze(1).repeat(1, 128, 1).permute(0, 2,
                                                                                                                 1)
                    cam_extractor = GradCAM(self.temporal_cnn, target_layer='conv2')
                    outputs = self.temporal_cnn(temporal_features)

                    predicted_class = torch.argmax(outputs[0]).item()
                    cams = cam_extractor(predicted_class, outputs)
                    cam = cams[0].squeeze().cpu().numpy()
                    cam = (cam - cam.min()) / (cam.max() - cam.min())

                    cam_aggregated = cam.mean(axis=0)
                    cam_aggregated = (cam_aggregated - cam_aggregated.min()) / (
                            cam_aggregated.max() - cam_aggregated.min())

                    input_image = x_temporal[0].detach().cpu().numpy()
                    time_steps = input_image.shape[1]
                    channel_names = test_loader.dataset.channel_names

                    # Plot Grad-CAM with better readability
                    fig, ax = plt.subplots(figsize=(12, 12))  # Larger figure
                    spacing = 25  # Increase spacing between channels

                    # Plot fewer raw signals with slightly thicker lines
                    for i in range(0, input_image.shape[0], 2):  # Plot every 2nd channel
                        ax.plot(range(time_steps), input_image[i] + i * spacing,
                                alpha=0.8, color='black', linewidth=1.5)  # Thicker lines

                    # Overlay Grad-CAM
                    extent = [0, time_steps, 0, input_image.shape[0] * spacing]
                    grad_cam_image = cam_aggregated[np.newaxis, :]
                    ax.imshow(grad_cam_image, aspect='auto', extent=extent, cmap='jet', alpha=0.4, origin='lower')

                    # Set Y-ticks with fewer channel names
                    if channel_names:
                        y_positions = np.arange(0, len(channel_names) * spacing, spacing * 2)
                        ax.set_yticks(y_positions)
                        ax.set_yticklabels(channel_names[::2], fontsize=10, rotation=45, ha="right")

                    # Add colorbar and labels
                    cbar = plt.colorbar(
                        ax.imshow(grad_cam_image, aspect='auto', extent=extent, cmap='jet', alpha=0.4, origin='lower'),
                        ax=ax)
                    cbar.set_label("Temporal attention level", fontsize=12)

                    ax.set_title(f"Grad-CAM Visualization for Batch {batch_idx + 1}", fontsize=14)
                    ax.set_xlabel("Time steps (Milliseconds)", fontsize=12)
                    ax.set_ylabel("Channels", fontsize=12)
                    plt.tight_layout()
                    plt.show()

    @contextmanager
    def evaluation_mode(self):
        original_mode = self.training
        self.eval()
        try:
            yield self
        finally:
            if original_mode:
                self.train()
