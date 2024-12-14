from torchcam.methods import GradCAM
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from .connectivity_graph_generator import ConnectivityGraphGenerator
from .spatial_decoder import SpatialDecoder
from .temporal_cnn import TemporalCNN
from .temporal_encoder import TemporalEncoder
from .classifier import GGNClassifier
from contextlib import contextmanager

class GGN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, device):
        super(GGN, self).__init__()
        self.in_channels = in_channels
        self.connectivity_graph = ConnectivityGraphGenerator(in_channels, hidden_channels, out_channels)
        self.temporal_encoder = TemporalEncoder(65, 128, 2)
        self.temporal_cnn = TemporalCNN(in_channels=128, hidden_channels=64)

        self.spatial_decoder = SpatialDecoder(hidden_channels, hidden_channels, out_channels, 4, device)

        self.classifier = GGNClassifier(hidden_channels, out_channels)

    def forward(self, x_temporal, x_topology):
        batch_size = x_temporal.size(0)
        num_nodes = self.in_channels

        # Step 1: Generate adjacency matrices and edge indices
        sampled_edge_indices = self.connectivity_graph(x_topology)

        # Step 2: Obtain node features from Temporal Encoder
        temporal_features = self.temporal_encoder(x_temporal)
        temporal_features_for_spatial = temporal_features.unsqueeze(1).repeat(1, num_nodes, 1)

        spatial_features = self.spatial_decoder(sampled_edge_indices, temporal_features_for_spatial)
        spatial_features = spatial_features.mean(dim=-1)

        temporal_features = temporal_features.unsqueeze(1).repeat(1, 128, 1).permute(0,2,1)#temporal_features.unsqueeze(2)
        print(temporal_features.shape)
        temporal_features = self.temporal_cnn(temporal_features)

        cat_features = torch.cat((temporal_features, spatial_features), 1)

        output = self.classifier(cat_features)

        return output
    
    def explain_temporal_cnn(self, test_loader, device):
        """
        Generate Grad-CAM and visualize the results to explain the predictions.

        Parameters:
        - test_loader: Data loader to retrieve a batch of input.
        - device: Device (CPU/GPU).

        Returns:
        - Grad-CAM visualized on the temporal data.
        """

        self.temporal_cnn.to(device)

        with self.temporal_cnn.evaluation_mode():  # Utilisation du context manager
            # Activer temporairement les gradients
            with torch.set_grad_enabled(True):
                # Charger un batch de données
                for x_temporal_batch, _, targets in test_loader:
                    x_temporal = x_temporal_batch.to(device).requires_grad_()  # Activer le suivi des gradients
                    targets = targets.to(device)
                    break  # Prendre un seul batch

                # Passer les données à travers l'encodeur
                num_nodes = self.in_channels
                
                temporal_features = self.temporal_encoder(x_temporal)
                temporal_features =  temporal_features.unsqueeze(1).repeat(1, 128, 1).permute(0,2,1)#temporal_features.unsqueeze(2)

                # Initialiser Grad-CAM
                cam_extractor = GradCAM(self.temporal_cnn, target_layer='conv2')

                # Passe avant
                outputs = self.temporal_cnn(temporal_features)

                # Extraire la classe prédite
                predicted_class = torch.argmax(outputs[0]).item()
                cams = cam_extractor(predicted_class, outputs)

                # Générer la carte Grad-CAM
                cam = cams[0].squeeze().cpu().numpy()
                cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalisation

                # Normalisation de cam_aggregated
                cam_aggregated = cam.mean(axis=0)
                cam_aggregated = (cam_aggregated - cam_aggregated.min()) / (cam_aggregated.max() - cam_aggregated.min())

                # Configuration de l'image
                input_image = x_temporal[0].detach().cpu().numpy()
                time_steps = input_image.shape[1]

                fig, ax = plt.subplots(figsize=(10, 5))

                # Tracer les données temporelles brutes
                for i in range(input_image.shape[0]):  # Pour chaque canal
                    ax.plot(range(time_steps), input_image[i] + i * 10, alpha=0.8, color='black', linewidth=0.8)

                # Ajustement de l'extension
                extent = [0, time_steps, 0, input_image.shape[0] * 10]  # Corrige ymin à 0 pour éviter inversion
                grad_cam_image = cam_aggregated[np.newaxis, :]  # Préparer pour affichage 2D
                ax.imshow(grad_cam_image, aspect='auto', extent=extent, cmap='jet', alpha=0.5, origin='lower')

                # Ajouter une barre de couleur
                cbar = plt.colorbar(ax.imshow(grad_cam_image, aspect='auto', extent=extent, cmap='jet', alpha=0.5, origin='lower'), ax=ax)
                cbar.set_label("Niveau d'attention temporelle")

                # Étiquettes et titre
                ax.set_title("Segment de données avec Grad-CAM")
                ax.set_xlabel("Pas de temps (Milliseconde)")
                ax.set_ylabel("Canaux (64 canaux)")
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

