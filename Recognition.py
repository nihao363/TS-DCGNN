import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGFeatureEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(EEGFeatureEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        # Temporal feature extraction
        self.cwt = nn.Conv1d(input_dim, embedding_dim, kernel_size=3, padding=1)
        # Spatial feature extraction
        self.spatial_filter = nn.Conv2d(1, embedding_dim, kernel_size=(3, 3), padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Additional pooling to get fixed-size output
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, eeg_signal):
        # Temporal features [batch, emb_dim, time]
        temporal_features = self.cwt(eeg_signal)

        # Spatial features [batch, 1, chans, time] -> [batch, emb_dim, chans, time]
        eeg_signal_4d = eeg_signal.unsqueeze(1)
        spatial_features = self.spatial_filter(eeg_signal_4d)
        spatial_features = self.adaptive_pool(spatial_features)  # [batch, emb_dim, 1, 1]
        spatial_features = spatial_features.view(spatial_features.size(0), -1).unsqueeze(2)  # [batch, emb_dim, 1]

        # Concatenate and pool
        fused_features = torch.cat((temporal_features, spatial_features), dim=2)  # [batch, emb_dim, time+1]
        fused_features = self.temporal_pool(fused_features).squeeze(2)  # [batch, emb_dim]

        return fused_features


class EmotionDecouplingModule(nn.Module):
    def __init__(self, input_dim):
        super(EmotionDecouplingModule, self).__init__()
        # Adjusted MLP input dimensions
        self.mlp_z = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, 64)
        )
        self.mlp_r = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.PReLU(),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Linear(256, 128)
        )
        self.mlp_h = nn.Sequential(
            nn.Linear(64 + 128, 256),  # 64 (z) + 128 (r) = 192
            nn.PReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, input_features):
        # Input features already flattened by encoder
        z = self.mlp_z(input_features)
        r = self.mlp_r(input_features)
        h = self.mlp_h(torch.cat((z, r), dim=1))
        return z, r, h


class RelationMetricNetwork(nn.Module):
    def __init__(self, input_dim_z, input_dim_r):
        super(RelationMetricNetwork, self).__init__()
        # Adjust the dimensions of z and r to match before computing similarity
        self.proj_z = nn.Linear(input_dim_z, input_dim_r)  # Project z to match r's dimension

    def forward(self, features_i, features_j):
        # Project features_i (z) to match features_j (r)
        features_i = self.proj_z(features_i)
        return F.cosine_similarity(features_i, features_j, dim=-1)


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes, args):
        super(Classifier, self).__init__()
        self.encoder = EEGFeatureEncoder(input_dim, embedding_dim=512)

        # Dynamically determine decoder input dimension
        with torch.no_grad():
            dummy = torch.randn(1, input_dim, 100)
            dummy_out = self.encoder(dummy)
            decoder_input_dim = dummy_out.shape[1]

        self.decoder = EmotionDecouplingModule(decoder_input_dim)
        self.relation_net = RelationMetricNetwork(64, 128)  # Match z (64) and r (128) dimensions

        # Adjust TCN input channels to encoder output
        # Changed kernel size to 1
        self.tcn = nn.Conv1d(
            in_channels=decoder_input_dim,
            out_channels=hidden_size,
            kernel_size=1,  # Changed from 5 to 1
            dilation=2
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.PReLU(),
            nn.Linear(256, num_classes)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, eeg_signal):
        # Encode EEG signal
        encoded_features = self.encoder(eeg_signal)

        # Decouple into emotional subspaces
        z, r, h = self.decoder(encoded_features)

        # Relation metric network to capture dependencies between features
        similarity = self.relation_net(z, r)

        # Add temporal dimension for TCN
        tcn_input = encoded_features.unsqueeze(2)  # [batch, feat, 1]
        temporal_features = self.tcn(tcn_input).squeeze(2)

        # Pass through the classification head
        output = self.mlp(temporal_features)
        return output, similarity

    def compute_loss(self, output, labels):
        return self.loss_fn(output, labels)


# Example usage remains the same
args = {'batch_size': 32, 'modalities': 'eeg', 'dataset': 'emotion', 'drop_rate': 0.5, 'use_highway': True}
model = Classifier(input_dim=64, hidden_size=128, num_classes=6, args=args)
eeg_signal = torch.randn(32, 64, 100)  # Example EEG signal batch with 32 samples, 64 channels, and 100 timepoints

output, similarity = model(eeg_signal)
loss = model.compute_loss(output, torch.randint(0, 6, (32,)))  # Random labels for illustration


