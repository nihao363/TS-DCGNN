import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from FeatureEncode import SeqContext
from GCN import GCNLayer  # Graph Convolutional Network Layer

from GAT import GATLayer  # Graph Attention Network
from functions import batch_graphify_label, batch_feat
from IB import IB
import numpy as np
import utils

log = utils.get_logger()

class D2GNN(nn.Module):
    def __init__(self, args):
        super(D2GNN, self).__init__()

        self.wp = args.wp
        self.wf = args.wf
        self.device = args.device

        # Prepare for graph creation, considering EEG as a graph with speakers as edge types.
        edge_type_to_idx = {'00': 0, '01': 1, '10': 2, '11': 3}
        self.edge_type_to_idx = edge_type_to_idx
        log.debug(self.edge_type_to_idx)

        u_dim = 1380  # Input dimension (adjust as needed for EEG features)
        self.hidden_dim = args.hidden_size
        self.num_view = args.n_views
        self.view_list = [100, 768, 512]  # Example dimensions (adjust as needed for EEG modalities)
        self.num_classes = args.n_classes
        self.output_dropout = 0.5

        self.beta = 10  # Regularization factor

        # Initialize RNN for feature extraction
        self.rnn = SeqContext(u_dim, self.hidden_dim, args)

        # Initialize encoder for each modality
        self.encoders = []
        for v in range(self.num_view):
            self.encoders.append(Encoder(self.view_list[v], self.hidden_dim).to(self.device))
        self.encoders = nn.ModuleList(self.encoders)

        # Decoder for both concatenated and unimodal features
        self.decoder_concat = Decoder(self.hidden_dim, self.hidden_dim * 2).to(self.device)
        self.decoders = []
        for v in range(self.num_view):
            self.decoders.append(Decoder(self.hidden_dim, self.hidden_dim * 2).to(self.device))
        self.decoders = nn.ModuleList(self.decoders)

        # Information Bottleneck (IB) loss for feature decoupling
        self.IB = IB(shape_x=self.hidden_dim, shape_z=self.hidden_dim, shape_y=self.num_classes,
                     per_class=self.num_classes, device=self.device, beta=self.beta)

        # MLP layers for initial projection
        self.judge_dim = self.hidden_dim
        self.proj1_init_concat = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.proj2_init_concat = nn.Linear(self.hidden_dim, self.judge_dim)
        self.out_layer_init_concat = nn.Linear(self.judge_dim, self.num_classes)

        # Additional MLP projections for different modalities (audio, text, video)
        self.proj1_init_a = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.proj2_init_a = nn.Linear(self.hidden_dim, self.judge_dim)
        self.out_layer_init_a = nn.Linear(self.judge_dim, self.num_classes)

        self.proj1_init_l = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.proj2_init_l = nn.Linear(self.hidden_dim, self.judge_dim)
        self.out_layer_init_l = nn.Linear(self.judge_dim, self.num_classes)

        self.proj1_init_v = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.proj2_init_v = nn.Linear(self.hidden_dim, self.judge_dim)
        self.out_layer_init_v = nn.Linear(self.judge_dim, self.num_classes)

        # Graph part (GAT and GCN)
        g_dim = self.hidden_dim
        h1_dim = self.hidden_dim
        h2_dim = self.hidden_dim

        # Initialize GAT and GCN models for graph-based feature enhancement
        self.gcn_j = GNN(g_dim, h1_dim, h2_dim, args)  # GCN for implicit emotional features
        self.gat_z = GAT(g_dim, h1_dim, h2_dim, args)  # GAT for explicit emotional features

        # Prediction layers
        self.proj1_out_pred = nn.Linear(h2_dim * 2, self.hidden_dim)
        self.proj2_out_pred = nn.Linear(self.hidden_dim, self.judge_dim)
        self.out_layer_pred = nn.Linear(self.judge_dim, self.num_classes)

        # Loss functions
        self.MSE = MSE()

        # NLL loss for classification with optional class weights
        if args.class_weight:
            self.loss_weights = torch.tensor([1 / 0.086747, 1 / 0.144406, 1 / 0.227883]).to(args.device)
            self.nll_loss = nn.NLLLoss(self.loss_weights)
            print("*******weighted loss*******")
        else:
            self.nll_loss = nn.NLLLoss()

    def get_rep(self, data):
        # Get features, edge index, and edge type for graph-based processing
        node_features = self.rnn(data["text_len_tensor"], data["input_tensor"])
        features, edge_index, edge_type, edge_index_lengths = batch_graphify_label(
            node_features,
            data["text_len_tensor"],
            data["speaker_tensor"],
            self.wp,
            self.wf,
            self.edge_type_to_idx,
            self.device,
        )
        return features, edge_index, edge_type, edge_index_lengths

    def forward(self, data):
        out_features, edge_index, edge_type, edge_index_lengths = self.get_rep(data)

        data_input = batch_feat(data["input_tensor"], data["text_len_tensor"], self.device)
        feat_a = data_input[:, :100]
        feat_l = data_input[:, 100:868]
        feat_v = data_input[:, 868:]
        data_unimodal = [feat_a, feat_l, feat_v]

        out_dims = []
        for v in range(self.num_view):
            out_dim = self.encoders[v](data_unimodal[v])
            out_dims.append(out_dim)

        # --------------------------- Feature Decoupling ---------------------------
        # IB for explicit emotional features
        z_features = self.IB.forward(x=out_features, y=data['label_tensor'])

        # Process features for each modality
        z_a_features = self.IB_a.forward(x=out_dims[0], y=data['label_tensor'])
        z_l_features = self.IB_l.forward(x=out_dims[1], y=data['label_tensor'])
        z_v_features = self.IB_v.forward(x=out_dims[2], y=data['label_tensor'])

        # --------------------------- Judge and Projection ---------------------------
        j_proj_con = self.proj2_init_concat(F.dropout(F.relu(self.proj1_init_concat(out_features)), p=self.output_dropout))
        j_proj_a = self.proj2_init_a(F.dropout(F.relu(self.proj1_init_a(out_dims[0])), p=self.output_dropout))
        j_proj_l = self.proj2_init_l(F.dropout(F.relu(self.proj1_init_l(out_dims[1])), p=self.output_dropout))
        j_proj_v = self.proj2_init_v(F.dropout(F.relu(self.proj1_init_v(out_dims[2])), p=self.output_dropout))

        # --------------------------- Distillation of Judge and IB ---------------------------
        # Distilling judge and IB features (GAT and GCN processing)
        j_dw_proj_concat = self.proj_j_fusion_concat(j_proj_con)
        j_dw_proj_a = self.proj_j_fusion_a(j_proj_a)
        j_dw_proj_l = self.proj_j_fusion_l(j_proj_l)
        j_dw_proj_v = self.proj_j_fusion_v(j_proj_v)

        w_j_a = self.W_weight_j_a(torch.cat([j_dw_proj_concat, j_dw_proj_a], dim=1))
        w_j_l = self.W_weight_j_l(torch.cat([j_dw_proj_concat, j_dw_proj_l], dim=1))
        w_j_v = self.W_weight_j_v(torch.cat([j_dw_proj_concat, j_dw_proj_v], dim=1))

        # Final joint features for explicit-implicit emotional resonance
        w_j = torch.cat([w_j_a, w_j_l, w_j_v], dim=1)
        w_j = torch.tanh(w_j)
        w_j = F.softmax(w_j, dim=1).transpose(0, 1)

        j_feats = j_proj_con + torch.mul((w_j[0]).unsqueeze(1), j_proj_a) + torch.mul((w_j[1]).unsqueeze(1), j_proj_l) + torch.mul((w_j[2]).unsqueeze(1), j_proj_v)

        # Final graph-based propagation
        node_z_feats = torch.cat([out_features, j_feats], dim=0)
        graph_z_feat_out = self.gcn_z(node_z_feats, edge_index, edge_type)
        graph_z_out = graph_z_feat_out[:out_features.shape[0], :]

        node_j_feats = torch.cat([out_features, j_feats], dim=0)
        graph_j_feat_out = self.gcn_j(node_j_feats, edge_index, edge_type)
        graph_j_out = graph_j_feat_out[:out_features.shape[0], :]

        final_feats = torch.cat([graph_z_out, graph_j_out], dim=1)
        logits = self.proj2_out_pred(F.dropout(F.relu(self.proj1_out_pred(final_feats)), p=self.output_dropout))
        logits_out = self.out_layer_pred(logits)
        log_prob_last_judge = F.log_softmax(logits_out, dim=1)

        out = torch.argmax(log_prob_last_judge, dim=-1)
        return out

    def get_loss(self, data):
        out_features, edge_index, edge_type, edge_index_lengths = self.get_rep(data)

        data_input = batch_feat(data["input_tensor"], data["text_len_tensor"], self.device)
        feat_a = data_input[:, :100]
        feat_l = data_input[:, 100:868]
        feat_v = data_input[:, 868:]
        data_unimodal = [feat_a, feat_l, feat_v]

        out_dims = []
        for v in range(self.num_view):
            out_dim = self.encoders[v](data_unimodal[v])
            out_dims.append(out_dim)

        z_features = self.IB.forward(x=out_features, y=data['label_tensor'])
        z_a_features = self.IB_a.forward(x=out_dims[0], y=data['label_tensor'])
        z_l_features = self.IB_l.forward(x=out_dims[1], y=data['label_tensor'])
        z_v_features = self.IB_v.forward(x=out_dims[2], y=data['label_tensor'])

        j_proj_con = self.proj2_init_concat(F.dropout(F.relu(self.proj1_init_concat(out_features)), p=self.output_dropout))
        loss_IB_concat = self.IB.get_IB_loss()

        # Adding losses for each modality
        loss = loss_IB_concat
        loss += self.nll_loss(log_prob_last_judge, data['label_tensor'])

        return loss
