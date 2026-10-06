from __future__ import annotations

"""Neural network modules for drug-disease association prediction."""

import torch
from torch import nn
import torch.nn.functional as F


GRAPH_ENCODER_GCN = "gcn"
GRAPH_ENCODER_GAT = "gat"
GRAPH_ENCODER_GATV2 = "gatv2"
GRAPH_ENCODERS = (GRAPH_ENCODER_GCN, GRAPH_ENCODER_GAT, GRAPH_ENCODER_GATV2)


def _edge_softmax(logits: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Normalize attention scores over incoming edges for each node."""

    heads = logits.size(1)
    scatter_index = dst.view(-1, 1).expand(-1, heads)
    max_per_dst = torch.full(
        (num_nodes, heads),
        torch.finfo(logits.dtype).min,
        dtype=logits.dtype,
        device=logits.device,
    )
    max_per_dst.scatter_reduce_(0, scatter_index, logits, reduce="amax", include_self=True)
    exp_logits = torch.exp(logits - max_per_dst[dst])
    denom = torch.zeros(num_nodes, heads, dtype=logits.dtype, device=logits.device)
    denom.scatter_add_(0, scatter_index, exp_logits)
    return exp_logits / denom[dst].clamp_min(torch.finfo(logits.dtype).eps)


class GATv2Layer(nn.Module):
    """Lightweight pure-PyTorch GATv2 layer."""

    def __init__(self, in_dim: int, out_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if out_dim % heads != 0:
            raise ValueError("out_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = out_dim // heads
        self.out_dim = out_dim
        self.lin_src = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_dst = nn.Linear(in_dim, out_dim, bias=False)
        self.att = nn.Parameter(torch.empty(heads, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if edge_index.numel() == 0:
            raise ValueError("edge_index must contain at least one edge")

        num_nodes = x.size(0)
        src, dst = edge_index[0].long(), edge_index[1].long()
        h_src = self.lin_src(x).view(num_nodes, self.heads, self.head_dim)
        h_dst = self.lin_dst(x).view(num_nodes, self.heads, self.head_dim)

        edge_features = self.leaky_relu(h_src[src] + h_dst[dst])
        logits = (edge_features * self.att.unsqueeze(0)).sum(dim=-1)
        alpha = _edge_softmax(logits, dst, num_nodes)
        alpha = self.dropout(alpha)

        messages = h_src[src] * alpha.unsqueeze(-1)
        out = torch.zeros(
            num_nodes,
            self.heads,
            self.head_dim,
            dtype=x.dtype,
            device=x.device,
        )
        scatter_index = dst.view(-1, 1, 1).expand(-1, self.heads, self.head_dim)
        out.scatter_add_(0, scatter_index, messages)
        return out.reshape(num_nodes, self.out_dim) + self.bias

    @staticmethod
    def _edge_softmax(logits: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Normalize attention scores over incoming edges for each node."""

        return _edge_softmax(logits, dst, num_nodes)


class GATLayer(nn.Module):
    """Original GAT layer with static additive source/destination attention."""

    def __init__(self, in_dim: int, out_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if out_dim % heads != 0:
            raise ValueError("out_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = out_dim // heads
        self.out_dim = out_dim
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if edge_index.numel() == 0:
            raise ValueError("edge_index must contain at least one edge")

        num_nodes = x.size(0)
        src, dst = edge_index[0].long(), edge_index[1].long()
        h = self.lin(x).view(num_nodes, self.heads, self.head_dim)
        logits = self.leaky_relu(
            (h[src] * self.att_src.unsqueeze(0)).sum(dim=-1)
            + (h[dst] * self.att_dst.unsqueeze(0)).sum(dim=-1)
        )
        alpha = self.dropout(_edge_softmax(logits, dst, num_nodes))
        messages = h[src] * alpha.unsqueeze(-1)
        out = torch.zeros(
            num_nodes,
            self.heads,
            self.head_dim,
            dtype=x.dtype,
            device=x.device,
        )
        scatter_index = dst.view(-1, 1, 1).expand(-1, self.heads, self.head_dim)
        out.scatter_add_(0, scatter_index, messages)
        return out.reshape(num_nodes, self.out_dim) + self.bias


class GCNLayer(nn.Module):
    """GCN layer with symmetric degree normalization and one self-loop per node."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if edge_index.numel() == 0:
            raise ValueError("edge_index must contain at least one edge")

        num_nodes = x.size(0)
        src, dst = edge_index[0].long(), edge_index[1].long()
        non_self = src != dst
        loops = torch.arange(num_nodes, device=edge_index.device)
        src = torch.cat([src[non_self], loops])
        dst = torch.cat([dst[non_self], loops])

        degree = torch.zeros(num_nodes, dtype=x.dtype, device=x.device)
        degree.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        inv_sqrt_degree = degree.clamp_min(1.0).pow(-0.5)
        edge_weight = inv_sqrt_degree[src] * inv_sqrt_degree[dst]

        h = self.lin(x)
        out = torch.zeros(num_nodes, h.size(1), dtype=x.dtype, device=x.device)
        out.index_add_(0, dst, h[src] * edge_weight.unsqueeze(-1))
        return out + self.bias


class PairRepresentation(nn.Module):
    """Build drug-disease pair features for fusion and prediction."""

    def forward(self, drug_z: torch.Tensor, disease_z: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                drug_z,
                disease_z,
                drug_z * disease_z,
                torch.abs(drug_z - disease_z),
            ],
            dim=-1,
        )


class SimilaritySpectrumEncoder(nn.Module):
    """从药物和疾病相似性矩阵行提取 pair-level 相似性谱特征。"""

    def __init__(
        self,
        drug_dim: int,
        disease_dim: int,
        output_dim: int,
        channels: int = 16,
    ) -> None:
        super().__init__()
        if min(drug_dim, disease_dim, output_dim, channels) <= 0:
            raise ValueError("SimilaritySpectrumEncoder dimensions must be positive")
        self.drug_encoder = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.disease_encoder = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Sequential(
            nn.Linear(channels * 2, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, drug_rows: torch.Tensor, disease_rows: torch.Tensor) -> torch.Tensor:
        if drug_rows.ndim != 2 or disease_rows.ndim != 2:
            raise ValueError("Similarity spectrum inputs must have shape [batch_size, feature_dim]")
        if drug_rows.size(0) != disease_rows.size(0):
            raise ValueError("Drug and disease spectrum batches must have the same size")
        drug_features = self.drug_encoder(drug_rows.unsqueeze(1)).squeeze(-1)
        disease_features = self.disease_encoder(disease_rows.unsqueeze(1)).squeeze(-1)
        return self.projection(torch.cat([drug_features, disease_features], dim=-1))


class AttentionFusion(nn.Module):
    """Learn view weights for each drug-disease pair."""

    def __init__(self, pair_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, view_pairs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack(view_pairs, dim=1)
        scores = self.scorer(stacked).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        return fused, weights


class MeanFusion(nn.Module):
    """Average view-specific pair representations without trainable parameters."""

    def forward(self, view_pairs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack(view_pairs, dim=1)
        weights = stacked.new_full(stacked.shape[:2], 1.0 / stacked.size(1))
        return stacked.mean(dim=1), weights


class ReliabilityGuidedGatedFusion(nn.Module):
    """Use pair-level reliability features to gate view-specific pair representations."""

    def __init__(self, pair_dim: int, hidden_dim: int, reliability_feature_dim: int) -> None:
        super().__init__()
        if reliability_feature_dim <= 0:
            raise ValueError("reliability_feature_dim must be positive for reliability_gate fusion")
        self.reliability_feature_dim = reliability_feature_dim
        self.scorer = nn.Sequential(
            nn.Linear(pair_dim + reliability_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        view_pairs: list[torch.Tensor],
        reliability_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if reliability_features is None:
            raise ValueError("reliability_features must be provided for reliability_gate fusion")
        stacked = torch.stack(view_pairs, dim=1)
        if reliability_features.ndim != 2:
            raise ValueError("reliability_features must have shape [batch_size, feature_dim]")
        if reliability_features.size(0) != stacked.size(0):
            raise ValueError("reliability_features batch size must match view pair batch size")
        if reliability_features.size(1) != self.reliability_feature_dim:
            raise ValueError("reliability_features feature dimension does not match model configuration")

        reliability_features = reliability_features.to(dtype=stacked.dtype, device=stacked.device)
        repeated_features = reliability_features.unsqueeze(1).expand(-1, stacked.size(1), -1)
        scores = self.scorer(torch.cat([stacked, repeated_features], dim=-1)).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        return fused, weights


class TransformerFusion(nn.Module):
    """Use lightweight self-attention to model interactions among view-specific pair tokens."""

    def __init__(
        self,
        pair_dim: int,
        hidden_dim: int,
        dropout: float,
        num_views: int,
        fusion_dim: int = 256,
        heads: int = 4,
        layers: int = 1,
    ) -> None:
        super().__init__()
        if fusion_dim % heads != 0:
            raise ValueError("fusion_dim must be divisible by heads")
        self.input_projection = nn.Linear(pair_dim, fusion_dim)
        self.view_embedding = nn.Parameter(torch.zeros(1, num_views, fusion_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=heads,
            dim_feedforward=max(hidden_dim, fusion_dim * 2),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.scorer = nn.Linear(fusion_dim, 1)
        self.output_projection = nn.Linear(fusion_dim, pair_dim)

    def forward(self, view_pairs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack(view_pairs, dim=1)
        tokens = self.input_projection(stacked) + self.view_embedding[:, : stacked.size(1), :]
        encoded = self.encoder(tokens)
        scores = self.scorer(encoded).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        fused_token = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        return self.output_projection(fused_token), weights


class LinkPredictor(nn.Module):
    """MLP decoder that maps fused pair features to association scores."""

    def __init__(self, pair_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, pair_z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(pair_z).squeeze(-1))


class GATv2ViewEncoder(nn.Module):
    """Two-layer residual GATv2 encoder for one graph view."""

    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        dropout: float,
        heads: int = 4,
        feature_dim: int = 0,
    ) -> None:
        super().__init__()
        if embedding_dim % heads != 0:
            raise ValueError("embedding_dim must be divisible by heads")
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.feature_projection = (
            nn.Linear(feature_dim, embedding_dim, bias=False) if feature_dim > 0 else None
        )
        self.gat1 = GATv2Layer(embedding_dim, embedding_dim, heads=heads, dropout=dropout)
        self.gat2 = GATv2Layer(embedding_dim, embedding_dim, heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def encode_all(
        self,
        edge_index: torch.Tensor,
        node_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode all nodes in the current graph view."""

        x = self.embedding.weight
        if self.feature_projection is not None:
            if node_features is None:
                raise ValueError("node_features must be provided when feature_dim > 0")
            x = x + self.feature_projection(node_features.to(dtype=x.dtype, device=x.device))
        h = F.elu(self.gat1(x, edge_index))
        x = self.norm1(x + self.dropout(h))
        h = F.elu(self.gat2(x, edge_index))
        x = self.norm2(x + self.dropout(h))
        return x

    def forward(
        self,
        node_ids: torch.Tensor,
        edge_index: torch.Tensor,
        node_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encode_all(edge_index, node_features=node_features)[node_ids]


class GATViewEncoder(nn.Module):
    """Two-layer residual original-GAT encoder for one graph view."""

    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        dropout: float,
        heads: int = 4,
        feature_dim: int = 0,
    ) -> None:
        super().__init__()
        if embedding_dim % heads != 0:
            raise ValueError("embedding_dim must be divisible by heads")
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.feature_projection = (
            nn.Linear(feature_dim, embedding_dim, bias=False) if feature_dim > 0 else None
        )
        self.gat1 = GATLayer(embedding_dim, embedding_dim, heads=heads, dropout=dropout)
        self.gat2 = GATLayer(embedding_dim, embedding_dim, heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def encode_all(
        self,
        edge_index: torch.Tensor,
        node_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embedding.weight
        if self.feature_projection is not None:
            if node_features is None:
                raise ValueError("node_features must be provided when feature_dim > 0")
            x = x + self.feature_projection(node_features.to(dtype=x.dtype, device=x.device))
        h = F.elu(self.gat1(x, edge_index))
        x = self.norm1(x + self.dropout(h))
        h = F.elu(self.gat2(x, edge_index))
        return self.norm2(x + self.dropout(h))

    def forward(
        self,
        node_ids: torch.Tensor,
        edge_index: torch.Tensor,
        node_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encode_all(edge_index, node_features=node_features)[node_ids]


class GCNViewEncoder(nn.Module):
    """Two-layer residual GCN encoder for one graph view."""

    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        dropout: float,
        heads: int = 4,
        feature_dim: int = 0,
    ) -> None:
        super().__init__()
        del heads
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.feature_projection = (
            nn.Linear(feature_dim, embedding_dim, bias=False) if feature_dim > 0 else None
        )
        self.gcn1 = GCNLayer(embedding_dim, embedding_dim)
        self.gcn2 = GCNLayer(embedding_dim, embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def encode_all(
        self,
        edge_index: torch.Tensor,
        node_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embedding.weight
        if self.feature_projection is not None:
            if node_features is None:
                raise ValueError("node_features must be provided when feature_dim > 0")
            x = x + self.feature_projection(node_features.to(dtype=x.dtype, device=x.device))
        h = F.elu(self.gcn1(x, edge_index))
        x = self.norm1(x + self.dropout(h))
        h = F.elu(self.gcn2(x, edge_index))
        return self.norm2(x + self.dropout(h))

    def forward(
        self,
        node_ids: torch.Tensor,
        edge_index: torch.Tensor,
        node_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encode_all(edge_index, node_features=node_features)[node_ids]


class NormalizedGraphPropagation(nn.Module):
    """Parameter-free symmetric graph propagation for one semantic graph."""

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("x must have shape [num_nodes, feature_dim]")
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if edge_index.numel() == 0:
            return x

        src, dst = edge_index
        if int(edge_index.min()) < 0 or int(edge_index.max()) >= x.size(0):
            raise ValueError("edge_index contains a node id outside the node feature matrix")
        degree = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        degree.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        inv_sqrt_degree = degree.clamp_min(1.0).pow(-0.5)
        edge_weight = inv_sqrt_degree[src] * inv_sqrt_degree[dst]

        propagated = torch.zeros_like(x)
        propagated.index_add_(0, dst, x[src] * edge_weight.unsqueeze(-1))
        return propagated


class MultiScaleSemanticGraphEncoder(nn.Module):
    """Shared local GATv2 plus recurrent multi-hop graph signal extraction."""

    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        dropout: float,
        spectral_hops: int,
        gat_heads: int = 4,
        feature_dim: int = 0,
    ) -> None:
        super().__init__()
        if spectral_hops < 1:
            raise ValueError("spectral_hops must be at least 1")
        if embedding_dim % gat_heads != 0:
            raise ValueError("embedding_dim must be divisible by gat_heads")
        self.spectral_hops = int(spectral_hops)
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.feature_projection = (
            nn.Linear(feature_dim, embedding_dim, bias=False) if feature_dim > 0 else None
        )
        self.local_gat = GATv2Layer(
            embedding_dim,
            embedding_dim,
            heads=gat_heads,
            dropout=dropout,
        )
        self.local_norm = nn.LayerNorm(embedding_dim)
        self.hop_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.hop_norm = nn.LayerNorm(embedding_dim)
        self.propagation = NormalizedGraphPropagation()
        self.dropout = nn.Dropout(dropout)

    def initial_signal(self, node_features: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding.weight
        if self.feature_projection is not None:
            if node_features is None:
                raise ValueError("node_features must be provided when feature_dim > 0")
            if node_features.ndim != 2 or node_features.size(0) != x.size(0):
                raise ValueError("node_features must align with the shared node embedding table")
            x = x + self.feature_projection(node_features.to(dtype=x.dtype, device=x.device))
        return x

    def encode_view(
        self,
        edge_index: torch.Tensor,
        node_features: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        base = self.initial_signal(node_features)
        local = F.gelu(self.local_gat(base, edge_index))
        current = self.local_norm(base + self.dropout(local))
        bands = [current]
        for _ in range(1, self.spectral_hops):
            propagated = self.propagation(current, edge_index)
            update = F.gelu(self.hop_projection(propagated))
            current = self.hop_norm(current + self.dropout(update))
            bands.append(current)
        if not all(bool(torch.isfinite(band).all()) for band in bands):
            raise FloatingPointError("multi-scale semantic graph encoder produced NaN or Inf")
        return bands

    def forward(
        self,
        edge_indices: list[torch.Tensor] | tuple[torch.Tensor, ...],
        node_features: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        return [
            band
            for edge_index in edge_indices
            for band in self.encode_view(edge_index, node_features=node_features)
        ]


class ReliabilityGuidedSemanticBandGate(nn.Module):
    """Assign pair-specific weights to semantic view-hop graph signal bands."""

    def __init__(
        self,
        pair_dim: int,
        hidden_dim: int,
        reliability_feature_dim: int,
    ) -> None:
        super().__init__()
        if reliability_feature_dim <= 0:
            raise ValueError("reliability_feature_dim must be positive")
        self.reliability_feature_dim = int(reliability_feature_dim)
        self.score = nn.Sequential(
            nn.Linear(pair_dim + reliability_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        band_pairs: list[torch.Tensor],
        reliability_features: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not band_pairs:
            raise ValueError("band_pairs must contain at least one semantic band")
        stacked = torch.stack(band_pairs, dim=1)
        if reliability_features is None:
            raise ValueError("reliability_features must be provided for semantic band gating")
        if reliability_features.ndim != 2:
            raise ValueError("reliability_features must have shape [batch_size, feature_dim]")
        if reliability_features.size(0) != stacked.size(0):
            raise ValueError("reliability_features batch size must match band pair batch size")
        if reliability_features.size(1) != self.reliability_feature_dim:
            raise ValueError("reliability_features dimension does not match model configuration")
        reliability_features = reliability_features.to(dtype=stacked.dtype, device=stacked.device)
        repeated = reliability_features.unsqueeze(1).expand(-1, stacked.size(1), -1)
        weights = torch.softmax(self.score(torch.cat([stacked, repeated], dim=-1)).squeeze(-1), dim=1)
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        return fused, weights


class SemanticPathSpectralDDAModel(nn.Module):
    """Reliability-guided multi-view and multi-hop graph signal model for DDA."""

    VIEW_NAMES = ("association", "similarity", "biology")

    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int = 512,
        hidden_dim: int = 1024,
        dropout: float = 0.2,
        num_views: int = 3,
        spectral_hops: int = 3,
        spectral_pair_dim: int = 256,
        gat_heads: int = 4,
        feature_dim: int = 0,
        reliability_feature_dim: int = 7,
    ) -> None:
        super().__init__()
        if num_views != len(self.VIEW_NAMES):
            raise ValueError("semantic_path_spectral currently requires exactly three graph views")
        if spectral_pair_dim <= 0:
            raise ValueError("spectral_pair_dim must be positive")
        self.num_views = int(num_views)
        self.spectral_hops = int(spectral_hops)
        self.band_names = [
            f"{view_name}_hop_{hop}"
            for view_name in self.VIEW_NAMES
            for hop in range(1, self.spectral_hops + 1)
        ]
        self.encoder = MultiScaleSemanticGraphEncoder(
            num_nodes=num_nodes,
            embedding_dim=embedding_dim,
            dropout=dropout,
            spectral_hops=spectral_hops,
            gat_heads=gat_heads,
            feature_dim=feature_dim,
        )
        self.pair_builder = PairRepresentation()
        self.pair_projection = nn.Sequential(
            nn.Linear(embedding_dim * 4, spectral_pair_dim),
            nn.LayerNorm(spectral_pair_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = ReliabilityGuidedSemanticBandGate(
            pair_dim=spectral_pair_dim,
            hidden_dim=max(16, spectral_pair_dim // 2),
            reliability_feature_dim=reliability_feature_dim,
        )
        self.predictor = nn.Sequential(
            nn.Linear(spectral_pair_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        drug_node_ids: torch.Tensor,
        disease_node_ids: torch.Tensor,
        edge_indices: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        node_features: torch.Tensor | None = None,
        reliability_features: torch.Tensor | None = None,
        similarity_spectrum: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_view_nodes: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]
    ):
        del similarity_spectrum
        encoded = self.encode_pair_features(
            drug_node_ids,
            disease_node_ids,
            edge_indices,
            node_features=node_features,
            reliability_features=reliability_features,
            return_view_nodes=return_view_nodes,
        )
        if return_view_nodes:
            fused_pair, band_weights, band_pairs, band_nodes = encoded
            predictions = torch.sigmoid(self.predictor(fused_pair).squeeze(-1))
            return predictions, band_weights, band_pairs, band_nodes
        fused_pair, band_weights, band_pairs = encoded
        predictions = torch.sigmoid(self.predictor(fused_pair).squeeze(-1))
        return predictions, band_weights, band_pairs

    def encode_pair_features(
        self,
        drug_node_ids: torch.Tensor,
        disease_node_ids: torch.Tensor,
        edge_indices: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        node_features: torch.Tensor | None = None,
        reliability_features: torch.Tensor | None = None,
        similarity_spectrum: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_view_nodes: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]
    ):
        del similarity_spectrum
        normalized_edges = self._normalize_edge_indices(edge_indices)
        band_nodes = self.encoder(normalized_edges, node_features=node_features)
        band_pairs = [
            self.pair_projection(
                self.pair_builder(
                    all_node_z[drug_node_ids],
                    all_node_z[disease_node_ids],
                )
            )
            for all_node_z in band_nodes
        ]
        fused_pair, band_weights = self.fusion(
            band_pairs,
            reliability_features=reliability_features,
        )
        if not bool(torch.isfinite(fused_pair).all()) or not bool(torch.isfinite(band_weights).all()):
            raise FloatingPointError("semantic path spectral model produced NaN or Inf")
        if return_view_nodes:
            return fused_pair, band_weights, band_pairs, band_nodes
        return fused_pair, band_weights, band_pairs

    def _normalize_edge_indices(
        self,
        edge_indices: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> list[torch.Tensor]:
        if torch.is_tensor(edge_indices):
            return [edge_indices for _ in range(self.num_views)]
        normalized = list(edge_indices)
        if len(normalized) != self.num_views:
            raise ValueError("edge_indices must contain one edge index per semantic view")
        return normalized


class MultiViewDrugDiseaseModel(nn.Module):
    """Multi-view graph-encoder link prediction model for DDA."""

    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int = 512,
        hidden_dim: int = 1024,
        dropout: float = 0.3,
        num_views: int = 3,
        gat_heads: int = 4,
        feature_dim: int = 0,
        fusion_type: str = "transformer",
        fusion_dim: int = 256,
        fusion_heads: int = 4,
        fusion_layers: int = 1,
        reliability_feature_dim: int = 0,
        use_ss_encoder: bool = False,
        ss_drug_dim: int = 0,
        ss_disease_dim: int = 0,
        ss_dim: int = 128,
        ss_channels: int = 16,
        graph_encoder: str = GRAPH_ENCODER_GATV2,
    ) -> None:
        super().__init__()
        if graph_encoder not in GRAPH_ENCODERS:
            raise ValueError(f"graph_encoder must be one of {GRAPH_ENCODERS}")
        encoder_class = {
            GRAPH_ENCODER_GCN: GCNViewEncoder,
            GRAPH_ENCODER_GAT: GATViewEncoder,
            GRAPH_ENCODER_GATV2: GATv2ViewEncoder,
        }[graph_encoder]
        self.graph_encoder = graph_encoder
        self.encoders = nn.ModuleList(
            [
                encoder_class(
                    num_nodes,
                    embedding_dim,
                    dropout,
                    heads=gat_heads,
                    feature_dim=feature_dim,
                )
                for _ in range(num_views)
            ]
        )
        self.pair_builder = PairRepresentation()
        pair_dim = embedding_dim * 4
        if fusion_type == "mean":
            self.fusion = MeanFusion()
        elif fusion_type == "attention":
            self.fusion = AttentionFusion(pair_dim, hidden_dim)
        elif fusion_type == "reliability_gate":
            self.fusion = ReliabilityGuidedGatedFusion(
                pair_dim,
                hidden_dim,
                reliability_feature_dim=reliability_feature_dim,
            )
        elif fusion_type == "transformer":
            self.fusion = TransformerFusion(
                pair_dim,
                hidden_dim,
                dropout,
                num_views=num_views,
                fusion_dim=fusion_dim,
                heads=fusion_heads,
                layers=fusion_layers,
            )
        else:
            raise ValueError("fusion_type must be 'mean', 'attention', 'transformer', or 'reliability_gate'")
        self.fusion_type = fusion_type
        self.use_ss_encoder = bool(use_ss_encoder)
        self.ss_encoder: SimilaritySpectrumEncoder | None = None
        self.ss_projection: nn.Linear | None = None
        if self.use_ss_encoder:
            self.ss_encoder = SimilaritySpectrumEncoder(
                drug_dim=ss_drug_dim,
                disease_dim=ss_disease_dim,
                output_dim=ss_dim,
                channels=ss_channels,
            )
            self.ss_projection = nn.Linear(ss_dim, pair_dim)
        self.predictor = LinkPredictor(pair_dim, hidden_dim, dropout)

    def forward(
        self,
        drug_node_ids: torch.Tensor,
        disease_node_ids: torch.Tensor,
        edge_indices: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        node_features: torch.Tensor | None = None,
        reliability_features: torch.Tensor | None = None,
        similarity_spectrum: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_view_nodes: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]
    ):
        """基于融合的 pair 表示计算 MLP 关联预测分数。"""

        if return_view_nodes:
            fused_pair, view_weights, view_pairs, view_nodes = self.encode_pair_features(
                drug_node_ids,
                disease_node_ids,
                edge_indices,
                node_features=node_features,
                reliability_features=reliability_features,
                similarity_spectrum=similarity_spectrum,
                return_view_nodes=True,
            )
            return self.predictor(fused_pair), view_weights, view_pairs, view_nodes

        fused_pair, view_weights, view_pairs = self.encode_pair_features(
            drug_node_ids,
            disease_node_ids,
            edge_indices,
            node_features=node_features,
            reliability_features=reliability_features,
            similarity_spectrum=similarity_spectrum,
        )
        return self.predictor(fused_pair), view_weights, view_pairs

    def encode_pair_features(
        self,
        drug_node_ids: torch.Tensor,
        disease_node_ids: torch.Tensor,
        edge_indices: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        node_features: torch.Tensor | None = None,
        reliability_features: torch.Tensor | None = None,
        similarity_spectrum: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_view_nodes: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]
    ):
        """导出供 MLP 或外部分类器使用的融合药物-疾病 pair 表示。"""

        edge_indices = self._normalize_edge_indices(edge_indices)
        view_pairs = []
        view_nodes = []
        for encoder, edge_index in zip(self.encoders, edge_indices):
            all_node_z = encoder.encode_all(edge_index, node_features=node_features)
            view_nodes.append(all_node_z)
            drug_z = all_node_z[drug_node_ids]
            disease_z = all_node_z[disease_node_ids]
            view_pairs.append(self.pair_builder(drug_z, disease_z))
        if self.fusion_type == "reliability_gate":
            fused_pair, view_weights = self.fusion(view_pairs, reliability_features=reliability_features)
        else:
            fused_pair, view_weights = self.fusion(view_pairs)
        if self.ss_encoder is not None:
            if similarity_spectrum is None:
                raise ValueError("similarity_spectrum must be provided when SS-Encoder is enabled")
            drug_rows, disease_rows = similarity_spectrum
            fused_pair = fused_pair + self.ss_projection(self.ss_encoder(drug_rows, disease_rows))
        if return_view_nodes:
            return fused_pair, view_weights, view_pairs, view_nodes
        return fused_pair, view_weights, view_pairs

    def _normalize_edge_indices(
        self,
        edge_indices: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> list[torch.Tensor]:
        """Accept either one shared edge index or one edge index per view."""

        if torch.is_tensor(edge_indices):
            return [edge_indices for _ in self.encoders]
        edge_indices = list(edge_indices)
        if len(edge_indices) != len(self.encoders):
            raise ValueError("edge_indices must contain one edge_index per view")
        return edge_indices


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """InfoNCE loss for aligned representation batches."""

    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)


def pair_contrastive_loss(view_pairs: list[torch.Tensor], temperature: float = 0.2) -> torch.Tensor:
    """Align drug-disease pair representations across views."""

    if len(view_pairs) < 2:
        return torch.tensor(0.0, device=view_pairs[0].device)
    losses = []
    for i in range(len(view_pairs)):
        for j in range(i + 1, len(view_pairs)):
            losses.append(info_nce_loss(view_pairs[i], view_pairs[j], temperature))
    return torch.stack(losses).mean()


def node_contrastive_loss(view_nodes: list[torch.Tensor], temperature: float = 0.2) -> torch.Tensor:
    """对同一节点在不同图视图中的表示进行跨视图对齐。"""

    return pair_contrastive_loss(view_nodes, temperature=temperature)


def weighted_bce_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary cross-entropy with optional sample weights."""

    loss = F.binary_cross_entropy(predictions, labels.float(), reduction="none")
    if sample_weights is not None:
        loss = loss * sample_weights
    return loss.mean()
