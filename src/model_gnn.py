"""Message-passing GNN for the multitask Direct Inhibition regression track.

Hand-rolled (no Chemprop) to keep full control over the architecture, per the
decision to represent molecules as graphs (nodes = atoms, edges = bonds) instead of
Morgan fingerprints. GINEConv is used because it natively consumes edge features
(bond type/conjugation/ring membership), which a plain GCN would drop.

One shared encoder + trunk across all 4 isoforms (matches the tabular multitask
model's design, which beat single-task by ~18% MA-ST-RAE) with a single 4-way
output head and a masked loss so molecules missing some isoform labels still
contribute gradient for whatever labels they do have.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, TransformerConv, global_add_pool, global_mean_pool

N_TASKS = 4


class CypGNN(nn.Module):
    def __init__(
        self,
        atom_feature_dim: int,
        bond_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
        n_tasks: int = N_TASKS,
    ):
        super().__init__()
        self.lin0 = nn.Linear(atom_feature_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=bond_feature_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, n_tasks)

    def forward(self, x, edge_index, edge_attr, batch) -> torch.Tensor:
        h = self.lin0(x)
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, edge_attr)
            h_new = norm(h_new)
            h_new = torch.relu(h_new)
            h = h + self.dropout(h_new)  # residual

        pooled = torch.cat([global_mean_pool(h, batch), global_add_pool(h, batch)], dim=1)
        trunk_out = self.trunk(pooled)
        return self.head(trunk_out)


class CypSubgraphTransformerGNN(nn.Module):
    """Attention-based alternative to CypGNN, inspired by LumiNet's (Su et al.,
    Chem. Sci. 2025) Subgraph Transformer ligand encoder.

    Core idea carried over: replace GINEConv's plain sum-aggregation over a node's
    1-hop neighborhood with edge-feature-modulated MULTI-HEAD ATTENTION over that
    same neighborhood (PyG's TransformerConv implements essentially the same Q/K/V
    formulation as the paper's equations 2-10) -- a node's neighbors get weighted
    by learned relevance rather than summed uniformly.

    Scope note (not a full reproduction): LumiNet's SubGT also explicitly computes
    and concatenates THREE embeddings per interacting pair -- Emb(i|SubGraph_i),
    Emb(j|SubGraph_i), Emb(i|SubGraph_j) -- i.e. a node's representation depends on
    whether it's the center of its own subgraph or a guest in a neighbor's. That
    piece is designed for their protein-ligand interaction block specifically and
    was not reproduced here; this class only carries over the attention-based
    aggregation mechanism, applied to our existing topology-only atom/bond graph.
    """

    def __init__(
        self,
        atom_feature_dim: int,
        bond_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        self.lin0 = nn.Linear(atom_feature_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    edge_dim=bond_feature_dim,
                    beta=True,  # learnable gating on the skip connection
                    dropout=dropout,
                )
            )
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, N_TASKS)

    def forward(self, x, edge_index, edge_attr, batch) -> torch.Tensor:
        h = self.lin0(x)
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, edge_attr)
            h_new = norm(h_new)
            h_new = torch.relu(h_new)
            h = h + self.dropout(h_new)  # residual

        pooled = torch.cat([global_mean_pool(h, batch), global_add_pool(h, batch)], dim=1)
        trunk_out = self.trunk(pooled)
        return self.head(trunk_out)


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over only the (sample, task) entries where mask is True."""
    diff_sq = (pred - target) ** 2
    diff_sq = torch.where(mask, diff_sq, torch.zeros_like(diff_sq))
    return diff_sq.sum() / mask.sum().clamp(min=1)


class CypGNNWithUnionAux(nn.Module):
    """Same shared GINEConv encoder/trunk as CypGNN, but with TWO auxiliary output heads
    instead of one: the existing single-conc log2fc head (features_single_conc.py) plus a
    second head for external ChEMBL+PubChem pIC50 (features_external_pic50.py).

    The second head is what lets ~25,600 net-new molecules outside our own training set
    (no primary pIC50 label at all -- fully masked on that head) still shape the shared
    encoder via gradient, the same mechanism that made the log2fc head worth trying, but
    at roughly 10x the molecule count. Both aux heads are discarded at inference; only
    the primary head's predictions are scored.
    """

    def __init__(
        self,
        atom_feature_dim: int,
        bond_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
        n_aux_tasks: int = 4,
        n_aux2_tasks: int = 4,
    ):
        super().__init__()
        self.lin0 = nn.Linear(atom_feature_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=bond_feature_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, N_TASKS)
        self.aux_head = nn.Linear(hidden_dim, n_aux_tasks)
        self.aux2_head = nn.Linear(hidden_dim, n_aux2_tasks)

    def forward(self, x, edge_index, edge_attr, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.lin0(x)
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, edge_attr)
            h_new = norm(h_new)
            h_new = torch.relu(h_new)
            h = h + self.dropout(h_new)  # residual

        pooled = torch.cat([global_mean_pool(h, batch), global_add_pool(h, batch)], dim=1)
        trunk_out = self.trunk(pooled)
        return self.head(trunk_out), self.aux_head(trunk_out), self.aux2_head(trunk_out)


class CypGNNWithAuxiliary(nn.Module):
    """Same shared GINEConv encoder/trunk as CypGNN, but with a SECOND output
    head for the single-concentration-screen log2fc auxiliary target (4 tasks,
    one per isoform, z-scored -- see features_single_conc.py).

    Motivation: single-conc data covers ~2500-3000 MORE compounds per isoform
    than direct pIC50 dose-response labels do, and correlates strongly with
    true pIC50 (r=-0.70 to -0.89) -- but it doesn't exist for the blind test
    set, so it can't be a direct input feature. Training it as a second,
    lower-weighted masked-loss output head lets those extra compounds shape
    the SHARED encoder/trunk via gradient during training, even for molecules
    with no pIC50 label at all. Only the primary (pIC50) head's predictions
    are used at inference/OOF-scoring time -- the auxiliary head is discarded.
    """

    def __init__(
        self,
        atom_feature_dim: int,
        bond_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
        n_aux_tasks: int = 4,
    ):
        super().__init__()
        self.lin0 = nn.Linear(atom_feature_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=bond_feature_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, N_TASKS)
        self.aux_head = nn.Linear(hidden_dim, n_aux_tasks)

    def forward(self, x, edge_index, edge_attr, batch) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.lin0(x)
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, edge_attr)
            h_new = norm(h_new)
            h_new = torch.relu(h_new)
            h = h + self.dropout(h_new)  # residual

        pooled = torch.cat([global_mean_pool(h, batch), global_add_pool(h, batch)], dim=1)
        trunk_out = self.trunk(pooled)
        return self.head(trunk_out), self.aux_head(trunk_out)
