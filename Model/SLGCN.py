import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional

# --- Model Helper Functions (from Training Code that worked) ---
def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def find_drop_size(num_nodes: int, num_edges: int, K: int = 1) -> float:
    """
    Calculates the drop size for DropGraphSpatial based on training code version.
    """
    B_sum = 0.0
    if num_nodes == 0 or num_edges < 0: # num_edges can be 0
        return 0.0
    avg_degree_term = (2 * num_edges / num_nodes)
    for i in range(1, K + 1):
        # This was the structure from the training code.
        # It relies on math.pow handling (base-1)^0 = 1.
        # If avg_degree_term - 1 is negative, math.pow might error for non-integer exponents,
        # but i-1 is always integer here.
        try:
            term = avg_degree_term * math.pow(avg_degree_term - 1, i - 1)
        except ValueError: # Happens if (avg_degree_term - 1) is negative and (i-1) is fractional, not an issue here.
                           # Or if avg_degree_term is 0 and i-1 is negative.
            if avg_degree_term == 0 and (i-1) < 0 : # e.g. 0^(-1))
                term = 0 # Define 0 * inf as 0 in this context if avg_degree is 0
            elif (avg_degree_term -1) < 0 and (i-1) % 2 != 0 : # Negative base to odd power
                 term = avg_degree_term * (-1 * math.pow(abs(avg_degree_term - 1), i-1))
            else: # Negative base to even power, or positive base
                 term = avg_degree_term * math.pow(abs(avg_degree_term - 1), i-1)


        B_sum += term
    return B_sum

# --- Graph Utility Functions (Consistent with successful setup) ---
def get_hop_distance(num_node: int, edge: List[Tuple[int, int]], max_hop: int = 1) -> np.ndarray:
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

def normalize_digraph(A: np.ndarray) -> np.ndarray:
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn) # A D_in^{-1}
    return AD

def edge2mat(link: List[Tuple[int, int]], num_node: int) -> np.ndarray:
    A = np.zeros((num_node, num_node))
    for i, j in link: # Edge from i to j
        A[j, i] = 1   # A[target, source] = 1
    return A

def get_spatial_graph(num_node: int, self_link: List[Tuple[int, int]],
                      inward: List[Tuple[int, int]], outward: List[Tuple[int, int]]) -> np.ndarray:
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

class SpatialGraph:
    def __init__(self, num_nodes: int, inward_edges: List[Tuple[int, int]], strategy: str ="spatial"):
        self.num_nodes = num_nodes
        self.strategy = strategy
        self.self_edges = [(i, i) for i in range(num_nodes)]
        self.inward_edges = inward_edges
        self.outward_edges = [(j, i) for (i, j) in self.inward_edges]
        self.A = self.get_adjacency_matrix()

    def get_adjacency_matrix(self) -> np.ndarray:
        if self.strategy == "spatial":
            return get_spatial_graph(
                self.num_nodes, self.self_edges, self.inward_edges, self.outward_edges
            )
        else:
            raise ValueError(f"Unsupported graph strategy: {self.strategy}") # From training code: raise ValueError()

# --- Model Definitions (Aligned with Training Code that yielded good results) ---

class DropGraphTemporal(nn.Module): # From Training Code
    def __init__(self, block_size: int = 7):
        super(DropGraphTemporal, self).__init__()
        self.block_size = block_size
        self.keep_prob = 1.0

    def forward(self, x: torch.Tensor, keep_prob: float) -> torch.Tensor:
        self.keep_prob = keep_prob
        if not self.training or self.keep_prob == 1:
            return x
        n, c, t, v = x.size()
        if t == 0: return x
        input_abs = torch.mean(torch.mean(torch.abs(x), dim=3), dim=1).detach()
        
        sum_abs_all = torch.sum(input_abs)
        if sum_abs_all.item() < 1e-6: # Avoid division by zero if all abs are zero
             input_abs_norm = torch.zeros_like(input_abs)
        else:
            input_abs_norm = (input_abs / sum_abs_all * input_abs.numel()) # Global normalization
        input_abs_norm = input_abs_norm.view(n, 1, t).to(x.device)


        gamma = (1.0 - self.keep_prob) / self.block_size
        input1 = x.permute(0, 1, 3, 2).contiguous().view(n, c * v, t)
        M_seed = torch.bernoulli(torch.clamp(input_abs_norm * gamma, max=1.0)).to(x.device)
        M = M_seed.repeat(1, c * v, 1)

        m_sum = F.max_pool1d(
            M, kernel_size=self.block_size, stride=1, padding=self.block_size // 2 # Corrected list for kernel_size
        )
        mask = (1 - m_sum).to(device=m_sum.device, dtype=m_sum.dtype)
        
        masked_input = input1 * mask
        sum_mask = mask.sum()
        if sum_mask.item() < 1e-6: # Avoid division by zero if mask is all zeros
            return x # Or handle appropriately, e.g. return zeros
        
        scaled_input = masked_input * (mask.numel() / sum_mask)
        
        return (
            scaled_input
            .view(n, c, v, t)
            .permute(0, 1, 3, 2)
            .contiguous()
        )

class DropGraphSpatial(nn.Module): # From Training Code
    def __init__(self, num_points: int, drop_size: float):
        super(DropGraphSpatial, self).__init__()
        self.drop_size = drop_size
        self.num_points = num_points
        self.keep_prob = 1.0

    def forward(self, x: torch.Tensor, keep_prob: float, A: torch.Tensor) -> torch.Tensor:
        self.keep_prob = keep_prob
        if not self.training or self.keep_prob == 1:
            return x
        n, c, t, v = x.size()
        if v == 0: return x
        if v != self.num_points:
            raise ValueError(f"Input V ({v}) doesn't match num_points ({self.num_points}) in DropGraphSpatial")

        input_abs = torch.mean(torch.mean(torch.abs(x), dim=2), dim=1).detach()
        
        sum_abs_all = torch.sum(input_abs)
        if sum_abs_all.item() < 1e-6:
            input_abs_norm = torch.zeros_like(input_abs)
        else:
            input_abs_norm = input_abs / sum_abs_all * input_abs.numel() # Global normalization
        input_abs_norm = input_abs_norm.to(x.device)

        gamma = (1.0 - self.keep_prob) / (1.0 + self.drop_size)
        
        M_seed = torch.bernoulli(torch.clamp(input_abs_norm * gamma, max=1.0)).to(
            device=x.device, dtype=x.dtype
        )
        
        A_tensor = A.to(device=x.device, dtype=x.dtype)
        M = torch.matmul(M_seed, A_tensor)
        M[M > 0.001] = 1.0
        M[M < 0.5] = 0.0
        mask = (1 - M).view(n, 1, 1, self.num_points)
        
        masked_x = x * mask
        sum_mask = mask.sum()
        if sum_mask.item() < 1e-6:
            return x
        scaled_x = masked_x * (mask.numel() / sum_mask)
        return scaled_x

class TCNUnit(nn.Module): # Aligned with Training Code that worked
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 9, stride: int = 1,
        use_drop: bool = True, drop_size: float = 1.92, num_points: int = 25, block_size: int = 41,
    ):
        super(TCNUnit, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(kernel_size, 1),
            padding=(pad, 0), stride=(stride, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        # ReLU is defined here as per training code, but applied by the wrapper module (DecoupledGCNUnit/DecoupledGCN_TCN_unit)
        self.relu = nn.ReLU() 
        conv_init(self.conv)
        bn_init(self.bn, 1)
        self.use_drop = use_drop
        if use_drop:
            self.dropS = DropGraphSpatial(num_points=num_points, drop_size=drop_size)
            self.dropT = DropGraphTemporal(block_size=block_size)

    def forward(self, x: torch.Tensor, keep_prob: Optional[float] = None, A: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Training code order: BN -> Conv. This was key for the working version.
        # However, the *training script fine_tune_slgcn* had self.bn(self.conv(x)).
        # The successful inference script used Conv -> BN.
        # Sticking to what was in the fine_tune_slgcn `TCNUnit`:
        x = self.bn(self.conv(x)) # BN then Conv
        
        if self.use_drop and self.training and keep_prob is not None and keep_prob < 1.0 and A is not None:
            x = self.dropT(self.dropS(x, keep_prob, A), keep_prob)
        return x

class DecoupledGCNUnit(nn.Module): # Aligned with Training Code that worked
    def __init__(self, in_channels: int, out_channels: int, A: np.ndarray, # A is (3,V,V)
                 groups: int, num_points: int, num_subset: int = 3):
        super(DecoupledGCNUnit, self).__init__()
        self.num_points = num_points
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.num_subset = num_subset

        self.decoupled_A = nn.Parameter(
            torch.tensor(
                np.reshape(A.astype(np.float32), [num_subset, 1, num_points, num_points]),
                dtype=torch.float32
            ).repeat(1, groups, 1, 1),
            requires_grad=True,
        )
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x
        
        self.bn0 = nn.BatchNorm2d(out_channels * num_subset)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU() # ReLU at the end of THIS unit

        self.linear_weight = nn.Parameter(
            torch.zeros(in_channels, out_channels * num_subset), requires_grad=True
        )
        self.linear_bias = nn.Parameter(
            torch.zeros(1, out_channels * num_subset, 1, 1), requires_grad=True
        )
        self.eye_list = nn.Parameter( # From training code (was param, not buffer)
            torch.stack([torch.eye(num_points, dtype=torch.float32) for _ in range(out_channels)]),
            requires_grad=False,
        )
        for m in self.modules(): # Initialize all submodules
            if isinstance(m, nn.Conv2d): conv_init(m)
            elif isinstance(m, nn.BatchNorm2d): bn_init(m, 1)
        if hasattr(self, 'down') and isinstance(self.down, nn.Sequential): # Re-init self.down specifically if it was Conv2d
             conv_init(self.down[0])
             bn_init(self.down[1],1)

        bn_init(self.bn, 1e-6) # Specific for the final BN
        nn.init.normal_(self.linear_weight, 0, math.sqrt(0.5 / (out_channels * num_subset)))
        nn.init.constant_(self.linear_bias, 1e-6)

    def norm(self, A_sub: torch.Tensor) -> torch.Tensor: # A_sub is (1, C_eff, V, V)
        # This is A D_out^-1, where D_out is from row sums, matching training code's effective norm
        b, c_eff, h, w = A_sub.size()
        A_flat = A_sub.view(c_eff, self.num_points, self.num_points) # (C_eff, V, V)
        
        # D_list from row sums (axis 1 of A_flat). This corresponds to D_out if A_uv is u->v.
        D_list_out = torch.sum(A_flat, 1).view(c_eff, 1, self.num_points) # (C_eff, 1, V)
        D_list_out_inv = (D_list_out + 0.001).pow(-1) # (C_eff, 1, V)
        
        current_eye_list = self.eye_list[:c_eff, :, :] # (C_eff, V, V)
        D_out_inv_diag = current_eye_list * D_list_out_inv # (C_eff, V, V) with D_list_out_inv on diagonal

        # A_normed = A_flat @ D_out_inv_diag (Right multiplication)
        A_normed = torch.bmm(A_flat, D_out_inv_diag)
        return A_normed.view(b, c_eff, h, w)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        learn_adj = self.decoupled_A.repeat(1, self.out_channels // self.groups, 1, 1)
        normed_adj_list = []
        for i in range(self.num_subset):
            normed_adj_list.append(self.norm(learn_adj[i:i+1, ...]))
        normed_adj = torch.cat(normed_adj_list, 0)

        x = torch.einsum("nctw,cd->ndtw", (x0, self.linear_weight)).contiguous()
        x = x + self.linear_bias
        x = self.bn0(x)

        n, kc, t, v_ = x.size()
        x = x.view(n, self.num_subset, kc // self.num_subset, t, v_)
        x = torch.einsum("nkctv,kcvw->nctw", (x, normed_adj))
        
        x = self.bn(x)
        x += self.down(x0)
        x = self.relu(x) # ReLU at the end of THIS unit, as per training code
        return x

class DecoupledGCN_TCN_unit(nn.Module): # Aligned with Training Code that worked
    def __init__(
        self, in_channels: int, out_channels: int, A: np.ndarray, groups: int, num_points: int,
        block_size: int, drop_size: float, stride: int = 1, residual: bool = True,
        use_attention: bool = True,
    ):
        super(DecoupledGCN_TCN_unit, self).__init__()
        num_joints = A.shape[-1]

        self.gcn1 = DecoupledGCNUnit(in_channels, out_channels, A, groups, num_points)
        self.tcn1 = TCNUnit(out_channels, out_channels, stride=stride, num_points=num_points,
                            drop_size=drop_size, use_drop=True, block_size=block_size)
        self.relu = nn.ReLU() # Final ReLU for the whole block

        A_summed_for_drop = np.sum(A.astype(np.float32).reshape([A.shape[0], num_points, num_points]), axis=0)
        self.A_matrix_for_drop = nn.Parameter( # From training code (Param not buffer)
            torch.tensor(A_summed_for_drop, dtype=torch.float32),
            requires_grad=False
        )

        if not residual: self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1): self.residual = lambda x: x
        else: self.residual = TCNUnit(in_channels, out_channels, kernel_size=1, stride=stride, use_drop=False, num_points=num_points, block_size=block_size)

        self.drop_spatial_skip = DropGraphSpatial(num_points=num_points, drop_size=drop_size)
        self.drop_temporal_skip = DropGraphTemporal(block_size=block_size)
        
        self.use_attention = use_attention
        if self.use_attention: # Attention layers from training code
            self.sigmoid = nn.Sigmoid()
            self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4); nn.init.constant_(self.conv_ta.weight, 0); nn.init.constant_(self.conv_ta.bias, 0)
            ker_jpt = num_joints - 1 if not num_joints % 2 else num_joints; pad = (ker_jpt - 1) // 2
            self.conv_sa = nn.Conv1d(out_channels, 1, ker_jpt, padding=pad); nn.init.xavier_normal_(self.conv_sa.weight); nn.init.constant_(self.conv_sa.bias, 0)
            rr = 2; self.fc1c = nn.Linear(out_channels, out_channels // rr); self.fc2c = nn.Linear(out_channels // rr, out_channels)
            nn.init.kaiming_normal_(self.fc1c.weight); nn.init.constant_(self.fc1c.bias, 0)
            nn.init.constant_(self.fc2c.weight, 0); nn.init.constant_(self.fc2c.bias, 0)

    def forward(self, x: torch.Tensor, keep_prob: float) -> torch.Tensor:
        # gcn1 output already has ReLU from DecoupledGCNUnit's own forward method
        y = self.gcn1(x) 

        if self.use_attention: # Attention application y*W+y, active if use_attention=True
            se_sa_in = y.mean(dim=-2)
            se_sa_out = self.sigmoid(self.conv_sa(se_sa_in))
            y = y * se_sa_out.unsqueeze(-2) + y

            se_ta_in = y.mean(dim=-1)
            se_ta_out = self.sigmoid(self.conv_ta(se_ta_in))
            y = y * se_ta_out.unsqueeze(-1) + y
            
            se_ca_in = y.mean(dim=-1).mean(dim=-1)
            se_ca_hidden = self.relu(self.fc1c(se_ca_in))
            se_ca_out = self.sigmoid(self.fc2c(se_ca_hidden))
            y = y * se_ca_out.unsqueeze(-1).unsqueeze(-1) + y
        
        # TCNUnit output does not have ReLU applied yet by itself
        y_tcn_features = self.tcn1(y, keep_prob, self.A_matrix_for_drop)
        
        x_skip_val = self.residual(x)

        apply_dropout_to_skip = False
        if isinstance(self.residual, TCNUnit) and self.residual.use_drop:
            pass 
        elif isinstance(x_skip_val, torch.Tensor):
            if x_skip_val.abs().sum() > 1e-9:
                apply_dropout_to_skip = True
        
        if apply_dropout_to_skip and self.training and keep_prob is not None and keep_prob < 1.0:
            x_skip_val = self.drop_spatial_skip(x_skip_val, keep_prob, self.A_matrix_for_drop)
            x_skip_val = self.drop_temporal_skip(x_skip_val, keep_prob)
            
        if isinstance(x_skip_val, int) and x_skip_val == 0:
            return self.relu(y_tcn_features)
        else:
            return self.relu(y_tcn_features + x_skip_val)

class DecoupledGCN(nn.Module): # Aligned with Training Code that worked
    def __init__(self, in_channels: int, graph_args: dict, groups: int = 8,
                 block_size: int = 41, n_out_features: int = 256):
        super(DecoupledGCN, self).__init__()
        if not isinstance(graph_args, dict):
            try: graph_args = OmegaConf.to_container(graph_args)
            except Exception as e: raise ValueError(f"graph_args must be dict or OmegaConf. Error: {e}")

        num_points = graph_args["num_nodes"]
        inward_edges = graph_args["inward_edges"]
        self.graph = SpatialGraph(num_points, inward_edges)
        A = self.graph.A

        self.data_bn = nn.BatchNorm1d(in_channels * num_points)
        num_edges_for_drop = len(self.graph.inward_edges) if self.graph.inward_edges else 0
        drop_size = find_drop_size(self.graph.num_nodes, num_edges_for_drop)

        common_args_unit = {'A': A, 'groups': groups, 'num_points': num_points,
                            'block_size': block_size, 'drop_size': drop_size}

        self.l1 = DecoupledGCN_TCN_unit(in_channels, 64, **common_args_unit, residual=False)
        self.l2 = DecoupledGCN_TCN_unit(64, 64, **common_args_unit)
        self.l3 = DecoupledGCN_TCN_unit(64, 64, **common_args_unit)
        self.l4 = DecoupledGCN_TCN_unit(64, 64, **common_args_unit)
        self.l5 = DecoupledGCN_TCN_unit(64, 128, **common_args_unit, stride=2)
        self.l6 = DecoupledGCN_TCN_unit(128, 128, **common_args_unit)
        self.l7 = DecoupledGCN_TCN_unit(128, 128, **common_args_unit)
        self.l8 = DecoupledGCN_TCN_unit(128, 256, **common_args_unit, stride=2)
        self.l9 = DecoupledGCN_TCN_unit(256, 256, **common_args_unit)
        self.n_out_features = n_out_features
        self.l10 = DecoupledGCN_TCN_unit(256, self.n_out_features, **common_args_unit)
        
        bn_init(self.data_bn, 1)
        self.classifier = None 

    def forward(self, x: torch.Tensor, keep_prob: float = 0.9) -> torch.Tensor:
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = (x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous())

        # Keep_prob scheduling from training code (fine_tune_slgcn)
        kp_l1_l6 = 1.0 
        kp_l7_l10 = keep_prob if self.training else 1.0
        
        # Apply layers with respective keep_probs
        x = self.l1(x, kp_l1_l6); x = self.l2(x, kp_l1_l6); x = self.l3(x, kp_l1_l6)
        x = self.l4(x, kp_l1_l6); x = self.l5(x, kp_l1_l6); x = self.l6(x, kp_l1_l6)
        x = self.l7(x, kp_l7_l10); x = self.l8(x, kp_l7_l10)
        x = self.l9(x, kp_l7_l10); x = self.l10(x, kp_l7_l10)
        
        # Global Average Pooling from training code (fine_tune_slgcn)
        c_new = x.size(1) # n_out_features
        x = x.reshape(N, c_new, -1) # (N, n_out_features, T_final * V)
        output_features = x.mean(2) # (N, n_out_features)
        return output_features