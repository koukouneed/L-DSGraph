import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math


class SBFLTokenCrossAttention(nn.Module):
    def __init__(self, in_dim_half, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert out_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.sbfl_proj = nn.Linear(in_dim_half, out_dim)
        self.token_proj = nn.Linear(in_dim_half, out_dim)
        self.q_proj = nn.Linear(out_dim, out_dim)
        self.k_proj = nn.Linear(out_dim, out_dim)
        self.v_proj = nn.Linear(out_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sbfl_feat, token_feat):
        N = sbfl_feat.shape[0]
        D = self.sbfl_proj.out_features
        H = self.num_heads
        d = self.head_dim

        s = F.relu(self.sbfl_proj(sbfl_feat))
        t = F.relu(self.token_proj(token_feat))

        Q = self.q_proj(s).view(N, H, d).transpose(0, 1)
        K = self.k_proj(t).view(N, H, d).transpose(0, 1)
        V = self.v_proj(t).view(N, H, d).transpose(0, 1)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(0, 1).contiguous().view(N, D)

        out = self.out_proj(out)
        out = self.norm(out + s)
        return out


class Propogator(nn.Module):
    def __init__(self, state_dim):
        super(Propogator, self).__init__()
        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.Tanh()
        )

    def forward(self, state, state_cur, A):
        a_t = torch.bmm(A, state)
        a = torch.cat((a_t, state_cur), 2)
        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_t, r * state_cur), 2)
        h_hat = self.tansform(joined_input)
        output = (1 - z) * state_cur + z * h_hat
        return output


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, A):
        Wh = torch.matmul(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        adj = A.to_dense()
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            h_prime = F.elu(h_prime)
        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        _, nodenum, _ = Wh.size()
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.transpose(2, 1)
        return self.leakyrelu(e)


class SpGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.out = nn.Linear(out_features, 1)

    def forward(self, x, ast_type_ids, edge_index, num_nodes):
        N = num_nodes
        dv = 'cuda' if x.is_cuda else 'cpu'

        adj = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.shape[1], device=dv),
            (N, N)
        )

        h = torch.mm(x, self.W)
        edge = edge_index

        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        e_rowsum = torch.sparse.mm(adj, torch.ones(size=(N, 1), device=dv))
        edge_e = self.dropout(edge_e)
        h_prime = torch.sparse.mm(adj, edge_e.unsqueeze(1) * h)
        h_prime = h_prime.div(e_rowsum)
        zero_vec = -9e15 * torch.ones_like(h_prime)
        h_prime = torch.where(torch.isnan(h_prime), zero_vec, h_prime)

        out = self.out(h_prime)
        return out, h_prime


class FaultLocGCN(nn.Module):
    def __init__(self, input_dim=25, hidden_dim=64, visualize=False):
        super(FaultLocGCN, self).__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.visualize = visualize
        self.step = 0

    def forward(self, x, ast_type_ids, edge_index, num_nodes):
        edge_weights = torch.ones(edge_index.size(1), device=x.device)
        deg = torch.bincount(edge_index[0], minlength=num_nodes).float()
        deg = deg.clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)
        norm_weights = deg_inv_sqrt[edge_index[0]] * edge_weights * deg_inv_sqrt[edge_index[1]]
        adj_norm = torch.sparse_coo_tensor(edge_index, norm_weights, (num_nodes, num_nodes))

        h = F.relu(self.embed(x))
        h = F.relu(self.fc1(h))
        h = torch.sparse.mm(adj_norm, h)
        h = F.relu(self.fc2(h))
        h = torch.sparse.mm(adj_norm, h)
        out = self.out(h)
        return out, h


class FaultLocGAT(nn.Module):
    def __init__(self, input_dim=25, hidden_dim=64, num_heads=2, visualize=False):
        super(FaultLocGAT, self).__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.attn_heads = nn.ModuleList()
        for _ in range(num_heads):
            self.attn_heads.append(nn.Linear(hidden_dim * 2, 1))
        self.num_heads = num_heads
        self.fc1 = nn.Linear(hidden_dim * num_heads, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.visualize = visualize
        self.step = 0

    def forward(self, x, ast_type_ids, edge_index, num_nodes):
        h = F.relu(self.embed(x))
        row, col = edge_index[0], edge_index[1]

        attn_scores = []
        for head in self.attn_heads:
            h_row = h[row]
            h_col = h[col]
            h_cat = torch.cat([h_row, h_col], dim=1)
            score = F.leaky_relu(head(h_cat))
            attn_scores.append(score)

        attn_weights = []
        for score in attn_scores:
            src_nodes = edge_index[0]
            max_score = torch.zeros(num_nodes, device=score.device)
            max_score.scatter_reduce_(0, src_nodes, score.squeeze(), reduce='amax')
            exp_score = torch.exp(score - max_score[src_nodes].unsqueeze(1))
            sum_exp = torch.zeros(num_nodes, device=score.device)
            sum_exp.scatter_reduce_(0, src_nodes, exp_score.squeeze(), reduce='sum')
            norm_score = exp_score / sum_exp[src_nodes].unsqueeze(1)
            attn_weights.append(norm_score)

        head_outs = []
        for i, weight in enumerate(attn_weights):
            out = torch.zeros_like(h)
            out.index_add_(0, row, weight * h[col])
            head_outs.append(out)

        h = torch.cat(head_outs, dim=1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        out = self.out(h)
        return out, h


class FaultLocDepGraph(nn.Module):
    def __init__(self, input_dim=25, hidden_dim=64, num_steps=5, visualize=False):
        super(FaultLocDepGraph, self).__init__()
        self.num_steps = num_steps
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.visualize = visualize
        self.step = 0

    def forward(self, x, ast_type_ids, edge_index, num_nodes):
        h = F.relu(self.linear(x))
        for _ in range(self.num_steps):
            messages = torch.zeros_like(h)
            messages.index_add_(0, edge_index[1], h[edge_index[0]])
            h = self.gru(messages, h)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        out = self.out(h)
        return out, h


class FaultLocDepGraphC(nn.Module):
    def __init__(self, input_dim=25, hidden_dim=64, num_steps=4, visualize=False):
        super().__init__()
        self.num_steps = num_steps
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.visualize = visualize
        self.step = 0

    def forward(self, x, edge_index, num_nodes):
        h = F.relu(self.linear(x))
        for _ in range(self.num_steps):
            messages = torch.zeros_like(h)
            messages.index_add_(0, edge_index[1], h[edge_index[0]])
            h_new = self.gru(messages, h)
            h = h + h_new
            h = self.norm(h)
            h = self.dropout(h)
        h_res = h
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h)) + h_res
        out = self.out(h)
        return out, h


class FaultLocDepGraphC2(nn.Module):
    def __init__(self, input_dim=155, hidden_dim=128, num_steps=4, dropout=0.3):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.num_steps = num_steps
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, ast_type_ids=None, edge_index=None, num_nodes=None):
        if edge_index is None:
            edge_index = ast_type_ids
        x = self.input_norm(x)
        h = F.relu(self.linear(x))
        for _ in range(self.num_steps):
            messages = torch.zeros_like(h)
            messages.index_add_(0, edge_index[1], h[edge_index[0]])
            h_new = self.gru(messages, h)
            h = h + h_new
            h = self.norm(h)
            h = self.dropout(h)
        h_res = h
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h)) + h_res
        out = self.out(h)
        return out, h


class L_DSGraph(nn.Module):
    def __init__(self, num_sbfl, num_ast_types=200, ast_emb_dim=16, hidden_dim=128,
                 num_steps=5, dropout=0.3, num_token_types=None, token_emb_dim=16,
                 num_edge_types=3, use_edge_type=False,
                 edge_gate_bias_init=-1, edge_gate_l1_lambda=0.01):
        super().__init__()
        self.num_steps = num_steps
        self.use_edge_type = use_edge_type
        self.use_sbfl = num_sbfl > 0
        self.edge_gate_l1_lambda = edge_gate_l1_lambda
        self._last_edge_gate_values = None

        self.alpha = nn.Parameter(torch.tensor([0.1]))
        self.beta = nn.Parameter(torch.tensor([0.3]))

        if self.use_sbfl:
            self.sbfl_proj = nn.Linear(num_sbfl, hidden_dim // 2)

        self.token_emb = nn.Embedding(num_token_types or 500, token_emb_dim)
        self.token_proj = nn.Linear(token_emb_dim, hidden_dim // 2)

        self.ast_emb = nn.Embedding(num_ast_types, ast_emb_dim)
        self.syntax_proj = nn.Linear(ast_emb_dim, hidden_dim)

        if self.use_sbfl:
            self.sbfl_norm = nn.LayerNorm(hidden_dim // 2)
        self.tok_norm = nn.LayerNorm(hidden_dim // 2)
        self.syn_norm = nn.LayerNorm(hidden_dim)

        self.gate_core = nn.Linear(hidden_dim, hidden_dim)
        self.gate_syn = nn.Linear(hidden_dim, hidden_dim)
        nn.init.constant_(self.gate_core.bias, 0.5)
        nn.init.constant_(self.gate_syn.bias, -0.5)

        self.res_weight = nn.Parameter(torch.tensor(0.8))

        if use_edge_type:
            self.edge_type_embeddings = nn.Embedding(num_edge_types, hidden_dim)
            self.edge_type_gate = nn.Linear(hidden_dim, hidden_dim)
            nn.init.constant_(self.edge_type_gate.bias, edge_gate_bias_init)
        else:
            self.edge_type_embeddings = None

        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def get_edge_gate_l1_loss(self):
        if self._last_edge_gate_values is not None and self.edge_gate_l1_lambda > 0:
            return self.edge_gate_l1_lambda * self._last_edge_gate_values.abs().mean()
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def forward(self, x_sbfl, x_ast, edge_index, x_token_ids=None, edge_type_ids=None, adj=None):
        if self.use_sbfl:
            h_sbfl = self.sbfl_proj(x_sbfl)
        else:
            h_sbfl = torch.zeros(x_sbfl.shape[0], self.token_proj.out_features, device=x_sbfl.device)

        if x_token_ids is not None:
            h_tok = self.token_proj(self.token_emb(x_token_ids))
        else:
            h_tok = torch.zeros_like(h_sbfl)

        h_syn = F.relu(self.syntax_proj(self.ast_emb(x_ast)))

        if self.use_sbfl:
            h_sbfl = self.sbfl_norm(h_sbfl)
        h_tok = self.tok_norm(h_tok)
        h_syn = self.syn_norm(h_syn)
        h_core = F.relu(torch.cat([h_sbfl, h_tok], dim=-1))

        g = torch.sigmoid(self.gate_core(h_core) + self.gate_syn(h_syn))
        h = g * h_core + (1 - g) * h_syn

        h = h + self.res_weight * h_core

        self._last_edge_gate_values = None
        for _ in range(self.num_steps):
            if adj is not None and adj.numel() > 0:
                messages = torch.mm(adj, h)
            elif self.use_edge_type and edge_type_ids is not None and self.edge_type_embeddings is not None:
                edge_emb = self.edge_type_embeddings(edge_type_ids)
                edge_gate = torch.sigmoid(self.edge_type_gate(edge_emb))
                self._last_edge_gate_values = edge_gate
                src_h = h[edge_index[0]] * edge_gate
                messages = torch.zeros_like(h)
                messages.index_add_(0, edge_index[1], src_h)
            else:
                messages = torch.zeros_like(h)
                messages.index_add_(0, edge_index[1], h[edge_index[0]])
            h_new = self.gru(messages, h)
            h = h + h_new + self.alpha * h_core
            h = self.norm(h)
            h = self.dropout(h)

        h_res = h
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h)) + h_res + self.beta * self.res_weight * h_core
        return self.out(h), h


class L_DSGraph_Ablation(nn.Module):
    def __init__(self, num_sbfl, num_lexical, num_ast_types=200, ast_emb_dim=16, hidden_dim=128,
                 variant='full', num_steps=5, dropout=0.3, num_token_types=None, token_emb_dim=16,
                 num_edge_types=3, use_edge_type=False,
                 use_fusion_norm=False, fusion_norm_type='none',
                 edge_gate_bias_init=-1, edge_gate_l1_lambda=0.01,
                 fusion_type='auto', use_cross_attn_fusion=False,
                 cross_attn_heads=4):
        super().__init__()
        self.num_steps = num_steps
        self.use_edge_type = use_edge_type
        self.fusion_norm_type = fusion_norm_type
        self.edge_gate_l1_lambda = edge_gate_l1_lambda
        self.use_cross_attn_fusion = use_cross_attn_fusion
        self._last_edge_gate_values = None

        if use_fusion_norm and fusion_norm_type == 'none':
            self.fusion_norm_type = 'layer_fusion'

        if isinstance(variant, str):
            self.variants = set([variant]) if variant != 'full' else set()
        else:
            self.variants = set(variant)

        if 'wo_lexical' in self.variants and 'wo_log1p' in self.variants:
            self.variants.discard('wo_log1p')

        self.variant_str = '_'.join(sorted(self.variants)) if self.variants else 'full'

        use_token_emb = 'token_emb' in self.variants
        use_hash = 'wo_lexical' not in self.variants and not use_token_emb

        if fusion_type != 'auto':
            self.fusion_type = fusion_type
        elif 'concat' in self.variants:
            self.fusion_type = 'concat'
        else:
            self.fusion_type = 'gate'

        self.alpha = nn.Parameter(torch.tensor([0.1]))
        self.beta = nn.Parameter(torch.tensor([0.3]))

        self.sbfl_proj = nn.Linear(num_sbfl, hidden_dim // 2)

        if use_token_emb:
            self.token_emb = nn.Embedding(num_token_types or 500, token_emb_dim)
            self.token_proj = nn.Linear(token_emb_dim, hidden_dim // 2)
        elif use_hash:
            self.lex_proj = nn.Linear(num_lexical, hidden_dim // 2)

        if use_cross_attn_fusion and 'wo_syntax' not in self.variants:
            self.cross_attn_fusion = SBFLTokenCrossAttention(
                hidden_dim // 2, hidden_dim,
                num_heads=cross_attn_heads, dropout=dropout
            )

        if 'wo_syntax' not in self.variants:
            self.ast_emb = nn.Embedding(num_ast_types, ast_emb_dim)
            self.syntax_proj = nn.Linear(ast_emb_dim, hidden_dim)

        if self.fusion_norm_type == 'layer':
            self.sbfl_norm = nn.LayerNorm(hidden_dim // 2)
            if use_token_emb:
                self.tok_norm = nn.LayerNorm(hidden_dim // 2)
            elif use_hash:
                self.lex_norm = nn.LayerNorm(hidden_dim // 2)
            if 'wo_syntax' not in self.variants:
                self.syn_norm = nn.LayerNorm(hidden_dim)
        elif self.fusion_norm_type == 'layer_fusion' and 'wo_syntax' not in self.variants:
            self.core_norm = nn.LayerNorm(hidden_dim)
            self.syn_norm = nn.LayerNorm(hidden_dim)

        if 'wo_syntax' not in self.variants:
            if self.fusion_type == 'concat':
                self.concat_proj = nn.Linear(hidden_dim * 2, hidden_dim)
            elif self.fusion_type == 'gate':
                self.fusion_gate = nn.Linear(hidden_dim * 2, hidden_dim)
                nn.init.constant_(self.fusion_gate.bias, -2.5)
            elif self.fusion_type == 'balanced_gate':
                self.gate_core = nn.Linear(hidden_dim, hidden_dim)
                self.gate_syn = nn.Linear(hidden_dim, hidden_dim)
                nn.init.constant_(self.gate_core.bias, 0.5)
                nn.init.constant_(self.gate_syn.bias, -0.5)

        self.res_weight = nn.Parameter(torch.tensor(0.8))

        if use_edge_type:
            self.edge_type_embeddings = nn.Embedding(num_edge_types, hidden_dim)
            self.edge_type_gate = nn.Linear(hidden_dim, hidden_dim)
            nn.init.constant_(self.edge_type_gate.bias, edge_gate_bias_init)
        else:
            self.edge_type_embeddings = None

        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def get_edge_gate_l1_loss(self):
        if self._last_edge_gate_values is not None and self.edge_gate_l1_lambda > 0:
            return self.edge_gate_l1_lambda * self._last_edge_gate_values.abs().mean()
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def forward(self, x_sbfl, x_lexical, x_ast, edge_index, x_token_ids=None, edge_type_ids=None, adj=None):
        h_sbfl = self.sbfl_proj(x_sbfl)

        use_token_emb = 'token_emb' in self.variants

        if 'wo_lexical' in self.variants and not use_token_emb:
            h_lex_feat = torch.zeros_like(h_sbfl)
        elif use_token_emb:
            if x_token_ids is not None:
                h_lex_feat = self.token_proj(self.token_emb(x_token_ids))
            else:
                h_lex_feat = torch.zeros_like(h_sbfl)
        else:
            if 'wo_log1p' in self.variants:
                x_lex_processed = x_lexical
            else:
                x_lex_processed = torch.log1p(x_lexical)
            h_lex_feat = self.lex_proj(x_lex_processed)

        if self.use_cross_attn_fusion:
            h_core = self.cross_attn_fusion(h_sbfl, h_lex_feat)
        else:
            if self.fusion_norm_type == 'layer':
                h_sbfl = self.sbfl_norm(h_sbfl)
                if hasattr(self, 'tok_norm'):
                    h_lex_feat = self.tok_norm(h_lex_feat)
                elif hasattr(self, 'lex_norm'):
                    h_lex_feat = self.lex_norm(h_lex_feat)
            elif self.fusion_norm_type == 'l2':
                h_sbfl = F.normalize(h_sbfl, p=2, dim=-1)
                h_lex_feat = F.normalize(h_lex_feat, p=2, dim=-1)
            h_core = F.relu(torch.cat([h_sbfl, h_lex_feat], dim=-1))

        if 'wo_syntax' in self.variants:
            h = h_core
        else:
            h_syn = F.relu(self.syntax_proj(self.ast_emb(x_ast)))

            if self.fusion_norm_type == 'layer':
                h_syn = self.syn_norm(h_syn)
            elif self.fusion_norm_type == 'l2':
                h_syn = F.normalize(h_syn, p=2, dim=-1)
            elif self.fusion_norm_type == 'layer_fusion':
                h_core = self.core_norm(h_core)
                h_syn = self.syn_norm(h_syn)

            if self.fusion_type == 'concat':
                h_concat = torch.cat([h_core, h_syn], dim=-1)
                h = F.relu(self.concat_proj(h_concat))
            elif self.fusion_type == 'gate':
                gate_input = torch.cat([h_core, h_syn], dim=-1)
                g = torch.sigmoid(self.fusion_gate(gate_input))
                h = h_core + g * h_syn
            elif self.fusion_type == 'balanced_gate':
                g = torch.sigmoid(self.gate_core(h_core) + self.gate_syn(h_syn))
                h = g * h_core + (1 - g) * h_syn

        h = h + self.res_weight * h_core

        self._last_edge_gate_values = None
        for _ in range(self.num_steps):
            if adj is not None and adj.numel() > 0:
                messages = torch.mm(adj, h)
            elif self.use_edge_type and edge_type_ids is not None and self.edge_type_embeddings is not None:
                edge_emb = self.edge_type_embeddings(edge_type_ids)
                edge_gate = torch.sigmoid(self.edge_type_gate(edge_emb))
                self._last_edge_gate_values = edge_gate
                src_h = h[edge_index[0]] * edge_gate
                messages = torch.zeros_like(h)
                messages.index_add_(0, edge_index[1], src_h)
            else:
                messages = torch.zeros_like(h)
                messages.index_add_(0, edge_index[1], h[edge_index[0]])
            h_new = self.gru(messages, h)
            h = h + h_new + self.alpha * h_core
            h = self.norm(h)
            h = self.dropout(h)

        h_res = h
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h)) + h_res + self.beta * self.res_weight * h_core
        return self.out(h), h


class GRAMUS(nn.Module):
    def __init__(self, input_dim=25, hidden_dim=64, nheads=1, dropout=0.1, alpha=0.2, use_sparse=False):
        super(GRAMUS, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.alpha = alpha
        self.use_sparse = use_sparse
        self.nheads = nheads

        nhid = int(hidden_dim / nheads)
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.propogator = Propogator(hidden_dim)
        self.out = nn.Sequential(nn.Tanh(), nn.Linear(hidden_dim, 1))
        self._initialization()

        if use_sparse:
            self.attentions = [SpGraphAttentionLayer(hidden_dim, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        else:
            self.attentions = [GraphAttentionLayer(hidden_dim, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        attention_output_dim = nhid * nheads
        if attention_output_dim != hidden_dim:
            self.attention_proj = nn.Linear(attention_output_dim, hidden_dim)
        else:
            self.attention_proj = None

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x, ast_type_ids, edge_index, num_nodes):
        h = F.relu(self.linear(x))
        h = F.dropout(h, self.dropout, training=self.training)

        row, col = edge_index
        values = torch.ones_like(row, dtype=torch.float)
        adj = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes))

        h = h.unsqueeze(0)
        adj = adj.unsqueeze(0)
        h = torch.cat([att(h, adj) for att in self.attentions], dim=-1)
        h = h.squeeze(0)

        if self.attention_proj is not None:
            h = self.attention_proj(h)

        adj_dense = adj.to_dense().squeeze(0)
        h = self.propogator(h.unsqueeze(0), h.unsqueeze(0), adj_dense.unsqueeze(0))
        h = h.squeeze(0)
        out = self.out(h)
        return out, h


class SpGRAMUS(nn.Module):
    def __init__(self, input_dim=25, hidden_dim=64, nheads=1, dropout=0.1, alpha=0.2):
        super(SpGRAMUS, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.alpha = alpha
        self.nheads = nheads

        nhid = int(hidden_dim / nheads)
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.propogator = Propogator(hidden_dim)
        self.out = nn.Sequential(nn.Tanh(), nn.Linear(hidden_dim, 1))
        self._initialization()

        self.attentions = [SpGraphAttentionLayer(hidden_dim, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        attention_output_dim = nhid * nheads
        if attention_output_dim != hidden_dim:
            self.attention_proj = nn.Linear(attention_output_dim, hidden_dim)
        else:
            self.attention_proj = None

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x, ast_type_ids, edge_index, num_nodes):
        h = F.relu(self.linear(x))
        h = F.dropout(h, self.dropout, training=self.training)

        attention_outputs = []
        for att in self.attentions:
            _, h_prime = att(h, ast_type_ids, edge_index, num_nodes)
            attention_outputs.append(h_prime)
        h = torch.cat(attention_outputs, dim=-1)

        if self.attention_proj is not None:
            h = self.attention_proj(h)

        row, col = edge_index
        values = torch.ones_like(row, dtype=torch.float)
        adj = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes))
        adj_dense = adj.to_dense()
        h = self.propogator(h.unsqueeze(0), h.unsqueeze(0), adj_dense.unsqueeze(0))
        h = h.squeeze(0).squeeze(0)
        out = self.out(h)
        return out, h


GGAT = GRAMUS
SpGGAT = SpGRAMUS


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class GraceAttention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            if len(list(mask.size())) != 4:
                mask = mask.unsqueeze(1).repeat(1, query.size(2), 1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class GraceMultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = GraceAttention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        x, attn = self.attention(query, key, value, mask=mask, dropout=None)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x)


class GraceCombinationLayer(nn.Module):
    def forward(self, query, key, value, dropout=None):
        query_key = query * key / math.sqrt(query.size(-1))
        query_value = query * value / math.sqrt(query.size(-1))
        tmpW = torch.stack([query_key, query_value], -1)
        tmpsum = torch.softmax(tmpW, dim=-1)
        tmpV = torch.stack([key, value], dim=-1)
        tmpsum = tmpsum * tmpV
        tmpsum = torch.squeeze(torch.sum(tmpsum, dim=-1), -1)
        if dropout:
            tmpsum = dropout(tmpsum)
        return tmpsum


class GraceMultiHeadedCombination(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.combination = GraceCombinationLayer()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        x = self.combination(query, key, value, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x)


class GraceConvolutionLayer(nn.Module):
    def __init__(self, dmodel, layernum, kernelsize=3, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(dmodel, layernum, kernelsize, padding=(kernelsize - 1) // 2)
        self.conv2 = nn.Conv1d(dmodel, layernum, kernelsize, padding=(kernelsize - 1) // 2)
        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        mask = mask.unsqueeze(-1).repeat(1, 1, x.size(2))
        x = x.masked_fill(mask == 0, 0)
        convx = self.conv1(x.permute(0, 2, 1))
        convx = self.dropout(self.activation(convx))
        out = self.conv2(convx).permute(0, 2, 1)
        return out


class GraceDenseLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class GracePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len <= self.max_len:
            return self.pe[:, :seq_len]
        position = torch.arange(0, seq_len, device=x.device).float().unsqueeze(1)
        div_term = (torch.arange(0, self.d_model, 2, device=x.device).float() * -(math.log(10000.0) / self.d_model)).exp()
        pe_ext = torch.zeros(seq_len, self.d_model, device=x.device)
        pe_ext[:, 0::2] = torch.sin(position * div_term)
        pe_ext[:, 1::2] = torch.cos(position * div_term)
        return pe_ext.unsqueeze(0)


class GraceGCNN(nn.Module):
    def __init__(self, dmodel, dropout=0.1):
        super().__init__()
        self.hiddensize = dmodel
        self.linear = nn.Linear(dmodel, dmodel)
        self.linearSecond = nn.Linear(dmodel, dmodel)
        self.activate = GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.subconnect = SublayerConnection(dmodel, dropout)
        self.lstm = nn.LSTMCell(dmodel, dmodel)

    def forward(self, state, left, inputad):
        if left is not None:
            state = torch.cat([left, state], dim=1)
        state = self.linear(state)
        s = state.size(1)
        state = self.subconnect(state, lambda _x: self.lstm(
            torch.bmm(inputad, state).reshape(-1, self.hiddensize),
            (torch.zeros(_x.reshape(-1, self.hiddensize).size(), device=state.device),
             _x.reshape(-1, self.hiddensize))
        )[1].reshape(-1, s, self.hiddensize))
        state = self.linearSecond(state)
        if left is not None:
            state = state[:, left.size(1):, :]
        return state


class GraceTransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.Tconv_forward = GraceGCNN(dmodel=hidden, dropout=dropout)
        self.sublayer4 = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, mask, inputP):
        x = self.sublayer4(x, lambda _x: self.Tconv_forward.forward(_x, None, inputP))
        x = self.norm(x)
        return self.dropout(x)


class GraceRightTransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = GraceMultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.combination = GraceMultiHeadedCombination(h=attn_heads, d_model=hidden)
        self.feed_forward = GraceDenseLayer(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.conv_forward = GraceConvolutionLayer(dmodel=hidden, layernum=hidden)
        self.sublayer1 = SublayerConnection(size=hidden, dropout=dropout)
        self.sublayer2 = SublayerConnection(size=hidden, dropout=dropout)
        self.sublayer3 = SublayerConnection(size=hidden, dropout=dropout)
        self.sublayer4 = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask, charEm):
        x = self.sublayer1(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.sublayer2(x, lambda _x: self.combination.forward(_x, _x, charEm))
        x = self.sublayer3(x, lambda _x: self.conv_forward.forward(_x, mask))
        return self.dropout(x)


class FaultLocGrace(nn.Module):
    def __init__(self, input_dim=155, hidden_dim=64, num_layers=5, num_heads=8, dropout=0.1, max_nodes=1000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        self.feed_forward_hidden = 4 * hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.ast_embedding = nn.Embedding(500, hidden_dim)
        self.pos = GracePositionalEmbedding(hidden_dim, max_len=max_nodes)
        self.transformerBlocks = nn.ModuleList([
            GraceTransformerBlock(hidden_dim, num_heads, self.feed_forward_hidden, dropout)
            for _ in range(num_layers)
        ])
        self.transformerBlocksTree = nn.ModuleList([
            GraceRightTransformerBlock(hidden_dim, num_heads, self.feed_forward_hidden, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.resLinear2 = nn.Linear(hidden_dim, 1)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, ast_type_ids, edge_index, num_nodes):
        N = num_nodes
        device = x.device
        h = self.input_proj(x)
        if ast_type_ids is not None:
            ast_emb = self.ast_embedding(ast_type_ids.clamp(0, 499))
            h = h + ast_emb
        h = h + self.pos(h.unsqueeze(0)).squeeze(0)
        h = self.dropout_layer(h)
        adj = torch.zeros(N, N, device=device)
        if edge_index.size(1) > 0:
            row, col = edge_index[0].clamp(0, N - 1), edge_index[1].clamp(0, N - 1)
            adj[row, col] = 1.0
        h = h.unsqueeze(0)
        adj = adj.unsqueeze(0)
        nlmask = torch.gt(ast_type_ids, 0).unsqueeze(0) if ast_type_ids is not None else None
        for trans in self.transformerBlocks:
            h = trans(h, nlmask, adj)
        h = h.squeeze(0)
        h = self.norm(h)
        out = self.resLinear2(h)
        return out, h


class FaultLocGraceLite(nn.Module):
    def __init__(self, input_dim=155, hidden_dim=128, num_layers=4, num_heads=4, dropout=0.1, max_nodes=1000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        self.feed_forward_hidden = 4 * hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.ast_embedding = nn.Embedding(500, hidden_dim)
        self.pos = GracePositionalEmbedding(hidden_dim, max_len=max_nodes)
        self.transformerBlocks = nn.ModuleList([
            GraceTransformerBlock(hidden_dim, num_heads, self.feed_forward_hidden, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.resLinear2 = nn.Linear(hidden_dim, 1)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, ast_type_ids, edge_index, num_nodes):
        N = num_nodes
        device = x.device
        h = self.input_proj(x)
        if ast_type_ids is not None:
            ast_emb = self.ast_embedding(ast_type_ids.clamp(0, 499))
            h = h + ast_emb
        h = h + self.pos(h.unsqueeze(0)).squeeze(0)
        h = self.dropout_layer(h)
        adj = torch.zeros(N, N, device=device)
        if edge_index.size(1) > 0:
            row, col = edge_index[0].clamp(0, N - 1), edge_index[1].clamp(0, N - 1)
            adj[row, col] = 1.0
        h = h.unsqueeze(0)
        adj = adj.unsqueeze(0)
        nlmask = torch.gt(ast_type_ids, 0).unsqueeze(0) if ast_type_ids is not None else None
        for trans in self.transformerBlocks:
            h = trans(h, nlmask, adj)
        h = h.squeeze(0)
        h = self.norm(h)
        out = self.resLinear2(h)
        return out, h


class depgraphO(nn.Module):
    def __init__(self, input_dim=25, hidden_dim=64, num_steps=5, dropout=0.3, visualize=False):
        super(depgraphO, self).__init__()
        self.embedding_size = hidden_dim
        self.nl_len = 100
        self.word_len = 10
        self.vocab_size = 10000
        self.nl_vocab_size = 500
        self.feed_forward_hidden = 4 * self.embedding_size

        self.char_embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, 10))

        self.transformerBlocks = nn.ModuleList([
            GraceTransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, dropout)
            for _ in range(num_steps)
        ])

        self.token_embedding = nn.Embedding(self.nl_vocab_size, self.embedding_size - 1)
        self.token_embedding1 = nn.Embedding(self.nl_vocab_size, self.embedding_size)

        self.text_embedding = nn.Embedding(20, self.embedding_size)

        self.transformerBlocksTree = nn.ModuleList([
            GraceRightTransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, dropout)
            for _ in range(num_steps)
        ])

        self.resLinear = nn.Linear(self.embedding_size, 2)
        self.pos = GracePositionalEmbedding(self.embedding_size)
        self.norm = nn.LayerNorm(self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size // 2, int(self.embedding_size / 4), batch_first=True, bidirectional=True)
        self.resLinear2 = nn.Linear(self.embedding_size, 1)

        self.input_proj = nn.Linear(input_dim, 1)

    def forward(self, x, ast_type_ids, edge_index, num_nodes):
        N = num_nodes
        device = x.device

        inputad = torch.zeros(N, N, device=device)
        if edge_index is not None and edge_index.size(1) > 0:
            row = edge_index[0].clamp(0, N - 1)
            col = edge_index[1].clamp(0, N - 1)
            inputad[row, col] = 1.0
        inputad = inputad.unsqueeze(0).float()

        if ast_type_ids is not None:
            nlmask = torch.gt(ast_type_ids, 0).unsqueeze(0)
        else:
            nlmask = torch.ones(1, N, dtype=torch.bool, device=device)

        if ast_type_ids is not None:
            nodeem = self.token_embedding(ast_type_ids.clamp(0, self.nl_vocab_size - 1))
        else:
            nodeem = torch.zeros(N, self.embedding_size - 1, device=device)
        text_feature = self.input_proj(x)
        nodeem = torch.cat([nodeem, text_feature], dim=-1)

        if ast_type_ids is not None:
            lineem = self.token_embedding1(ast_type_ids.clamp(0, self.nl_vocab_size - 1))
        else:
            lineem = torch.zeros(N, self.embedding_size, device=device)

        h = torch.cat([nodeem, lineem], dim=0).unsqueeze(0)

        padded_ad = torch.zeros(1, 2 * N, 2 * N, device=device)
        padded_ad[0, :N, :N] = inputad[0]

        nlmask_ext = torch.cat([nlmask, torch.ones(1, N, dtype=torch.bool, device=device)], dim=1)

        for trans in self.transformerBlocks:
            h = trans(h, nlmask_ext, padded_ad)

        h = h[:, :N, :].squeeze(0)

        h = self.norm(h)
        out = self.resLinear2(h)
        return out, h


class SageConv(Module):
    def __init__(self, in_features, out_features, bias=False):
        super(SageConv, self).__init__()
        self.proj = nn.Linear(in_features * 2, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.)

    def forward(self, features, adj):
        if not isinstance(adj, torch.sparse.FloatTensor):
            a = adj.to_dense()
            b = a.sum(dim=1).reshape(adj.shape[0], -1) + 1
            neigh_feature = torch.mm(a, features) / b
        else:
            a = adj.to_dense()
            b = a.sum(dim=1).reshape(adj.shape[0], -1) + 1
            neigh_feature = torch.spmm(adj, features) / b
        data = torch.cat([features, neigh_feature], dim=-1)
        combined = self.proj(data)
        return combined


class Sage_En(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout):
        super(Sage_En, self).__init__()
        self.sage1 = SageConv(nfeat, nembed)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.sage1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class Sage_Classifier(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout):
        super(Sage_Classifier, self).__init__()
        self.sage1 = SageConv(nembed, nhid)
        self.mlp = nn.Linear(nhid, nclass)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight, std=0.05)

    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)
        return x


class Classifier(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout):
        super(Classifier, self).__init__()
        self.mlp = nn.Linear(nhid, nclass)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight, std=0.05)

    def forward(self, x, adj):
        x = self.mlp(x)
        return x


class Decoder(Module):
    def __init__(self, nembed, dropout=0.1):
        super(Decoder, self).__init__()
        self.dropout = dropout
        self.de_weight = Parameter(torch.FloatTensor(nembed, nembed))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.de_weight.size(1))
        self.de_weight.data.uniform_(-stdv, stdv)

    def forward(self, node_embed):
        combine = F.linear(node_embed, self.de_weight)
        adj_out = torch.sigmoid(torch.mm(combine, combine.transpose(-1, -2)))
        return adj_out


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=100):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 16, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        return w


class GNET4FL(nn.Module):
    def __init__(self, input_dim=25, hidden_dim=64, nclass=1, dropout=0.3,
                 use_decoder=False, use_semantic_attention=False):
        super(GNET4FL, self).__init__()
        nembed = hidden_dim
        nhid = hidden_dim
        self.use_decoder = use_decoder
        self.use_semantic_attention = use_semantic_attention

        self.encoder = Sage_En(input_dim, nhid, nembed, dropout)
        self.classifier = Sage_Classifier(nembed, nhid, nclass, dropout)

        if use_decoder:
            self.decoder = Decoder(nembed, dropout)

        if use_semantic_attention:
            self.semantic_attn = SemanticAttention(in_size=nembed, hidden_size=hidden_dim)

    def forward(self, x, ast_type_ids, edge_index, num_nodes):
        adj = self._edge_index_to_adj(edge_index, num_nodes, x.device)

        embed = self.encoder(x, adj)

        if self.use_semantic_attention:
            embed = self.semantic_attn(embed)

        out = self.classifier(embed, adj)

        if self.use_decoder and self.training:
            adj_recon = self.decoder(embed)
            return out, embed, adj_recon

        return out, embed

    def _edge_index_to_adj(self, edge_index, num_nodes, device):
        self_loop = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1)
        edge_index_with_loop = torch.cat([edge_index, self_loop], dim=1)
        edge_weights = torch.ones(edge_index_with_loop.size(1), device=device)
        adj = torch.sparse_coo_tensor(edge_index_with_loop, edge_weights, (num_nodes, num_nodes))
        return adj


class DeepFL(nn.Module):
    """
    DEEP-FL 主模型架构，从 DeepFaultLocalization-master 中提取，适配 GNN 项目。

    原始 DEEP-FL 架构（mutation_first）特征分组：
      - Spectrum:     34 dims
      - Mutation×4:  4×35 = 140 dims
      - Complexity:   37 dims
      - Similarity:   15 dims
      总计: 226 dims

    在 GNN 项目中适配为：
      - SBFL 特征:     num_sbfl_cols 维（覆盖矩阵 + SBFL 统计量 + 公式分数）
      - Lexical 特征:  num_lexical_cols 维

    SBFL 特征内部进一步拆分为子组：
      - 覆盖/公式特征 (cov_dim)
      - SBFL 统计量 (5: ep, ef, np, nf, total)

    核心架构：多组特征分别通过独立 FC 层处理 → 拼接融合 → 最终输出
    """
    def __init__(self, input_dim, num_sbfl=None, num_lexical=None,
                 hidden_dim=128, model_size_times=2, dropout=0.3):
        super().__init__()
        model_times = model_size_times

        self.num_sbfl = num_sbfl
        self.num_lexical = num_lexical

        # SBFL 特征内部子组
        self.sbfl_stats_dim = 5
        self.sbfl_formula_dim = 0
        self.cov_dim = 0
        self.lex_dim = num_lexical if num_lexical else 64

        if num_sbfl and num_sbfl > self.sbfl_stats_dim:
            self.sbfl_formula_dim = num_sbfl - self.sbfl_stats_dim
            self.cov_dim = self.sbfl_formula_dim
        elif num_sbfl:
            self.sbfl_stats_dim = num_sbfl

        # 覆盖矩阵特征处理
        if self.cov_dim > 0:
            self.cov_fc = nn.Linear(self.cov_dim, self.cov_dim * model_times)
            self.cov_norm = nn.LayerNorm(self.cov_dim * model_times)
            self.cov_dropout = nn.Dropout(dropout)
        else:
            self.cov_fc = None

        # SBFL 统计量处理
        if self.sbfl_stats_dim > 0:
            self.sbfl_stats_fc = nn.Linear(
                self.sbfl_stats_dim, self.sbfl_stats_dim * model_times * 2
            )
            self.sbfl_stats_norm = nn.LayerNorm(self.sbfl_stats_dim * model_times * 2)
            self.sbfl_stats_dropout = nn.Dropout(dropout)
        else:
            self.sbfl_stats_fc = None

        # 词法特征处理
        self.lex_fc = nn.Linear(self.lex_dim, self.lex_dim * model_times)
        self.lex_norm = nn.LayerNorm(self.lex_dim * model_times)
        self.lex_dropout = nn.Dropout(dropout)

        # 计算融合后的总维度
        fused_dim = 0
        if self.cov_fc is not None:
            fused_dim += self.cov_dim * model_times
        if self.sbfl_stats_fc is not None:
            fused_dim += self.sbfl_stats_dim * model_times * 2
        fused_dim += self.lex_dim * model_times

        # 特征融合层（对应 DEEP-FL 的 mut_concat + fc 层）
        self.fusion_fc = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 残差连接权重
        self.res_weight = nn.Parameter(torch.tensor(0.8))

        # 输出层
        self.out = nn.Linear(hidden_dim, 1)

        # 兼容 model_forward 接口的维度推断属性
        self._sbfl_dim = num_sbfl if num_sbfl else 0
        self._lex_dim = self.lex_dim

    @property
    def sbfl_proj(self):
        class _DimProxy:
            def __init__(self, dim):
                self.in_features = dim
        return _DimProxy(self._sbfl_dim)

    @property
    def lex_proj(self):
        class _DimProxy:
            def __init__(self, dim):
                self.in_features = dim
        return _DimProxy(self._lex_dim)

    def forward(self, x_sbfl, x_lexical, x_ast, edge_index,
                x_token_ids=None, edge_type_ids=None, adj=None):
        feature_parts = []

        # 1. 处理覆盖/公式特征
        if self.cov_fc is not None and self.num_sbfl:
            cov_feat = x_sbfl[:, :self.cov_dim]
            cov_h = F.relu(self.cov_fc(cov_feat))
            cov_h = self.cov_norm(cov_h)
            cov_h = self.cov_dropout(cov_h)
            feature_parts.append(cov_h)

        # 2. 处理 SBFL 统计量
        if self.sbfl_stats_fc is not None and self.num_sbfl:
            stats_start = self.cov_dim
            stats_end = stats_start + self.sbfl_stats_dim
            if stats_end <= x_sbfl.shape[1]:
                stats_feat = x_sbfl[:, stats_start:stats_end]
            else:
                stats_feat = x_sbfl[:, stats_start:]
            if stats_feat.shape[1] < self.sbfl_stats_dim:
                pad = torch.zeros(stats_feat.shape[0], self.sbfl_stats_dim - stats_feat.shape[1],
                                  device=stats_feat.device)
                stats_feat = torch.cat([stats_feat, pad], dim=-1)
            stats_h = F.relu(self.sbfl_stats_fc(stats_feat))
            stats_h = self.sbfl_stats_norm(stats_h)
            stats_h = self.sbfl_stats_dropout(stats_h)
            feature_parts.append(stats_h)

        # 3. 处理词法特征
        if x_lexical.shape[1] >= self.lex_dim:
            lex_feat = x_lexical[:, :self.lex_dim]
        else:
            lex_feat = x_lexical
            if lex_feat.shape[1] < self.lex_dim:
                pad = torch.zeros(lex_feat.shape[0], self.lex_dim - lex_feat.shape[1],
                                  device=lex_feat.device)
                lex_feat = torch.cat([lex_feat, pad], dim=-1)
        lex_h = F.relu(self.lex_fc(lex_feat))
        lex_h = self.lex_norm(lex_h)
        lex_h = self.lex_dropout(lex_h)
        feature_parts.append(lex_h)

        # 4. 拼接所有特征组
        h_concat = torch.cat(feature_parts, dim=-1)

        # 5. 融合层
        h = self.fusion_fc(h_concat)

        # 6. 输出
        out = self.out(h)
        return out, h


class CNNFL(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, kernel_size=3, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))

        self.out_conv = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, features, ast_type_ids, edges, nodes):
        h = self.input_proj(features)
        h = h.unsqueeze(0).transpose(1, 2)

        for conv in self.convs:
            residual = h
            h = conv(h)
            if h.shape == residual.shape:
                h = h + residual

        out = self.out_conv(h)
        out = out.squeeze(1).transpose(0, 1)
        h_out = h.transpose(1, 2).squeeze(0)
        return out, h_out


class RNNFL(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        rnn_hidden = hidden_dim // 2 if bidirectional else hidden_dim
        self.rnn = nn.GRU(
            hidden_dim, rnn_hidden,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, features, ast_type_ids, edges, nodes):
        h = self.input_proj(features)
        h_seq = h.unsqueeze(0)
        h_rnn, _ = self.rnn(h_seq)
        h_rnn = h_rnn.squeeze(0)

        h = self.fc(h_rnn)
        out = self.out(h)
        return out, h


def create_model(gnn_type, input_dim, num_ast_types=200, **kwargs):
    hidden_dim = kwargs.get('hidden_dim', 128)
    num_heads = kwargs.get('num_heads', 5)
    num_steps = kwargs.get('num_steps', 5)
    dropout = kwargs.get('dropout', 0.3)
    ast_emb_dim = kwargs.get('ast_emb_dim', 16)
    variant = kwargs.get('variant', None)
    num_sbfl = kwargs.get('num_sbfl', None)
    num_lexical = kwargs.get('num_lexical', None)
    alpha = kwargs.get('alpha', 0.2)
    num_layers = kwargs.get('num_layers', 5)
    num_token_types = kwargs.get('num_token_types', None)
    token_emb_dim = kwargs.get('token_emb_dim', 16)
    use_edge_type = kwargs.get('use_edge_type', False)
    num_edge_types = kwargs.get('num_edge_types', 3)
    use_fusion_norm = kwargs.get('use_fusion_norm', False)
    fusion_norm_type = kwargs.get('fusion_norm_type', 'none')
    edge_gate_bias_init = kwargs.get('edge_gate_bias_init', -1)
    edge_gate_l1_lambda = kwargs.get('edge_gate_l1_lambda', 0.01)
    fusion_type = kwargs.get('fusion_type', 'concat')
    use_cross_attn_fusion = kwargs.get('use_cross_attn_fusion', False)
    cross_attn_heads = kwargs.get('cross_attn_heads', 4)

    gnn_type_lower = gnn_type.lower()

    if gnn_type_lower == 'gcn':
        return FaultLocGCN(input_dim, hidden_dim)
    elif gnn_type_lower == 'gat':
        return FaultLocGAT(input_dim, hidden_dim, num_heads)
    elif gnn_type_lower == 'depgraph':
        return FaultLocDepGraph(input_dim, hidden_dim, num_steps=num_steps)
    elif gnn_type_lower == 'depgraphc':
        return FaultLocDepGraphC(input_dim, hidden_dim, num_steps=num_steps)
    elif gnn_type_lower == 'depgraphc2':
        return FaultLocDepGraphC2(input_dim, hidden_dim, num_steps=num_steps, dropout=dropout)
    elif gnn_type_lower == 'depgrapho':
        return depgraphO(input_dim, hidden_dim, num_steps=num_steps, dropout=dropout)
    elif gnn_type_lower in ['ggat', 'gramus']:
        return GRAMUS(input_dim=input_dim, hidden_dim=hidden_dim, nheads=num_heads, dropout=dropout, alpha=alpha, use_sparse=False)
    elif gnn_type_lower in ['spggat', 'spgramus']:
        return SpGRAMUS(input_dim=input_dim, hidden_dim=hidden_dim, nheads=num_heads, dropout=dropout, alpha=alpha)
    elif gnn_type_lower == 'l_dsgraph':
        return L_DSGraph(
            num_sbfl=num_sbfl if num_sbfl is not None else input_dim - 64,
            num_ast_types=num_ast_types,
            ast_emb_dim=ast_emb_dim,
            hidden_dim=hidden_dim,
            num_steps=num_steps,
            dropout=dropout,
            num_token_types=num_token_types,
            token_emb_dim=token_emb_dim,
            num_edge_types=num_edge_types,
            use_edge_type=use_edge_type,
            edge_gate_bias_init=edge_gate_bias_init,
            edge_gate_l1_lambda=edge_gate_l1_lambda,
        )
    elif gnn_type_lower == 'l_dsgraph_ablation':
        return L_DSGraph_Ablation(
            num_sbfl=num_sbfl if num_sbfl is not None else input_dim - 64,
            num_lexical=num_lexical if num_lexical is not None else 64,
            num_ast_types=num_ast_types,
            ast_emb_dim=ast_emb_dim,
            hidden_dim=hidden_dim,
            variant=variant if variant is not None else 'full',
            num_steps=num_steps,
            dropout=dropout,
            num_token_types=num_token_types,
            token_emb_dim=token_emb_dim,
            num_edge_types=num_edge_types,
            use_edge_type=use_edge_type,
            use_fusion_norm=use_fusion_norm,
            fusion_norm_type=fusion_norm_type,
            edge_gate_bias_init=edge_gate_bias_init,
            edge_gate_l1_lambda=edge_gate_l1_lambda,
            fusion_type=fusion_type if fusion_type != 'concat' else 'auto',
            use_cross_attn_fusion=use_cross_attn_fusion,
            cross_attn_heads=cross_attn_heads,
        )
    elif gnn_type_lower == 'grace':
        return FaultLocGrace(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
    elif gnn_type_lower == 'grace_lite':
        return FaultLocGraceLite(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
    elif gnn_type_lower == 'gnet4fl':
        return GNET4FL(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            nclass=1,
            dropout=dropout,
            use_decoder=kwargs.get('use_decoder', False),
            use_semantic_attention=kwargs.get('use_semantic_attention', False),
        )
    elif gnn_type_lower == 'deepfl':
        return DeepFL(
            input_dim=input_dim,
            num_sbfl=kwargs.get('num_sbfl'),
            num_lexical=kwargs.get('num_lexical'),
            hidden_dim=hidden_dim,
            model_size_times=kwargs.get('model_size_times', 2),
            dropout=dropout,
        )
    elif gnn_type_lower == 'cnnfl':
        return CNNFL(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=kwargs.get('num_layers', 3),
            kernel_size=kwargs.get('kernel_size', 3),
            dropout=dropout,
        )
    elif gnn_type_lower == 'rnnfl':
        return RNNFL(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=kwargs.get('num_layers', 2),
            dropout=dropout,
            bidirectional=kwargs.get('bidirectional', True),
        )
    else:
        raise ValueError(f"未知的模型类型: {gnn_type}")
