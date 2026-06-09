import os
import random
import numpy as np
import torch
import inspect
import time
import datetime
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

STATEMENT_ROOT_TYPES = {
    'Assign', 'AugAssign', 'Expr', 'Return',
    'If', 'For', 'While', 'With', 'Raise', 'Try',
    'FunctionDef', 'ClassDef'
}


def get_statement_root_type_ids(ast_type_vocab):
    stmt_root_ids = set()
    for type_name, type_id in ast_type_vocab.items():
        if type_name in STATEMENT_ROOT_TYPES:
            stmt_root_ids.add(type_id)
    return stmt_root_ids


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id, seed=0):
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


class GraphDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        if 'ast_type_ids' in d:
            ast_type_ids = d['ast_type_ids']
        else:
            ast_type_ids = torch.zeros(len(d['features']), dtype=torch.long)
        token_ids = d.get('token_ids', torch.zeros(len(d['features']), dtype=torch.long))
        line_numbers = d.get('line_numbers', torch.zeros(len(d['features']), dtype=torch.long))
        edge_type_ids = d.get('edge_type_ids', torch.zeros(d['edge_index'].shape[1], dtype=torch.long))
        adj = d.get('adj', torch.zeros(0, 0))
        return d['features'], ast_type_ids, token_ids, d['edge_index'], edge_type_ids, d['labels'], len(d['features']), line_numbers, adj


def collate_fn(batch):
    features_list = []
    ast_type_ids_list = []
    token_ids_list = []
    edge_index_list = []
    edge_type_ids_list = []
    labels_list = []
    num_nodes_list = []
    line_numbers_list = []
    adj_list = []

    for item in batch:
        features, ast_type_ids, token_ids, edge_index, edge_type_ids, labels, num_nodes, line_numbers, adj = item
        features_list.append(features)
        ast_type_ids_list.append(ast_type_ids)
        token_ids_list.append(token_ids)
        edge_index_list.append(edge_index)
        edge_type_ids_list.append(edge_type_ids)
        labels_list.append(labels)
        num_nodes_list.append(num_nodes)
        line_numbers_list.append(line_numbers)
        adj_list.append(adj)

    return features_list, ast_type_ids_list, token_ids_list, edge_index_list, edge_type_ids_list, labels_list, num_nodes_list, line_numbers_list, adj_list


def model_forward(model, features, ast_type_ids, edges, nodes, num_sbfl=None, num_lexical=None, token_ids=None, edge_type_ids=None, adj=None):
    sig = inspect.signature(model.forward)
    param_names = list(sig.parameters.keys())

    if len(param_names) == 3:
        return model(features, edges, nodes)
    elif 'x_sbfl' in param_names and 'x_lexical' in param_names:
        model_num_sbfl = model.sbfl_proj.in_features
        model_num_lexical = model.lex_proj.in_features if hasattr(model, 'lex_proj') else 0

        actual_sbfl = num_sbfl if num_sbfl else features.shape[1] - 64
        actual_lexical = num_lexical if num_lexical else 64

        if actual_sbfl < model_num_sbfl:
            pad_sbfl = torch.zeros(features.shape[0], model_num_sbfl - actual_sbfl, device=features.device)
            x_sbfl = torch.cat([features[:, :actual_sbfl], pad_sbfl], dim=-1)
        elif actual_sbfl > model_num_sbfl:
            x_sbfl = features[:, :model_num_sbfl]
        else:
            x_sbfl = features[:, :actual_sbfl]

        if model_num_lexical > 0:
            if actual_lexical < model_num_lexical:
                pad_lex = torch.zeros(features.shape[0], model_num_lexical - actual_lexical, device=features.device)
                x_lexical = torch.cat([features[:, actual_sbfl:actual_sbfl + actual_lexical], pad_lex], dim=-1)
            elif actual_lexical > model_num_lexical:
                x_lexical = features[:, actual_sbfl:actual_sbfl + model_num_lexical]
            else:
                x_lexical = features[:, actual_sbfl:actual_sbfl + actual_lexical]
        else:
            x_lexical = torch.zeros(features.shape[0], 0, device=features.device)

        kwargs = {}
        if 'x_token_ids' in param_names and token_ids is not None:
            kwargs['x_token_ids'] = token_ids
        if 'edge_type_ids' in param_names and edge_type_ids is not None:
            kwargs['edge_type_ids'] = edge_type_ids
        if 'adj' in param_names and adj is not None and adj.numel() > 0:
            kwargs['adj'] = adj
        return model(x_sbfl, x_lexical, ast_type_ids, edges, **kwargs)
    elif 'x_sbfl' in param_names and 'x_ast' in param_names and 'x_lexical' not in param_names:
        if hasattr(model, 'sbfl_proj'):
            model_num_sbfl = model.sbfl_proj.in_features
        else:
            model_num_sbfl = 0

        actual_sbfl = num_sbfl if num_sbfl else features.shape[1]

        if model_num_sbfl > 0:
            if actual_sbfl < model_num_sbfl:
                pad_sbfl = torch.zeros(features.shape[0], model_num_sbfl - actual_sbfl, device=features.device)
                x_sbfl = torch.cat([features[:, :actual_sbfl], pad_sbfl], dim=-1)
            elif actual_sbfl > model_num_sbfl:
                x_sbfl = features[:, :model_num_sbfl]
            else:
                x_sbfl = features[:, :actual_sbfl]
        else:
            x_sbfl = features[:, :0] if actual_sbfl > 0 else torch.zeros(features.shape[0], 0, device=features.device)

        kwargs = {}
        if 'x_token_ids' in param_names and token_ids is not None:
            kwargs['x_token_ids'] = token_ids
        if 'edge_type_ids' in param_names and edge_type_ids is not None:
            kwargs['edge_type_ids'] = edge_type_ids
        if 'adj' in param_names and adj is not None and adj.numel() > 0:
            kwargs['adj'] = adj
        return model(x_sbfl, ast_type_ids, edges, **kwargs)
    elif 'x_num' in param_names and 'x_cat' in param_names:
        return model(features, ast_type_ids, edges, nodes)
    else:
        result = model(features, ast_type_ids, edges, nodes)
        if isinstance(result, tuple) and len(result) == 3:
            return result[0], result[1]
        return result


def evaluate_model(model, dataloader, top_ks=None, return_details=False,
                   difficulties=None, num_sbfl=None, num_lexical=None,
                   statement_root_type_ids=None, difficulty_levels=None):
    if top_ks is None:
        top_ks = [1, 3, 5]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    top_k_accuracies = {k: 0 for k in top_ks}
    total_samples = 0
    total_mrr = total_mar = total_mfr = 0
    total_exam = 0
    sample_details = []
    inference_times = []
    exam_scores = []

    stmt_root_top_k_accuracies = {k: 0 for k in top_ks}
    stmt_root_total_samples = 0
    stmt_root_total_mrr = 0
    stmt_root_total_mfr = 0
    stmt_root_total_exam = 0
    stmt_root_exam_scores = []

    if difficulty_levels is None:
        difficulty_levels = ['0-10', '10-100', '100-500', '500-1000', '1000-2000', '>2000']

    difficulty_top_k = {
        level: {k: 0 for k in top_ks}
        for level in difficulty_levels
    }
    difficulty_counts = {level: 0 for level in difficulty_levels}
    sample_idx = 0

    with torch.no_grad():
        for batch in dataloader:
            features_list, ast_type_ids_list, token_ids_list, edge_index_list, edge_type_ids_list, labels_list, num_nodes_list, line_numbers_list, adj_list = batch

            for i in range(len(features_list)):
                features = features_list[i].to(device)
                ast_type_ids = ast_type_ids_list[i].to(device)
                token_ids = token_ids_list[i].to(device)
                edges = edge_index_list[i].to(device)
                edge_type_ids = edge_type_ids_list[i].to(device)
                labels = labels_list[i].to(device)
                nodes = num_nodes_list[i]
                line_numbers = line_numbers_list[i].cpu().numpy()
                adj = adj_list[i].to(device) if adj_list[i].numel() > 0 else None

                start_time = time.time()
                out, _ = model_forward(model, features, ast_type_ids, edges, nodes, num_sbfl, num_lexical, token_ids=token_ids, edge_type_ids=edge_type_ids, adj=adj)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                true_positions = torch.where(labels == 1)[0]
                if len(true_positions) == 0:
                    total_samples += 1
                    continue

                probabilities = torch.sigmoid(out.squeeze()).cpu().numpy()

                true_line_numbers = set()
                for pos in true_positions:
                    ln = line_numbers[pos.item()]
                    if ln > 0:
                        true_line_numbers.add(ln)

                if len(true_line_numbers) == 0:
                    total_samples += 1
                    continue

                line_to_max_prob = {}
                for node_idx, prob in enumerate(probabilities):
                    ln = line_numbers[node_idx]
                    if ln > 0:
                        if ln not in line_to_max_prob:
                            line_to_max_prob[ln] = prob
                        else:
                            line_to_max_prob[ln] = max(line_to_max_prob[ln], prob)

                line_to_stmt_root_max_prob = {}
                if statement_root_type_ids is not None:
                    ast_type_ids_cpu = ast_type_ids_list[i].cpu().numpy()
                    stmt_root_mask = np.zeros(len(ast_type_ids_cpu), dtype=bool)
                    for type_id in statement_root_type_ids:
                        stmt_root_mask |= (ast_type_ids_cpu == type_id)
                    for node_idx in range(len(probabilities)):
                        if stmt_root_mask[node_idx]:
                            ln = line_numbers[node_idx]
                            if ln > 0:
                                if ln not in line_to_stmt_root_max_prob:
                                    line_to_stmt_root_max_prob[ln] = probabilities[node_idx]
                                else:
                                    line_to_stmt_root_max_prob[ln] = max(line_to_stmt_root_max_prob[ln], probabilities[node_idx])

                if not line_to_max_prob:
                    total_samples += 1
                    continue

                sorted_lines = sorted(line_to_max_prob.keys(), key=lambda x: line_to_max_prob[x], reverse=True)
                total_lines = len(sorted_lines)

                ranks = []
                for ln in sorted(true_line_numbers):
                    if ln in sorted_lines:
                        rank = sorted_lines.index(ln) + 1
                        ranks.append(rank)

                if not ranks:
                    total_samples += 1
                    continue

                best_rank = min(ranks)
                mrr = 1.0 / best_rank
                total_mrr += mrr
                mar = sum(ranks) / len(ranks)
                total_mar += mar
                mfr = best_rank
                total_mfr += mfr

                exam_score = (best_rank / total_lines * 100) if total_lines > 0 else 100
                total_exam += exam_score
                exam_scores.append(exam_score)

                for k in top_ks:
                    if best_rank <= k:
                        top_k_accuracies[k] += 1

                if statement_root_type_ids is not None and line_to_stmt_root_max_prob:
                    stmt_root_has_fault = any(ln in line_to_stmt_root_max_prob for ln in true_line_numbers)
                    if stmt_root_has_fault:
                        sorted_stmt_lines = sorted(
                            line_to_stmt_root_max_prob.keys(),
                            key=lambda x: line_to_stmt_root_max_prob[x], reverse=True
                        )
                        total_stmt_lines = len(sorted_stmt_lines)
                        stmt_ranks = []
                        for ln in sorted(true_line_numbers):
                            if ln in sorted_stmt_lines:
                                rank = sorted_stmt_lines.index(ln) + 1
                                stmt_ranks.append(rank)
                        if stmt_ranks:
                            best_stmt_rank = min(stmt_ranks)
                            stmt_root_total_mrr += 1.0 / best_stmt_rank
                            stmt_root_total_mfr += best_stmt_rank
                            stmt_exam = (best_stmt_rank / total_stmt_lines * 100) if total_stmt_lines > 0 else 100
                            stmt_root_total_exam += stmt_exam
                            stmt_root_exam_scores.append(stmt_exam)
                            for k in top_ks:
                                if best_stmt_rank <= k:
                                    stmt_root_top_k_accuracies[k] += 1
                            stmt_root_total_samples += 1

                if return_details:
                    sample_details.append({
                        'sample_idx': sample_idx,
                        'rank': best_rank,
                        'mrr': mrr,
                        'mar': mar,
                        'mfr': mfr,
                        'top_k_hits': {k: best_rank <= k for k in top_ks}
                    })

                if difficulties is not None and len(difficulties) > 0 and sample_idx < len(difficulties):
                    diff_level = difficulties[sample_idx]
                    if diff_level in difficulty_top_k:
                        for k in top_ks:
                            if best_rank <= k:
                                difficulty_top_k[diff_level][k] += 1
                        difficulty_counts[diff_level] += 1

                sample_idx += 1
                total_samples += 1

    for k in top_ks:
        top_k_accuracies[k] = top_k_accuracies[k] / total_samples if total_samples > 0 else 0
    avg_mrr = total_mrr / total_samples if total_samples > 0 else 0
    avg_mar = total_mar / total_samples if total_samples > 0 else 0
    avg_mfr = total_mfr / total_samples if total_samples > 0 else 0
    avg_exam = total_exam / total_samples if total_samples > 0 else 0

    best_exam = min(exam_scores) if exam_scores else 0
    worst_exam = max(exam_scores) if exam_scores else 0

    min_inference_time = min(inference_times) if inference_times else 0
    max_inference_time = max(inference_times) if inference_times else 0
    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0

    difficulty_results = None
    if difficulties is not None:
        difficulty_results = {}
        for diff_level in difficulty_levels:
            if diff_level in difficulty_counts and difficulty_counts[diff_level] > 0:
                difficulty_results[diff_level] = {
                    k: difficulty_top_k[diff_level][k] / difficulty_counts[diff_level] for k in top_ks
                }
            else:
                difficulty_results[diff_level] = {k: 0 for k in top_ks}

    for k in top_ks:
        stmt_root_top_k_accuracies[k] = stmt_root_top_k_accuracies[k] / stmt_root_total_samples if stmt_root_total_samples > 0 else 0
    stmt_root_avg_mrr = stmt_root_total_mrr / stmt_root_total_samples if stmt_root_total_samples > 0 else 0
    stmt_root_avg_mfr = stmt_root_total_mfr / stmt_root_total_samples if stmt_root_total_samples > 0 else 0
    stmt_root_avg_exam = stmt_root_total_exam / stmt_root_total_samples if stmt_root_total_samples > 0 else 0
    stmt_root_best_exam = min(stmt_root_exam_scores) if stmt_root_exam_scores else 0
    stmt_root_worst_exam = max(stmt_root_exam_scores) if stmt_root_exam_scores else 0

    shortcut_degree = {}
    for k in top_ks:
        shortcut_degree[k] = top_k_accuracies[k] - stmt_root_top_k_accuracies[k]

    result = {
        'top_k_accuracies': top_k_accuracies,
        'avg_mrr': avg_mrr,
        'avg_mar': avg_mar,
        'avg_mfr': avg_mfr,
        'avg_exam': avg_exam,
        'best_exam': best_exam,
        'worst_exam': worst_exam,
        'difficulty_results': difficulty_results,
        'min_inference_time': min_inference_time,
        'max_inference_time': max_inference_time,
        'avg_inference_time': avg_inference_time,
        'stmt_root_top_k_accuracies': stmt_root_top_k_accuracies,
        'stmt_root_avg_mrr': stmt_root_avg_mrr,
        'stmt_root_avg_mfr': stmt_root_avg_mfr,
        'stmt_root_avg_exam': stmt_root_avg_exam,
        'stmt_root_best_exam': stmt_root_best_exam,
        'stmt_root_worst_exam': stmt_root_worst_exam,
        'stmt_root_sample_count': stmt_root_total_samples,
        'shortcut_degree': shortcut_degree,
    }

    if return_details:
        result['sample_details'] = sample_details

    return result


def _compute_rank_with_ties(score_dict):
    if not score_dict:
        return {}
    sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    rank_map = {}
    i = 0
    while i < len(sorted_items):
        score = sorted_items[i][1]
        j = i
        while j < len(sorted_items) and sorted_items[j][1] == score:
            j += 1
        for k in range(i, j):
            rank_map[sorted_items[k][0]] = j
        i = j
    return rank_map


def collect_per_program_ranks(model, processed_items, device, num_sbfl=None, num_lexical=None):
    model.eval()
    ranks = {}
    with torch.no_grad():
        for item in processed_items:
            bug_id = item['bug_id']
            features = item['features'].to(device)
            ast_type_ids = item['ast_type_ids'].to(device)
            token_ids = item['token_ids'].to(device)
            edge_index = item['edge_index'].to(device)
            edge_type_ids = item['edge_type_ids'].to(device)
            labels = item['labels']
            nodes = item['num_nodes']
            line_numbers = item['line_numbers'].numpy()

            out, _ = model_forward(model, features, ast_type_ids, edge_index, nodes,
                                   num_sbfl, num_lexical,
                                   token_ids=token_ids, edge_type_ids=edge_type_ids)
            probs = torch.sigmoid(out.squeeze()).cpu().numpy()

            true_line_numbers = set()
            true_positions = torch.where(labels == 1)[0]
            for pos in true_positions:
                ln = line_numbers[pos.item()]
                if ln > 0:
                    true_line_numbers.add(ln)
            if not true_line_numbers:
                continue

            line_to_max_prob = {}
            for node_idx, prob in enumerate(probs):
                ln = line_numbers[node_idx]
                if ln > 0:
                    line_to_max_prob[ln] = max(line_to_max_prob.get(ln, prob), prob)
            if not line_to_max_prob:
                continue

            line_ranks = _compute_rank_with_ties(line_to_max_prob)
            best_rank = None
            for ln in true_line_numbers:
                if ln in line_ranks:
                    r = line_ranks[ln]
                    if best_rank is None or r < best_rank:
                        best_rank = r
            if best_rank is not None:
                ranks[bug_id] = best_rank
    return ranks


def collect_per_program_statement_ranks(model, processed_items, device, num_sbfl=None, num_lexical=None):
    model.eval()
    statement_ranks = {}
    with torch.no_grad():
        for item in processed_items:
            bug_id = item['bug_id']
            features = item['features'].to(device)
            ast_type_ids = item['ast_type_ids'].to(device)
            token_ids = item['token_ids'].to(device)
            edge_index = item['edge_index'].to(device)
            edge_type_ids = item['edge_type_ids'].to(device)
            labels = item['labels']
            nodes = item['num_nodes']
            line_numbers = item['line_numbers'].numpy()

            out, _ = model_forward(model, features, ast_type_ids, edge_index, nodes,
                                   num_sbfl, num_lexical,
                                   token_ids=token_ids, edge_type_ids=edge_type_ids)
            probs = torch.sigmoid(out.squeeze()).cpu().numpy()

            true_line_numbers = set()
            true_positions = torch.where(labels == 1)[0]
            for pos in true_positions:
                ln = line_numbers[pos.item()]
                if ln > 0:
                    true_line_numbers.add(ln)
            if not true_line_numbers:
                continue

            line_to_max_prob = {}
            for node_idx, prob in enumerate(probs):
                ln = line_numbers[node_idx]
                if ln > 0:
                    line_to_max_prob[ln] = max(line_to_max_prob.get(ln, prob), prob)
            if not line_to_max_prob:
                continue

            line_ranks = _compute_rank_with_ties(line_to_max_prob)
            bug_statement_ranks = {}
            for ln in sorted(true_line_numbers):
                if ln in line_ranks:
                    bug_statement_ranks[ln] = line_ranks[ln]
            if bug_statement_ranks:
                statement_ranks[bug_id] = bug_statement_ranks
    return statement_ranks


def collect_sbfl_per_program_ranks(processed_items, sbfl_data_path, formula_name='zoltar'):
    import json as _json
    ranks = {}
    for item in processed_items:
        bug_id = item['bug_id']
        parts = bug_id.rsplit('_', 1)
        if len(parts) != 2:
            continue
        project, case = parts[0], parts[1]

        formula_path = os.path.join(sbfl_data_path, project, 'Python', case, f"{formula_name}.txt")
        if not os.path.exists(formula_path):
            continue

        try:
            sbfl_scores_by_line = {}
            with open(formula_path, 'r') as f:
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    if '→' in line:
                        scores_str = line.split('→')[1]
                    else:
                        scores_str = line
                    val = _json.loads(scores_str)
                    score = val if isinstance(val, (int, float)) else (max(val) if isinstance(val, list) and val else 0.0)
                    code_line = line_idx + 1
                    sbfl_scores_by_line[code_line] = max(sbfl_scores_by_line.get(code_line, 0), score)
        except Exception:
            continue

        if not sbfl_scores_by_line:
            continue

        true_line_numbers = set()
        labels = item['labels']
        line_numbers = item['line_numbers'].numpy()
        true_positions = torch.where(labels == 1)[0]
        for pos in true_positions:
            ln = line_numbers[pos.item()]
            if ln > 0:
                true_line_numbers.add(ln)
        if not true_line_numbers:
            continue

        line_ranks = _compute_rank_with_ties(sbfl_scores_by_line)
        best_rank = None
        for ln in true_line_numbers:
            if ln in line_ranks:
                r = line_ranks[ln]
                if best_rank is None or r < best_rank:
                    best_rank = r
        if best_rank is not None:
            ranks[bug_id] = best_rank
    return ranks


def a12_effect_size(x, y):
    """计算 Vargha-Delaney Â₁₂ 效应量。

    Â₁₂ = P(X > Y) + 0.5 * P(X = Y)
    - Â₁₂ > 0.5: X 倾向于比 Y 大
    - Â₁₂ < 0.5: Y 倾向于比 X 大
    - Â₁₂ = 0.5: 无差异

    Returns:
        a12: Â₁₂ 值
        greater: X > Y 的次数
        less: X < Y 的次数
        equal: X == Y 的次数
    """
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0.5, 0, 0, 0
    greater = 0
    equal = 0
    for xi in x:
        for yi in y:
            if xi > yi:
                greater += 1
            elif xi == yi:
                equal += 1
    less = nx * ny - greater - equal
    a12 = (greater + 0.5 * equal) / (nx * ny)
    return a12, greater, less, equal


def interpret_a12(a12):
    """解释 Â₁₂ 效应量大小（Vargha & Delaney, 2000）。

    thresholds (based on |Â₁₂ - 0.50|):
        < 0.06: Negligible  (0.44 ~ 0.56)
        < 0.14: Small       (0.36 ~ 0.44 或 0.56 ~ 0.64)
        < 0.21: Medium      (0.29 ~ 0.36 或 0.64 ~ 0.71)
        >= 0.21: Large      (< 0.29 或 > 0.71)
    """
    diff = abs(a12 - 0.50)
    if diff < 0.06:
        return 'Negligible'
    elif diff < 0.14:
        return 'Small'
    elif diff < 0.21:
        return 'Medium'
    else:
        return 'Large'


def compute_statistical_comparison(gnn_ranks, sbfl_ranks, gnn_name='GNN', sbfl_name='SBFL'):
    from scipy import stats as _stats
    common_ids = sorted(set(gnn_ranks.keys()) & set(sbfl_ranks.keys()))
    if len(common_ids) < 5:
        return None

    gnn_vals = np.array([gnn_ranks[bid] for bid in common_ids], dtype=float)
    sbfl_vals = np.array([sbfl_ranks[bid] for bid in common_ids], dtype=float)

    try:
        wilcoxon_stat, p_value = _stats.wilcoxon(gnn_vals, sbfl_vals, alternative='two-sided')
    except Exception:
        return None

    a12_val, greater_count, less_count, equal_count = a12_effect_size(gnn_vals, sbfl_vals)

    # 逐程序胜出次数（配对比，非 n×n 交叉）
    gnn_program_wins = int(np.sum(gnn_vals < sbfl_vals))
    sbfl_program_wins = int(np.sum(gnn_vals > sbfl_vals))
    program_ties = int(np.sum(gnn_vals == sbfl_vals))

    return {
        'method_a': gnn_name,
        'method_b': sbfl_name,
        'n': len(common_ids),
        'mean_a': float(np.mean(gnn_vals)),
        'mean_b': float(np.mean(sbfl_vals)),
        'median_a': float(np.median(gnn_vals)),
        'median_b': float(np.median(sbfl_vals)),
        'std_a': float(np.std(gnn_vals, ddof=1)),
        'std_b': float(np.std(sbfl_vals, ddof=1)),
        'wilcoxon_stat': float(wilcoxon_stat),
        'p_value': float(p_value),
        'significant_0.05': p_value < 0.05,
        'significant_0.01': p_value < 0.01,
        'significant_0.001': p_value < 0.001,
        'a12': float(a12_val),
        'effect_size': interpret_a12(a12_val),
        'a_wins': gnn_program_wins,
        'b_wins': sbfl_program_wins,
        'tied': program_ties,
    }


def compute_all_statistical_tests(gnn_ranks, sbfl_data_path, processed_items, gnn_name='GNN'):
    all_tests = {}
    for formula in ['zoltar', 'ochiai']:
        sbfl_ranks = collect_sbfl_per_program_ranks(processed_items, sbfl_data_path, formula)
        result = compute_statistical_comparison(gnn_ranks, sbfl_ranks, gnn_name, formula.upper())
        if result:
            all_tests[formula] = result
    return all_tests


def load_difficulty_mapping(difficulty_file):
    difficulty_map = {}
    if not os.path.exists(difficulty_file):
        return difficulty_map
    with open(difficulty_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                bug_id = parts[0]
                diff_val = parts[1]
                if diff_val.isdigit():
                    difficulty_map[bug_id] = int(diff_val)
                else:
                    difficulty_map[bug_id] = None
    return difficulty_map


def get_variant_str(variant):
    if variant is None:
        return ""
    if isinstance(variant, list):
        return "_" + "_".join(sorted(variant))
    if variant == 'full':
        return ""
    return f"_{variant}"


def generate_cross_validation_splits(indices, bug_ids, difficulty_levels, cv_method, k_folds, seed):
    if cv_method == 'leave_one_out':
        from collections import defaultdict as _defaultdict
        project_groups = _defaultdict(list)
        for i, bug_id in enumerate(bug_ids):
            project = bug_id.split('_')[0]
            project_groups[project].append(i)

        project_names = sorted(project_groups.keys())
        loo_rng = np.random.RandomState(seed)
        unified_splits = []

        for test_project in project_names:
            test_idx = np.array(project_groups[test_project])
            remaining_projects = [p for p in project_names if p != test_project]

            if len(remaining_projects) < 1:
                continue

            val_project_idx = loo_rng.randint(0, len(remaining_projects))
            val_project = remaining_projects[val_project_idx]
            val_idx = np.array(project_groups[val_project])

            train_projects = [p for p in remaining_projects if p != val_project]
            if not train_projects:
                train_idx = val_idx.copy()
            else:
                train_idx = np.concatenate([np.array(project_groups[p]) for p in train_projects])

            unified_splits.append((train_idx, val_idx, test_idx, test_project))

        return unified_splits
    else:
        sgkf = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=seed)
        raw_splits = list(sgkf.split(indices, y=difficulty_levels, groups=bug_ids))

        unified_splits = []
        for fold_idx, (train_val_idx, test_idx) in enumerate(raw_splits):
            current_project = None
            if len(test_idx) > 0:
                test_bug_id = bug_ids[test_idx[0]]
                current_project = test_bug_id.split('_', 1)[0] if '_' in test_bug_id else test_bug_id

            if len(train_val_idx) <= 1:
                train_idx = train_val_idx
                val_idx = train_val_idx
            else:
                train_val_bug_ids = [bug_ids[i] for i in train_val_idx]
                gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                train_idx_rel, val_idx_rel = next(gss_val.split(
                    np.arange(len(train_val_idx)), groups=train_val_bug_ids
                ))
                train_idx = train_val_idx[train_idx_rel]
                val_idx = train_val_idx[val_idx_rel]

            if len(train_idx) == 0 or len(val_idx) == 0:
                continue

            unified_splits.append((train_idx, val_idx, test_idx, current_project))

        return unified_splits


def save_results_to_excel(results, detailed_results, model_dir, config_str="results"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_file = os.path.join(model_dir, f"training_results_{config_str}_{timestamp}.xlsx")

    def result_to_row(result):
        variant = result.get('variant')
        if variant:
            if isinstance(variant, list):
                variant_display = '+'.join(variant)
            else:
                variant_display = str(variant)
        else:
            variant_display = ''

        row = {
            '模型名称': result.get('model_name', ''),
            '隐藏维度': result['hidden_dim'],
            '学习率': result['lr'],
            '训练轮数': result['epochs'],
            'GNN类型': result['gnn_type'],
            '变体': variant_display,
            '交叉验证方法': result.get('cv_method', 'kfold'),
            'SBFL类型': result.get('sbfl_type', 'mergeSBFLALL'),
            '注意力头数': result['num_heads'],
            '总参数量': result.get('total_params', 0),
            '验证集Top-1准确率': result.get('best_val_top1_accuracy', 0),
            '测试集Top-1准确率': result['test_top_k_accuracies'].get(1, 0),
            '测试集Top-3准确率': result['test_top_k_accuracies'].get(3, 0),
            '测试集Top-5准确率': result['test_top_k_accuracies'].get(5, 0),
            '测试集Top-10准确率': result['test_top_k_accuracies'].get(10, 0),
            '测试集MRR': result.get('test_mrr', 0),
            '测试集MAR': result.get('test_mar', 0),
            '测试集MFR': result.get('test_mfr', 0),
            '测试集EXAM(%)': result.get('test_exam', 0),
            'Best EXAM(%)': result.get('best_exam', 0),
            'Worst EXAM(%)': result.get('worst_exam', 0),
            '训练时间(秒)': result.get('fold_training_time', 0),
            '最大显存使用(MB)': result.get('max_memory_allocated', 0),
            '最小推理耗时(ms)': result.get('min_inference_time', 0) * 1000,
            '最大推理耗时(ms)': result.get('max_inference_time', 0) * 1000,
            '平均推理耗时(ms)': result.get('avg_inference_time', 0) * 1000,
            '语句根Top-1准确率': result.get('stmt_root_top_k_accuracies', {}).get(1, 0),
            '语句根Top-3准确率': result.get('stmt_root_top_k_accuracies', {}).get(3, 0),
            '语句根Top-5准确率': result.get('stmt_root_top_k_accuracies', {}).get(5, 0),
            '语句根Top-10准确率': result.get('stmt_root_top_k_accuracies', {}).get(10, 0),
            '语句根MRR': result.get('stmt_root_avg_mrr', 0),
            '语句根MFR': result.get('stmt_root_avg_mfr', 0),
            '语句根EXAM(%)': result.get('stmt_root_avg_exam', 0),
            '捷径程度Top-1': result.get('shortcut_degree', {}).get(1, 0),
            '捷径程度Top-3': result.get('shortcut_degree', {}).get(3, 0),
            '捷径程度Top-5': result.get('shortcut_degree', {}).get(5, 0),
            '捷径程度Top-10': result.get('shortcut_degree', {}).get(10, 0),
            '加权标签': result.get('use_weighted_labels', False),
            '语句根权重': result.get('statement_root_weight', 1.0),
            '子节点权重': result.get('child_node_weight', 0.5),
            '对比学习权重': result.get('contrastive_weight', 0.0),
            '对比学习温度': result.get('contrastive_temperature', 0.07),
            '损失调度': str(result.get('loss_schedule', None)) if result.get('loss_schedule') else '固定',
            'SBFL模式': result.get('sbfl_mode', 'normal'),
            'SBFL特征': '启用' if result.get('use_sbfl', True) else '禁用',
            '原始覆盖矩阵': '启用' if result.get('use_original_coverage_matrix', True) else '禁用',
            'SBFL公式': result.get('sbfl_formula', ''),
        }
        if 'test_top_k_std' in result:
            row['测试集Top-1标准差'] = result['test_top_k_std'].get(1, 0)
            row['测试集Top-3标准差'] = result['test_top_k_std'].get(3, 0)
            row['测试集Top-5标准差'] = result['test_top_k_std'].get(5, 0)
            row['测试集Top-10标准差'] = result['test_top_k_std'].get(10, 0)
        if 'final_test_difficulty_results' in result and result['final_test_difficulty_results']:
            diff_res = result['final_test_difficulty_results']
            for diff_level in sorted(diff_res.keys()):
                if diff_level in diff_res:
                    row[f'{diff_level}测试Top-1'] = diff_res[diff_level].get(1, 0)
                    row[f'{diff_level}测试Top-3'] = diff_res[diff_level].get(3, 0)
                    row[f'{diff_level}测试Top-5'] = diff_res[diff_level].get(5, 0)
                    row[f'{diff_level}测试Top-10'] = diff_res[diff_level].get(10, 0)
        return row

    def get_config_key(result):
        parts = [result.get('gnn_type', '')]
        if result.get('variant'):
            v = result['variant']
            parts.append('+'.join(v) if isinstance(v, list) else str(v))
        sbfl_mode = result.get('sbfl_mode', 'normal')
        if sbfl_mode and sbfl_mode != 'normal':
            parts.append(f"sbfl_{sbfl_mode}")
        sbfl_formula = result.get('sbfl_formula', '')
        if sbfl_formula:
            parts.append(str(sbfl_formula))
        sbfl_type = result.get('sbfl_type', '')
        if sbfl_type:
            parts.append(sbfl_type)
        if result.get('use_weighted_labels', False):
            parts.append('weighted')
        if result.get('contrastive_weight', 0) > 0:
            parts.append(f"cl{result['contrastive_weight']}")
        return '_'.join(parts) if parts else 'default'

    excel_data = [result_to_row(result) for result in results]

    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        df = pd.DataFrame(excel_data)
        df.to_excel(writer, index=False, sheet_name='训练结果')

        config_groups = {}
        for result in results:
            config_key = get_config_key(result)
            if config_key not in config_groups:
                config_groups[config_key] = []
            config_groups[config_key].append(result_to_row(result))

        for config_key, rows in config_groups.items():
            sheet_df = pd.DataFrame(rows)
            sheet_name = config_key[:31]
            sheet_df.to_excel(writer, index=False, sheet_name=sheet_name)

        if detailed_results:
            epoch_groups = {}
            for record in detailed_results:
                ekey = f"{record['gnn_type']}_h{record['hidden_dim']}_lr{record['lr']}"
                if record.get('sbfl_mode') and record['sbfl_mode'] != 'normal':
                    ekey += f"_{record['sbfl_mode']}"
                if record.get('sbfl_formula'):
                    ekey += f"_{record['sbfl_formula']}"
                if ekey not in epoch_groups:
                    epoch_groups[ekey] = []
                epoch_groups[ekey].append(record)

            for ekey, records in epoch_groups.items():
                epoch_records = [r for r in records if isinstance(r.get('epoch'), int)]
                if epoch_records:
                    sorted_records = sorted(epoch_records, key=lambda x: (x['fold'], x['epoch']))
                    sheet_df = pd.DataFrame(sorted_records)
                    sheet_name = ekey[:31]
                    sheet_df.to_excel(writer, index=False, sheet_name=sheet_name)

        # 检查是否有任何结果包含统计检验数据
        has_statistical_tests = any(r.get('statistical_tests') for r in results)
        if has_statistical_tests:
            stats_rows = []
            for result in results:
                model_name = result.get('model_name', '')
                tests = result.get('statistical_tests', {})
                if not tests:
                    continue
                for formula, test_result in tests.items():
                    row = {
                        '模型名称': model_name,
                        '对比方法': f"{test_result['method_a']} vs {test_result['method_b']}",
                        '样本数N': test_result['n'],
                        f"{test_result['method_a']}均值排名": test_result['mean_a'],
                        f"{test_result['method_b']}均值排名": test_result['mean_b'],
                        f"{test_result['method_a']}中位数排名": test_result['median_a'],
                        f"{test_result['method_b']}中位数排名": test_result['median_b'],
                        f"{test_result['method_a']}标准差": test_result['std_a'],
                        f"{test_result['method_b']}标准差": test_result['std_b'],
                        'Wilcoxon统计量': test_result['wilcoxon_stat'],
                        'p值': test_result['p_value'],
                        '显著性(0.05)': '是' if test_result['significant_0.05'] else '否',
                        '显著性(0.01)': '是' if test_result['significant_0.01'] else '否',
                        '显著性(0.001)': '是' if test_result['significant_0.001'] else '否',
                        'Â₁₂': test_result['a12'],
                        '效应量': test_result['effect_size'],
                        f"{test_result['method_a']}胜出次数": test_result['a_wins'],
                        f"{test_result['method_b']}胜出次数": test_result['b_wins'],
                        '平局次数': test_result['tied'],
                    }
                    stats_rows.append(row)
            if stats_rows:
                stats_df = pd.DataFrame(stats_rows)
                stats_df.to_excel(writer, index=False, sheet_name='统计检验')

        # 逐程序语句排名 Sheet（每折测试集，每个程序真实错误行的排名）
        all_stmt_rows = []
        for result in results:
            model_name = result.get('model_name', '')
            fold_ranks = result.get('fold_statement_ranks', [])
            if not fold_ranks:
                continue
            for item in fold_ranks:
                item['model_name'] = model_name
                all_stmt_rows.append(item)
        if all_stmt_rows:
            stmt_df = pd.DataFrame(all_stmt_rows)
            column_order = ['model_name', 'fold', 'bug_id', 'project', 'case',
                           'true_line', 'rank']
            stmt_df = stmt_df[[c for c in column_order if c in stmt_df.columns]]
            stmt_df.to_excel(writer, index=False, sheet_name='逐程序语句排名')

    print(f"Excel结果文件已生成: {excel_file}")
    return excel_file


def append_result_to_excel(result, detailed_records, model_dir, config_str="results"):
    results_dir = os.path.join(model_dir, "excel_tmp")
    os.makedirs(results_dir, exist_ok=True)
    pkl_file = os.path.join(results_dir, f"incremental_{config_str}.pkl")

    existing_results = []
    existing_detailed = []
    if os.path.exists(pkl_file):
        try:
            import pickle
            with open(pkl_file, 'rb') as f:
                saved = pickle.load(f)
                existing_results = saved.get('results', [])
                existing_detailed = saved.get('detailed', [])
        except:
            pass

    existing_results.append(result)
    if detailed_records:
        existing_detailed.extend(detailed_records)

    try:
        import pickle
        with open(pkl_file, 'wb') as f:
            pickle.dump({'results': existing_results, 'detailed': existing_detailed}, f)
    except:
        pass

    excel_file = save_results_to_excel(existing_results, existing_detailed, model_dir, config_str)
    return excel_file
