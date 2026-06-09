import os
import sys
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
import datetime
import time
import random
import math
from collections import Counter

from DataConfig import DataConfig
from ModelAll import create_model
from utils import (
    set_seed, worker_init_fn, GraphDataset, collate_fn,
    model_forward, evaluate_model, load_difficulty_mapping,
    get_variant_str, generate_cross_validation_splits, save_results_to_excel,
    STATEMENT_ROOT_TYPES, get_statement_root_type_ids, append_result_to_excel,
    collect_per_program_ranks, collect_per_program_statement_ranks,
    collect_sbfl_per_program_ranks,
    compute_statistical_comparison, compute_all_statistical_tests
)


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def compute_weighted_labels(labels, ast_type_ids, statement_root_type_ids,
                            statement_root_weight=1.0, child_node_weight=0.5):
    weighted = labels.clone()
    pos_mask = labels > 0
    if pos_mask.sum() == 0:
        return weighted
    stmt_root_mask = torch.zeros(len(ast_type_ids), dtype=torch.bool, device=labels.device)
    for type_id in statement_root_type_ids:
        stmt_root_mask |= (ast_type_ids == type_id)
    weighted[pos_mask & stmt_root_mask] = statement_root_weight
    weighted[pos_mask & ~stmt_root_mask] = child_node_weight
    return weighted


def statement_contrastive_loss(node_embeddings, line_numbers, labels, temperature=0.07):
    unique_lines = line_numbers.unique()
    line_reprs_mean = []
    line_reprs_max = []
    line_labels_list = []

    for ln in unique_lines:
        if ln == 0:
            continue
        mask = line_numbers == ln
        if mask.sum() == 0:
            continue
        nodes_on_line = node_embeddings[mask]
        line_reprs_mean.append(nodes_on_line.mean(dim=0))
        line_reprs_max.append(nodes_on_line.max(dim=0)[0])
        line_labels_list.append(labels[mask].max().item())

    if len(line_reprs_mean) <= 1:
        return torch.tensor(0.0, device=node_embeddings.device)

    line_reprs_mean = torch.stack(line_reprs_mean)
    line_reprs_max = torch.stack(line_reprs_max)
    line_labels_t = torch.tensor(line_labels_list, device=node_embeddings.device)

    pos_mask = line_labels_t > 0
    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=node_embeddings.device)

    line_reprs_mean = F.normalize(line_reprs_mean, dim=-1)
    line_reprs_max = F.normalize(line_reprs_max, dim=-1)

    pos_indices = pos_mask.nonzero(as_tuple=True)[0]

    total_loss = torch.tensor(0.0, device=node_embeddings.device)
    for anchor_idx in pos_indices:
        anchor = line_reprs_mean[anchor_idx]
        positive = line_reprs_max[anchor_idx]

        sim_pos = (anchor * positive).sum() / temperature
        sim_all = torch.mm(anchor.unsqueeze(0), line_reprs_max.t()).squeeze(0) / temperature

        loss = -sim_pos + torch.logsumexp(sim_all, dim=0)
        total_loss = total_loss + loss

    return total_loss / len(pos_indices)


def evaluate_pure_sbfl(processed_data, sbfl_data_path, sbfl_type='mergeSBFL1.01', sbfl_formula=None,
                       top_ks=None, difficulties=None, difficulty_levels=None):
    if top_ks is None:
        top_ks = [1, 3, 5, 10]

    top_k_accuracies = {k: 0 for k in top_ks}
    total_samples = 0
    total_mrr = 0
    total_mfr = 0
    total_exam = 0
    exam_scores = []

    if difficulty_levels is None:
        difficulty_levels = ['0-10', '10-100', '100-500', '500-1000', '1000-2000', '>2000']
    difficulty_top_k = {level: {k: 0 for k in top_ks} for level in difficulty_levels}
    difficulty_counts = {level: 0 for level in difficulty_levels}

    for sample_idx, item in enumerate(processed_data):
        fault_lines = item.get('fault_lines', [])
        if not fault_lines:
            continue
        true_line_numbers = fault_lines

        bug_id = item.get('bug_id', '')
        parts = bug_id.rsplit('_', 1)
        if len(parts) != 2:
            continue
        project, case = parts[0], parts[1]

        sbfl_scores_by_line = {}

        if sbfl_formula:
            formula_path = os.path.join(sbfl_data_path, project, 'Python', case, f"{sbfl_formula}.txt")
            if not os.path.exists(formula_path):
                continue
            try:
                with open(formula_path, 'r') as f:
                    for line_idx, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        if '→' in line:
                            scores_str = line.split('→')[1]
                        else:
                            scores_str = line
                        val = json.loads(scores_str)
                        score = val if isinstance(val, (int, float)) else (max(val) if isinstance(val, list) and val else 0.0)
                        code_line = line_idx + 1
                        sbfl_scores_by_line[code_line] = max(sbfl_scores_by_line.get(code_line, 0), score)
            except:
                continue
        else:
            sbfl_features = item.get('sbfl_features', [])
            for idx, scores in enumerate(sbfl_features):
                code_line = idx + 1
                if isinstance(scores, list):
                    max_score = max(scores) if scores else 0.0
                else:
                    max_score = scores
                sbfl_scores_by_line[code_line] = max(sbfl_scores_by_line.get(code_line, 0), max_score)

        if not sbfl_scores_by_line:
            continue

        sorted_lines = sorted(sbfl_scores_by_line.keys(),
                              key=lambda x: sbfl_scores_by_line[x], reverse=True)
        total_lines = len(sorted_lines)

        ranks = []
        for ln in true_line_numbers:
            if ln in sorted_lines:
                rank = sorted_lines.index(ln) + 1
                ranks.append(rank)

        if not ranks:
            continue

        best_rank = min(ranks)
        total_samples += 1
        total_mrr += 1.0 / best_rank
        total_mfr += best_rank
        exam = (best_rank / total_lines * 100) if total_lines > 0 else 100
        total_exam += exam
        exam_scores.append(exam)

        for k in top_ks:
            if best_rank <= k:
                top_k_accuracies[k] += 1

        if difficulties is not None and sample_idx < len(difficulties):
            diff_level = difficulties[sample_idx]
            if diff_level in difficulty_counts:
                difficulty_counts[diff_level] += 1
                for k in top_ks:
                    if best_rank <= k:
                        difficulty_top_k[diff_level][k] = difficulty_top_k[diff_level].get(k, 0) + 1

    if total_samples == 0:
        return None

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

    result = {
        'top_k_accuracies': {k: top_k_accuracies[k] / total_samples for k in top_ks},
        'avg_mrr': total_mrr / total_samples,
        'avg_mfr': total_mfr / total_samples,
        'avg_exam': total_exam / total_samples,
        'best_exam': min(exam_scores) if exam_scores else 0,
        'worst_exam': max(exam_scores) if exam_scores else 0,
        'total_samples': total_samples,
        'sbfl_type': sbfl_type,
        'sbfl_formula': sbfl_formula,
        'difficulty_results': difficulty_results,
    }
    return result


def train_model(model, input_dim, hidden_dim, lr, epochs, gnn_type, num_heads,
                train_dataloader, val_dataloader, test_dataloader,
                training_config, exp_config, data_config,
                fold=None, save_dir=None, val_difficulties=None, test_difficulties=None,
                statement_root_type_ids=None, difficulty_levels=None,
                test_dataset_processed=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = training_config.get('seed', 0)
    set_seed(seed)

    model.to(device)

    patience = training_config.get('patience', 30)
    use_validation = training_config.get('use_validation', True)
    save_model = training_config.get('save_model', True)
    weight_decay = training_config.get('weight_decay', 0.001)
    loss_alpha = training_config.get('loss_alpha', 0.7)
    loss_beta = training_config.get('loss_beta', 0.3)
    loss_schedule = training_config.get('loss_schedule', None)
    l1_lambda = training_config.get('l1_lambda', 0.0005)
    scheduler_T0 = training_config.get('scheduler_T0', 20)
    scheduler_T_mult = training_config.get('scheduler_T_mult', 2)
    grad_clip = training_config.get('grad_clip', 1.0)
    model_dir = training_config.get('model_dir', 'model')

    use_weighted_labels = exp_config.get('use_weighted_labels', False)
    statement_root_weight = exp_config.get('statement_root_weight', 1.0)
    child_node_weight = exp_config.get('child_node_weight', 0.5)
    contrastive_weight = exp_config.get('contrastive_weight', 0.0)
    contrastive_temperature = exp_config.get('contrastive_temperature', 0.07)
    ranking_weight = exp_config.get('ranking_weight', 0.0)
    ranking_margin = exp_config.get('ranking_margin', 0.1)

    num_sbfl = exp_config.get('num_sbfl')
    num_lexical = exp_config.get('num_lexical')
    variant = exp_config.get('variant')
    fusion_norm_type = exp_config.get('fusion_norm_type', 'none')
    edge_gate_bias_init = exp_config.get('edge_gate_bias_init', -1)
    edge_gate_l1_lambda = exp_config.get('edge_gate_l1_lambda', 0.01)
    fusion_type = exp_config.get('fusion_type', 'concat')
    use_cross_attn_fusion = exp_config.get('use_cross_attn_fusion', False)
    cross_attn_heads = exp_config.get('cross_attn_heads', 4)
    edge_gate_lr_scale = exp_config.get('edge_gate_lr_scale', 0.1)

    def get_param_stats(m):
        param_counts = []
        for name, param in m.named_parameters():
            if param.requires_grad:
                param_counts.append(param.numel())
        if param_counts:
            return {
                'total_params': sum(param_counts),
                'min_params': min(param_counts),
                'max_params': max(param_counts),
                'avg_params': sum(param_counts) / len(param_counts),
                'num_layers': len(param_counts)
            }
        return {'total_params': 0, 'min_params': 0, 'max_params': 0, 'avg_params': 0, 'num_layers': 0}

    param_stats = get_param_stats(model)
    print(f"模型参数统计: 总参数={param_stats['total_params']:,}")

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    edge_gate_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'edge_type_gate' in name or 'edge_type_embeddings' in name:
                edge_gate_params.append(param)
            else:
                other_params.append(param)

    if edge_gate_params and edge_gate_lr_scale != 1.0:
        optimizer = Adam([
            {'params': other_params},
            {'params': edge_gate_params, 'lr': lr * edge_gate_lr_scale},
        ], lr=lr, weight_decay=weight_decay)
        print(f"边门控参数使用独立学习率: {lr * edge_gate_lr_scale:.6f} (缩放: {edge_gate_lr_scale})")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=scheduler_T0, T_mult=scheduler_T_mult, eta_min=lr * 0.01
    )

    def l1_loss(m, lambda_l1=l1_lambda):
        return lambda_l1 * sum(p.abs().sum() for p in m.parameters())

    def listwise_loss(logits, labels):
        scores = logits - logits.max(dim=0, keepdim=True)[0]
        p = F.softmax(scores, dim=0)
        return -torch.sum(labels.float() * torch.log(p + 1e-12))

    def bce_loss(logits, labels):
        return F.binary_cross_entropy_with_logits(logits, labels.float())

    def pairwise_ranking_loss(logits, labels, margin=0.1):
        pos_mask = labels > 0
        neg_mask = labels == 0
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        pos_scores = logits[pos_mask]
        neg_scores = logits[neg_mask]
        diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
        loss = F.relu(margin - diff).mean()
        return loss

    def get_scheduled_loss_weights(epoch_idx, total_epochs, base_alpha, base_beta, schedule_cfg):
        if schedule_cfg is None:
            return base_alpha, base_beta
        mode = schedule_cfg.get('mode', 'linear')
        alpha_start = schedule_cfg.get('alpha_start', base_alpha)
        alpha_end = schedule_cfg.get('alpha_end', base_alpha)
        beta_start = schedule_cfg.get('beta_start', base_beta)
        beta_end = schedule_cfg.get('beta_end', base_beta)
        warmup_epochs = schedule_cfg.get('warmup_epochs', 0)
        if epoch_idx < warmup_epochs:
            return alpha_start, beta_start
        if total_epochs <= warmup_epochs:
            progress = 1.0
        else:
            progress = (epoch_idx - warmup_epochs) / (total_epochs - warmup_epochs)
            progress = min(max(progress, 0.0), 1.0)
        if mode == 'linear':
            alpha = alpha_start + (alpha_end - alpha_start) * progress
            beta = beta_start + (beta_end - beta_start) * progress
        elif mode == 'cosine':
            cos_progress = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
            alpha = alpha_start + (alpha_end - alpha_start) * cos_progress
            beta = beta_start + (beta_end - beta_start) * cos_progress
        else:
            alpha = base_alpha
            beta = base_beta
        return alpha, beta

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    variant_str = get_variant_str(variant)
    if fold is not None:
        model_name = f"{gnn_type}{variant_str}_h{hidden_dim}_lr{lr}_fold{fold}_e{epochs}_{timestamp}.pth"
    else:
        model_name = f"{gnn_type}{variant_str}_h{hidden_dim}_lr{lr}_e{epochs}_{timestamp}.pth"

    actual_save_dir = save_dir if save_dir is not None else model_dir
    os.makedirs(actual_save_dir, exist_ok=True)
    best_model_path = os.path.join(actual_save_dir, model_name)

    print(f"开始训练模型: {model_name}")

    best_val_top1_accuracy = 0.0
    early_stopping_counter = 0

    train_losses = []
    train_top1_accuracies = []
    val_losses = []
    val_top1_accuracies = []
    test_top1_accuracies = []
    test_mrrs = []
    test_mars = []
    test_mfrs = []

    epoch_records = []

    best_val_top1_for_model = 0
    best_val_epoch = 0
    best_val_model_state = None

    fold_start_time = time.time()
    max_memory_allocated = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(epochs):
        cur_alpha, cur_beta = get_scheduled_loss_weights(epoch, epochs, loss_alpha, loss_beta, loss_schedule)
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            optimizer.zero_grad()
            batch_loss = 0

            features_list, ast_type_ids_list, token_ids_list, edge_index_list, edge_type_ids_list, labels_list, num_nodes_list, line_numbers_list, adj_list = batch

            for i in range(len(features_list)):
                features = features_list[i].to(device)
                ast_type_ids = ast_type_ids_list[i].to(device)
                token_ids = token_ids_list[i].to(device)
                edges = edge_index_list[i].to(device)
                edge_type_ids = edge_type_ids_list[i].to(device)
                labels = labels_list[i].to(device)
                nodes = num_nodes_list[i]
                line_numbers = line_numbers_list[i].to(device)
                adj = adj_list[i].to(device) if adj_list[i].numel() > 0 else None

                original_labels = labels.clone()

                if use_weighted_labels and statement_root_type_ids:
                    labels = compute_weighted_labels(
                        labels, ast_type_ids, statement_root_type_ids,
                        statement_root_weight, child_node_weight
                    )

                out, h = model_forward(model, features, ast_type_ids, edges, nodes, num_sbfl, num_lexical, token_ids=token_ids, edge_type_ids=edge_type_ids, adj=adj)

                logits = out.squeeze(-1) if out.dim() > 1 else out.squeeze()
                loss = cur_alpha * listwise_loss(logits, labels.float()) + \
                       cur_beta * bce_loss(logits, labels.float()) + \
                       l1_loss(model)

                if ranking_weight > 0:
                    loss = loss + ranking_weight * pairwise_ranking_loss(
                        logits, original_labels.float(), ranking_margin
                    )

                if hasattr(model, 'get_edge_gate_l1_loss'):
                    loss = loss + model.get_edge_gate_l1_loss()

                if contrastive_weight > 0:
                    c_loss = statement_contrastive_loss(
                        h, line_numbers, original_labels,
                        temperature=contrastive_temperature
                    )
                    loss = loss + contrastive_weight * c_loss

                batch_loss += loss
                total_loss += loss.item()

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            if torch.cuda.is_available():
                current_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
                if current_memory > max_memory_allocated:
                    max_memory_allocated = current_memory

        train_avg_loss = total_loss / len(train_dataloader)
        train_losses.append(train_avg_loss)

        train_eval = evaluate_model(model, train_dataloader, [1, 3, 5, 10], num_sbfl=num_sbfl, num_lexical=num_lexical, statement_root_type_ids=statement_root_type_ids, difficulty_levels=difficulty_levels)
        train_top1 = train_eval['top_k_accuracies'][1]
        train_top1_accuracies.append(train_top1)

        if use_validation:
            model.eval()
            val_total_loss = 0

            with torch.no_grad():
                for batch in val_dataloader:
                    features_list, ast_type_ids_list, token_ids_list, edge_index_list, edge_type_ids_list, labels_list, num_nodes_list, line_numbers_list, adj_list = batch
                    for i in range(len(features_list)):
                        features = features_list[i].to(device)
                        ast_type_ids = ast_type_ids_list[i].to(device)
                        token_ids = token_ids_list[i].to(device)
                        edges = edge_index_list[i].to(device)
                        edge_type_ids = edge_type_ids_list[i].to(device)
                        labels = labels_list[i].to(device)
                        nodes = num_nodes_list[i]
                        line_numbers = line_numbers_list[i].to(device)
                        adj = adj_list[i].to(device) if adj_list[i].numel() > 0 else None

                        original_labels = labels.clone()

                        if use_weighted_labels and statement_root_type_ids:
                            labels = compute_weighted_labels(
                                labels, ast_type_ids, statement_root_type_ids,
                                statement_root_weight, child_node_weight
                            )

                        out, h = model_forward(model, features, ast_type_ids, edges, nodes, num_sbfl, num_lexical, token_ids=token_ids, edge_type_ids=edge_type_ids, adj=adj)
                        logits = out.squeeze(-1) if out.dim() > 1 else out.squeeze()
                        loss = cur_alpha * listwise_loss(logits, labels.float()) + \
                               cur_beta * bce_loss(logits, labels.float()) + \
                               l1_loss(model)

                        if hasattr(model, 'get_edge_gate_l1_loss'):
                            loss = loss + model.get_edge_gate_l1_loss()

                        if contrastive_weight > 0:
                            c_loss = statement_contrastive_loss(
                                h, line_numbers, original_labels,
                                temperature=contrastive_temperature
                            )
                            loss = loss + contrastive_weight * c_loss

                        val_total_loss += loss.item()

            val_avg_loss = val_total_loss / len(val_dataloader)
            val_losses.append(val_avg_loss)

            val_eval = evaluate_model(model, val_dataloader, [1, 3, 5, 10],
                                      difficulties=val_difficulties, num_sbfl=num_sbfl, num_lexical=num_lexical,
                                      statement_root_type_ids=statement_root_type_ids, difficulty_levels=difficulty_levels)
            val_top1 = val_eval['top_k_accuracies'][1]
            val_top1_accuracies.append(val_top1)

            print(f'Epoch {epoch + 1}/{epochs}:')
            if loss_schedule is not None:
                print(f'  损失权重: alpha={cur_alpha:.4f}, beta={cur_beta:.4f}')
            print(f'  训练集: Loss {train_avg_loss:.4f}, Top-1: {train_eval["top_k_accuracies"][1]:.4f}, Top-3: {train_eval["top_k_accuracies"][3]:.4f}, Top-5: {train_eval["top_k_accuracies"][5]:.4f}, Top-10: {train_eval["top_k_accuracies"][10]:.4f}')
            print(f'  验证集: Loss {val_avg_loss:.4f}, Top-1: {val_eval["top_k_accuracies"][1]:.4f}, Top-3: {val_eval["top_k_accuracies"][3]:.4f}, Top-5: {val_eval["top_k_accuracies"][5]:.4f}, Top-10: {val_eval["top_k_accuracies"][10]:.4f}')

            if val_top1 >= best_val_top1_accuracy:
                best_val_top1_accuracy = val_top1
                if save_model:
                    torch.save(model.state_dict(), best_model_path)
                    print(f"验证集性能提升，保存最佳模型，Top-1: {val_top1:.4f}")
                early_stopping_counter = 0

            if val_top1 >= best_val_top1_for_model:
                best_val_top1_for_model = val_top1
                best_val_epoch = epoch
                best_val_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                early_stopping_counter += 1
                print(f"早停计数器: {early_stopping_counter}/{patience}")
        else:
            print(f'Epoch {epoch + 1}/{epochs}:', end='')
            if loss_schedule is not None:
                print(f' alpha={cur_alpha:.4f}, beta={cur_beta:.4f},', end='')
            print(f' 训练集 Loss {train_avg_loss:.4f}, Top-1: {train_eval["top_k_accuracies"][1]:.4f}, Top-3: {train_eval["top_k_accuracies"][3]:.4f}, Top-5: {train_eval["top_k_accuracies"][5]:.4f}, Top-10: {train_eval["top_k_accuracies"][10]:.4f}')
            val_losses.append(0)
            val_top1_accuracies.append(0)

        test_eval = evaluate_model(model, test_dataloader, [1, 3, 5, 10],
                                   difficulties=test_difficulties, num_sbfl=num_sbfl, num_lexical=num_lexical,
                                   statement_root_type_ids=statement_root_type_ids, difficulty_levels=difficulty_levels)
        test_top1 = test_eval['top_k_accuracies'][1]
        test_top1_accuracies.append(test_top1)
        test_mrrs.append(test_eval['avg_mrr'])
        test_mars.append(test_eval['avg_mar'])
        test_mfrs.append(test_eval['avg_mfr'])
        print(f'  测试集 Top-1: {test_eval["top_k_accuracies"][1]:.4f}, Top-3: {test_eval["top_k_accuracies"][3]:.4f}, Top-5: {test_eval["top_k_accuracies"][5]:.4f}, Top-10: {test_eval["top_k_accuracies"][10]:.4f}, MRR: {test_eval["avg_mrr"]:.4f}, MFR: {test_eval["avg_mfr"]:.4f}, MAR: {test_eval["avg_mar"]:.4f}')

        epoch_record = {
            'fold': fold if fold is not None else 0,
            'epoch': epoch + 1,
            'gnn_type': gnn_type,
            'hidden_dim': hidden_dim,
            'lr': lr,
            'loss_alpha': cur_alpha,
            'loss_beta': cur_beta,
            'train_loss': train_avg_loss,
            'train_top1': train_eval['top_k_accuracies'][1],
            'train_top3': train_eval['top_k_accuracies'][3],
            'train_top5': train_eval['top_k_accuracies'][5],
            'train_top10': train_eval['top_k_accuracies'][10],
            'train_mrr': train_eval['avg_mrr'],
            'train_mfr': train_eval['avg_mfr'],
            'train_mar': train_eval['avg_mar'],
        }
        if use_validation:
            epoch_record.update({
                'val_loss': val_avg_loss,
                'val_top1': val_eval['top_k_accuracies'][1],
                'val_top3': val_eval['top_k_accuracies'][3],
                'val_top5': val_eval['top_k_accuracies'][5],
                'val_top10': val_eval['top_k_accuracies'][10],
                'val_mrr': val_eval['avg_mrr'],
                'val_mfr': val_eval['avg_mfr'],
                'val_mar': val_eval['avg_mar'],
            })
        epoch_record.update({
            'test_top1': test_eval['top_k_accuracies'][1],
            'test_top3': test_eval['top_k_accuracies'][3],
            'test_top5': test_eval['top_k_accuracies'][5],
            'test_top10': test_eval['top_k_accuracies'][10],
            'test_mrr': test_eval['avg_mrr'],
            'test_mfr': test_eval['avg_mfr'],
            'test_mar': test_eval['avg_mar'],
            'test_exam': test_eval['avg_exam'],
        })
        epoch_records.append(epoch_record)

        if use_validation and early_stopping_counter >= patience:
            print(f"早停触发！验证集性能在{patience}个epoch内没有提升。")
            break

        if use_validation:
            scheduler.step()

    if use_validation and best_val_model_state is not None:
        print(f"\n加载验证集最优模型 (Epoch {best_val_epoch + 1}) 进行最终评估")
        model.load_state_dict(best_val_model_state)
    elif save_model and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))

    final_eval = evaluate_model(model, test_dataloader, [1, 3, 5, 10],
                                difficulties=test_difficulties, num_sbfl=num_sbfl, num_lexical=num_lexical,
                                statement_root_type_ids=statement_root_type_ids, difficulty_levels=difficulty_levels)

    # 收集逐程序排名数据（用于统计检验）
    per_program_ranks = {}
    per_statement_ranks = {}
    if test_dataset_processed is not None:
        per_program_ranks = collect_per_program_ranks(
            model, test_dataset_processed, device, num_sbfl=num_sbfl, num_lexical=num_lexical
        )
        per_statement_ranks = collect_per_program_statement_ranks(
            model, test_dataset_processed, device, num_sbfl=num_sbfl, num_lexical=num_lexical
        )

    fold_end_time = time.time()
    fold_training_time = fold_end_time - fold_start_time

    return {
        'model_name': model_name,
        'hidden_dim': hidden_dim,
        'lr': lr,
        'epochs': epochs,
        'gnn_type': gnn_type,
        'num_heads': num_heads,
        'variant': variant,
        'best_val_top1_accuracy': best_val_top1_accuracy,
        'test_top_k_accuracies': final_eval['top_k_accuracies'],
        'test_mrr': final_eval['avg_mrr'],
        'test_mar': final_eval['avg_mar'],
        'test_mfr': final_eval['avg_mfr'],
        'test_exam': final_eval['avg_exam'],
        'best_exam': final_eval['best_exam'],
        'worst_exam': final_eval['worst_exam'],
        'min_inference_time': final_eval['min_inference_time'],
        'max_inference_time': final_eval['max_inference_time'],
        'avg_inference_time': final_eval['avg_inference_time'],
        'final_test_difficulty_results': final_eval['difficulty_results'],
        'stmt_root_top_k_accuracies': final_eval['stmt_root_top_k_accuracies'],
        'stmt_root_avg_mrr': final_eval['stmt_root_avg_mrr'],
        'stmt_root_avg_mfr': final_eval['stmt_root_avg_mfr'],
        'stmt_root_avg_exam': final_eval['stmt_root_avg_exam'],
        'stmt_root_sample_count': final_eval['stmt_root_sample_count'],
        'shortcut_degree': final_eval['shortcut_degree'],
        'use_weighted_labels': use_weighted_labels,
        'statement_root_weight': statement_root_weight,
        'child_node_weight': child_node_weight,
        'contrastive_weight': contrastive_weight,
        'contrastive_temperature': contrastive_temperature,
        'loss_schedule': loss_schedule,
        'train_losses': train_losses,
        'train_top1_accuracies': train_top1_accuracies,
        'val_losses': val_losses,
        'val_top1_accuracies': val_top1_accuracies,
        'test_top1_accuracies': test_top1_accuracies,
        'test_mrrs': test_mrrs,
        'test_mars': test_mars,
        'test_mfrs': test_mfrs,
        'epoch_records': epoch_records,
        'fold_training_time': fold_training_time,
        'max_memory_allocated': max_memory_allocated,
        'per_program_ranks': per_program_ranks,
        'per_statement_ranks': per_statement_ranks,
        'best_model_path': best_model_path,
        'best_val_epoch': best_val_epoch if use_validation else 0,
        'total_params': param_stats['total_params'],
    }


def run_experiment(exp_config, data_config_obj, training_config, difficulty_map):
    gnn_type = exp_config['gnn_type']
    hidden_dim = exp_config.get('hidden_dim', 128)
    lr = exp_config.get('lr', 0.001)
    epochs = exp_config.get('epochs', 120)
    num_heads = exp_config.get('num_heads', 5)
    cv_method = exp_config.get('cv_method', 'kfold')
    k_folds = exp_config.get('k_folds', 5)
    sbfl_type = exp_config.get('sbfl_type', training_config.get('sbfl_type', 'mergeSBFLALL'))
    variant = exp_config.get('variant')
    num_steps = exp_config.get('num_steps', 5)
    dropout = exp_config.get('dropout', 0.3)
    ast_emb_dim = exp_config.get('ast_emb_dim', 16)
    num_layers = exp_config.get('num_layers', 5)
    alpha = exp_config.get('alpha', 0.2)
    token_emb_dim = exp_config.get('token_emb_dim', 16)
    use_edge_type = exp_config.get('use_edge_type', False)
    seed = training_config.get('seed', 0)
    batch_size = training_config.get('batch_size', 60)
    model_dir = training_config.get('model_dir', 'model')

    print(f"\n{'=' * 60}")
    print(f"实验: {exp_config.get('name', 'unnamed')}")
    print(f"配置: gnn_type={gnn_type}, hidden_dim={hidden_dim}, lr={lr}, epochs={epochs}")
    if variant:
        print(f"变体: {variant}")
    print(f"交叉验证: {cv_method}, k_folds={k_folds}")
    print(f"{'=' * 60}")

    sbfl_mode = exp_config.get('sbfl_mode', None)
    sbfl_formula = exp_config.get('sbfl_formula', None)
    stratify_by = exp_config.get('stratify_by', 'difficulty')
    data_config_obj.load_dataset(sbfl_type=sbfl_type, sbfl_mode=sbfl_mode, sbfl_formula=sbfl_formula)
    processed_data = data_config_obj.processed

    if len(processed_data) == 0:
        print("警告: 数据集为空，跳过此配置")
        return None

    input_dim = processed_data[0]['features'].shape[1]
    num_ast_types = len(data_config_obj.ast_type_vocab) + 1
    num_sbfl = processed_data[0].get('num_sbfl')
    num_lexical = processed_data[0].get('num_lexical')
    num_token_types = len(data_config_obj.token_vocab) + 1
    use_fusion_norm = exp_config.get('use_fusion_norm', False)
    fusion_norm_type = exp_config.get('fusion_norm_type', 'none')
    edge_gate_bias_init = exp_config.get('edge_gate_bias_init', -1)
    edge_gate_l1_lambda = exp_config.get('edge_gate_l1_lambda', 0.01)
    fusion_type = exp_config.get('fusion_type', 'concat')
    use_cross_attn_fusion = exp_config.get('use_cross_attn_fusion', False)
    cross_attn_heads = exp_config.get('cross_attn_heads', 4)
    edge_gate_lr_scale = exp_config.get('edge_gate_lr_scale', 0.1)

    statement_root_type_ids = get_statement_root_type_ids(data_config_obj.ast_type_vocab)
    print(f"语句根节点类型数: {len(statement_root_type_ids)}, 类型ID: {sorted(statement_root_type_ids)}")

    stmt_count_boundaries = exp_config.get('stmt_count_boundaries', None)
    stmt_count_num_groups = exp_config.get('stmt_count_num_groups', 5)

    stmt_length_map = data_config_obj.load_stmt_length_map()
    stmt_counts = []
    for item in processed_data:
        bug_id = item.get('bug_id', '')
        stmt_len = stmt_length_map.get(bug_id, None)
        if stmt_len is None:
            stmt_len = item.get('num_nodes', 0)
        stmt_counts.append(stmt_len)

    if stratify_by == 'stmt_count':
        if stmt_count_boundaries is not None:
            stmt_count_boundaries = list(stmt_count_boundaries)
        else:
            stmt_count_boundaries = DataConfig.compute_adaptive_stmt_boundaries(
                stmt_counts, stmt_count_num_groups
            )

    if sbfl_mode == 'pure_sbfl':
        print("\n=== 纯SBFL基线评估（无GNN） ===")
        sbfl_formula_name = sbfl_formula if sbfl_formula else sbfl_type

        if stratify_by == 'stmt_count':
            sbfl_stratify_levels = [
                DataConfig.get_stmt_count_level(sc, stmt_count_boundaries) for sc in stmt_counts
            ]
        else:
            sbfl_difficulty_values = []
            for item in processed_data:
                bug_id = item.get('bug_id', '')
                base_bug_id = '_'.join(bug_id.split('_')[:2])
                if base_bug_id in difficulty_map:
                    sbfl_difficulty_values.append(difficulty_map[base_bug_id])
                else:
                    sbfl_difficulty_values.append(None)
            sbfl_stratify_levels = [
                DataConfig.get_difficulty_level(v) if v is not None else '10-100'
                for v in sbfl_difficulty_values
            ]

        pure_sbfl_result = evaluate_pure_sbfl(
            processed_data, data_config_obj.sbfl_data_path, sbfl_type=sbfl_type, sbfl_formula=sbfl_formula,
            top_ks=[1, 3, 5, 10], difficulties=sbfl_stratify_levels,
            difficulty_levels=sorted(set(sbfl_stratify_levels))
        )
        if pure_sbfl_result is None:
            print("纯SBFL评估失败：无有效样本")
            return None

        print(f"纯SBFL ({sbfl_formula_name}) Top-1: {pure_sbfl_result['top_k_accuracies'][1]:.4f}, "
              f"Top-3: {pure_sbfl_result['top_k_accuracies'][3]:.4f}, "
              f"Top-5: {pure_sbfl_result['top_k_accuracies'][5]:.4f}, "
              f"Top-10: {pure_sbfl_result['top_k_accuracies'][10]:.4f}")
        print(f"MRR: {pure_sbfl_result['avg_mrr']:.4f}, MFR: {pure_sbfl_result['avg_mfr']:.4f}, EXAM: {pure_sbfl_result['avg_exam']:.2f}%")

        return {
            'model_name': f'PureSBFL_{sbfl_formula_name}',
            'gnn_type': 'pure_sbfl',
            'hidden_dim': 0,
            'lr': 0,
            'epochs': 0,
            'num_heads': 0,
            'variant': f'sbfl_{sbfl_formula_name}',
            'best_val_top1_accuracy': 0,
            'test_top_k_accuracies': pure_sbfl_result['top_k_accuracies'],
            'test_mrr': pure_sbfl_result['avg_mrr'],
            'test_mar': 0,
            'test_mfr': pure_sbfl_result['avg_mfr'],
            'test_exam': pure_sbfl_result['avg_exam'],
            'best_exam': pure_sbfl_result['best_exam'],
            'worst_exam': pure_sbfl_result['worst_exam'],
            'min_inference_time': 0,
            'max_inference_time': 0,
            'avg_inference_time': 0,
            'final_test_difficulty_results': pure_sbfl_result.get('difficulty_results', {}),
            'stmt_root_top_k_accuracies': {k: 0 for k in [1, 3, 5, 10]},
            'stmt_root_avg_mrr': 0,
            'stmt_root_avg_mfr': 0,
            'stmt_root_avg_exam': 0,
            'stmt_root_sample_count': 0,
            'shortcut_degree': {k: 0 for k in [1, 3, 5, 10]},
            'use_weighted_labels': False,
            'statement_root_weight': 1.0,
            'child_node_weight': 0.5,
            'contrastive_weight': 0.0,
            'contrastive_temperature': 0.07,
            'stratify_by': stratify_by,
            'stmt_count_boundaries': None,
            'sbfl_mode': sbfl_mode,
            'sbfl_formula': sbfl_formula_name,
            'use_sbfl': data_config_obj.use_sbfl,
            'use_original_coverage_matrix': data_config_obj.use_original_coverage_matrix,
            'train_losses': [],
            'train_top1_accuracies': [],
            'val_losses': [],
            'val_top1_accuracies': [],
            'test_top1_accuracies': [pure_sbfl_result['top_k_accuracies'][1]],
            'test_mrrs': [pure_sbfl_result['avg_mrr']],
            'test_mars': [0],
            'test_mfrs': [pure_sbfl_result['avg_mfr']],
            'fold_training_time': 0,
            'max_memory_allocated': 0,
            'best_model_path': '',
            'best_val_epoch': 0,
            'total_params': 0,
        }

    model = create_model(
        gnn_type, input_dim,
        num_ast_types=num_ast_types,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_steps=num_steps,
        dropout=dropout,
        ast_emb_dim=ast_emb_dim,
        variant=variant,
        num_sbfl=num_sbfl,
        num_lexical=num_lexical,
        alpha=alpha,
        num_layers=num_layers,
        num_token_types=num_token_types,
        token_emb_dim=token_emb_dim,
        use_edge_type=use_edge_type,
        use_fusion_norm=use_fusion_norm,
        fusion_norm_type=fusion_norm_type,
        edge_gate_bias_init=edge_gate_bias_init,
        edge_gate_l1_lambda=edge_gate_l1_lambda,
        fusion_type=fusion_type,
        use_cross_attn_fusion=use_cross_attn_fusion,
        cross_attn_heads=cross_attn_heads,
    )

    dataset = GraphDataset(processed_data)
    indices = np.array(range(len(dataset)))
    bug_ids = [item['bug_id'] for item in processed_data]

    difficulty_values = []
    for bug_id in bug_ids:
        base_bug_id = '_'.join(bug_id.split('_')[:2])
        if base_bug_id in difficulty_map:
            difficulty_values.append(difficulty_map[base_bug_id])
        else:
            difficulty_values.append(None)

    difficulty_levels = [
        DataConfig.get_difficulty_level(v) if v is not None else '10-100'
        for v in difficulty_values
    ]

    difficulty_adaptive_num_groups = exp_config.get('difficulty_adaptive_num_groups', 5)

    if stratify_by == 'stmt_count':
        stratify_levels = [
            DataConfig.get_stmt_count_level(sc, stmt_count_boundaries) for sc in stmt_counts
        ]
        print(f"分层方式: 语句数量, 分界点: {stmt_count_boundaries}")
        data_config_obj.stratify_data_by_statement_count(
            boundaries=stmt_count_boundaries, num_groups=stmt_count_num_groups,
            stmt_counts=stmt_counts
        )
    elif stratify_by == 'difficulty_adaptive':
        diff_boundaries = DataConfig.compute_adaptive_difficulty_boundaries(
            difficulty_values, difficulty_adaptive_num_groups
        )
        stratify_levels = [
            DataConfig.get_difficulty_level_adaptive(v, diff_boundaries) if v is not None else f'<{diff_boundaries[0]}' if diff_boundaries else '10-100'
            for v in difficulty_values
        ]
        print(f"分层方式: 难度(自适应), 分界点: {diff_boundaries}")
        data_config_obj.stratify_data_by_difficulty_adaptive(
            difficulty_values, diff_boundaries
        )
    else:
        stratify_levels = difficulty_levels
        print(f"分层方式: 难度(固定)")

    unified_splits = generate_cross_validation_splits(
        indices, bug_ids, stratify_levels, cv_method, k_folds, seed
    )

    K_FOLDS = len(unified_splits)
    fold_results = []
    all_epoch_records = []
    all_test_topk = {1: [], 3: [], 5: [], 10: []}
    all_val_top1 = []
    all_training_times = []
    all_memory_usage = []
    all_test_mrr = []
    all_test_mar = []
    all_test_mfr = []
    all_test_exam = []
    all_best_exam = []
    all_worst_exam = []
    all_min_inference_time = []
    all_max_inference_time = []
    all_avg_inference_time = []
    all_total_params = []

    all_stmt_root_topk = {1: [], 3: [], 5: [], 10: []}
    all_stmt_root_mrr = []
    all_stmt_root_mfr = []
    all_stmt_root_exam = []
    all_shortcut_degree = {1: [], 3: [], 5: [], 10: []}

    all_difficulty_topk = {}
    unique_stratify_levels = sorted(set(stratify_levels))
    for diff_level in unique_stratify_levels:
        all_difficulty_topk[diff_level] = {1: [], 3: [], 5: [], 10: []}

    for fold, (train_idx, val_idx, test_idx, current_project) in enumerate(unified_splits):
        if cv_method == 'leave_one_out':
            print(f"\n--- Fold {fold + 1}/{K_FOLDS} (Leave-One-Out, 测试项目: {current_project}) ---")
        else:
            print(f"\n--- Fold {fold + 1}/{K_FOLDS} ---")

        if len(train_idx) < 1 or len(val_idx) == 0:
            print(f"  跳过此折：数据不足")
            continue

        train_level_counts = {lv: 0 for lv in unique_stratify_levels}
        val_level_counts = {lv: 0 for lv in unique_stratify_levels}
        test_level_counts = {lv: 0 for lv in unique_stratify_levels}
        for i in train_idx:
            train_level_counts[stratify_levels[i]] += 1
        for i in val_idx:
            val_level_counts[stratify_levels[i]] += 1
        for i in test_idx:
            test_level_counts[stratify_levels[i]] += 1
        print(f"  {'层级':15s} {'训练集':>6s} {'验证集':>6s} {'测试集':>6s}")
        for lv in unique_stratify_levels:
            print(f"  {lv:15s} {train_level_counts[lv]:6d} {val_level_counts[lv]:6d} {test_level_counts[lv]:6d}")
        print(f"  {'总计':15s} {len(train_idx):6d} {len(val_idx):6d} {len(test_idx):6d}")

        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        test_dataset = Subset(dataset, test_idx)

        val_difficulties = [stratify_levels[i] for i in val_idx]
        test_difficulties = [stratify_levels[i] for i in test_idx]

        _worker_init = lambda wid: worker_init_fn(wid, seed)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      collate_fn=collate_fn, worker_init_fn=_worker_init,
                                      num_workers=0, generator=torch.Generator().manual_seed(seed))
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                    collate_fn=collate_fn, worker_init_fn=_worker_init, num_workers=0)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                     collate_fn=collate_fn, worker_init_fn=_worker_init, num_workers=0)

        fold_model = create_model(
            gnn_type, input_dim,
            num_ast_types=num_ast_types,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_steps=num_steps,
            dropout=dropout,
            ast_emb_dim=ast_emb_dim,
            variant=variant,
            num_sbfl=num_sbfl,
            num_lexical=num_lexical,
            alpha=alpha,
            num_layers=num_layers,
            num_token_types=num_token_types,
            token_emb_dim=token_emb_dim,
            use_edge_type=use_edge_type,
            use_fusion_norm=use_fusion_norm,
            fusion_norm_type=fusion_norm_type,
            edge_gate_bias_init=edge_gate_bias_init,
            edge_gate_l1_lambda=edge_gate_l1_lambda,
            fusion_type=fusion_type,
            use_cross_attn_fusion=use_cross_attn_fusion,
            cross_attn_heads=cross_attn_heads,
        )

        test_dataset_processed = [data_config_obj.processed[i] for i in test_idx]

        result = train_model(
            fold_model, input_dim, hidden_dim, lr, epochs, gnn_type, num_heads,
            train_dataloader, val_dataloader, test_dataloader,
            training_config, exp_config, data_config_obj,
            fold=fold + 1, save_dir=model_dir,
            val_difficulties=val_difficulties, test_difficulties=test_difficulties,
            statement_root_type_ids=statement_root_type_ids,
            difficulty_levels=list(set(stratify_levels)),
            test_dataset_processed=test_dataset_processed
        )

        fold_results.append(result)
        all_training_times.append(result.get('fold_training_time', 0))
        all_memory_usage.append(result.get('max_memory_allocated', 0))

        if 'epoch_records' in result:
            all_epoch_records.extend(result['epoch_records'])

        topk = result['test_top_k_accuracies']
        for k in [1, 3, 5, 10]:
            all_test_topk[k].append(topk[k])
        all_test_mrr.append(result.get('test_mrr', 0))
        all_test_mar.append(result.get('test_mar', 0))
        all_test_mfr.append(result.get('test_mfr', 0))
        all_test_exam.append(result.get('test_exam', 0))
        all_best_exam.append(result.get('best_exam', 0))
        all_worst_exam.append(result.get('worst_exam', 0))
        all_min_inference_time.append(result.get('min_inference_time', 0))
        all_max_inference_time.append(result.get('max_inference_time', 0))
        all_avg_inference_time.append(result.get('avg_inference_time', 0))
        all_val_top1.append(result.get('best_val_top1_accuracy', 0))
        all_total_params.append(result.get('total_params', 0))

        stmt_root_topk = result.get('stmt_root_top_k_accuracies', {})
        for k in [1, 3, 5, 10]:
            all_stmt_root_topk[k].append(stmt_root_topk.get(k, 0))
        all_stmt_root_mrr.append(result.get('stmt_root_avg_mrr', 0))
        all_stmt_root_mfr.append(result.get('stmt_root_avg_mfr', 0))
        all_stmt_root_exam.append(result.get('stmt_root_avg_exam', 0))
        shortcut_deg = result.get('shortcut_degree', {})
        for k in [1, 3, 5, 10]:
            all_shortcut_degree[k].append(shortcut_deg.get(k, 0))

        if 'final_test_difficulty_results' in result and result['final_test_difficulty_results']:
            diff_res = result['final_test_difficulty_results']
            for diff_level in unique_stratify_levels:
                if diff_level in diff_res:
                    for k in [1, 3, 5, 10]:
                        all_difficulty_topk[diff_level][k].append(diff_res[diff_level].get(k, 0))

        print(f"Fold {fold + 1} Test Top-1: {topk[1]:.4f} | Top-3: {topk[3]:.4f} | Top-5: {topk[5]:.4f} | Top-10: {topk[10]:.4f}")

    # 合并所有折的逐程序排名并计算统计检验
    statistical_tests = {}
    merged_gnn_ranks = {}
    for result in fold_results:
        fold_ranks = result.get('per_program_ranks', {})
        for bug_id, rank in fold_ranks.items():
            if bug_id not in merged_gnn_ranks:
                merged_gnn_ranks[bug_id] = []
            merged_gnn_ranks[bug_id].append(rank)
    
    # 取平均排名
    merged_gnn_ranks = {bug_id: np.mean(ranks) for bug_id, ranks in merged_gnn_ranks.items()}
    
    if merged_gnn_ranks and data_config_obj.sbfl_data_path:
        print(f"\n计算统计检验 (GNN vs SBFL)...")
        print(f"  GNN逐程序排名: {len(merged_gnn_ranks)} 个程序")
        
        sbfl_data_path = data_config_obj.sbfl_data_path
        all_processed = data_config_obj.processed
        
        for formula in ['zoltar', 'ochiai']:
            sbfl_ranks = collect_sbfl_per_program_ranks(all_processed, sbfl_data_path, formula)
            test_result = compute_statistical_comparison(
                merged_gnn_ranks, sbfl_ranks,
                gnn_name=f"{gnn_type}",
                sbfl_name=formula.upper()
            )
            if test_result:
                statistical_tests[formula] = test_result
                print(f"  vs {formula.upper()}: N={test_result['n']}, "
                      f"p={test_result['p_value']:.4f}, Â₁₂={test_result['a12']:.3f} ({test_result['effect_size']})")

    # 按折收集逐语句排名
    all_fold_statement_ranks = []
    for fi, result in enumerate(fold_results):
        fold_num = fi + 1
        stmt_ranks = result.get('per_statement_ranks', {})
        for bug_id, line_ranks in stmt_ranks.items():
            for line_num, rank in line_ranks.items():
                parts = bug_id.rsplit('_', 1)
                all_fold_statement_ranks.append({
                    'fold': fold_num,
                    'bug_id': bug_id,
                    'project': parts[0] if len(parts) == 2 else '',
                    'case': parts[1] if len(parts) == 2 else '',
                    'true_line': line_num,
                    'rank': rank,
                })

    if not fold_results:
        print("没有成功的训练折数")
        return None

    variant_str = get_variant_str(variant)
    avg_result = {
        'model_name': f"{gnn_type}{variant_str}_h{hidden_dim}_lr{lr}_{K_FOLDS}fold_avg",
        'hidden_dim': hidden_dim,
        'lr': lr,
        'epochs': epochs,
        'gnn_type': gnn_type,
        'num_heads': num_heads,
        'variant': variant,
        'cv_method': cv_method,
        'sbfl_type': sbfl_type,
        'best_val_top1_accuracy': np.mean(all_val_top1) if all_val_top1 else 0,
        'test_top_k_accuracies': {k: np.mean(all_test_topk[k]) for k in [1, 3, 5, 10]},
        'test_top_k_std': {k: np.std(all_test_topk[k]) for k in [1, 3, 5, 10]},
        'test_mrr': np.mean(all_test_mrr) if all_test_mrr else 0,
        'test_mar': np.mean(all_test_mar) if all_test_mar else 0,
        'test_mfr': np.mean(all_test_mfr) if all_test_mfr else 0,
        'test_exam': np.mean(all_test_exam) if all_test_exam else 0,
        'best_exam': np.mean(all_best_exam) if all_best_exam else 0,
        'worst_exam': np.mean(all_worst_exam) if all_worst_exam else 0,
        'fold_training_time': np.mean(all_training_times) if all_training_times else 0,
        'max_memory_allocated': np.mean(all_memory_usage) if all_memory_usage else 0,
        'min_inference_time': min(all_min_inference_time) if all_min_inference_time else 0,
        'max_inference_time': max(all_max_inference_time) if all_max_inference_time else 0,
        'avg_inference_time': np.mean(all_avg_inference_time) if all_avg_inference_time else 0,
        'final_test_difficulty_results': {
            diff_level: {
                k: np.mean(all_difficulty_topk[diff_level][k]) if all_difficulty_topk[diff_level][k] else 0
                for k in [1, 3, 5, 10]
            }
            for diff_level in unique_stratify_levels
        },
        'total_params': np.mean(all_total_params) if all_total_params else 0,
        'stmt_root_top_k_accuracies': {k: np.mean(all_stmt_root_topk[k]) if all_stmt_root_topk[k] else 0 for k in [1, 3, 5, 10]},
        'stmt_root_avg_mrr': np.mean(all_stmt_root_mrr) if all_stmt_root_mrr else 0,
        'stmt_root_avg_mfr': np.mean(all_stmt_root_mfr) if all_stmt_root_mfr else 0,
        'stmt_root_avg_exam': np.mean(all_stmt_root_exam) if all_stmt_root_exam else 0,
        'shortcut_degree': {k: np.mean(all_shortcut_degree[k]) if all_shortcut_degree[k] else 0 for k in [1, 3, 5, 10]},
        'use_weighted_labels': exp_config.get('use_weighted_labels', False),
        'statement_root_weight': exp_config.get('statement_root_weight', 1.0),
        'child_node_weight': exp_config.get('child_node_weight', 0.5),
        'contrastive_weight': exp_config.get('contrastive_weight', 0.0),
        'contrastive_temperature': exp_config.get('contrastive_temperature', 0.07),
        'stratify_by': stratify_by,
        'stmt_count_boundaries': stmt_count_boundaries if stratify_by == 'stmt_count' else None,
        'sbfl_mode': sbfl_mode,
        'sbfl_formula': sbfl_formula,
        'use_sbfl': data_config_obj.use_sbfl,
        'use_original_coverage_matrix': data_config_obj.use_original_coverage_matrix,
        'epoch_records': all_epoch_records,
        'statistical_tests': statistical_tests,
        'fold_statement_ranks': all_fold_statement_ranks,
    }

    print(f"\n=== {K_FOLDS}折平均性能（{gnn_type}） ===")
    print(f"平均 Top-1: {avg_result['test_top_k_accuracies'][1]:.4f} ± {avg_result['test_top_k_std'][1]:.4f}")
    print(f"平均 Top-3: {avg_result['test_top_k_accuracies'][3]:.4f} ± {avg_result['test_top_k_std'][3]:.4f}")
    print(f"平均 Top-5: {avg_result['test_top_k_accuracies'][5]:.4f} ± {avg_result['test_top_k_std'][5]:.4f}")
    print(f"平均 Top-10: {avg_result['test_top_k_accuracies'][10]:.4f} ± {avg_result['test_top_k_std'][10]:.4f}")
    print(f"平均 MRR: {avg_result['test_mrr']:.4f}")
    print(f"平均 MFR: {avg_result['test_mfr']:.4f}")
    print(f"平均 MAR: {avg_result['test_mar']:.4f}")
    print(f"平均 EXAM: {avg_result['test_exam']:.2f}%")
    print(f"--- 语句根节点分析 ---")
    print(f"语句根 Top-1: {avg_result['stmt_root_top_k_accuracies'][1]:.4f}, Top-3: {avg_result['stmt_root_top_k_accuracies'][3]:.4f}, Top-5: {avg_result['stmt_root_top_k_accuracies'][5]:.4f}, Top-10: {avg_result['stmt_root_top_k_accuracies'][10]:.4f}")
    print(f"语句根 MRR: {avg_result['stmt_root_avg_mrr']:.4f}, MFR: {avg_result['stmt_root_avg_mfr']:.4f}, EXAM: {avg_result['stmt_root_avg_exam']:.2f}%")
    print(f"捷径程度 Top-1: {avg_result['shortcut_degree'][1]:.4f}, Top-3: {avg_result['shortcut_degree'][3]:.4f}, Top-5: {avg_result['shortcut_degree'][5]:.4f}, Top-10: {avg_result['shortcut_degree'][10]:.4f}")

    if avg_result.get('final_test_difficulty_results'):
        stratify_label = "语句数量" if stratify_by == 'stmt_count' else "难度"
        print(f"--- {stratify_label}分层 Top-K 检出率 ---")
        diff_res = avg_result['final_test_difficulty_results']
        for diff_level in unique_stratify_levels:
            if diff_level in diff_res:
                vals = diff_res[diff_level]
                print(f"  {diff_level:15s} Top-1: {vals.get(1, 0):.4f}, Top-3: {vals.get(3, 0):.4f}, Top-5: {vals.get(5, 0):.4f}, Top-10: {vals.get(10, 0):.4f}")

    return avg_result


def main():
    parser = argparse.ArgumentParser(description='GNN缺陷定位训练脚本')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='配置文件路径')
    parser.add_argument('--experiments', type=str, nargs='*', default=None,
                        help='指定要运行的实验名称（不指定则运行全部）')
    args = parser.parse_args()

    config = load_config(args.config)

    data_config_obj = DataConfig(config)
    training_config = config.get('training', {})
    experiments = config.get('experiments', [])

    seed = training_config.get('seed', 0)
    set_seed(seed)

    difficulty_map = load_difficulty_mapping(data_config_obj.difficulty_file)

    data_config_obj.stratify_data_by_difficulty()

    results = []
    detailed_results = []

    current_sbfl_type = None
    current_ast_granularity = data_config_obj.ast_granularity
    model_dir = training_config.get('model_dir', 'model')
    os.makedirs(model_dir, exist_ok=True)

    results_tmp_dir = os.path.join(model_dir, "excel_tmp")
    if os.path.exists(results_tmp_dir):
        import shutil
        shutil.rmtree(results_tmp_dir)

    for exp_config in experiments:
        if args.experiments is not None:
            exp_name = exp_config.get('name', '')
            if exp_name not in args.experiments:
                print(f"跳过实验: {exp_name}")
                continue

        sbfl_type = exp_config.get('sbfl_type', data_config_obj.sbfl_type)
        ast_granularity = exp_config.get('ast_granularity', data_config_obj.ast_granularity)
        if ast_granularity != current_ast_granularity:
            print(f"\nAST粒度变更: {current_ast_granularity} -> {ast_granularity}，更新数据源...")
            data_config_obj.ast_granularity = ast_granularity
            current_ast_granularity = ast_granularity
            current_sbfl_type = None
        if sbfl_type != current_sbfl_type:
            print(f"\nSBFL特征类型变更: {current_sbfl_type} -> {sbfl_type}，重新加载数据...")
            current_sbfl_type = sbfl_type

        try:
            result = run_experiment(exp_config, data_config_obj, training_config, difficulty_map)
        except Exception as e:
            print(f"实验执行出错: {e}")
            import traceback
            traceback.print_exc()
            result = None

        if result is not None:
            results.append(result)
            epoch_records = result.get('epoch_records', [])
            if epoch_records:
                detailed_results.extend(epoch_records)

            try:
                append_result_to_excel(result, epoch_records, model_dir)
                print(f"已保存实验结果到Excel: {result['model_name']}")
            except Exception as e:
                print(f"保存Excel时出错: {e}")

    print(f"\n{'=' * 60}")
    print("所有模型性能比较")
    print(f"{'=' * 60}")
    for i, result in enumerate(results):
        print(f"模型 {i + 1}: {result['model_name']}")
        print(f"  Top-1: {result['test_top_k_accuracies'][1]:.4f}")
        print(f"  Top-3: {result['test_top_k_accuracies'][3]:.4f}")
        print(f"  Top-5: {result['test_top_k_accuracies'][5]:.4f}")
        print(f"  Top-10: {result['test_top_k_accuracies'][10]:.4f}")
        print(f"  MRR: {result.get('test_mrr', 0):.4f}")
        print(f"  MFR: {result.get('test_mfr', 0):.4f}")
        print(f"  MAR: {result.get('test_mar', 0):.4f}")
        if 'test_top_k_std' in result:
            print(f"  Top-1 std: {result['test_top_k_std'][1]:.4f}")
        print()

    if results:
        best_model = max(results, key=lambda x: x['test_top_k_accuracies'][1])
        print(f"最佳模型: {best_model['model_name']}")
        print(f"  Top-1: {best_model['test_top_k_accuracies'][1]:.4f}")

        save_results_to_excel(results, detailed_results, model_dir)

    print("\n所有实验完成！")


if __name__ == "__main__":
    main()
