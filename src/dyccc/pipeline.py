#!/usr/bin/env python3
import argparse
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import heapq
import time
import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

from dyccc.nichenet_prior import ligand_target_activity, load_ligand_target_prior


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PACKAGE_ROOT / 'configs' / 'default.yaml'


def parse_age(a):
    s = str(a)
    if s.lower() == 'adult':
        return 100
    if s.startswith('P') and s[1:].isdigit():
        return int(s[1:])
    digits = ''.join([c for c in s if c.isdigit()])
    return int(digits) if digits else 999


def _align_meta_to_cells(meta: pd.DataFrame, cells):
    cells = pd.Index(pd.Series(cells).astype(str).values)
    meta = meta.copy()
    if 'cell' in meta.columns:
        key = meta['cell'].astype(str)
        if key.is_unique and cells.isin(key).all():
            aligned = meta.set_index(key).loc[cells].copy()
            aligned.index = cells
            aligned['_dyccc_cell'] = cells.values
            return aligned.reset_index(drop=True)
    idx = pd.Index(meta.index.astype(str))
    if idx.is_unique and cells.isin(idx).all():
        aligned = meta.loc[cells].copy()
        aligned['_dyccc_cell'] = cells.values
        return aligned.reset_index(drop=True)
    if len(meta) != len(cells):
        raise ValueError('metadata rows do not match expression cells')
    meta['_dyccc_cell'] = cells.values
    return meta.reset_index(drop=True)


def _read_table_expression(path: Path, meta: pd.DataFrame):
    sep = '\t' if path.suffix.lower() in {'.tsv', '.txt'} else ','
    df = pd.read_csv(path, sep=sep, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    meta_cells = set(meta['cell'].astype(str)) if 'cell' in meta.columns else set(meta.index.astype(str))
    row_hits = sum(x in meta_cells for x in df.index)
    col_hits = sum(x in meta_cells for x in df.columns)
    if col_hits >= row_hits:
        genes = df.index.astype(str).values
        cells = df.columns.astype(str).values
        X = csr_matrix(df.T.values)
    else:
        cells = df.index.astype(str).values
        genes = df.columns.astype(str).values
        X = csr_matrix(df.values)
    return X, genes, cells


def load_expression_metadata(expression_path: Path, metadata_path: Path, cache_dir: Path):
    meta = pd.read_csv(metadata_path, index_col=0)
    suffixes = ''.join(expression_path.suffixes).lower()
    if suffixes.endswith('.csv') or suffixes.endswith('.csv.gz') or suffixes.endswith('.tsv') or suffixes.endswith('.tsv.gz') or suffixes.endswith('.txt'):
        X, genes, cells = _read_table_expression(expression_path, meta)
    else:
        raise ValueError('expression_path must be a CSV/TSV table with gene and cell names')
    if X.shape[0] != len(cells):
        raise ValueError(f'cell mismatch: X={X.shape[0]}, cells={len(cells)}')
    if X.shape[1] != len(genes):
        raise ValueError(f'gene mismatch: X={X.shape[1]}, genes={len(genes)}')
    meta = _align_meta_to_cells(meta, cells)
    return X, meta, genes, cells


def normalize_log1p(X: csr_matrix, target_sum: float = 1e4):
    sums = np.asarray(X.sum(axis=1)).flatten()
    sums[sums == 0] = 1.0
    scale = target_sum / sums
    Xn = X.multiply(scale[:, None])
    Xn.data = np.log1p(Xn.data)
    return Xn.tocsr()


def qc_mask(meta: pd.DataFrame, n_genes_min=200, pct_mt_max=10.0):
    mask = np.ones(len(meta), dtype=bool)
    if 'nFeature_RNA' in meta.columns:
        mask &= (meta['nFeature_RNA'].values >= n_genes_min)
    if 'percent.mt' in meta.columns:
        mask &= (meta['percent.mt'].values <= pct_mt_max)
    return mask


def compute_embedding(X, n_components=50, seed=42):
    return TruncatedSVD(n_components=n_components, random_state=seed).fit_transform(X)


def time_embedding(ages):
    t = np.array([parse_age(a) for a in ages], dtype=float)
    t = (t - t.min()) / max(t.max() - t.min(), 1e-6)
    return t[:, None]


def build_knn_graph(emb, k=15):
    nn = NearestNeighbors(n_neighbors=k, metric='cosine')
    nn.fit(emb)
    _, idx = nn.kneighbors(emb)
    edges = []
    for i in range(len(emb)):
        for j in idx[i]:
            if i != j:
                edges.append((i, j))
    return np.array(edges).T


def _top_cartesian_product(left_scores, right_scores, k):
    """Return top-k product indices from two descending non-negative score arrays."""
    if k <= 0 or len(left_scores) == 0 or len(right_scores) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    if left_scores[0] <= 0 or right_scores[0] <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)

    heap = [(-(float(left_scores[0]) * float(right_scores[0])), 0, 0)]
    seen = {(0, 0)}
    li, ri, vals = [], [], []
    while heap and len(vals) < k:
        neg_val, i, j = heapq.heappop(heap)
        val = -neg_val
        if val <= 0:
            break
        li.append(i)
        ri.append(j)
        vals.append(val)
        if i + 1 < len(left_scores) and (i + 1, j) not in seen:
            seen.add((i + 1, j))
            heapq.heappush(heap, (-(float(left_scores[i + 1]) * float(right_scores[j])), i + 1, j))
        if j + 1 < len(right_scores) and (i, j + 1) not in seen:
            seen.add((i, j + 1))
            heapq.heappush(heap, (-(float(left_scores[i]) * float(right_scores[j + 1])), i, j + 1))
    return (
        np.asarray(li, dtype=np.int64),
        np.asarray(ri, dtype=np.int64),
        np.asarray(vals, dtype=np.float32),
    )


def _split_complex_name(name: str):
    s = str(name).strip()
    if not s:
        return []
    for ch in ['(', ')', ' ']:
        s = s.replace(ch, '')
    s = s.replace('+', '_').replace('-', '_')
    return [x for x in s.split('_') if x]


def _load_lr_pairs_from_csv(path: Path, genes, ligand_col='ligand', receptor_col='receptor'):
    if not path or not Path(path).exists():
        return []
    df = pd.read_csv(path)
    if ligand_col not in df.columns or receptor_col not in df.columns:
        return []
    gset = set(genes)
    pairs = []
    for _, row in df.iterrows():
        lig = row[ligand_col]
        rec = row[receptor_col]
        if pd.isna(lig) or pd.isna(rec):
            continue
        lig_genes = [g for g in _split_complex_name(lig) if g in gset]
        rec_genes = [g for g in _split_complex_name(rec) if g in gset]
        if lig_genes and rec_genes:
            pairs.append((sorted(set(lig_genes)), sorted(set(rec_genes))))
    uniq = []
    seen = set()
    for l, r in pairs:
        key = (tuple(l), tuple(r))
        if key not in seen:
            seen.add(key)
            uniq.append((l, r))
    return uniq


def _load_lr_pairs_from_cellphonedb(genes, cellphonedb_dir: Path):
    interaction_csv = cellphonedb_dir / 'interaction_input.csv'
    return _load_lr_pairs_from_csv(interaction_csv, genes, ligand_col='partner_a', receptor_col='partner_b')


def _load_lr_pairs_from_cellchat(genes, cellchat_dir: Path):
    gset = set(genes)
    out_csv = cellchat_dir / 'lr_pairs_cellchat_mouse.csv'
    if not out_csv.exists():
        rda_path = cellchat_dir / 'CellChatDB.mouse.rda'
        if rda_path.exists():
            r_cmd = (
                "load('{rda}'); "
                "db <- CellChatDB.mouse; int <- as.data.frame(db$interaction); "
                "if ('ligand' %in% colnames(int) && 'receptor' %in% colnames(int)) "
                "{{ write.csv(unique(int[,c('ligand','receptor')]), '{out}', row.names=FALSE) }} "
                "else {{ stop('CellChat interaction table has no ligand/receptor columns') }}"
            ).format(rda=str(rda_path).replace("'", "\\'"), out=str(out_csv).replace("'", "\\'"))
            r_runner = os.environ.get('DYCCC_R_COMMAND')
            if r_runner:
                r_command = shlex.split(r_runner)
            else:
                common_runners = [
                    shutil.which('Rscript'),
                    shutil.which('R'),
                ]
                r_command = [next((runner for runner in common_runners if runner and Path(runner).exists()), None)]
            if not r_command[0]:
                return []
            try:
                subprocess.run(
                    [*r_command, '-e', r_cmd],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except Exception:
                return []
    if not out_csv.exists():
        return []
    df = pd.read_csv(out_csv)
    if 'ligand' not in df.columns or 'receptor' not in df.columns:
        return []
    pairs = []
    for _, row in df.iterrows():
        lig = row['ligand']; rec = row['receptor']
        if pd.isna(lig) or pd.isna(rec):
            continue
        lig_genes = [g for g in _split_complex_name(lig) if g in gset]
        rec_genes = [g for g in _split_complex_name(rec) if g in gset]
        if lig_genes and rec_genes:
            pairs.append((sorted(set(lig_genes)), sorted(set(rec_genes))))
    uniq = []
    seen = set()
    for l, r in pairs:
        key = (tuple(l), tuple(r))
        if key not in seen:
            seen.add(key)
            uniq.append((l, r))
    return uniq


def get_lr_pairs_priority(genes, cellchat_dir: Path, cellphonedb_dir: Path, lr_pairs_path: Path | None = None):
    """Load ligand-receptor pairs in priority order: user CSV, CellChat, CellPhoneDB."""
    pairs_user = _load_lr_pairs_from_csv(lr_pairs_path, genes) if lr_pairs_path else []
    if len(pairs_user) > 0:
        return pairs_user, f'user_csv:{lr_pairs_path}'
    pairs_cellchat = _load_lr_pairs_from_cellchat(genes, cellchat_dir)
    if len(pairs_cellchat) > 0:
        return pairs_cellchat, 'CellChat'
    pairs_cpdb = _load_lr_pairs_from_cellphonedb(genes, cellphonedb_dir)
    if len(pairs_cpdb) > 0:
        return pairs_cpdb, 'CellPhoneDB'
    return [], 'none'


def build_lr_prior_graph(
    X,
    genes,
    lr_pairs,
    max_edges=30000,
    n_top=1500,
    per_pair_topk=4000,
    ligand_target_prior=None,
    ligand_target_weight=1.0,
    n_jobs=1,
    age_labels=None,
    same_age_only=False,
):
    var_idx = {g: i for i, g in enumerate(genes)}
    n_cells = X.shape[0]
    pair_specs = []
    complex_keys = set()
    ligand_names = set()
    for lig_genes, rec_genes in lr_pairs:
        lig_idx = tuple(var_idx[g] for g in lig_genes if g in var_idx)
        rec_idx = tuple(var_idx[g] for g in rec_genes if g in var_idx)
        if not lig_idx or not rec_idx:
            continue
        ligand_name = lig_genes[0]
        lig_key = tuple(sorted(lig_idx))
        rec_key = tuple(sorted(rec_idx))
        complex_keys.add(lig_key)
        complex_keys.add(rec_key)
        ligand_names.add(ligand_name)
        pair_specs.append((lig_genes, rec_genes, lig_key, rec_key, ligand_name))

    expr_cache = {
        key: np.asarray(X[:, list(key)].mean(axis=1)).flatten().astype(np.float32, copy=False)
        for key in complex_keys
    }
    zero_activity = np.zeros(n_cells, dtype=np.float32)
    target_cache = {}
    for ligand_name in ligand_names:
        if ligand_name in (ligand_target_prior or {}):
            target_cache[ligand_name] = ligand_target_activity(
                X, var_idx, ligand_name, ligand_target_prior or {}
            )
        else:
            target_cache[ligand_name] = (zero_activity, '')
    age_labels = np.asarray(age_labels) if age_labels is not None else None
    age_groups = []
    if same_age_only:
        if age_labels is None or len(age_labels) != n_cells:
            raise ValueError('age_labels are required when same_age_only=True')
        for age in pd.Series(age_labels).dropna().unique():
            idx = np.where(age_labels == age)[0]
            if len(idx) > 1:
                age_groups.append(idx)

    def process_pair(spec):
        lig_genes, rec_genes, lig_key, rec_key, ligand_name = spec
        lig_expr = expr_cache[lig_key]
        rec_expr = expr_cache[rec_key]
        target_activity, top_targets = target_cache[ligand_name]
        receiver_score = rec_expr * (1.0 + ligand_target_weight * target_activity)

        if same_age_only:
            group_outputs = []
            k_per_group = max(1, int(np.ceil(per_pair_topk / max(len(age_groups), 1))))
            for group_idx in age_groups:
                senders = group_idx[np.argsort(-lig_expr[group_idx])[:min(n_top, len(group_idx))]]
                receivers = group_idx[np.argsort(-receiver_score[group_idx])[:min(n_top, len(group_idx))]]
                sender_scores = lig_expr[senders]
                receiver_scores = receiver_score[receivers]
                k_local = min(k_per_group, len(senders) * len(receivers))
                sender_rank, receiver_rank, vals = _top_cartesian_product(sender_scores, receiver_scores, k_local)
                if len(vals) > 0:
                    group_outputs.append((senders[sender_rank], receivers[receiver_rank], vals))
            if not group_outputs:
                return None
            r = np.concatenate([x[0] for x in group_outputs])
            c = np.concatenate([x[1] for x in group_outputs])
            v = np.concatenate([x[2] for x in group_outputs])
            if len(v) > per_pair_topk:
                keep_top = np.argpartition(v, -per_pair_topk)[-per_pair_topk:]
                keep_top = keep_top[np.argsort(-v[keep_top])]
                r, c, v = r[keep_top], c[keep_top], v[keep_top]
        else:
            senders = np.argsort(-lig_expr)[:min(n_top, n_cells)]
            receivers = np.argsort(-receiver_score)[:min(n_top, n_cells)]
            sender_scores = lig_expr[senders]
            receiver_scores = receiver_score[receivers]
            k_local = min(per_pair_topk, len(senders) * len(receivers))
            sender_rank, receiver_rank, v = _top_cartesian_product(sender_scores, receiver_scores, k_local)
            if len(v) == 0:
                return None
            r = senders[sender_rank]
            c = receivers[receiver_rank]

        if len(v) == 0:
            return None
        keep = r != c
        if not np.any(keep):
            return None
        r = r[keep].astype(np.int32, copy=False)
        c = c[keep].astype(np.int32, copy=False)
        v = v[keep].astype(np.float32, copy=False)
        lr_vals = (lig_expr[r] * rec_expr[c]).astype(np.float32, copy=False)
        target_vals = target_activity[c].astype(np.float32, copy=False)
        n_keep = len(r)
        return {
            'rows': r,
            'cols': c,
            'vals': v,
            'lr_vals': lr_vals,
            'target_vals': target_vals,
            'ligands': ['_'.join(lig_genes)] * n_keep,
            'receptors': ['_'.join(rec_genes)] * n_keep,
            'targets': [top_targets] * n_keep,
        }

    if n_jobs and n_jobs > 1 and len(lr_pairs) > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            pair_results = list(executor.map(process_pair, pair_specs))
    else:
        pair_results = [process_pair(spec) for spec in pair_specs]
    pair_results = [res for res in pair_results if res is not None]

    if not pair_results:
        return (
            np.zeros((2, 0), dtype=int),
            np.zeros(0, dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            pd.DataFrame(),
        )

    from scipy.sparse import coo_matrix

    rows = np.concatenate([res['rows'] for res in pair_results])
    cols = np.concatenate([res['cols'] for res in pair_results])
    vals = np.concatenate([res['vals'] for res in pair_results])
    lr_vals = np.concatenate([res['lr_vals'] for res in pair_results])
    target_vals = np.concatenate([res['target_vals'] for res in pair_results])
    lig_all = [x for res in pair_results for x in res['ligands']]
    rec_all = [x for res in pair_results for x in res['receptors']]
    targets_all = [x for res in pair_results for x in res['targets']]
    candidate_df = pd.DataFrame({
        'source_idx': rows,
        'target_idx': cols,
        'ligand_genes': lig_all,
        'receptor_genes': rec_all,
        'lr_expr_score': lr_vals,
        'ligand_target_activity': target_vals,
        'combined_score': vals,
        'top_nichenet_targets': targets_all,
    })

    mat = coo_matrix((vals, (rows, cols)), shape=(n_cells, n_cells), dtype=np.float32).tocsr()
    mat.setdiag(0)
    mat.eliminate_zeros()
    mat = mat.tocoo()
    if mat.nnz == 0:
        return (
            np.zeros((2, 0), dtype=int),
            np.zeros(0, dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            pd.DataFrame(),
        )

    k_global = min(max_edges, mat.nnz)
    pick = np.argpartition(mat.data, -k_global)[-k_global:]
    pick = pick[np.argsort(-mat.data[pick])]
    edge_index = np.vstack([mat.row[pick], mat.col[pick]]).astype(np.int32, copy=False)
    edge_weight = mat.data[pick].astype(np.float32, copy=False)
    edge_df = pd.DataFrame({'source_idx': edge_index[0], 'target_idx': edge_index[1], 'edge_weight': edge_weight})
    best = (
        candidate_df.sort_values('combined_score', ascending=False)
        .drop_duplicates(['source_idx', 'target_idx'])
    )
    edge_df = edge_df.merge(best, on=['source_idx', 'target_idx'], how='left')
    features = edge_df[['lr_expr_score', 'ligand_target_activity', 'combined_score']].fillna(0).to_numpy(dtype=np.float32)
    return edge_index, edge_weight, features, edge_df


def build_transition_edges(meta, age_col, celltype_col, n_sample_per_pair=50, seed=42):
    ages = sorted(meta[age_col].dropna().unique(), key=parse_age)
    age_map = {a: i for i, a in enumerate(ages)}
    t = meta[age_col].map(age_map).values
    ct = meta[celltype_col].values

    groups = defaultdict(list)
    for i in range(len(meta)):
        groups[(ct[i], int(t[i]))].append(i)

    edges = []
    weights = []
    keys = list(groups.keys())
    rng = np.random.default_rng(seed)
    for ka in keys:
        cta, ta = ka
        idx_a = np.array(groups[ka])
        for kb in keys:
            ctb, tb = kb
            if cta != ctb or abs(ta - tb) != 1:
                continue
            idx_b = np.array(groups[kb])
            if len(idx_a) == 0 or len(idx_b) == 0:
                continue
            n_edges = min(n_sample_per_pair, len(idx_a) * len(idx_b))
            ia = rng.choice(idx_a, size=n_edges, replace=True)
            ib = rng.choice(idx_b, size=n_edges, replace=True)
            for i, j in zip(ia, ib):
                edges.append((i, j)); edges.append((j, i))
                weights.extend([1.0, 1.0])

    if not edges:
        return np.zeros((2, 0), dtype=int), np.zeros(0, dtype=np.float32)
    return np.array(edges, dtype=int).T, np.array(weights, dtype=np.float32)


def sample_neg_edges(edge_pos, n_cells, n_neg, seed=42):
    pos = set(zip(edge_pos[0].tolist(), edge_pos[1].tolist()))
    neg = set()
    rng = np.random.default_rng(seed)
    while len(neg) < n_neg:
        i = int(rng.integers(0, n_cells))
        j = int(rng.integers(0, n_cells))
        if i != j and (i, j) not in pos:
            neg.add((i, j))
    return np.array(list(neg), dtype=int).T


def _quantile_bins(values, n_bins=5):
    values = np.asarray(values, dtype=float)
    if len(np.unique(values)) <= 1:
        return np.zeros(len(values), dtype=int)
    qs = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    if len(qs) <= 2:
        return np.zeros(len(values), dtype=int)
    return np.digitize(values, qs[1:-1], right=True)


def sample_matched_neg_edges(edge_pos, meta, celltype_col, age_col, abundance_scores, n_neg=None, forbidden_edges=None, seed=42):
    """Sample difficult negatives matched on sender/receiver cell type, age, and expression abundance bins."""
    n_neg = n_neg or edge_pos.shape[1]
    forbidden = set(zip(edge_pos[0].tolist(), edge_pos[1].tolist()))
    if forbidden_edges is not None and forbidden_edges.shape[1] > 0:
        forbidden.update(zip(forbidden_edges[0].tolist(), forbidden_edges[1].tolist()))

    bins = _quantile_bins(abundance_scores, n_bins=5)
    group_to_cells = defaultdict(list)
    for i, row in meta.reset_index(drop=True).iterrows():
        key = (str(row[celltype_col]), str(row[age_col]), int(bins[i]))
        group_to_cells[key].append(i)

    rng = np.random.default_rng(seed)
    negatives = []
    pos_pairs = list(zip(edge_pos[0].tolist(), edge_pos[1].tolist()))
    attempts = 0
    max_attempts = max(10000, n_neg * 100)
    while len(negatives) < n_neg and attempts < max_attempts:
        attempts += 1
        src_pos, tgt_pos = pos_pairs[int(rng.integers(0, len(pos_pairs)))]
        src_key = (
            str(meta[celltype_col].iloc[src_pos]),
            str(meta[age_col].iloc[src_pos]),
            int(bins[src_pos]),
        )
        tgt_key = (
            str(meta[celltype_col].iloc[tgt_pos]),
            str(meta[age_col].iloc[tgt_pos]),
            int(bins[tgt_pos]),
        )
        src_pool = group_to_cells.get(src_key, [])
        tgt_pool = group_to_cells.get(tgt_key, [])
        if not src_pool or not tgt_pool:
            continue
        i = int(rng.choice(src_pool))
        j = int(rng.choice(tgt_pool))
        if i == j or (i, j) in forbidden:
            continue
        forbidden.add((i, j))
        negatives.append((i, j))

    if len(negatives) < n_neg:
        fallback = sample_neg_edges(edge_pos, len(meta), n_neg - len(negatives), seed=seed)
        negatives.extend(zip(fallback[0].tolist(), fallback[1].tolist()))
    return np.array(negatives[:n_neg], dtype=int).T


def split_edges_by_age(edge_idx, ages, train_ages, val_ages):
    train_set, val_set = set(train_ages), set(val_ages)
    tr = np.array([(ages[i] in train_set and ages[j] in train_set) for i, j in zip(edge_idx[0], edge_idx[1])])
    va = np.array([(ages[i] in val_set and ages[j] in val_set) for i, j in zip(edge_idx[0], edge_idx[1])])
    return edge_idx[:, tr], edge_idx[:, va], tr, va


def scale_edge_features(features):
    if features is None or len(features) == 0:
        return features
    x = np.asarray(features, dtype=np.float32).copy()
    x[:, 0] = np.log1p(np.maximum(x[:, 0], 0))
    x[:, 2] = np.log1p(np.maximum(x[:, 2], 0))
    for j in range(x.shape[1]):
        lo, hi = np.nanmin(x[:, j]), np.nanmax(x[:, j])
        if hi > lo:
            x[:, j] = (x[:, j] - lo) / (hi - lo)
        else:
            x[:, j] = 0
    return x.astype(np.float32, copy=False)


def train_two_head_gnn(
    X,
    edge_index,
    edge_pos,
    edge_neg,
    edge_trans,
    w_trans,
    ages_num,
    celltypes_num,
    edge_pos_val=None,
    edge_neg_val=None,
    edge_pos_features=None,
    edge_neg_features=None,
    edge_pos_val_features=None,
    edge_neg_val_features=None,
    readiness_target=None,
    predict_edges=None,
    predict_edge_features=None,
    n_epochs=80,
    lambda_r=0.1,
    lambda_t=0.05,
    lambda_readiness_target=0.0,
    n_label_permutations=0,
    edge_batch_size=0,
    torch_threads=0,
    seed=42,
):
    import torch
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
    from sklearn.metrics import roc_auc_score, average_precision_score

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch_threads and torch_threads > 0:
        torch.set_num_threads(torch_threads)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    class Model(torch.nn.Module):
        def __init__(self, d_in, d_edge=0):
            super().__init__()
            self.g1 = GCNConv(d_in, 32)
            self.g2 = GCNConv(32, 16)
            self.d_edge = d_edge
            self.edge_head = torch.nn.Sequential(
                torch.nn.Linear(32 + d_edge, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1)
            )
            self.node_head = torch.nn.Sequential(
                torch.nn.Linear(16, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1), torch.nn.Sigmoid()
            )

        def encode(self, x, ei):
            h = self.g1(x, ei).relu()
            h = self.g2(h, ei)
            return h

        def edge_prob(self, h, e, ef=None):
            a = h[e[0]]; b = h[e[1]]
            x = torch.cat([a, b], dim=1)
            if self.d_edge:
                if ef is None:
                    ef = torch.zeros((x.shape[0], self.d_edge), dtype=x.dtype, device=x.device)
                x = torch.cat([x, ef], dim=1)
            return torch.sigmoid(self.edge_head(x).squeeze(-1))

        def readiness(self, h):
            return self.node_head(h).squeeze(-1)

    x = torch.tensor(X, dtype=torch.float32, device=device)
    ei = torch.tensor(edge_index, dtype=torch.long, device=device)
    data = Data(x=x, edge_index=ei)

    e_all = np.hstack([edge_pos, edge_neg])
    y_all = np.concatenate([np.ones(edge_pos.shape[1]), np.zeros(edge_neg.shape[1])])
    e_all_t = torch.tensor(e_all, dtype=torch.long, device=device)
    y_all_t = torch.tensor(y_all, dtype=torch.float32, device=device)
    d_edge = 0
    edge_features_all_t = None
    if edge_pos_features is not None and edge_neg_features is not None:
        edge_features_all = np.vstack([edge_pos_features, edge_neg_features]).astype(np.float32, copy=False)
        d_edge = edge_features_all.shape[1]
        edge_features_all_t = torch.tensor(edge_features_all, dtype=torch.float32, device=device)

    et = torch.tensor(edge_trans, dtype=torch.long, device=device) if edge_trans.shape[1] > 0 else None
    wt = torch.tensor(w_trans, dtype=torch.float32, device=device) if len(w_trans) > 0 else None
    rt = torch.tensor(readiness_target, dtype=torch.float32, device=device) if readiness_target is not None else None

    model = Model(X.shape[1], d_edge=d_edge).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    rng = np.random.default_rng(seed)
    n_edge_train = e_all.shape[1]

    for ep in range(n_epochs):
        model.train(); opt.zero_grad()
        h = model.encode(data.x, data.edge_index)
        if edge_batch_size and edge_batch_size > 0 and n_edge_train > edge_batch_size:
            batch_idx_np = rng.choice(n_edge_train, size=edge_batch_size, replace=False)
            batch_idx = torch.tensor(batch_idx_np, dtype=torch.long, device=device)
            ef_batch = edge_features_all_t[batch_idx] if edge_features_all_t is not None else None
            p = model.edge_prob(h, e_all_t[:, batch_idx], ef_batch)
            loss_edge = torch.nn.functional.binary_cross_entropy(p, y_all_t[batch_idx])
        else:
            p = model.edge_prob(h, e_all_t, edge_features_all_t)
            loss_edge = torch.nn.functional.binary_cross_entropy(p, y_all_t)

        r = model.readiness(h)
        loss_r = torch.tensor(0.0, device=device)
        if et is not None:
            src, tgt = et[0], et[1]
            w = wt if wt is not None else torch.ones(src.shape[0], device=device)
            loss_r = (w * (r[src] - r[tgt]) ** 2).mean()

        loss_t = torch.tensor(0.0, device=device)
        groups = defaultdict(list)
        for i in range(len(ages_num)):
            groups[(celltypes_num[i], ages_num[i])].append(i)
        keys = sorted(groups.keys(), key=lambda z: (z[0], z[1]))
        if len(keys) > 1:
            num, den = torch.tensor(0.0, device=device), 0
            for i in range(len(keys)-1):
                ct1, a1 = keys[i]; ct2, a2 = keys[i+1]
                if ct1 == ct2:
                    m1 = r[torch.tensor(groups[keys[i]], dtype=torch.long, device=device)].mean()
                    m2 = r[torch.tensor(groups[keys[i+1]], dtype=torch.long, device=device)].mean()
                    num = num + (m1 - m2) ** 2
                    den += 1
            if den > 0:
                loss_t = num / den

        loss_target = torch.tensor(0.0, device=device)
        if rt is not None and lambda_readiness_target > 0:
            valid = ~torch.isnan(rt)
            if valid.any():
                loss_target = torch.nn.functional.mse_loss(r[valid], rt[valid])

        loss = loss_edge + lambda_r * loss_r + lambda_t * loss_t + lambda_readiness_target * loss_target
        loss.backward(); opt.step()

        if (ep + 1) % 20 == 0:
            with torch.no_grad():
                p_np = model.edge_prob(h, e_all_t, edge_features_all_t).detach().cpu().numpy()
                try:
                    auc = roc_auc_score(y_all, p_np)
                except Exception:
                    auc = 0.5
            print(f'Epoch {ep+1}: loss={loss.item():.4f}, AUROC={auc:.4f}')

    model.eval()
    with torch.no_grad():
        h = model.encode(data.x, data.edge_index)
        p_train = model.edge_prob(h, e_all_t, edge_features_all_t).detach().cpu().numpy()
        readiness = model.readiness(h).detach().cpu().numpy()

    metrics = {}
    try:
        metrics['auroc'] = float(roc_auc_score(y_all, p_train))
    except Exception:
        metrics['auroc'] = 0.5
    try:
        metrics['auprc'] = float(average_precision_score(y_all, p_train))
    except Exception:
        metrics['auprc'] = 0.0

    if edge_pos_val is not None and edge_neg_val is not None and edge_pos_val.shape[1] > 0 and edge_neg_val.shape[1] > 0:
        e_val = np.hstack([edge_pos_val, edge_neg_val])
        y_val = np.concatenate([np.ones(edge_pos_val.shape[1]), np.zeros(edge_neg_val.shape[1])])
        ef_val_t = None
        if edge_pos_val_features is not None and edge_neg_val_features is not None and d_edge:
            ef_val = np.vstack([edge_pos_val_features, edge_neg_val_features]).astype(np.float32, copy=False)
            ef_val_t = torch.tensor(ef_val, dtype=torch.float32, device=device)
        with torch.no_grad():
            p_val = model.edge_prob(h, torch.tensor(e_val, dtype=torch.long, device=device), ef_val_t).detach().cpu().numpy()
        try:
            metrics['val_auroc'] = float(roc_auc_score(y_val, p_val))
        except Exception:
            metrics['val_auroc'] = 0.5
        try:
            metrics['val_auprc'] = float(average_precision_score(y_val, p_val))
        except Exception:
            metrics['val_auprc'] = 0.0

    if rt is not None:
        valid = ~np.isnan(readiness_target)
        if valid.any():
            try:
                metrics['readiness_target_corr'] = float(np.corrcoef(readiness[valid], readiness_target[valid])[0, 1])
            except Exception:
                metrics['readiness_target_corr'] = np.nan
            metrics['readiness_target_mse'] = float(np.mean((readiness[valid] - readiness_target[valid]) ** 2))

    if n_label_permutations > 0:
        rng = np.random.default_rng(seed)
        aucs, auprcs = [], []
        for _ in range(n_label_permutations):
            yp = rng.permutation(y_all)
            try:
                aucs.append(float(roc_auc_score(yp, p_train)))
                auprcs.append(float(average_precision_score(yp, p_train)))
            except Exception:
                pass
        if aucs:
            metrics['label_perm_auroc_mean'] = float(np.mean(aucs))
            metrics['label_perm_auroc_std'] = float(np.std(aucs))
        if auprcs:
            metrics['label_perm_auprc_mean'] = float(np.mean(auprcs))
            metrics['label_perm_auprc_std'] = float(np.std(auprcs))

    predicted = {}
    if predict_edges is not None and predict_edges.shape[1] > 0:
        with torch.no_grad():
            pred_t = torch.tensor(predict_edges, dtype=torch.long, device=device)
            pred_ef_t = torch.tensor(predict_edge_features, dtype=torch.float32, device=device) if predict_edge_features is not None and d_edge else None
            predicted['edges'] = predict_edges
            predicted['scores'] = model.edge_prob(h, pred_t, pred_ef_t).detach().cpu().numpy()

    return readiness, metrics, predicted


def _save_current_figure(path_without_suffix: Path, plot_format='pdf', dpi=150):
    import matplotlib.pyplot as plt
    formats = ['png', 'pdf'] if plot_format == 'both' else [plot_format]
    for fmt in formats:
        plt.savefig(path_without_suffix.with_suffix(f'.{fmt}'), dpi=dpi, bbox_inches='tight')


def parse_plot_parts(value=None, skip_plots=False):
    if value is None:
        return set() if skip_plots else {'heatmap', 'line', 'readiness'}
    raw = str(value).strip().lower()
    if raw == 'none':
        return set()
    if raw == 'all':
        return {'heatmap', 'line', 'readiness'}
    parts = {x.strip() for x in raw.split(',') if x.strip()}
    allowed = {'heatmap', 'line', 'readiness'}
    unknown = parts - allowed
    if unknown:
        raise ValueError(f'unknown plot parts: {sorted(unknown)}')
    return parts


def plot_interaction(
    meta,
    edge_index,
    out_dir: Path,
    celltype_col,
    age_col,
    label='interaction',
    make_plots=True,
    plot_format='pdf',
    plot_parts=None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    src, tgt = edge_index[0], edge_index[1]
    ct_src = meta[celltype_col].iloc[src].values
    ct_tgt = meta[celltype_col].iloc[tgt].values

    def matrix(mask):
        df = pd.DataFrame({'source': ct_src[mask], 'target': ct_tgt[mask]})
        m = df.groupby(['source', 'target']).size().unstack(fill_value=0)
        all_ct = m.index.union(m.columns)
        return m.reindex(index=all_ct, columns=all_ct, fill_value=0)

    mat = matrix(np.ones(len(src), dtype=bool))
    mat.to_csv(out_dir / f'{label}_matrix.csv')
    parts = {'heatmap', 'line'} if plot_parts is None else set(plot_parts)
    if not make_plots or not parts:
        return

    import matplotlib.pyplot as plt
    import seaborn as sns

    if 'heatmap' in parts:
        for std, fn, title in [
            (False, out_dir / f'{label}_heatmap', f'{label}: cell type interaction (raw)'),
            (True, out_dir / f'{label}_heatmap_normalized', f'{label}: cell type interaction (normalized by self)'),
        ]:
            p = mat.copy().astype(float)
            fmt = '.0f'; cbar = 'edge count'
            if std:
                for i in p.index:
                    d = p.loc[i, i] if i in p.columns else 0
                    if d > 0: p.loc[i, :] /= d
                    elif p.loc[i, :].max() > 0: p.loc[i, :] /= p.loc[i, :].max()
                fmt = '.2f'; cbar = 'normalized (by self=1)'
            plt.figure(figsize=(max(8, p.shape[1]*0.5), max(6, p.shape[0]*0.4)))
            sns.heatmap(p, cmap='YlOrRd', annot=True, fmt=fmt, cbar_kws={'label': cbar})
            plt.title(title)
            plt.xlabel('Target cell type'); plt.ylabel('Source cell type')
            plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
            plt.tight_layout(); _save_current_figure(fn, plot_format=plot_format); plt.close()

    if 'line' not in parts:
        return

    ages = sorted(meta[age_col].dropna().unique(), key=parse_age)
    targets = sorted(set(ct_tgt))
    by_ct = out_dir / f'{label}_by_celltype'
    by_ct.mkdir(parents=True, exist_ok=True)

    for ct_s in meta[celltype_col].dropna().unique():
        data = {t: [] for t in targets}
        for age in ages:
            cells_age = set(np.where(meta[age_col].values == age)[0])
            n_src = ((meta[age_col] == age) & (meta[celltype_col] == ct_s)).sum()
            if n_src == 0:
                for t in targets: data[t].append(0)
                continue
            src_age = np.isin(src, list(cells_age)); tgt_age = np.isin(tgt, list(cells_age))
            for ct_t in targets:
                m = (ct_src == ct_s) & (ct_tgt == ct_t) & src_age & tgt_age
                data[ct_t].append(m.sum() / max(n_src, 1))

        plt.figure(figsize=(10, 5)); x = np.arange(len(ages))
        for ct_t in targets:
            if max(data[ct_t]) > 0 or ct_t == ct_s:
                plt.plot(x, data[ct_t], 'o-', label=str(ct_t)[:25], alpha=0.8)
        plt.xticks(x, ages, rotation=45)
        plt.xlabel(age_col)
        plt.ylabel('Interaction (edges per source cell, normalized)')
        plt.title(f'{label}: {ct_s}\\nInteraction to targets over development')
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
        plt.tight_layout()
        safe = str(ct_s).replace('/', '_')[:50]
        _save_current_figure(by_ct / f'interaction_vs_age_{safe}', plot_format=plot_format)
        plt.close()


def select_top_scored_edges(edge_index, scores, top_k):
    if edge_index is None or edge_index.shape[1] == 0 or scores is None or len(scores) == 0:
        return np.zeros((2, 0), dtype=int), np.zeros(0, dtype=np.float32), np.array([], dtype=int)
    top_k = min(int(top_k), edge_index.shape[1])
    pick = np.argpartition(scores, -top_k)[-top_k:]
    pick = pick[np.argsort(-scores[pick])]
    return edge_index[:, pick], scores[pick].astype(np.float32, copy=False), pick


def write_evidence_tables(
    evidence: pd.DataFrame,
    out_dir: Path,
    meta: pd.DataFrame | None = None,
    celltype_col: str | None = None,
    age_col: str | None = None,
):
    evidence = evidence.copy()
    if {'ligand_genes', 'receptor_genes'}.issubset(evidence.columns):
        evidence['lr_pair'] = (
            evidence['ligand_genes'].fillna('').astype(str)
            + '-'
            + evidence['receptor_genes'].fillna('').astype(str)
        )
    if {'lr_pair', 'top_nichenet_targets'}.issubset(evidence.columns):
        targets = evidence['top_nichenet_targets'].fillna('').astype(str)
        evidence['lr_pair_with_nichenet_targets'] = np.where(
            targets.str.len() > 0,
            evidence['lr_pair'].astype(str) + ' | targets: ' + targets,
            evidence['lr_pair'].astype(str),
        )
    front_cols = [
        'gnn_score',
        'source_celltype',
        'target_celltype',
        'source_age',
        'target_age',
        'lr_pair',
        'lr_pair_with_nichenet_targets',
        'ligand_genes',
        'receptor_genes',
        'top_nichenet_targets',
        'ligand_target_activity',
        'lr_expr_score',
        'combined_score',
    ]
    ordered = [c for c in front_cols if c in evidence.columns]
    evidence = evidence[ordered + [c for c in evidence.columns if c not in ordered]]
    evidence.to_csv(out_dir / 'ligand_receptor_target_evidence.csv', index=False)

    same_age = evidence[evidence['source_age'].astype(str) == evidence['target_age'].astype(str)]
    same_age.to_csv(out_dir / 'ligand_receptor_target_evidence_same_age.csv', index=False)

    incoming_dir = out_dir / 'incoming'
    outgoing_dir = out_dir / 'outgoing'
    incoming_dir.mkdir(parents=True, exist_ok=True)
    outgoing_dir.mkdir(parents=True, exist_ok=True)
    incoming, outgoing, persistence = summarize_celltype_lr_evidence(
        evidence,
        meta=meta,
        celltype_col=celltype_col,
        age_col=age_col,
    )
    incoming.to_csv(incoming_dir / 'celltype_lr_summary.csv', index=False)
    outgoing.to_csv(outgoing_dir / 'celltype_lr_summary.csv', index=False)
    persistence.to_csv(out_dir / 'lr_pair_stage_persistence.csv', index=False)


def _first_nonempty(series):
    vals = series.dropna().astype(str)
    vals = vals[vals.str.len() > 0]
    return vals.iloc[0] if len(vals) else ''


def _celltype_age_counts(meta: pd.DataFrame | None, celltype_col: str | None, age_col: str | None):
    if meta is None or not celltype_col or not age_col:
        return None
    if celltype_col not in meta.columns or age_col not in meta.columns:
        return None
    counts = (
        meta.groupby([celltype_col, age_col], dropna=False)
        .size()
        .rename('n_cells')
        .reset_index()
    )
    totals = (
        meta.groupby(age_col, dropna=False)
        .size()
        .rename('n_cells_age')
        .reset_index()
    )
    counts = counts.merge(totals, on=age_col, how='left')
    counts['celltype_age_fraction'] = counts['n_cells'] / counts['n_cells_age'].replace(0, np.nan)
    return counts.rename(columns={celltype_col: 'celltype', age_col: 'age'})


def _add_cell_pair_prior(summary: pd.DataFrame, counts: pd.DataFrame | None):
    summary = summary.copy()
    summary['same_timepoint_score'] = (
        summary['source_age'].astype(str) == summary['target_age'].astype(str)
    ).astype(float)
    if counts is None or counts.empty:
        summary['source_celltype_age_n'] = np.nan
        summary['target_celltype_age_n'] = np.nan
        summary['source_celltype_age_fraction'] = np.nan
        summary['target_celltype_age_fraction'] = np.nan
        abundance_score = np.ones(len(summary), dtype=float)
    else:
        src_counts = counts.rename(columns={
            'celltype': 'source_celltype',
            'age': 'source_age',
            'n_cells': 'source_celltype_age_n',
            'celltype_age_fraction': 'source_celltype_age_fraction',
        })[['source_celltype', 'source_age', 'source_celltype_age_n', 'source_celltype_age_fraction']]
        tgt_counts = counts.rename(columns={
            'celltype': 'target_celltype',
            'age': 'target_age',
            'n_cells': 'target_celltype_age_n',
            'celltype_age_fraction': 'target_celltype_age_fraction',
        })[['target_celltype', 'target_age', 'target_celltype_age_n', 'target_celltype_age_fraction']]
        summary = summary.merge(src_counts, on=['source_celltype', 'source_age'], how='left')
        summary = summary.merge(tgt_counts, on=['target_celltype', 'target_age'], how='left')
        src_frac = summary['source_celltype_age_fraction'].fillna(0).astype(float)
        tgt_frac = summary['target_celltype_age_fraction'].fillna(0).astype(float)
        abundance_score = np.sqrt(np.minimum(src_frac / 0.01, 1.0) * np.minimum(tgt_frac / 0.01, 1.0))
    summary['cell_abundance_score'] = abundance_score
    summary['cell_pair_prior'] = 0.5 * summary['same_timepoint_score'] + 0.5 * summary['cell_abundance_score']
    return summary


def _stage_label(row):
    return str(row['source_age']) if str(row['source_age']) == str(row['target_age']) else f"{row['source_age']}->{row['target_age']}"


def _trend_from_stage_scores(stage_scores: pd.DataFrame):
    if stage_scores.empty:
        return 'none'
    ordered = stage_scores.sort_values('age_order')
    values = ordered['communication_score'].astype(float).values
    if len(values) <= 1:
        return 'single_stage'
    peak = int(np.argmax(values))
    if peak == 0 and values[-1] < values[0]:
        return 'early_peak'
    if peak == len(values) - 1 and values[-1] > values[0]:
        return 'late_increase'
    if len(values) >= 3 and 0 < peak < len(values) - 1:
        return 'mid_peak'
    if np.all(np.diff(values) >= 0):
        return 'increasing'
    if np.all(np.diff(values) <= 0):
        return 'decreasing'
    return 'dynamic'


def _make_stage_persistence(summary: pd.DataFrame):
    if summary.empty:
        return pd.DataFrame(columns=[
            'source_celltype', 'target_celltype', 'lr_pair', 'n_stages', 'stages',
            'trend', 'total_communication_score', 'max_communication_score',
            'mean_gnn_score', 'mean_ligand_target_activity', 'mean_lr_expr_score',
            'mean_cell_pair_prior', 'confidence_tier',
        ])
    work = summary.copy()
    work['stage'] = work.apply(_stage_label, axis=1)
    work['age_order'] = work['stage'].map(parse_age)
    rows = []
    for key, sub in work.groupby(['source_celltype', 'target_celltype', 'lr_pair'], dropna=False):
        stage_scores = (
            sub.groupby(['stage', 'age_order'], as_index=False)['communication_score']
            .sum()
            .sort_values('age_order')
        )
        rows.append({
            'source_celltype': key[0],
            'target_celltype': key[1],
            'lr_pair': key[2],
            'n_stages': int(stage_scores['stage'].nunique()),
            'stages': ';'.join(stage_scores['stage'].astype(str).tolist()),
            'trend': _trend_from_stage_scores(stage_scores),
            'total_communication_score': float(sub['communication_score'].sum()),
            'max_communication_score': float(sub['communication_score'].max()),
            'mean_gnn_score': float(sub['mean_gnn_score'].mean()),
            'mean_ligand_target_activity': float(sub['mean_ligand_target_activity'].mean()),
            'mean_lr_expr_score': float(sub['mean_lr_expr_score'].mean()),
            'mean_cell_pair_prior': float(sub['cell_pair_prior'].mean()),
        })
    persistence = pd.DataFrame(rows)
    return persistence.sort_values(['total_communication_score', 'n_stages'], ascending=[False, False])


def _rank01(series):
    s = pd.to_numeric(series, errors='coerce')
    if s.notna().sum() <= 1:
        return pd.Series(np.ones(len(s)), index=s.index, dtype=float)
    return s.rank(pct=True).fillna(0.0).astype(float)


def _add_confidence_tiers(summary: pd.DataFrame, persistence: pd.DataFrame):
    summary = summary.copy()
    if summary.empty:
        summary['confidence_tier'] = []
        return summary, persistence
    persist_cols = ['source_celltype', 'target_celltype', 'lr_pair', 'n_stages', 'trend']
    summary = summary.merge(
        persistence[persist_cols],
        on=['source_celltype', 'target_celltype', 'lr_pair'],
        how='left',
    )
    summary['n_stages'] = summary['n_stages'].fillna(1).astype(int)
    summary['gnn_score_rank'] = _rank01(summary['mean_gnn_score'])
    summary['lr_expr_score_rank'] = _rank01(summary['mean_lr_expr_score'])
    summary['ligand_target_activity_rank'] = _rank01(summary['mean_ligand_target_activity'])
    summary['communication_score_rank'] = _rank01(summary['communication_score'])
    summary['confidence_score'] = (
        0.30 * summary['gnn_score_rank']
        + 0.25 * summary['lr_expr_score_rank']
        + 0.25 * summary['ligand_target_activity_rank']
        + 0.10 * summary['cell_pair_prior'].fillna(0)
        + 0.10 * np.minimum(summary['n_stages'] / 2.0, 1.0)
    )
    high = (
        (summary['gnn_score_rank'] >= 0.60)
        & (summary['lr_expr_score_rank'] >= 0.60)
        & (summary['ligand_target_activity_rank'] >= 0.60)
        & ((summary['n_stages'] >= 2) | (summary['cell_pair_prior'] >= 0.75))
    )
    medium = (
        (summary['gnn_score_rank'] >= 0.50)
        & (summary['lr_expr_score_rank'] >= 0.50)
        & ((summary['ligand_target_activity_rank'] >= 0.40) | (summary['n_stages'] >= 2))
    )
    summary['confidence_tier'] = np.select(
        [high, medium],
        ['high', 'medium'],
        default='exploratory',
    )
    persistence = persistence.merge(
        summary.groupby(['source_celltype', 'target_celltype', 'lr_pair'], dropna=False)
        .agg(
            confidence_score=('confidence_score', 'max'),
            confidence_tier=('confidence_tier', lambda x: 'high' if (x == 'high').any() else ('medium' if (x == 'medium').any() else 'exploratory')),
        )
        .reset_index(),
        on=['source_celltype', 'target_celltype', 'lr_pair'],
        how='left',
    )
    return summary, persistence


def summarize_celltype_lr_evidence(
    evidence: pd.DataFrame,
    meta: pd.DataFrame | None = None,
    celltype_col: str | None = None,
    age_col: str | None = None,
):
    group_cols = ['source_celltype', 'target_celltype', 'source_age', 'target_age', 'lr_pair']
    if evidence.empty or not set(group_cols).issubset(evidence.columns):
        columns = group_cols + [
            'n_cell_edges', 'n_source_cells', 'n_target_cells',
            'mean_gnn_score', 'max_gnn_score', 'mean_ligand_target_activity',
            'mean_lr_expr_score', 'mean_combined_score', 'mean_edge_weight',
            'communication_score', 'top_nichenet_targets', 'ligand_genes',
            'receptor_genes', 'lr_pair_with_nichenet_targets',
        ]
        empty = pd.DataFrame(columns=columns)
        return empty.copy(), empty.copy(), pd.DataFrame()

    work = evidence.copy()
    numeric_cols = [
        'gnn_score', 'ligand_target_activity', 'lr_expr_score',
        'combined_score', 'edge_weight',
    ]
    for col in numeric_cols:
        if col not in work.columns:
            work[col] = np.nan

    grouped = work.groupby(group_cols, dropna=False)
    summary = grouped.agg(
        n_cell_edges=('lr_pair', 'size'),
        n_source_cells=('source_idx', 'nunique'),
        n_target_cells=('target_idx', 'nunique'),
        mean_gnn_score=('gnn_score', 'mean'),
        max_gnn_score=('gnn_score', 'max'),
        mean_ligand_target_activity=('ligand_target_activity', 'mean'),
        mean_lr_expr_score=('lr_expr_score', 'mean'),
        mean_combined_score=('combined_score', 'mean'),
        mean_edge_weight=('edge_weight', 'mean'),
        top_nichenet_targets=('top_nichenet_targets', _first_nonempty),
        ligand_genes=('ligand_genes', _first_nonempty),
        receptor_genes=('receptor_genes', _first_nonempty),
        lr_pair_with_nichenet_targets=('lr_pair_with_nichenet_targets', _first_nonempty),
    ).reset_index()
    summary['communication_score'] = (
        summary['n_cell_edges'].astype(float)
        * summary['mean_gnn_score'].fillna(0).astype(float)
    )
    counts = _celltype_age_counts(meta, celltype_col, age_col)
    summary = _add_cell_pair_prior(summary, counts)
    persistence = _make_stage_persistence(summary)
    summary, persistence = _add_confidence_tiers(summary, persistence)

    incoming_cols = [
        'target_celltype', 'target_age', 'source_celltype', 'source_age', 'lr_pair',
        'n_cell_edges', 'n_source_cells', 'n_target_cells', 'communication_score',
        'confidence_tier', 'confidence_score', 'cell_pair_prior',
        'same_timepoint_score', 'cell_abundance_score', 'n_stages', 'trend',
        'mean_gnn_score', 'max_gnn_score', 'mean_ligand_target_activity',
        'mean_lr_expr_score', 'mean_combined_score', 'mean_edge_weight',
        'ligand_genes', 'receptor_genes', 'top_nichenet_targets',
        'lr_pair_with_nichenet_targets',
    ]
    outgoing_cols = [
        'source_celltype', 'source_age', 'target_celltype', 'target_age', 'lr_pair',
        'n_cell_edges', 'n_source_cells', 'n_target_cells', 'communication_score',
        'confidence_tier', 'confidence_score', 'cell_pair_prior',
        'same_timepoint_score', 'cell_abundance_score', 'n_stages', 'trend',
        'mean_gnn_score', 'max_gnn_score', 'mean_ligand_target_activity',
        'mean_lr_expr_score', 'mean_combined_score', 'mean_edge_weight',
        'ligand_genes', 'receptor_genes', 'top_nichenet_targets',
        'lr_pair_with_nichenet_targets',
    ]
    incoming = summary[incoming_cols].sort_values(
        ['target_celltype', 'target_age', 'communication_score'],
        ascending=[True, True, False],
    )
    outgoing = summary[outgoing_cols].sort_values(
        ['source_celltype', 'source_age', 'communication_score'],
        ascending=[True, True, False],
    )
    return incoming, outgoing, persistence


def plot_incoming_celltype_summary(out_dir: Path, plot_format='pdf', top_sources=8):
    plot_dir = out_dir / 'incoming'
    path = plot_dir / 'celltype_lr_summary.csv'
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    required = {'target_celltype', 'target_age', 'source_celltype', 'communication_score'}
    if not required.issubset(df.columns):
        return

    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    agg = (
        df.groupby(['target_celltype', 'target_age', 'source_celltype'], dropna=False)['communication_score']
        .sum()
        .reset_index()
    )
    ages = sorted(agg['target_age'].dropna().unique(), key=parse_age)
    x = np.arange(len(ages))
    for target, sub in agg.groupby('target_celltype'):
        top = (
            sub.groupby('source_celltype')['communication_score']
            .sum()
            .sort_values(ascending=False)
            .head(top_sources)
            .index
        )
        if len(top) == 0:
            continue
        plt.figure(figsize=(10, 5))
        for source in top:
            y = []
            s = sub[sub['source_celltype'] == source]
            for age in ages:
                val = s.loc[s['target_age'] == age, 'communication_score'].sum()
                y.append(float(val))
            if max(y) > 0:
                plt.plot(x, y, 'o-', label=str(source)[:30], alpha=0.85)
        plt.xticks(x, ages, rotation=45)
        plt.xlabel('target age')
        plt.ylabel('incoming communication score')
        plt.title(f'Incoming regulation of {target}')
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
        plt.tight_layout()
        safe = str(target).replace('/', '_')[:50]
        _save_current_figure(plot_dir / f'incoming_score_vs_age_{safe}', plot_format=plot_format)
        plt.close()


def plot_outgoing_celltype_summary(out_dir: Path, plot_format='pdf', top_targets=8):
    plot_dir = out_dir / 'outgoing'
    path = plot_dir / 'celltype_lr_summary.csv'
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    required = {'source_celltype', 'source_age', 'target_celltype', 'communication_score'}
    if not required.issubset(df.columns):
        return

    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    agg = (
        df.groupby(['source_celltype', 'source_age', 'target_celltype'], dropna=False)['communication_score']
        .sum()
        .reset_index()
    )
    ages = sorted(agg['source_age'].dropna().unique(), key=parse_age)
    x = np.arange(len(ages))
    for source, sub in agg.groupby('source_celltype'):
        top = (
            sub.groupby('target_celltype')['communication_score']
            .sum()
            .sort_values(ascending=False)
            .head(top_targets)
            .index
        )
        if len(top) == 0:
            continue
        plt.figure(figsize=(10, 5))
        for target in top:
            y = []
            s = sub[sub['target_celltype'] == target]
            for age in ages:
                val = s.loc[s['source_age'] == age, 'communication_score'].sum()
                y.append(float(val))
            if max(y) > 0:
                plt.plot(x, y, 'o-', label=str(target)[:30], alpha=0.85)
        plt.xticks(x, ages, rotation=45)
        plt.xlabel('source age')
        plt.ylabel('outgoing communication score')
        plt.title(f'Outgoing communication from {source}')
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
        plt.tight_layout()
        safe = str(source).replace('/', '_')[:50]
        _save_current_figure(plot_dir / f'outgoing_score_vs_age_{safe}', plot_format=plot_format)
        plt.close()


def plot_readiness(meta, readiness, age_col, out_path, plot_format='pdf'):
    import matplotlib.pyplot as plt
    ages = sorted(meta[age_col].dropna().unique(), key=parse_age)
    means = [readiness[meta[age_col].values == a].mean() for a in ages]
    stds = [readiness[meta[age_col].values == a].std() for a in ages]
    plt.figure(figsize=(8, 4))
    plt.errorbar(range(len(ages)), means, yerr=stds, fmt='o-', capsize=3)
    plt.xticks(range(len(ages)), ages, rotation=45)
    plt.xlabel(age_col); plt.ylabel('Readiness score'); plt.title('Readiness along development')
    plt.tight_layout(); _save_current_figure(out_path.with_suffix(''), plot_format=plot_format); plt.close()


def export_top_lr_pairs_per_point(
    Xn,
    meta: pd.DataFrame,
    genes,
    lr_pairs,
    edge_index,
    out_dir: Path,
    celltype_col: str,
    age_col: str,
    top_n: int = 15,
    direction: str = 'outgoing',
):
    """Export top LR pairs for each source-target cell-type and age point."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if direction not in {'incoming', 'outgoing'}:
        raise ValueError("direction must be 'incoming' or 'outgoing'")

    src, tgt = edge_index[0], edge_index[1]
    ct_src = meta[celltype_col].iloc[src].values
    ct_tgt = meta[celltype_col].iloc[tgt].values
    age_src = meta[age_col].iloc[src].values
    age_tgt = meta[age_col].iloc[tgt].values
    same_age_mask = age_src == age_tgt
    df_edges = pd.DataFrame({
        'source_ct': ct_src[same_age_mask],
        'target_ct': ct_tgt[same_age_mask],
        'age': age_src[same_age_mask],
    })
    edge_counts = (
        df_edges.groupby(['source_ct', 'target_ct', 'age'])
        .size()
        .rename('edge_count')
        .reset_index()
    )
    if direction == 'outgoing':
        denom = (
            meta.groupby([celltype_col, age_col])
            .size()
            .rename('n_denominator_cells')
            .reset_index()
            .rename(columns={celltype_col: 'source_ct', age_col: 'age'})
        )
        line_df = edge_counts.merge(denom, on=['source_ct', 'age'], how='left')
    else:
        denom = (
            meta.groupby([celltype_col, age_col])
            .size()
            .rename('n_denominator_cells')
            .reset_index()
            .rename(columns={celltype_col: 'target_ct', age_col: 'age'})
        )
        line_df = edge_counts.merge(denom, on=['target_ct', 'age'], how='left')
    line_df['line_value'] = line_df['edge_count'] / line_df['n_denominator_cells'].clip(lower=1)
    line_map = {
        (r.source_ct, r.target_ct, r.age): float(r.line_value)
        for _, r in line_df.iterrows()
    }

    var_idx = {g: i for i, g in enumerate(genes)}
    required_genes = sorted({
        g for lig, rec in lr_pairs for g in (list(lig) + list(rec)) if g in var_idx
    })
    if len(required_genes) == 0:
        pd.DataFrame(columns=[
            'direction', 'source_ct', 'target_ct', 'age', 'line_value', 'rank',
            'lr_pair', 'ligand_genes', 'receptor_genes', 'score'
        ]).to_csv(out_dir / 'top_lr_pairs_per_point.csv', index=False)
        return
    req_idx = np.array([var_idx[g] for g in required_genes], dtype=int)
    req_gene_to_col = {g: i for i, g in enumerate(required_genes)}
    pair_specs = []
    complex_keys = []
    complex_to_idx = {}
    for lig_genes, rec_genes in lr_pairs:
        lig_cols = tuple(req_gene_to_col[g] for g in lig_genes if g in req_gene_to_col)
        rec_cols = tuple(req_gene_to_col[g] for g in rec_genes if g in req_gene_to_col)
        if not lig_cols or not rec_cols:
            continue
        lig_key = tuple(sorted(lig_cols))
        rec_key = tuple(sorted(rec_cols))
        for key in (lig_key, rec_key):
            if key not in complex_to_idx:
                complex_to_idx[key] = len(complex_keys)
                complex_keys.append(key)
        pair_specs.append((
            complex_to_idx[lig_key],
            complex_to_idx[rec_key],
            '_'.join(lig_genes),
            '_'.join(rec_genes),
        ))
    if len(pair_specs) == 0:
        pd.DataFrame(columns=[
            'direction', 'source_ct', 'target_ct', 'age', 'line_value', 'rank',
            'lr_pair', 'ligand_genes', 'receptor_genes', 'score'
        ]).to_csv(out_dir / 'top_lr_pairs_per_point.csv', index=False)
        return
    lig_complex_idx = np.array([x[0] for x in pair_specs], dtype=int)
    rec_complex_idx = np.array([x[1] for x in pair_specs], dtype=int)
    lig_names = np.array([x[2] for x in pair_specs], dtype=object)
    rec_names = np.array([x[3] for x in pair_specs], dtype=object)

    group_means = {}
    for (ct, age), idxs in meta.groupby([celltype_col, age_col]).groups.items():
        idxs = np.asarray(list(idxs), dtype=int)
        if len(idxs) == 0:
            continue
        mean_expr = np.asarray(Xn[idxs][:, req_idx].mean(axis=0)).flatten().astype(np.float32)
        complex_mean = np.zeros(len(complex_keys), dtype=np.float32)
        for complex_idx, cols in enumerate(complex_keys):
            complex_mean[complex_idx] = float(mean_expr[list(cols)].mean())
        group_means[(ct, age)] = complex_mean

    ages = sorted(meta[age_col].dropna().unique(), key=parse_age)
    celltypes = sorted(meta[celltype_col].dropna().astype(str).unique())
    rows = []
    for age in ages:
        for sct in celltypes:
            src_key = (sct, age)
            if src_key not in group_means:
                continue
            src_vec = group_means[src_key]
            for tct in celltypes:
                tgt_key = (tct, age)
                if tgt_key not in group_means:
                    continue
                tgt_vec = group_means[tgt_key]
                scores = src_vec[lig_complex_idx] * tgt_vec[rec_complex_idx]
                valid = np.where(scores > 0)[0]
                if len(valid) == 0:
                    continue
                k = min(top_n, len(valid))
                if len(valid) > k:
                    local = valid[np.argpartition(scores[valid], -k)[-k:]]
                    local = local[np.argsort(-scores[local])]
                else:
                    local = valid[np.argsort(-scores[valid])]
                lv = line_map.get((sct, tct, age), 0.0)
                for rk, pair_idx in enumerate(local, start=1):
                    lig_s = lig_names[pair_idx]
                    rec_s = rec_names[pair_idx]
                    rows.append({
                        'direction': direction,
                        'source_ct': sct,
                        'target_ct': tct,
                        'age': age,
                        'line_value': lv,
                        'rank': rk,
                        'lr_pair': f'{lig_s}->{rec_s}',
                        'ligand_genes': lig_s,
                        'receptor_genes': rec_s,
                        'score': float(scores[pair_idx]),
                    })

    out_csv = out_dir / 'top_lr_pairs_per_point.csv'
    df = pd.DataFrame(rows)
    if not df.empty and direction == 'incoming':
        cols = [
            'direction', 'target_ct', 'source_ct', 'age', 'line_value', 'rank',
            'lr_pair', 'ligand_genes', 'receptor_genes', 'score',
        ]
        df = df[cols]
    df.to_csv(out_csv, index=False)
    print(f'Top LR pairs per point saved: {out_csv}')


def load_readiness_target(path: Path | None, meta: pd.DataFrame, cell_col='cell', target_col='pseudotime_like'):
    if path is None:
        return None
    df = pd.read_csv(path)
    if cell_col not in df.columns:
        raise ValueError(f'readiness target cell column not found: {cell_col}')
    if target_col not in df.columns:
        raise ValueError(f'readiness target column not found: {target_col}')
    target_map = df.set_index(cell_col)[target_col].astype(float)
    target = meta['_dyccc_cell'].map(target_map).astype(float).values
    valid = ~np.isnan(target)
    if valid.sum() == 0:
        raise ValueError('readiness target did not match any cells')
    lo, hi = np.nanmin(target), np.nanmax(target)
    if hi > lo:
        target = (target - lo) / (hi - lo)
    return target.astype(np.float32)


def shuffle_lr_pairs(lr_pairs, rng):
    ligands = [lig for lig, _ in lr_pairs]
    receptors = [rec for _, rec in lr_pairs]
    order = rng.permutation(len(receptors))
    return [(ligands[i], receptors[order[i]]) for i in range(len(ligands))]


def edge_jaccard(a, b):
    if a.shape[1] == 0 and b.shape[1] == 0:
        return 1.0
    sa = set(zip(a[0].tolist(), a[1].tolist()))
    sb = set(zip(b[0].tolist(), b[1].tolist()))
    union = len(sa | sb)
    return len(sa & sb) / union if union else 0.0


def lr_permutation_diagnostics(
    Xn,
    genes,
    lr_pairs,
    real_g1,
    n_permutations,
    max_edges,
    n_top,
    per_pair_topk,
    max_pairs=None,
    ligand_target_prior=None,
    ligand_target_weight=1.0,
    n_jobs=1,
    age_labels=None,
    same_age_only=False,
    seed=42,
):
    if n_permutations <= 0 or len(lr_pairs) < 2 or real_g1.shape[1] == 0:
        return {}
    rng = np.random.default_rng(seed)
    pairs_used = list(lr_pairs)
    real_ref = real_g1
    if max_pairs is not None and max_pairs > 0 and len(pairs_used) > max_pairs:
        pair_idx = rng.choice(len(pairs_used), size=max_pairs, replace=False)
        pairs_used = [pairs_used[i] for i in pair_idx]
        real_ref, _, _, _ = build_lr_prior_graph(
            Xn,
            genes,
            lr_pairs=pairs_used,
            max_edges=max_edges,
            n_top=n_top,
            per_pair_topk=per_pair_topk,
            ligand_target_prior=ligand_target_prior,
            ligand_target_weight=ligand_target_weight,
            n_jobs=n_jobs,
            age_labels=age_labels,
            same_age_only=same_age_only,
        )
    jaccards = []
    edge_counts = []
    for _ in range(n_permutations):
        shuffled = shuffle_lr_pairs(pairs_used, rng)
        g_perm, _, _, _ = build_lr_prior_graph(
            Xn,
            genes,
            lr_pairs=shuffled,
            max_edges=max_edges,
            n_top=n_top,
            per_pair_topk=per_pair_topk,
            ligand_target_prior=ligand_target_prior,
            ligand_target_weight=ligand_target_weight,
            n_jobs=n_jobs,
            age_labels=age_labels,
            same_age_only=same_age_only,
        )
        jaccards.append(edge_jaccard(real_ref, g_perm))
        edge_counts.append(g_perm.shape[1])
    return {
        'lr_perm_edge_jaccard_mean': float(np.mean(jaccards)),
        'lr_perm_edge_jaccard_std': float(np.std(jaccards)),
        'lr_perm_edges_mean': float(np.mean(edge_counts)),
        'lr_perm_pairs_used': int(len(pairs_used)),
    }


def apply_runtime_preset(runtime_preset, params, protected=None):
    if runtime_preset != 'large':
        return params
    capped = dict(params)
    protected = set(protected or [])
    caps = {
        'max_edges_g1': 10000,
        'lr_top_cells': 500,
        'lr_per_pair_topk': 1000,
        'predicted_top_edges': 3000,
        'n_epochs': 40,
        'k': 10,
        'max_train_edges': 10000,
    }
    for key, cap in caps.items():
        if key not in protected:
            capped[key] = min(capped[key], cap)
    return capped


def _explicit_cli_settings(args, names):
    out = set()
    for name in names:
        if getattr(args, name, None) is not None:
            out.add(name)
    return out


def print_large_run_note(n_cells, lr_pairs_count, lr_top_cells, lr_per_pair_topk, n_lr_permutations, lr_permutation_max_pairs):
    if n_cells < 10000:
        return
    effective_top = min(lr_top_cells, n_cells)
    local_candidates = effective_top * effective_top
    print(
        'Runtime note: large dataset detected '
        f'(cells={n_cells}, lr_pairs={lr_pairs_count}). '
        f'Each LR pair can score up to {local_candidates:,} sender-receiver candidates '
        f'before retaining {lr_per_pair_topk:,}.'
    )
    print(
        'Runtime note: for large full-data runs, consider '
        '--runtime-preset large, --n-lr-permutations 0, and keep '
        '--lr-permutation-max-pairs around 500 for later null diagnostics.'
    )
    if n_lr_permutations > 0 and (lr_permutation_max_pairs is None or lr_permutation_max_pairs <= 0):
        print('Runtime warning: LR permutations are using all LR pairs; set --lr-permutation-max-pairs 500 to cap cost.')


def apply_metadata_filters(X, meta: pd.DataFrame, cells, filters):
    if not filters:
        return X, meta, cells
    mask = np.ones(len(meta), dtype=bool)
    for expr in filters:
        if '=' not in expr:
            raise ValueError(f'metadata filter must use COLUMN=VALUE syntax: {expr}')
        col, value = expr.split('=', 1)
        col = col.strip()
        value = value.strip()
        if col not in meta.columns:
            raise ValueError(f'metadata filter column not found: {col}')
        allowed = {v.strip() for v in value.split('|')}
        mask &= meta[col].astype(str).isin(allowed).values
    if mask.sum() == 0:
        raise ValueError(f'metadata filters removed all cells: {filters}')
    idx = np.where(mask)[0]
    return X[idx], meta.iloc[idx].reset_index(drop=True), cells[idx]


def write_run_summary(
    out_dir: Path,
    meta: pd.DataFrame,
    readiness,
    metrics,
    g1,
    g2,
    predicted_edges,
    lr_source,
    lr_pairs,
    celltype_col,
    age_col,
    params,
):
    src, tgt = g2[0], g2[1]
    same_celltype = meta[celltype_col].iloc[src].values == meta[celltype_col].iloc[tgt].values
    same_age = meta[age_col].iloc[src].values == meta[age_col].iloc[tgt].values
    readiness_series = pd.Series(readiness)
    summary = {
        'n_cells': int(len(meta)),
        'n_celltypes': int(meta[celltype_col].nunique()),
        'n_ages': int(meta[age_col].nunique()),
        'celltype_col': celltype_col,
        'age_col': age_col,
        'lr_source': str(lr_source),
        'n_lr_pairs_used': int(len(lr_pairs)),
        'g1_edges': int(g1.shape[1]),
        'g2_edges': int(g2.shape[1]),
        'gnn_predicted_edges': int(predicted_edges.shape[1]) if predicted_edges is not None else 0,
        'g2_same_celltype_fraction': float(same_celltype.mean()) if len(same_celltype) else 0.0,
        'g2_same_age_fraction': float(same_age.mean()) if len(same_age) else 0.0,
        'readiness_min': float(readiness_series.min()),
        'readiness_median': float(readiness_series.median()),
        'readiness_mean': float(readiness_series.mean()),
        'readiness_max': float(readiness_series.max()),
        **{f'metric_{k}': v for k, v in metrics.items()},
        **{f'param_{k}': v for k, v in params.items()},
    }
    pd.DataFrame([summary]).to_csv(out_dir / 'run_summary.csv', index=False)

    by_age = (
        pd.DataFrame({age_col: meta[age_col].values, 'readiness': readiness})
        .groupby(age_col)['readiness']
        .agg(['count', 'mean', 'std', 'min', 'median', 'max'])
        .reset_index()
    )
    by_age.to_csv(out_dir / 'readiness_by_age_summary.csv', index=False)


def _load_config(path: Path | None):
    cfg = {}
    if path and path.exists():
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError('PyYAML is required when using --config') from exc
        with path.open('r', encoding='utf-8') as handle:
            cfg = yaml.safe_load(handle) or {}
    return cfg


def _path_from_config(value, default):
    return Path(value if value is not None else default).expanduser()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Infer temporal cell-cell interaction potential and developmental readiness.'
    )
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG, help='YAML config file.')
    parser.add_argument('--expression-path', type=Path, help='Expression matrix file with cell and gene names.')
    parser.add_argument('--metadata-path', type=Path, help='Cell metadata CSV.')
    parser.add_argument('--out-dir', type=Path, help='Directory for result tables, arrays, and figures.')
    parser.add_argument('--celltype-col', help='Cell metadata column with cell-type labels.')
    parser.add_argument('--age-col', help='Cell metadata column with developmental stage labels.')
    parser.add_argument('--seed', type=int, help='Random seed for SVD, sampling, GNN initialization, and permutation diagnostics.')
    parser.add_argument('--runtime-preset', choices=['standard', 'large'], help='Runtime preset. Use large for datasets with >=10k cells.')
    parser.add_argument('--cellchat-dir', type=Path, help='Directory containing CellChat LR resources.')
    parser.add_argument('--cellphonedb-dir', type=Path, help='Directory containing CellPhoneDB interaction_input.csv.')
    parser.add_argument('--lr-pairs-path', type=Path, help='Optional normalized LR CSV with ligand and receptor columns.')
    parser.add_argument('--ligand-target-prior-path', type=Path, help='Optional NicheNet-style ligand,target,weight CSV/TSV or matrix CSV.')
    parser.add_argument('--ligand-target-weight', type=float, help='Weight of ligand-target activity in LR edge scoring.')
    parser.add_argument('--max-targets-per-ligand', type=int, help='Maximum NicheNet targets retained per ligand.')
    parser.add_argument('--lr-same-age-only', action=argparse.BooleanOptionalAction, default=None, help='Restrict LR-prior biological edges to cells from the same age/timepoint.')
    parser.add_argument('--metadata-filter', action='append', default=[], help='Filter cells before QC using COLUMN=VALUE or COLUMN=A|B. Repeatable.')
    parser.add_argument('--readiness-target-path', type=Path, help='Optional cell-level readiness/progression target CSV.')
    parser.add_argument('--readiness-target-cell-col', default=None, help='Cell ID column in readiness target CSV.')
    parser.add_argument('--readiness-target-col', default=None, help='Numeric target column in readiness target CSV.')
    parser.add_argument('--n-components', type=int, help='Number of SVD components for transcriptomic embedding.')
    parser.add_argument('--k', type=int, help='k for transcriptomic kNN graph.')
    parser.add_argument('--max-edges-g1', type=int, help='Maximum LR-prior edges retained globally.')
    parser.add_argument('--lr-top-cells', type=int, help='Top ligand/receptor expressing cells considered per LR pair.')
    parser.add_argument('--lr-per-pair-topk', type=int, help='Maximum candidate edges retained per LR pair before global merge.')
    parser.add_argument('--n-jobs', type=int, help='Worker threads for LR-prior graph construction.')
    parser.add_argument('--top-lr-per-point', type=int, help='Top LR pairs exported for each celltype-age line-plot point.')
    parser.add_argument('--predicted-top-edges', type=int, help='Top scored LR-candidate edges exported as GNN biological readout.')
    parser.add_argument('--max-train-edges', type=int, help='Maximum positive LR edges sampled for GNN training before matched negatives.')
    parser.add_argument('--edge-batch-size', type=int, help='Mini-batch size for edge prediction loss. Use 0 for full-batch edge loss.')
    parser.add_argument('--torch-threads', type=int, help='Number of PyTorch CPU threads. Use 0 for PyTorch default.')
    parser.add_argument('--skip-plots', action='store_true', help='Write tables and arrays but skip figure generation.')
    parser.add_argument('--plot-format', choices=['png', 'pdf', 'both'], help='Figure format when plots are enabled.')
    parser.add_argument('--plot-parts', help='Comma-separated plot parts: heatmap,line,readiness,all,none. Overrides --skip-plots when set.')
    parser.add_argument('--n-epochs', type=int, help='Number of GNN training epochs.')
    parser.add_argument('--lambda-readiness', type=float, help='Weight for transition-edge readiness smoothness.')
    parser.add_argument('--lambda-time', type=float, help='Weight for celltype-age readiness smoothness.')
    parser.add_argument('--lambda-readiness-target', type=float, help='Weight for supervised/weakly supervised readiness target loss.')
    parser.add_argument('--n-label-permutations', type=int, help='Number of label permutations for edge-score null diagnostics.')
    parser.add_argument('--n-lr-permutations', type=int, help='Number of LR-pair shuffles for G1 graph null diagnostics.')
    parser.add_argument('--lr-permutation-max-pairs', type=int, help='Maximum LR pairs used per LR-permutation diagnostic. Use 0 for all pairs.')
    return parser.parse_args(argv)


def _setting(args, cfg, name, default):
    cli_name = name.replace('-', '_')
    value = getattr(args, cli_name)
    if value is not None:
        return value
    return cfg.get(name.replace('-', '_'), default)


def main(argv=None):
    t_run_start = time.perf_counter()
    args = parse_args(argv)
    cfg = _load_config(args.config)

    expression_path = (
        _path_from_config(args.expression_path or cfg.get('expression_path'), '')
        if (args.expression_path or cfg.get('expression_path')) else None
    )
    metadata_path = (
        _path_from_config(args.metadata_path or cfg.get('metadata_path'), '')
        if (args.metadata_path or cfg.get('metadata_path')) else None
    )
    if not (expression_path and metadata_path):
        raise ValueError('Provide both expression_path and metadata_path. DyCCC input is expression.csv plus metadata.csv.')
    out_dir = _path_from_config(args.out_dir or cfg.get('out_dir'), 'output')
    celltype_col = _setting(args, cfg, 'celltype-col', 'cell_type')
    age_col = _setting(args, cfg, 'age-col', 'age')
    seed = int(_setting(args, cfg, 'seed', 42))
    np.random.seed(seed)
    runtime_preset = _setting(args, cfg, 'runtime-preset', 'standard')
    n_components = int(_setting(args, cfg, 'n-components', 50))
    k = int(_setting(args, cfg, 'k', 10))
    max_edges_g1 = int(_setting(args, cfg, 'max-edges-g1', 30000))
    lr_top_cells = int(_setting(args, cfg, 'lr-top-cells', 500))
    lr_per_pair_topk = int(_setting(args, cfg, 'lr-per-pair-topk', 1000))
    n_jobs = int(_setting(args, cfg, 'n-jobs', 4))
    top_lr_per_point = int(_setting(args, cfg, 'top-lr-per-point', 15))
    predicted_top_edges = int(_setting(args, cfg, 'predicted-top-edges', 5000))
    max_train_edges = int(_setting(args, cfg, 'max-train-edges', 10000))
    edge_batch_size = int(_setting(args, cfg, 'edge-batch-size', 8192))
    torch_threads = int(_setting(args, cfg, 'torch-threads', 0))
    skip_plots = bool(args.skip_plots or cfg.get('skip_plots', True))
    plot_parts_arg = args.plot_parts if args.plot_parts is not None else cfg.get('plot_parts')
    n_epochs = int(_setting(args, cfg, 'n-epochs', 40))
    lambda_readiness = float(_setting(args, cfg, 'lambda-readiness', 0.1))
    lambda_time = float(_setting(args, cfg, 'lambda-time', 0.05))
    lambda_readiness_target_setting = _setting(args, cfg, 'lambda-readiness-target', None)
    n_label_permutations = int(_setting(args, cfg, 'n-label-permutations', 20))
    n_lr_permutations = int(_setting(args, cfg, 'n-lr-permutations', 0))
    lr_permutation_max_pairs = int(_setting(args, cfg, 'lr-permutation-max-pairs', 500))
    preset_keys = {
        'k',
        'max_edges_g1',
        'lr_top_cells',
        'lr_per_pair_topk',
        'predicted_top_edges',
        'n_epochs',
        'max_train_edges',
    }
    protected_runtime = _explicit_cli_settings(args, preset_keys)
    runtime_params = apply_runtime_preset(
        runtime_preset,
        {
            'k': k,
            'max_edges_g1': max_edges_g1,
            'lr_top_cells': lr_top_cells,
            'lr_per_pair_topk': lr_per_pair_topk,
            'predicted_top_edges': predicted_top_edges,
            'n_epochs': n_epochs,
            'max_train_edges': max_train_edges,
        },
        protected=protected_runtime,
    )
    k = runtime_params['k']
    max_edges_g1 = runtime_params['max_edges_g1']
    lr_top_cells = runtime_params['lr_top_cells']
    lr_per_pair_topk = runtime_params['lr_per_pair_topk']
    predicted_top_edges = runtime_params['predicted_top_edges']
    n_epochs = runtime_params['n_epochs']
    max_train_edges = runtime_params['max_train_edges']
    if runtime_preset == 'large' and n_jobs <= 1:
        n_jobs = min(4, os.cpu_count() or 1)
    if runtime_preset == 'large' and edge_batch_size <= 0:
        edge_batch_size = 8192
    input_parent = expression_path.parent
    cellchat_dir = _path_from_config(args.cellchat_dir or cfg.get('cellchat_dir'), input_parent / 'CellChat')
    cellphonedb_dir = _path_from_config(args.cellphonedb_dir or cfg.get('cellphonedb_dir'), input_parent / 'CellPhoneDB')
    lr_pairs_path = _path_from_config(args.lr_pairs_path or cfg.get('lr_pairs_path'), '') if (args.lr_pairs_path or cfg.get('lr_pairs_path')) else None
    ligand_target_prior_path = (
        _path_from_config(args.ligand_target_prior_path or cfg.get('ligand_target_prior_path'), '')
        if (args.ligand_target_prior_path or cfg.get('ligand_target_prior_path')) else None
    )
    ligand_target_weight = float(_setting(args, cfg, 'ligand-target-weight', 1.0))
    max_targets_per_ligand = int(_setting(args, cfg, 'max-targets-per-ligand', 100))
    lr_same_age_only = bool(_setting(args, cfg, 'lr-same-age-only', True))
    metadata_filters = args.metadata_filter or cfg.get('metadata_filter', [])
    plot_format = _setting(args, cfg, 'plot-format', 'pdf')
    plot_parts = parse_plot_parts(plot_parts_arg, skip_plots=skip_plots)
    readiness_target_path = (
        _path_from_config(args.readiness_target_path or cfg.get('readiness_target_path'), '')
        if (args.readiness_target_path or cfg.get('readiness_target_path')) else None
    )
    readiness_target_cell_col = args.readiness_target_cell_col or cfg.get('readiness_target_cell_col', 'cell')
    readiness_target_col = args.readiness_target_col or cfg.get('readiness_target_col', 'pseudotime_like')
    if lambda_readiness_target_setting is None:
        lambda_readiness_target = 1.0 if readiness_target_path else 0.0
    else:
        lambda_readiness_target = float(lambda_readiness_target_setting)

    out_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir = out_dir / '.matplotlib'
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(mpl_config_dir))

    print('Step1 Input data...')
    t_step = time.perf_counter()
    X_raw, meta, genes, cells = load_expression_metadata(
        expression_path,
        metadata_path,
        out_dir / '.dyccc_input_cache',
    )
    X_raw, meta, cells = apply_metadata_filters(X_raw, meta, cells, metadata_filters)
    if celltype_col not in meta.columns:
        raise ValueError(f'missing celltype column: {celltype_col}')
    if age_col not in meta.columns:
        raise ValueError(f'missing age column: {age_col}')
    print(f'Step1 time={time.perf_counter() - t_step:.1f}s')
    print('Step2 Preprocessing...')
    t_step = time.perf_counter()
    mask = qc_mask(meta, n_genes_min=200, pct_mt_max=10.0)
    if mask.sum() < 100:
        raise RuntimeError('Too few cells after QC')
    X = X_raw[mask]
    meta = meta.iloc[np.where(mask)[0]].reset_index(drop=True)
    cells = cells[np.where(mask)[0]]
    Xn = normalize_log1p(X)
    emb = compute_embedding(Xn, n_components=n_components, seed=seed)
    t_emb = time_embedding(meta[age_col].values)
    node_feat = np.hstack([emb, t_emb])
    print(f'Step2 time={time.perf_counter() - t_step:.1f}s')

    print('Step3 Graph construction...')
    t_step = time.perf_counter()
    g2 = build_knn_graph(emb, k=k)
    lr_pairs, lr_source = get_lr_pairs_priority(genes, cellchat_dir, cellphonedb_dir, lr_pairs_path)
    print_large_run_note(
        n_cells=Xn.shape[0],
        lr_pairs_count=len(lr_pairs),
        lr_top_cells=lr_top_cells,
        lr_per_pair_topk=lr_per_pair_topk,
        n_lr_permutations=n_lr_permutations,
        lr_permutation_max_pairs=lr_permutation_max_pairs,
    )
    ligand_target_prior = load_ligand_target_prior(
        ligand_target_prior_path,
        genes,
        max_targets_per_ligand=max_targets_per_ligand,
    ) if ligand_target_prior_path else {}
    print(f'Ligand-target prior ligands={len(ligand_target_prior)}')
    g1, w1, g1_features_raw, g1_evidence = build_lr_prior_graph(
        Xn,
        genes,
        lr_pairs=lr_pairs,
        max_edges=max_edges_g1,
        n_top=lr_top_cells,
        per_pair_topk=lr_per_pair_topk,
        ligand_target_prior=ligand_target_prior,
        ligand_target_weight=ligand_target_weight,
        n_jobs=n_jobs,
        age_labels=meta[age_col].values,
        same_age_only=lr_same_age_only,
    )
    g1_features = scale_edge_features(g1_features_raw)
    t_trans, w_trans = build_transition_edges(meta, age_col, celltype_col, n_sample_per_pair=50, seed=seed)
    print(
        f'G1 edges={g1.shape[1]}, G2 edges={g2.shape[1]}, '
        f'T_trans edges={t_trans.shape[1]}, lr_pairs={len(lr_pairs)}, lr_source={lr_source}'
    )
    metrics_extra = lr_permutation_diagnostics(
        Xn,
        genes,
        lr_pairs,
        g1,
        n_permutations=n_lr_permutations,
        max_edges=max_edges_g1,
        n_top=lr_top_cells,
        per_pair_topk=lr_per_pair_topk,
        max_pairs=lr_permutation_max_pairs,
        ligand_target_prior=ligand_target_prior,
        ligand_target_weight=ligand_target_weight,
        n_jobs=n_jobs,
        age_labels=meta[age_col].values,
        same_age_only=lr_same_age_only,
        seed=seed,
    )
    print(f'Step3 time={time.perf_counter() - t_step:.1f}s')

    if g1.shape[1] == 0:
        n_pos = min(max_train_edges, g2.shape[1])
        idx = np.random.RandomState(seed).choice(g2.shape[1], n_pos, replace=False)
        e_pos = g2[:, idx]
        e_pos_features = np.zeros((e_pos.shape[1], g1_features.shape[1] if g1_features is not None and len(g1_features) else 3), dtype=np.float32)
    else:
        n_pos = min(max_train_edges, g1.shape[1])
        idx = np.random.RandomState(seed).choice(g1.shape[1], n_pos, replace=False)
        e_pos = g1[:, idx]
        e_pos_features = g1_features[idx]
    abundance_scores = np.asarray(Xn.sum(axis=1)).flatten()
    e_neg = sample_matched_neg_edges(e_pos, meta, celltype_col, age_col, abundance_scores, n_pos, forbidden_edges=g1, seed=seed)
    e_neg_features = np.zeros_like(e_pos_features, dtype=np.float32)

    # Step6 age-stratified validation + fallback
    ages = meta[age_col].values
    uniq_ages = sorted(pd.Series(ages).dropna().unique(), key=parse_age)
    train_ages = uniq_ages[:-1] if len(uniq_ages) > 1 else uniq_ages
    val_ages = [uniq_ages[-1]] if len(uniq_ages) > 1 else []

    e_pos_tr, e_pos_val = e_pos, np.zeros((2, 0), dtype=int)
    e_neg_tr, e_neg_val = e_neg, np.zeros((2, 0), dtype=int)
    if len(val_ages) > 0:
        e_pos_tr, e_pos_val, pos_tr_mask, pos_val_mask = split_edges_by_age(e_pos, ages, train_ages, val_ages)
        e_neg_tr, e_neg_val, neg_tr_mask, neg_val_mask = split_edges_by_age(e_neg, ages, train_ages, val_ages)
        e_pos_features_tr, e_pos_features_val = e_pos_features[pos_tr_mask], e_pos_features[pos_val_mask]
        e_neg_features_tr, e_neg_features_val = e_neg_features[neg_tr_mask], e_neg_features[neg_val_mask]
        if e_pos_tr.shape[1] < 10:
            print('Warning: train positives sparse, fallback to full-data training')
            e_pos_tr, e_pos_val = e_pos, np.zeros((2, 0), dtype=int)
            e_neg_tr, e_neg_val = e_neg, np.zeros((2, 0), dtype=int)
            e_pos_features_tr, e_pos_features_val = e_pos_features, np.zeros((0, e_pos_features.shape[1]), dtype=np.float32)
            e_neg_features_tr, e_neg_features_val = e_neg_features, np.zeros((0, e_neg_features.shape[1]), dtype=np.float32)
        else:
            print(f'Age-stratified split: train pos={e_pos_tr.shape[1]} val pos={e_pos_val.shape[1]}')
    else:
        e_pos_features_tr, e_pos_features_val = e_pos_features, np.zeros((0, e_pos_features.shape[1]), dtype=np.float32)
        e_neg_features_tr, e_neg_features_val = e_neg_features, np.zeros((0, e_neg_features.shape[1]), dtype=np.float32)

    print('Step4-5 Two-head GNN + composite loss...')
    t_step = time.perf_counter()
    cts = meta[celltype_col].fillna('NA').astype(str).values
    ct_map = {c: i for i, c in enumerate(sorted(set(cts)))}
    ct_num = np.array([ct_map[c] for c in cts], dtype=int)
    age_num = np.array([parse_age(a) for a in ages], dtype=int)
    readiness_target = load_readiness_target(
        readiness_target_path,
        meta,
        readiness_target_cell_col,
        readiness_target_col,
    ) if readiness_target_path else None

    readiness, metrics, predicted = train_two_head_gnn(
        node_feat, g2, e_pos_tr, e_neg_tr, t_trans, w_trans,
        age_num, ct_num,
        edge_pos_val=e_pos_val if e_pos_val.shape[1] > 0 else None,
        edge_neg_val=e_neg_val if e_neg_val.shape[1] > 0 else None,
        edge_pos_features=e_pos_features_tr,
        edge_neg_features=e_neg_features_tr,
        edge_pos_val_features=e_pos_features_val if e_pos_val.shape[1] > 0 else None,
        edge_neg_val_features=e_neg_features_val if e_neg_val.shape[1] > 0 else None,
        readiness_target=readiness_target,
        predict_edges=g1 if g1.shape[1] > 0 else g2,
        predict_edge_features=g1_features if g1.shape[1] > 0 else None,
        n_epochs=n_epochs,
        lambda_r=lambda_readiness,
        lambda_t=lambda_time,
        lambda_readiness_target=lambda_readiness_target,
        n_label_permutations=n_label_permutations,
        edge_batch_size=edge_batch_size,
        torch_threads=torch_threads,
        seed=seed,
    )
    print(f'Step4-5 time={time.perf_counter() - t_step:.1f}s')
    metrics.update(metrics_extra)
    predicted_edges, predicted_scores, predicted_pick = select_top_scored_edges(
        predicted.get('edges'),
        predicted.get('scores'),
        predicted_top_edges,
    )
    if g1_evidence is not None and not g1_evidence.empty and len(predicted_pick) > 0:
        evidence = g1_evidence.iloc[predicted_pick].copy()
        evidence.insert(0, 'gnn_score', predicted_scores)
        evidence['source_celltype'] = meta[celltype_col].iloc[evidence['source_idx'].values].values
        evidence['target_celltype'] = meta[celltype_col].iloc[evidence['target_idx'].values].values
        evidence['source_age'] = meta[age_col].iloc[evidence['source_idx'].values].values
        evidence['target_age'] = meta[age_col].iloc[evidence['target_idx'].values].values
        write_evidence_tables(
            evidence,
            out_dir,
            meta=meta,
            celltype_col=celltype_col,
            age_col=age_col,
        )

    print('Step7 Biological readouts...')
    t_step = time.perf_counter()
    lr_prior_plot_parts = set(plot_parts) & {'heatmap'}
    plot_interaction(
        meta,
        g1,
        out_dir / 'lr_prior',
        celltype_col,
        age_col,
        label='lr_prior',
        make_plots=bool(lr_prior_plot_parts),
        plot_format=plot_format,
        plot_parts=lr_prior_plot_parts,
    )
    plot_interaction(
        meta,
        predicted_edges,
        out_dir / 'outgoing',
        celltype_col,
        age_col,
        label='outgoing',
        make_plots=bool(plot_parts),
        plot_format=plot_format,
        plot_parts=plot_parts,
    )
    if 'readiness' in plot_parts:
        plot_readiness(meta, readiness, age_col, out_dir / 'readiness_by_age.png', plot_format=plot_format)
    if 'line' in plot_parts:
        plot_incoming_celltype_summary(out_dir, plot_format=plot_format)
        plot_outgoing_celltype_summary(out_dir, plot_format=plot_format)
    export_top_lr_pairs_per_point(
        Xn=Xn,
        meta=meta,
        genes=genes,
        lr_pairs=lr_pairs,
        edge_index=predicted_edges if predicted_edges.shape[1] > 0 else g1,
        out_dir=out_dir / 'outgoing',
        celltype_col=celltype_col,
        age_col=age_col,
        top_n=top_lr_per_point,
        direction='outgoing',
    )
    export_top_lr_pairs_per_point(
        Xn=Xn,
        meta=meta,
        genes=genes,
        lr_pairs=lr_pairs,
        edge_index=predicted_edges if predicted_edges.shape[1] > 0 else g1,
        out_dir=out_dir / 'incoming',
        celltype_col=celltype_col,
        age_col=age_col,
        top_n=top_lr_per_point,
        direction='incoming',
    )

    pd.DataFrame([metrics]).to_csv(out_dir / 'metrics.csv', index=False)
    np.save(out_dir / 'edge_index_g1.npy', g1)
    np.save(out_dir / 'edge_index_g2.npy', g2)
    np.save(out_dir / 'edge_index_gnn_predicted.npy', predicted_edges)
    np.save(out_dir / 'edge_scores_gnn_predicted.npy', predicted_scores)
    np.save(out_dir / 'readiness.npy', readiness)
    meta.to_csv(out_dir / 'metadata.csv', index=False)
    write_run_summary(
        out_dir=out_dir,
        meta=meta,
        readiness=readiness,
        metrics=metrics,
        g1=g1,
        g2=g2,
        predicted_edges=predicted_edges,
        lr_source=lr_source,
        lr_pairs=lr_pairs,
        celltype_col=celltype_col,
        age_col=age_col,
        params={
            'expression_path': str(expression_path) if expression_path else '',
            'metadata_path': str(metadata_path) if metadata_path else '',
            'runtime_preset': runtime_preset,
            'seed': seed,
            'n_components': n_components,
            'k': k,
            'max_edges_g1': max_edges_g1,
            'lr_top_cells': lr_top_cells,
            'lr_per_pair_topk': lr_per_pair_topk,
            'n_jobs': n_jobs,
            'max_train_edges': max_train_edges,
            'edge_batch_size': edge_batch_size,
            'torch_threads': torch_threads,
            'skip_plots': skip_plots,
            'plot_parts': ','.join(sorted(plot_parts)),
            'n_label_permutations': n_label_permutations,
            'n_lr_permutations': n_lr_permutations,
            'lr_permutation_max_pairs': lr_permutation_max_pairs,
            'ligand_target_prior_path': str(ligand_target_prior_path) if ligand_target_prior_path else '',
            'ligand_target_prior_ligands': len(ligand_target_prior),
            'ligand_target_weight': ligand_target_weight,
            'max_targets_per_ligand': max_targets_per_ligand,
            'lr_same_age_only': lr_same_age_only,
            'plot_format': plot_format,
            'predicted_top_edges': predicted_top_edges,
            'n_epochs': n_epochs,
            'lambda_readiness': lambda_readiness,
            'lambda_time': lambda_time,
            'lambda_readiness_target': lambda_readiness_target,
        },
    )
    print(f'Step7 time={time.perf_counter() - t_step:.1f}s')
    print(f'Total runtime={time.perf_counter() - t_run_start:.1f}s')

    print('Done. Metrics:', metrics)
    print('lr_prior_readout:', out_dir / 'lr_prior')
    print('incoming_readout:', out_dir / 'incoming')
    print('outgoing_readout:', out_dir / 'outgoing')


if __name__ == '__main__':
    main()
