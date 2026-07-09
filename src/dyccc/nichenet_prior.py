"""NicheNet-style ligand-target prior utilities."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_ligand_target_prior(path: Path | None, genes, max_targets_per_ligand=100):
    """Load a ligand-target prior from long table or matrix CSV."""
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ligand-target prior not found: {path}")
    if path.suffix.lower() in {".tsv", ".txt"}:
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}
    if {"ligand", "target", "weight"}.issubset(lower):
        ligand_col, target_col, weight_col = lower["ligand"], lower["target"], lower["weight"]
        long_df = df[[ligand_col, target_col, weight_col]].rename(
            columns={ligand_col: "ligand", target_col: "target", weight_col: "weight"}
        )
    else:
        first_col = df.columns[0]
        long_df = df.melt(id_vars=first_col, var_name="ligand", value_name="weight")
        long_df = long_df.rename(columns={first_col: "target"})

    gene_set = set(genes)
    prior = {}
    for ligand, sub in long_df.dropna().groupby("ligand"):
        sub = sub[sub["target"].isin(gene_set)].copy()
        sub["weight"] = pd.to_numeric(sub["weight"], errors="coerce")
        sub = sub.dropna(subset=["weight"])
        sub = sub[sub["weight"] > 0]
        if sub.empty:
            continue
        sub = sub.sort_values("weight", ascending=False).head(max_targets_per_ligand)
        prior[str(ligand)] = (
            sub["target"].astype(str).tolist(),
            sub["weight"].astype(float).to_numpy(dtype=np.float32),
        )
    return prior


def ligand_target_activity(X, gene_to_idx, ligand, ligand_target_prior):
    """Score each receiver cell by weighted expression of predicted target genes."""
    if ligand not in ligand_target_prior:
        return np.zeros(X.shape[0], dtype=np.float32), ""
    targets, weights = ligand_target_prior[ligand]
    kept = [(gene_to_idx[g], float(w), g) for g, w in zip(targets, weights) if g in gene_to_idx]
    if not kept:
        return np.zeros(X.shape[0], dtype=np.float32), ""

    idx = [x[0] for x in kept]
    weights = np.array([x[1] for x in kept], dtype=np.float32)
    kept_targets = [x[2] for x in kept]
    denom = float(np.abs(weights).sum()) or 1.0
    activity = np.asarray(X[:, idx].dot(weights / denom)).flatten().astype(np.float32)
    if activity.max() > activity.min():
        activity = (activity - activity.min()) / (activity.max() - activity.min())
    return activity, ";".join(kept_targets[:20])
