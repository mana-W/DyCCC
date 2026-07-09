#!/usr/bin/env python3
"""Summarize DyCCC outputs and flag common interpretation risks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--celltype-col", required=True)
    parser.add_argument("--age-col", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    out = args.out_dir
    metrics_path = out / "metrics.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    meta = pd.read_csv(out / "metadata.csv")
    readiness = np.load(out / "readiness.npy")
    g1 = np.load(out / "edge_index_g1.npy")
    g2 = np.load(out / "edge_index_g2.npy")
    pred_path = out / "edge_index_gnn_predicted.npy"
    gpred = np.load(pred_path) if pred_path.exists() else np.zeros((2, 0), dtype=int)

    same_ct = meta[args.celltype_col].iloc[g2[0]].values == meta[args.celltype_col].iloc[g2[1]].values
    same_age = meta[args.age_col].iloc[g2[0]].values == meta[args.age_col].iloc[g2[1]].values

    def diag_fraction(path):
        if not path.exists():
            return np.nan
        mat = pd.read_csv(path, index_col=0)
        total = mat.values.sum()
        return float(np.trace(mat.values) / total) if total else np.nan

    lr_matrix = out / "lr_prior" / "lr_prior_matrix.csv"
    pred_matrix = out / "outgoing" / "outgoing_matrix.csv"
    lr_prior_diag_fraction = diag_fraction(lr_matrix)
    predicted_diag_fraction = diag_fraction(pred_matrix)

    summary = {
        "n_cells": int(len(meta)),
        "n_celltypes": int(meta[args.celltype_col].nunique()),
        "n_ages": int(meta[args.age_col].nunique()),
        "g1_edges": int(g1.shape[1]),
        "g2_edges": int(g2.shape[1]),
        "gnn_predicted_edges": int(gpred.shape[1]),
        "g2_same_celltype_fraction": float(same_ct.mean()),
        "g2_same_age_fraction": float(same_age.mean()),
        "lr_prior_diag_fraction": lr_prior_diag_fraction,
        "gnn_predicted_diag_fraction": predicted_diag_fraction,
        "readiness_min": float(np.min(readiness)),
        "readiness_median": float(np.median(readiness)),
        "readiness_mean": float(np.mean(readiness)),
        "readiness_max": float(np.max(readiness)),
    }
    for key, value in (metrics.iloc[0].to_dict() if not metrics.empty else {}).items():
        summary[f"metric_{key}"] = value

    flags = []
    if not np.isnan(summary["gnn_predicted_diag_fraction"]) and summary["gnn_predicted_diag_fraction"] > 0.8:
        flags.append("GNN-predicted readout is dominated by same-celltype edges")
    if summary["readiness_median"] < 1e-8 and summary["readiness_max"] < 1e-2:
        flags.append("readiness scores are close to zero and may be poorly calibrated")
    if summary["g1_edges"] < 0.1 * summary["g2_edges"]:
        flags.append("LR-prior graph is much smaller than kNN graph")
    if summary.get("metric_label_perm_auroc_mean", 0.5) > 0.6:
        flags.append("label permutation AUROC is high")
    if summary.get("metric_readiness_target_corr", 1.0) < 0.2 and "metric_readiness_target_corr" in summary:
        flags.append("readiness has weak correlation with calibration target")

    report_path = out / "dyccc_run_assessment.csv"
    pd.DataFrame([{**summary, "flags": "; ".join(flags)}]).to_csv(report_path, index=False)
    print(pd.DataFrame([{**summary, "flags": "; ".join(flags)}]).T.to_string(header=False))
    print(f"Saved assessment to {report_path}")


if __name__ == "__main__":
    main()
