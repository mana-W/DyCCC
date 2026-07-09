#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from dyccc.pipeline import (
    parse_plot_parts,
    plot_incoming_celltype_summary,
    plot_interaction,
    plot_outgoing_celltype_summary,
    plot_readiness,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate DyCCC plots from an existing output directory without rerunning the model."
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Existing DyCCC output directory.")
    parser.add_argument("--celltype-col", help="Cell-type metadata column. Defaults to run_summary.csv value.")
    parser.add_argument("--age-col", help="Age/timepoint metadata column. Defaults to run_summary.csv value.")
    parser.add_argument("--plot-format", choices=["png", "pdf", "both"], default="pdf")
    parser.add_argument(
        "--plot-parts",
        default="line,readiness",
        help="Comma-separated plot parts: heatmap,line,readiness,all,none.",
    )
    parser.add_argument(
        "--which",
        choices=["lr_prior", "gnn_predicted", "both"],
        default="both",
        help="Which interaction readouts to plot.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir
    run_summary_path = out_dir / "run_summary.csv"
    run_summary = pd.read_csv(run_summary_path).iloc[0].to_dict() if run_summary_path.exists() else {}
    celltype_col = args.celltype_col or run_summary.get("celltype_col")
    age_col = args.age_col or run_summary.get("age_col")
    if not celltype_col or not age_col:
        raise ValueError("--celltype-col and --age-col are required when run_summary.csv is unavailable.")

    mpl_dir = out_dir / ".matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

    parts = parse_plot_parts(args.plot_parts, skip_plots=False)
    meta = pd.read_csv(out_dir / "metadata.csv")

    if args.which in {"lr_prior", "both"}:
        g1 = np.load(out_dir / "edge_index_g1.npy")
        lr_prior_parts = set(parts) & {"heatmap"}
        plot_interaction(
            meta,
            g1,
            out_dir / "lr_prior",
            celltype_col,
            age_col,
            label="lr_prior",
            make_plots=bool(lr_prior_parts),
            plot_format=args.plot_format,
            plot_parts=lr_prior_parts,
        )

    if args.which in {"gnn_predicted", "both"}:
        pred = np.load(out_dir / "edge_index_gnn_predicted.npy")
        plot_interaction(
            meta,
            pred,
            out_dir / "outgoing",
            celltype_col,
            age_col,
            label="outgoing",
            make_plots=bool(parts),
            plot_format=args.plot_format,
            plot_parts=parts,
        )

    if "readiness" in parts:
        readiness = np.load(out_dir / "readiness.npy")
        plot_readiness(
            meta,
            readiness,
            age_col,
            out_dir / "readiness_by_age.png",
            plot_format=args.plot_format,
        )

    if "line" in parts:
        plot_incoming_celltype_summary(out_dir, plot_format=args.plot_format)
        plot_outgoing_celltype_summary(out_dir, plot_format=args.plot_format)



if __name__ == "__main__":
    main()
