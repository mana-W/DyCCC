#!/usr/bin/env python3
"""Normalize ligand-receptor resources to DyCCC's ligand,receptor CSV format."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SOURCE_COLUMNS = {
    "cellchat": ("ligand", "receptor"),
    "cellphonedb": ("partner_a", "partner_b"),
    "generic": ("ligand", "receptor"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=sorted(SOURCE_COLUMNS), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    ligand_col, receptor_col = SOURCE_COLUMNS[args.source]
    df = pd.read_csv(args.input)
    missing = [col for col in [ligand_col, receptor_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {args.source}: {', '.join(missing)}")
    out = (
        df[[ligand_col, receptor_col]]
        .rename(columns={ligand_col: "ligand", receptor_col: "receptor"})
        .dropna()
        .drop_duplicates()
        .sort_values(["ligand", "receptor"])
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved {len(out)} ligand-receptor pairs to {args.output}")


if __name__ == "__main__":
    main()
