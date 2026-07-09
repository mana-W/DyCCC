"""Basic import tests for DyCCC package."""


def test_import_pipeline():
    import dyccc.pipeline as pipeline

    assert hasattr(pipeline, "main")


def test_cli_parser_accepts_core_options():
    from dyccc.pipeline import parse_args

    args = parse_args([
        "--expression-path", "data/example_expression.csv",
        "--metadata-path", "data/example_metadata.csv",
        "--out-dir", "output/example",
        "--celltype-col", "cell_type",
        "--age-col", "age",
        "--seed", "123",
        "--runtime-preset", "large",
        "--lr-pairs-path", "data/lr_pairs.csv",
        "--ligand-target-prior-path", "data/nichenet_prior.csv",
        "--ligand-target-weight", "2.0",
        "--max-targets-per-ligand", "50",
        "--lr-same-age-only",
        "--n-jobs", "4",
        "--metadata-filter", "condition=treated",
        "--readiness-target-path", "data/cytotrace.csv",
        "--readiness-target-col", "pseudotime_like",
        "--lambda-readiness-target", "1.0",
        "--n-label-permutations", "2",
        "--n-lr-permutations", "2",
        "--lr-permutation-max-pairs", "500",
        "--max-train-edges", "8000",
        "--edge-batch-size", "2048",
        "--torch-threads", "2",
        "--skip-plots",
        "--plot-format", "both",
        "--plot-parts", "line,readiness",
        "--n-epochs", "1",
    ])

    assert str(args.expression_path) == "data/example_expression.csv"
    assert str(args.metadata_path) == "data/example_metadata.csv"
    assert str(args.out_dir) == "output/example"
    assert args.celltype_col == "cell_type"
    assert args.age_col == "age"
    assert args.seed == 123
    assert args.runtime_preset == "large"
    assert str(args.lr_pairs_path) == "data/lr_pairs.csv"
    assert str(args.ligand_target_prior_path) == "data/nichenet_prior.csv"
    assert args.ligand_target_weight == 2.0
    assert args.max_targets_per_ligand == 50
    assert args.lr_same_age_only is True
    assert args.n_jobs == 4
    assert args.metadata_filter == ["condition=treated"]
    assert str(args.readiness_target_path) == "data/cytotrace.csv"
    assert args.readiness_target_col == "pseudotime_like"
    assert args.lambda_readiness_target == 1.0
    assert args.n_label_permutations == 2
    assert args.n_lr_permutations == 2
    assert args.lr_permutation_max_pairs == 500
    assert args.max_train_edges == 8000
    assert args.edge_batch_size == 2048
    assert args.torch_threads == 2
    assert args.skip_plots is True
    assert args.plot_format == "both"
    assert args.plot_parts == "line,readiness"
    assert args.n_epochs == 1


def test_large_preset_keeps_explicit_cli_runtime_values():
    from dyccc.pipeline import _explicit_cli_settings, apply_runtime_preset, parse_args

    args = parse_args([
        "--runtime-preset", "large",
        "--max-edges-g1", "30000",
    ])
    protected = _explicit_cli_settings(args, {"max_edges_g1", "predicted_top_edges"})
    params = apply_runtime_preset(
        "large",
        {
            "max_edges_g1": 30000,
            "predicted_top_edges": 30000,
            "k": 15,
            "lr_top_cells": 1500,
            "lr_per_pair_topk": 4000,
            "n_epochs": 80,
            "max_train_edges": 20000,
        },
        protected=protected,
    )

    assert params["max_edges_g1"] == 30000
    assert params["predicted_top_edges"] == 3000

def test_ligand_target_prior_loader(tmp_path):
    import pandas as pd
    from dyccc.nichenet_prior import load_ligand_target_prior

    path = tmp_path / "prior.csv"
    pd.DataFrame({
        "ligand": ["L1", "L1", "L2"],
        "target": ["G1", "missing", "G2"],
        "weight": [0.9, 0.8, 0.7],
    }).to_csv(path, index=False)

    prior = load_ligand_target_prior(path, genes=["G1", "G2"])

    assert set(prior) == {"L1", "L2"}
    assert prior["L1"][0] == ["G1"]


def test_lr_permutation_max_pairs_caps_diagnostics():
    import numpy as np
    from scipy.sparse import csr_matrix

    from dyccc.pipeline import build_lr_prior_graph, lr_permutation_diagnostics

    genes = np.array(["L1", "L2", "L3", "R1", "R2", "R3"])
    X = csr_matrix([
        [4, 0, 0, 0, 3, 1],
        [0, 4, 0, 3, 0, 1],
        [0, 0, 4, 1, 3, 0],
        [3, 0, 0, 0, 1, 4],
    ], dtype=float)
    lr_pairs = [
        (["L1"], ["R1"]),
        (["L2"], ["R2"]),
        (["L3"], ["R3"]),
    ]
    g1, _, _, _ = build_lr_prior_graph(X, genes, lr_pairs, max_edges=20, n_top=3, per_pair_topk=5)
    metrics = lr_permutation_diagnostics(
        X,
        genes,
        lr_pairs,
        g1,
        n_permutations=1,
        max_edges=20,
        n_top=3,
        per_pair_topk=5,
        max_pairs=2,
    )

    assert metrics["lr_perm_pairs_used"] == 2


def test_lr_same_age_only_keeps_edges_within_timepoint():
    import numpy as np
    from scipy.sparse import csr_matrix

    from dyccc.pipeline import build_lr_prior_graph

    genes = np.array(["L1", "R1"])
    X = csr_matrix([
        [4, 1],
        [3, 2],
        [2, 3],
        [1, 4],
    ], dtype=float)
    ages = np.array(["P1", "P1", "P4", "P4"])
    g1, _, _, _ = build_lr_prior_graph(
        X,
        genes,
        [(["L1"], ["R1"])],
        max_edges=10,
        n_top=4,
        per_pair_topk=8,
        age_labels=ages,
        same_age_only=True,
    )

    assert g1.shape[1] > 0
    assert np.all(ages[g1[0]] == ages[g1[1]])
