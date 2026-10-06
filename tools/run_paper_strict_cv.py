"""Run the paper's ten-fold RMGCL and contrastive-learning ablations.

Positive pairs are split first. Each outer fold independently reserves test
and validation pairs before scoring or selecting training negatives and PU
samples. Risk scoring and association graphs use only the remaining training
positives. Historical warm-start runs are not comparable to these outputs.

Example:
    python tools/run_paper_strict_cv.py --dataset Fdataset --strategy all \
        --output-root outputs/paper_strict_cv_10fold
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import quick_train as qt


DATASETS = {
    "Cdataset": ("data/cdataset_raw", "data/cdataset_disease_semantic_features"),
    "Fdataset": ("data/fdataset_raw", None),
    "LRSSL": ("data/lrssl_raw", None),
}
STRATEGY_WEIGHTS = {
    "full": (0.75, 0.75),
    "wo_cl": (0.0, 0.0),
    "pair_cl": (0.75, 0.0),
    "node_cl": (0.0, 0.75),
}
BASE_CONFIG = {
    "folds": 10,
    "cv_mode": "random",
    "epochs": 100,
    "batch_size": 2048,
    "embedding_dim": 1024,
    "hidden_dim": 2048,
    "lr": 0.001,
    "contrastive_temperature": 0.2,
    "negative_threshold": 0.75,
    "seed": 42,
    "device_name": "cuda",
    "similarity_top_k": 20,
    "fusion_type": "reliability_gate",
    "fusion_dim": 256,
    "fusion_heads": 4,
    "fusion_layers": 1,
    "decoder_type": "mlp",
    "validation_ratio": 0.1,
    "validation_metric": "AUPR",
    "pu_learning": True,
    "pu_unlabeled_ratio": 1.0,
    "pu_loss_weight": 0.1,
    "rns_strategy": "adaptive_topk",
    "negative_ratio": 1.0,
    "evaluation_protocol": "fold_wise",
    "model_variant": "current",
    "active_views": ("A", "S", "B"),
    "graph_encoder": "gatv2",
}


def config_for_strategy(strategy: str) -> dict[str, object]:
    pair_weight, node_weight = STRATEGY_WEIGHTS[strategy]
    return {
        **BASE_CONFIG,
        "contrastive_weight": pair_weight,
        "node_contrastive_weight": node_weight,
    }


def run_experiment(dataset: str, strategy: str, output_root: Path, device: str) -> Path:
    output_dir = output_root / dataset / strategy
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing experiment: {output_dir}")

    raw_dir, feature_dir = DATASETS[dataset]
    tables = qt.load_raw_tables(
        PROJECT_ROOT / raw_dir,
        feature_dir=PROJECT_ROOT / feature_dir if feature_dir else None,
    )
    config = config_for_strategy(strategy)
    config["device_name"] = device
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "dataset": dataset,
        "strategy": strategy,
        "config": config,
        "quick_train_sha256": hashlib.sha256(Path(qt.__file__).read_bytes()).hexdigest(),
        "negative_sampling_sha256": hashlib.sha256(
            (PROJECT_ROOT / "src" / "negative_sampling.py").read_bytes()
        ).hexdigest(),
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    results = qt.train_kfold_fold_aware(
        tables=tables,
        processed_dir=output_dir / "processed",
        output_dir=output_dir,
        **config,
    )
    if len(results) != config["folds"] or not all(
        result.get("strict_leakage_free")
        and result.get("test_pair_disjoint_from_train")
        and result.get("test_pair_excluded_from_pu")
        and result.get("test_positive_absent_from_association_graph")
        and result.get("validation_pair_disjoint_from_train")
        and result.get("validation_pair_excluded_from_pu")
        and result.get("validation_positive_absent_from_association_graph")
        and result.get("test_evaluation_count") == 1
        for result in results
    ):
        raise RuntimeError(f"Strict fold-wise audit failed: {output_dir}")
    return output_dir / f"summary_metrics_{config['folds']}fold.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], required=True)
    parser.add_argument("--strategy", choices=[*STRATEGY_WEIGHTS, "all"], required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    strategies = STRATEGY_WEIGHTS if args.strategy == "all" else (args.strategy,)
    for dataset in datasets:
        for strategy in strategies:
            summary_path = run_experiment(dataset, strategy, args.output_root, args.device)
            print(f"Completed {dataset}/{strategy}: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
