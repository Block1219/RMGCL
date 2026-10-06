from __future__ import annotations

"""Docstring."""

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_schema import validate_project_tables
    from model import (
        GRAPH_ENCODERS,
        MultiViewDrugDiseaseModel,
        SemanticPathSpectralDDAModel,
        node_contrastive_loss,
        pair_contrastive_loss,
        weighted_bce_loss,
    )
    from negative_sampling import NegativeSamplerConfig, ReliableNegativeSampler
else:
    from .data_schema import validate_project_tables
    from .model import (
        GRAPH_ENCODERS,
        MultiViewDrugDiseaseModel,
        SemanticPathSpectralDDAModel,
        node_contrastive_loss,
        pair_contrastive_loss,
        weighted_bce_loss,
    )
    from .negative_sampling import NegativeSamplerConfig, ReliableNegativeSampler


OPTIONAL_FEATURE_FILES = {
    "drug_smiles": ("drug_smiles.csv", "drug_smiles.tsv", "smiles.csv"),
    "drug_fingerprint": ("drug_fingerprint.csv", "DrugFingerprint.csv", "fingerprint.csv"),
    "disease_mesh": ("disease_mesh.csv", "mesh.csv", "disease_mesh.tsv"),
    "disease_gene": ("disease_gene.csv", "disease_gene.tsv", "gene_disease.csv"),
}

SPLIT_MODES = ("random", "cold-drug", "cold-disease", "double-cold")
EVALUATION_PROTOCOL_FOLD_WISE = "fold_wise"
EVALUATION_PROTOCOL_GLOBAL_PREPROCESS = "global_preprocess"
EVALUATION_PROTOCOL_P1_TRANSDUCTIVE = "p1_transductive"
EVALUATION_PROTOCOL_BALANCED_WARM_START = "balanced_warm_start"
EVALUATION_PROTOCOLS = (
    EVALUATION_PROTOCOL_FOLD_WISE,
    EVALUATION_PROTOCOL_GLOBAL_PREPROCESS,
    EVALUATION_PROTOCOL_P1_TRANSDUCTIVE,
    EVALUATION_PROTOCOL_BALANCED_WARM_START,
)
GLOBAL_PREPROCESSING_PROTOCOL_NAME = "Global-preprocessing protocol"
P1_TRANSDUCTIVE_PROTOCOL_NAME = "P1 transductive split protocol"
BALANCED_WARM_START_PROTOCOL_NAME = "Balanced warm-start protocol"
NCH_P1_RELIABLE_PROTOCOL_NAME = "NCH-P1-style reliable-negative protocol"
RNS_STRATEGY_THRESHOLD = "threshold"
RNS_STRATEGY_ADAPTIVE_TOPK = "adaptive_topk"
RNS_STRATEGIES = (RNS_STRATEGY_THRESHOLD, RNS_STRATEGY_ADAPTIVE_TOPK)
BALANCED_NEGATIVE_POOL_RANDOM = "random"
BALANCED_NEGATIVE_POOL_RELIABLE = "reliable"
BALANCED_NEGATIVE_POOL_STRATEGIES = (
    BALANCED_NEGATIVE_POOL_RANDOM,
    BALANCED_NEGATIVE_POOL_RELIABLE,
)
DEFAULT_NEGATIVE_RATIO = 0.5
MODEL_VARIANT_CURRENT = "current"
MODEL_VARIANT_NO_SS = "no_ss"
MODEL_VARIANT_SEMANTIC_PATH_SPECTRAL = "semantic_path_spectral"
MODEL_VARIANTS = (
    MODEL_VARIANT_CURRENT,
    MODEL_VARIANT_NO_SS,
    MODEL_VARIANT_SEMANTIC_PATH_SPECTRAL,
)


LIGHTGBM_DECODER_DEFAULTS = {
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_jobs": 4,
}
RISK_FEATURE_COLUMNS = [
    "reliability",
    "false_negative_risk",
    "drug_similarity_risk",
    "disease_similarity_risk",
    "biological_path_risk",
    "topology_risk",
    "cold_start_uncertainty",
]
DEFAULT_RISK_FEATURE_VALUES = {
    "reliability": 1.0,
    "false_negative_risk": 0.0,
    "drug_similarity_risk": 0.0,
    "disease_similarity_risk": 0.0,
    "biological_path_risk": 0.0,
    "topology_risk": 0.0,
    "cold_start_uncertainty": 0.0,
}
SAMPLE_ROLE_COLUMN = "sample_role"
ROLE_POSITIVE = "positive"
ROLE_RELIABLE_NEGATIVE = "reliable_negative"
ROLE_BALANCED_RANDOM_NEGATIVE = "balanced_random_negative"
ROLE_PU_UNLABELED = "pu_unlabeled"
ROLE_CODES = {
    ROLE_POSITIVE: 0,
    ROLE_RELIABLE_NEGATIVE: 1,
    ROLE_BALANCED_RANDOM_NEGATIVE: 1,
    ROLE_PU_UNLABELED: 2,
}
ROLE_NAMES = {value: key for key, value in ROLE_CODES.items()}


@dataclass
class GlobalPreprocessedData:
    """全局预处理阶段一次性生成的数据和统计信息。"""

    tables: dict[str, pd.DataFrame]
    positives: pd.DataFrame
    scored_unknown_pairs: pd.DataFrame
    reliable_negatives: pd.DataFrame
    metadata: dict[str, object]


@dataclass
class GlobalGraphContext:
    """Global-preprocessing protocol 中五折共享的固定图上下文。"""

    num_nodes: int
    node_maps: dict[str, dict[str, int]]
    view_edge_indices: list[torch.Tensor]
    node_features: torch.Tensor | None
    drug_spectrum: torch.Tensor | None
    disease_spectrum: torch.Tensor | None
    disease_node_offset: int
    node_feature_stats: dict[str, int | float | bool]
    ss_feature_stats: dict[str, int | bool]
    graph_fingerprint: str
    build_count: int = 1


def add_total_runtime_to_fold_logs(output_dir: Path, total_run_seconds: float) -> None:
    """在所有单折 epoch 日志中补充完整运行总时间。"""

    for log_path in output_dir.glob("training_log_fold_*.csv"):
        fold_log = pd.read_csv(log_path).copy()
        fold_log = fold_log.assign(total_run_seconds=float(total_run_seconds))
        fold_log.to_csv(log_path, index=False)


def load_optional_feature_tables(feature_dirs: list[Path]) -> dict[str, pd.DataFrame]:
    """从原始目录或独立特征目录加载可选药物和疾病属性表。"""

    optional_tables: dict[str, pd.DataFrame] = {}
    for table_name, candidates in OPTIONAL_FEATURE_FILES.items():
        for feature_dir in feature_dirs:
            if not feature_dir.exists():
                continue
            existing_files = {path.name.lower(): path for path in feature_dir.iterdir() if path.is_file()}
            for candidate in candidates:
                path = existing_files.get(candidate.lower())
                if path is None:
                    continue
                sep = "\t" if path.suffix.lower() == ".tsv" else ","
                optional_tables[table_name] = pd.read_csv(path, sep=sep)
                break
            if table_name in optional_tables:
                break
    return optional_tables


def load_raw_tables(raw_dir: Path, feature_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """加载必需原始表，并可额外注入独立目录中的属性特征。"""

    tables = validate_project_tables(raw_dir)
    feature_dirs = [raw_dir]
    if feature_dir is not None:
        feature_dirs.append(feature_dir)
    tables.update(load_optional_feature_tables(feature_dirs))
    return tables


def make_split(
    examples: pd.DataFrame,
    mode: str,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Docstring."""

    rng = np.random.default_rng(seed)
    examples = examples.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if mode == "random":
        test_size = max(1, int(round(len(examples) * test_ratio)))
        test_indices = set(rng.choice(examples.index.to_numpy(), size=test_size, replace=False))
        test = examples[examples.index.isin(test_indices)].copy()
        train = examples[~examples.index.isin(test_indices)].copy()
        return train.reset_index(drop=True), test.reset_index(drop=True)

    if mode == "double-cold":
        test_size = max(1, int(round(len(examples) * test_ratio)))
        if len(examples) > 1:
            test_size = min(test_size, len(examples) - 1)
        shuffled_indices = rng.permutation(examples.index.to_numpy())
        for size in range(test_size, 0, -1):
            for start in range(len(shuffled_indices)):
                rotated = np.roll(shuffled_indices, -start)
                train, test = make_double_cold_pair_split_from_indices(
                    examples,
                    test_indices=rotated[:size],
                )
                if not train.empty and not test.empty:
                    return train.reset_index(drop=True), test.reset_index(drop=True)

    if mode == "cold-drug":
        column = "drug_id"
    elif mode == "cold-disease":
        column = "disease_id"
    else:
        raise ValueError(f"Unknown split mode: {mode}")

    values = np.array(sorted(examples[column].astype(str).unique()))
    test_count = max(1, int(round(len(values) * test_ratio)))
    if len(values) > 1:
        test_count = min(test_count, len(values) - 1)
    test_values = set(rng.choice(values, size=test_count, replace=False))
    test = examples[examples[column].astype(str).isin(test_values)].copy()
    train = examples[~examples[column].astype(str).isin(test_values)].copy()

    # Fallback to random split only when a tiny dataset creates an empty cold split.
    if train.empty or test.empty:
        return make_split(examples, mode="random", test_ratio=test_ratio, seed=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def make_double_cold_pair_split_from_indices(
    examples: pd.DataFrame,
    test_indices: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Docstring."""

    test_set = set(pd.Index(test_indices).tolist())
    test = examples.loc[examples.index.isin(test_set)].copy()
    if test.empty:
        return examples.copy().reset_index(drop=True), test.reset_index(drop=True)

    cold_drugs = set(test["drug_id"].astype(str))
    cold_diseases = set(test["disease_id"].astype(str))
    train_mask = (
        ~examples["drug_id"].astype(str).isin(cold_drugs)
        & ~examples["disease_id"].astype(str).isin(cold_diseases)
        & ~examples.index.isin(test_set)
    )
    train = examples.loc[train_mask].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def make_kfold_splits(
    examples: pd.DataFrame,
    folds: int,
    seed: int,
    mode: str = "random",
) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Docstring."""

    if folds < 2:
        raise ValueError("folds must be at least 2")

    rng = np.random.default_rng(seed)
    if mode == "random":
        if folds > len(examples):
            raise ValueError("folds cannot exceed the number of examples")
        values = examples.index.to_numpy()
        shuffled = rng.permutation(values)
        test_chunks = np.array_split(shuffled, folds)
        out = []
        for fold_idx, test_indices in enumerate(test_chunks, start=1):
            test_set = set(test_indices.tolist())
            test = examples.loc[sorted(test_set)].copy()
            train = examples.loc[[idx for idx in examples.index if idx not in test_set]].copy()
            out.append((fold_idx, train, test))
        return out

    if mode == "double-cold":
        if folds > len(examples):
            raise ValueError("folds cannot exceed the number of examples")
        shuffled = rng.permutation(examples.index.to_numpy())
        test_chunks = np.array_split(shuffled, folds)
        out = []
        for fold_idx, test_indices in enumerate(test_chunks, start=1):
            train, test = make_double_cold_pair_split_from_indices(examples, test_indices)
            if train.empty or test.empty:
                raise ValueError(
                    "double-cold fold produced an empty train or test set; "
                    "use fewer folds or a larger dataset"
                )
            out.append((fold_idx, train, test))
        return out

    if mode == "cold-drug":
        column = "drug_id"
    elif mode == "cold-disease":
        column = "disease_id"
    else:
        raise ValueError(f"Unknown k-fold mode: {mode}")

    group_values = examples[column].astype(str)
    values = np.array(sorted(group_values.unique()))
    if folds > len(values):
        raise ValueError(f"folds cannot exceed the number of unique {column} values")

    shuffled = rng.permutation(values)
    test_chunks = np.array_split(shuffled, folds)
    out = []
    for fold_idx, test_values in enumerate(test_chunks, start=1):
        test_set = set(test_values.tolist())
        test_mask = group_values.isin(test_set)
        test = examples.loc[test_mask].copy()
        train = examples.loc[~test_mask].copy()
        out.append((fold_idx, train, test))
    return out


def make_examples_from_tables(
    tables: dict[str, pd.DataFrame],
    processed_dir: Path,
    negative_threshold: float,
) -> pd.DataFrame:
    """Docstring."""

    sampler = ReliableNegativeSampler(
        associations=tables["drug_disease"],
        drug_similarity=tables["drug_similarity"],
        disease_similarity=tables["disease_similarity"],
        drug_target=tables["drug_target"],
        target_disease=tables["target_disease"],
        config=NegativeSamplerConfig(reliability_threshold=negative_threshold),
    )
    negatives = sampler.sample()
    processed_dir.mkdir(parents=True, exist_ok=True)
    negatives.to_csv(processed_dir / "reliable_negatives.csv", index=False)

    positives = ensure_sample_role_column(
        ensure_risk_feature_columns(tables["drug_disease"][["drug_id", "disease_id", "label"]].copy())
    )
    negatives = ensure_risk_feature_columns(negatives)
    negatives = negatives[["drug_id", "disease_id", "label", *RISK_FEATURE_COLUMNS]].copy()
    negatives[SAMPLE_ROLE_COLUMN] = ROLE_RELIABLE_NEGATIVE
    examples = pd.concat([positives, negatives], ignore_index=True)
    return examples.sample(frac=1.0, random_state=42).reset_index(drop=True)


def make_examples(raw_dir: Path, processed_dir: Path, negative_threshold: float) -> pd.DataFrame:
    """Docstring."""

    tables = load_raw_tables(raw_dir)
    return make_examples_from_tables(tables, processed_dir, negative_threshold)


def positive_examples_from_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    positives = tables["drug_disease"][["drug_id", "disease_id", "label"]].copy()
    positives = positives[positives["label"] == 1].copy()
    positives["drug_id"] = positives["drug_id"].astype(str)
    positives["disease_id"] = positives["disease_id"].astype(str)
    positives = ensure_risk_feature_columns(positives)
    positives[SAMPLE_ROLE_COLUMN] = ROLE_POSITIVE
    return positives.reset_index(drop=True)


def sample_balanced_random_negatives(
    tables: dict[str, pd.DataFrame],
    *,
    negative_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """从全部未知 pair 中均匀无放回抽样，不使用风险分数进行排序。"""

    if negative_ratio <= 0.0:
        raise ValueError("negative_ratio 必须大于 0。")
    positives = positive_examples_from_tables(tables)
    positive_pairs = _pair_set(positives)
    drug_ids, disease_ids = _global_drug_disease_ids(tables)
    unknown_pairs = [
        (drug_id, disease_id)
        for drug_id in drug_ids
        for disease_id in disease_ids
        if (drug_id, disease_id) not in positive_pairs
    ]
    target_count = min(len(unknown_pairs), int(round(len(positives) * negative_ratio)))
    if target_count <= 0:
        return _empty_negative_frame()

    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(len(unknown_pairs), size=target_count, replace=False)
    negatives = pd.DataFrame(
        [unknown_pairs[int(index)] for index in selected_indices],
        columns=["drug_id", "disease_id"],
    )
    negatives["label"] = 0
    negatives = ensure_risk_feature_columns(negatives)
    negatives[SAMPLE_ROLE_COLUMN] = ROLE_BALANCED_RANDOM_NEGATIVE
    return negatives.reset_index(drop=True)


def build_balanced_warm_start_pool(
    tables: dict[str, pd.DataFrame],
    *,
    negative_ratio: float,
    seed: int,
    negative_pool_strategy: str = BALANCED_NEGATIVE_POOL_RANDOM,
    negative_threshold: float = 0.75,
) -> pd.DataFrame:
    """构造一次性的平衡样本池，供分层外层交叉验证使用。"""

    positives = positive_examples_from_tables(tables)
    if negative_pool_strategy == BALANCED_NEGATIVE_POOL_RANDOM:
        negatives = sample_balanced_random_negatives(
            tables,
            negative_ratio=negative_ratio,
            seed=seed,
        )
    elif negative_pool_strategy == BALANCED_NEGATIVE_POOL_RELIABLE:
        _, negatives = select_global_reliable_negatives(
            tables=tables,
            positives=positives,
            negative_threshold=negative_threshold,
            negative_ratio=negative_ratio,
            seed=seed,
        )
    else:
        raise ValueError(
            "negative_pool_strategy 必须是 "
            f"{', '.join(BALANCED_NEGATIVE_POOL_STRATEGIES)}。"
        )
    pool = pd.concat([positives, negatives], ignore_index=True)
    rng = np.random.default_rng(seed)
    return pool.iloc[rng.permutation(len(pool))].reset_index(drop=True)


def subsample_balanced_outer_train(
    outer_train: pd.DataFrame,
    *,
    positive_retention_rate: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Retain a deterministic fraction of positives and an equal number of negatives."""

    if not 0.0 < positive_retention_rate <= 1.0:
        raise ValueError("positive_retention_rate must be in (0, 1].")

    positives = outer_train.loc[outer_train["label"] == 1].copy()
    negatives = outer_train.loc[outer_train["label"] == 0].copy()
    if positives.empty:
        raise ValueError("outer_train must contain positive samples.")

    retained_positive_count = max(
        1,
        int(round(len(positives) * positive_retention_rate)),
    )
    if len(negatives) < retained_positive_count:
        raise ValueError("outer_train does not contain enough negatives for a 1:1 sample ratio.")

    rng = np.random.default_rng(seed)
    retained_positive_indices = rng.choice(
        positives.index.to_numpy(),
        size=retained_positive_count,
        replace=False,
    )
    retained_negative_indices = rng.choice(
        negatives.index.to_numpy(),
        size=retained_positive_count,
        replace=False,
    )
    sparse_train = pd.concat(
        [
            positives.loc[retained_positive_indices],
            negatives.loc[retained_negative_indices],
        ],
        ignore_index=True,
    )
    sparse_train = sparse_train.iloc[rng.permutation(len(sparse_train))].reset_index(drop=True)
    stats: dict[str, int | float] = {
        "positive_retention_rate": float(positive_retention_rate),
        "original_positive_count": int(len(positives)),
        "original_negative_count": int(len(negatives)),
        "retained_positive_count": int(retained_positive_count),
        "retained_negative_count": int(retained_positive_count),
        "realized_positive_retention_rate": float(retained_positive_count / len(positives)),
        "sparsity_sampling_seed": int(seed),
    }
    return sparse_train, stats


def inject_balanced_label_noise(
    outer_train: pd.DataFrame,
    *,
    label_noise_rate: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Flip equal fractions of positive and negative labels without changing pairs."""

    if not 0.0 < label_noise_rate < 0.5:
        raise ValueError("label_noise_rate must be in (0, 0.5).")

    noisy_train = outer_train.copy().reset_index(drop=True)
    positive_indices = noisy_train.index[noisy_train["label"] == 1].to_numpy()
    negative_indices = noisy_train.index[noisy_train["label"] == 0].to_numpy()
    if len(positive_indices) == 0 or len(negative_indices) == 0:
        raise ValueError("outer_train must contain positive and negative samples.")

    reference_class_count = min(len(positive_indices), len(negative_indices))
    flip_count_per_class = max(1, int(round(reference_class_count * label_noise_rate)))
    rng = np.random.default_rng(seed)
    flipped_positive_indices = rng.choice(
        positive_indices,
        size=flip_count_per_class,
        replace=False,
    )
    flipped_negative_indices = rng.choice(
        negative_indices,
        size=flip_count_per_class,
        replace=False,
    )
    noisy_train["original_label"] = noisy_train["label"].astype(int)
    noisy_train["label_noise_injected"] = False
    noisy_train.loc[flipped_positive_indices, "label"] = 0
    noisy_train.loc[flipped_negative_indices, "label"] = 1
    noisy_train.loc[flipped_positive_indices, SAMPLE_ROLE_COLUMN] = "label_noise_false_negative"
    noisy_train.loc[flipped_negative_indices, SAMPLE_ROLE_COLUMN] = "label_noise_false_positive"
    noisy_train.loc[
        np.concatenate([flipped_positive_indices, flipped_negative_indices]),
        "label_noise_injected",
    ] = True

    total_flipped = 2 * flip_count_per_class
    stats: dict[str, int | float] = {
        "label_noise_rate": float(label_noise_rate),
        "realized_label_noise_rate": float(total_flipped / len(noisy_train)),
        "flipped_positive_to_negative_count": int(flip_count_per_class),
        "flipped_negative_to_positive_count": int(flip_count_per_class),
        "total_flipped_label_count": int(total_flipped),
        "label_noise_sampling_seed": int(seed),
    }
    return noisy_train, stats


def create_balanced_warm_start_cv_splits(
    tables: dict[str, pd.DataFrame],
    *,
    folds: int,
    negative_ratio: float,
    seed: int,
    processed_dir: Path | None = None,
    negative_pool_strategy: str = BALANCED_NEGATIVE_POOL_RANDOM,
    negative_threshold: float = 0.75,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame, dict[str, object]]]:
    """对全局平衡样本池做分层外折，保证 pair 不重叠且类别比例稳定。"""

    pool = build_balanced_warm_start_pool(
        tables,
        negative_ratio=negative_ratio,
        seed=seed,
        negative_pool_strategy=negative_pool_strategy,
        negative_threshold=negative_threshold,
    )
    labels = pool["label"].to_numpy(dtype=int)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    positive_count = int((labels == 1).sum())
    negative_count = int((labels == 0).sum())
    all_drug_count = int(pool["drug_id"].astype(str).nunique())
    all_disease_count = int(pool["disease_id"].astype(str).nunique())

    if processed_dir is not None:
        processed_dir.mkdir(parents=True, exist_ok=True)
        pool_filename = (
            "balanced_reliable_sample_pool.csv"
            if negative_pool_strategy == BALANCED_NEGATIVE_POOL_RELIABLE
            else "balanced_random_sample_pool.csv"
        )
        pool.to_csv(processed_dir / pool_filename, index=False)

    splits: list[tuple[int, pd.DataFrame, pd.DataFrame, dict[str, object]]] = []
    for fold_id, (train_indices, test_indices) in enumerate(
        splitter.split(np.zeros(len(pool), dtype=np.int8), labels),
        start=1,
    ):
        train_df = pool.iloc[train_indices].copy().reset_index(drop=True)
        test_df = pool.iloc[test_indices].copy().reset_index(drop=True)
        train_pairs = _pair_set(train_df)
        test_pairs = _pair_set(test_df)
        uses_reliable_pool = negative_pool_strategy == BALANCED_NEGATIVE_POOL_RELIABLE
        stats: dict[str, object] = {
            "evaluation_protocol": EVALUATION_PROTOCOL_BALANCED_WARM_START,
            "protocol_name": (
                NCH_P1_RELIABLE_PROTOCOL_NAME
                if uses_reliable_pool
                else BALANCED_WARM_START_PROTOCOL_NAME
            ),
            "negative_sampling_scope": (
                "adaptive_topk_global_before_cv_train_risk_rescoring"
                if uses_reliable_pool
                else "uniform_random_before_cv_train_risk_rescoring"
            ),
            "negative_pool_strategy": (
                "adaptive_topk_before_cv"
                if uses_reliable_pool
                else "uniform_random_before_cv"
            ),
            "rns_strategy": (
                "adaptive_topk_global_pool"
                if uses_reliable_pool
                else "train_core_risk_rescoring_only"
            ),
            "negative_ratio": float(negative_ratio),
            "global_positive_count": positive_count,
            "global_random_negative_count": 0 if uses_reliable_pool else negative_count,
            "global_reliable_negative_count": negative_count if uses_reliable_pool else 0,
            "global_preprocessing_scope": (
                "all_known_positives_before_cv"
                if uses_reliable_pool
                else "uniform_unknown_sampling_before_cv"
            ),
            "strict_leakage_free": False,
            "global_unlabeled_pair_count": int(
                len(_global_drug_disease_ids(tables)[0])
                * len(_global_drug_disease_ids(tables)[1])
                - positive_count
            ),
            "global_gip_enabled": False,
            "train_positive_count": int((train_df["label"] == 1).sum()),
            "train_negative_count": int((train_df["label"] == 0).sum()),
            "test_positive_count": int((test_df["label"] == 1).sum()),
            "test_negative_count": int((test_df["label"] == 0).sum()),
            "test_pair_disjoint_from_train": train_pairs.isdisjoint(test_pairs),
            "warm_start_global_drug_count": all_drug_count,
            "warm_start_global_disease_count": all_disease_count,
        }
        if processed_dir is not None:
            fold_dir = processed_dir / f"fold_{fold_id}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            train_df.to_csv(fold_dir / "outer_train_samples.csv", index=False)
            test_df.to_csv(fold_dir / "test_samples.csv", index=False)
        splits.append((fold_id, train_df, test_df, stats))
    return splits


def load_auxiliary_positive_examples(path: Path) -> pd.DataFrame:
    """读取已对齐到当前数据集内部 ID 的外部辅助正样本。"""

    auxiliary = pd.read_csv(path)
    required_columns = {"drug_id", "disease_id"}
    if not required_columns.issubset(auxiliary.columns):
        raise ValueError(f"Auxiliary positives at {path} must contain drug_id and disease_id columns.")
    auxiliary = auxiliary[["drug_id", "disease_id"]].copy()
    auxiliary["drug_id"] = auxiliary["drug_id"].astype(str)
    auxiliary["disease_id"] = auxiliary["disease_id"].astype(str)
    auxiliary["label"] = 1
    auxiliary = auxiliary.drop_duplicates(["drug_id", "disease_id"])
    auxiliary = ensure_risk_feature_columns(auxiliary)
    auxiliary[SAMPLE_ROLE_COLUMN] = ROLE_POSITIVE
    return auxiliary.reset_index(drop=True)


def _pair_set(df: pd.DataFrame) -> set[tuple[str, str]]:
    if df.empty:
        return set()
    return set(map(tuple, df[["drug_id", "disease_id"]].astype(str).to_numpy()))


def _empty_negative_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["drug_id", "disease_id", "label", *RISK_FEATURE_COLUMNS, SAMPLE_ROLE_COLUMN])


def negative_sampling_scope_name(rns_strategy: str, pu_only: bool = False) -> str:
    if pu_only:
        return "pu_unlabeled_only_no_fixed_threshold"
    if rns_strategy == RNS_STRATEGY_ADAPTIVE_TOPK:
        return "adaptive_topk_fold_train_positives_only"
    return "fold_train_positives_only"


def ensure_risk_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Docstring."""

    out = df.copy()
    for column in RISK_FEATURE_COLUMNS:
        default = DEFAULT_RISK_FEATURE_VALUES[column]
        if column not in out.columns:
            out[column] = default
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(default).astype(float)
    return out


def ensure_sample_role_column(df: pd.DataFrame) -> pd.DataFrame:
    """Docstring."""

    out = df.copy()
    if SAMPLE_ROLE_COLUMN not in out.columns:
        labels = pd.to_numeric(out["label"], errors="coerce").fillna(0).astype(int)
        out[SAMPLE_ROLE_COLUMN] = np.where(labels == 1, ROLE_POSITIVE, ROLE_RELIABLE_NEGATIVE)
    out[SAMPLE_ROLE_COLUMN] = out[SAMPLE_ROLE_COLUMN].fillna(ROLE_RELIABLE_NEGATIVE).astype(str)
    return out


def make_repeated_cold_start_splits(
    examples: pd.DataFrame,
    repeats: int,
    test_ratio: float,
    seed: int,
    mode: str,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Create repeated entity-level holdouts at a fixed cold-start rate."""

    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("cold-start test ratio must be between 0 and 1")
    if mode not in {"cold-drug", "cold-disease"}:
        raise ValueError("fixed-rate cold start supports cold-drug or cold-disease")

    out: list[tuple[int, pd.DataFrame, pd.DataFrame]] = []
    for repeat_id in range(1, repeats + 1):
        train, test = make_split(
            examples,
            mode=mode,
            test_ratio=test_ratio,
            seed=seed + repeat_id - 1,
        )
        out.append((repeat_id, train, test))
    return out


def _global_drug_disease_ids(tables: dict[str, pd.DataFrame]) -> tuple[list[str], list[str]]:
    """从五类基础表收集全局药物和疾病 ID，保证 GIP 矩阵坐标固定。"""

    drugs: set[str] = set()
    diseases: set[str] = set()
    associations = tables["drug_disease"]
    drugs.update(associations["drug_id"].astype(str))
    diseases.update(associations["disease_id"].astype(str))
    for left, right in tables.get("drug_similarity", pd.DataFrame()).reindex(
        columns=["drug_id_1", "drug_id_2"]
    ).itertuples(index=False, name=None):
        if pd.notna(left):
            drugs.add(str(left))
        if pd.notna(right):
            drugs.add(str(right))
    for left, right in tables.get("disease_similarity", pd.DataFrame()).reindex(
        columns=["disease_id_1", "disease_id_2"]
    ).itertuples(index=False, name=None):
        if pd.notna(left):
            diseases.add(str(left))
        if pd.notna(right):
            diseases.add(str(right))
    for drug_id in tables.get("drug_target", pd.DataFrame()).reindex(columns=["drug_id"]).itertuples(index=False, name=None):
        if pd.notna(drug_id[0]):
            drugs.add(str(drug_id[0]))
    for disease_id in tables.get("target_disease", pd.DataFrame()).reindex(columns=["disease_id"]).itertuples(index=False, name=None):
        if pd.notna(disease_id[0]):
            diseases.add(str(disease_id[0]))
    return sorted(drugs), sorted(diseases)


def _global_association_matrix(
    positives: pd.DataFrame,
    drug_ids: list[str],
    disease_ids: list[str],
) -> np.ndarray:
    """按固定 ID 顺序构造完整已知正关联矩阵。"""

    matrix = np.zeros((len(drug_ids), len(disease_ids)), dtype=np.float64)
    drug_index = {drug_id: index for index, drug_id in enumerate(drug_ids)}
    disease_index = {disease_id: index for index, disease_id in enumerate(disease_ids)}
    for drug_id, disease_id in positives[["drug_id", "disease_id"]].astype(str).itertuples(index=False, name=None):
        drug_position = drug_index.get(drug_id)
        disease_position = disease_index.get(disease_id)
        if drug_position is not None and disease_position is not None:
            matrix[drug_position, disease_position] = 1.0
    return matrix


def _gip_similarity_from_profiles(profiles: np.ndarray) -> np.ndarray:
    """根据全局 Gaussian interaction profile 构造对称 GIP 相似性矩阵。"""

    if profiles.size == 0:
        return np.zeros((profiles.shape[0], profiles.shape[0]), dtype=np.float64)
    squared_norm = np.sum(profiles * profiles, axis=1)
    mean_norm = float(np.mean(squared_norm))
    gamma = 1.0 / max(mean_norm, 1e-12)
    squared_distance = (
        squared_norm[:, None]
        + squared_norm[None, :]
        - 2.0 * profiles @ profiles.T
    )
    similarity = np.exp(-gamma * np.maximum(squared_distance, 0.0))
    np.fill_diagonal(similarity, 1.0)
    return similarity


def _similarity_table_to_matrix(
    table: pd.DataFrame,
    entity_ids: list[str],
    left_column: str,
    right_column: str,
) -> np.ndarray:
    """将现有相似性边表转换为对称二维矩阵，保留最大的重复边分数。"""

    matrix = np.zeros((len(entity_ids), len(entity_ids)), dtype=np.float64)
    np.fill_diagonal(matrix, 1.0)
    index = {entity_id: position for position, entity_id in enumerate(entity_ids)}
    if table.empty:
        return matrix
    columns = [left_column, right_column, "score"]
    if not set(columns).issubset(table.columns):
        return matrix
    for left, right, score in table[columns].itertuples(index=False, name=None):
        left_position = index.get(str(left))
        right_position = index.get(str(right))
        numeric_score = pd.to_numeric(score, errors="coerce")
        if left_position is None or right_position is None or pd.isna(numeric_score):
            continue
        value = float(np.clip(numeric_score, 0.0, 1.0))
        matrix[left_position, right_position] = max(matrix[left_position, right_position], value)
        matrix[right_position, left_position] = max(matrix[right_position, left_position], value)
    return matrix


def _similarity_matrix_to_table(
    matrix: np.ndarray,
    entity_ids: list[str],
    left_column: str,
    right_column: str,
) -> pd.DataFrame:
    """将标准二维相似性矩阵转换为现有图构建器可消费的无向边表。"""

    rows = [
        (entity_ids[left], entity_ids[right], float(matrix[left, right]))
        for left in range(len(entity_ids))
        for right in range(left + 1, len(entity_ids))
    ]
    return pd.DataFrame(rows, columns=[left_column, right_column, "score"])


def build_global_gip_similarity_tables(
    tables: dict[str, pd.DataFrame],
    positives: pd.DataFrame,
    gip_weight: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """基于完整关联矩阵计算 GIP，并与原始相似性进行固定权重融合。"""

    if not 0.0 <= gip_weight <= 1.0:
        raise ValueError("global_gip_weight 必须在 [0, 1] 范围内。")
    drug_ids, disease_ids = _global_drug_disease_ids(tables)
    association_matrix = _global_association_matrix(positives, drug_ids, disease_ids)
    drug_gip = _gip_similarity_from_profiles(association_matrix)
    disease_gip = _gip_similarity_from_profiles(association_matrix.T)
    original_drug = _similarity_table_to_matrix(
        tables.get("drug_similarity", pd.DataFrame()),
        drug_ids,
        "drug_id_1",
        "drug_id_2",
    )
    original_disease = _similarity_table_to_matrix(
        tables.get("disease_similarity", pd.DataFrame()),
        disease_ids,
        "disease_id_1",
        "disease_id_2",
    )
    original_weight = 1.0 - float(gip_weight)
    global_drug_similarity = original_weight * original_drug + float(gip_weight) * drug_gip
    global_disease_similarity = original_weight * original_disease + float(gip_weight) * disease_gip
    np.fill_diagonal(global_drug_similarity, 1.0)
    np.fill_diagonal(global_disease_similarity, 1.0)
    return (
        _similarity_matrix_to_table(global_drug_similarity, drug_ids, "drug_id_1", "drug_id_2"),
        _similarity_matrix_to_table(global_disease_similarity, disease_ids, "disease_id_1", "disease_id_2"),
        {
            "global_gip_weight": float(gip_weight),
            "global_original_similarity_weight": float(original_weight),
            "global_drug_similarity_shape": [len(drug_ids), len(drug_ids)],
            "global_disease_similarity_shape": [len(disease_ids), len(disease_ids)],
            "global_association_matrix_shape": [len(drug_ids), len(disease_ids)],
        },
    )


def select_global_reliable_negatives(
    tables: dict[str, pd.DataFrame],
    positives: pd.DataFrame,
    negative_threshold: float,
    negative_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在五折前基于完整已知关联一次性评分并选择 adaptive-topk 可靠负样本。"""

    drug_ids, disease_ids = _global_drug_disease_ids(tables)
    sampler = ReliableNegativeSampler(
        associations=positives[["drug_id", "disease_id", "label"]],
        drug_similarity=tables["drug_similarity"],
        disease_similarity=tables["disease_similarity"],
        drug_target=tables["drug_target"],
        target_disease=tables["target_disease"],
        config=NegativeSamplerConfig(
            reliability_threshold=negative_threshold,
            negative_ratio=negative_ratio,
            random_state=seed,
        ),
        all_drugs=drug_ids,
        all_diseases=disease_ids,
        excluded_pairs=_pair_set(positives),
    )
    scored = ensure_risk_feature_columns(sampler.score_unknown_pairs())
    if scored.empty:
        empty = _empty_negative_frame()
        empty.insert(0, "pair_index", pd.Series(dtype=str))
        return empty.copy(), empty.copy()
    scored["drug_id"] = scored["drug_id"].astype(str)
    scored["disease_id"] = scored["disease_id"].astype(str)
    scored["pair_index"] = scored["drug_id"] + "::" + scored["disease_id"]
    scored["label"] = 0
    scored[SAMPLE_ROLE_COLUMN] = ROLE_RELIABLE_NEGATIVE
    n_negatives = min(len(scored), int(round(len(positives) * negative_ratio)))
    selected = scored.sort_values(
        ["reliability", "false_negative_risk", "drug_id", "disease_id"],
        ascending=[False, True, True, True],
    ).head(n_negatives).copy()
    columns = ["pair_index", "drug_id", "disease_id", "label", *RISK_FEATURE_COLUMNS, SAMPLE_ROLE_COLUMN]
    return scored[columns].reset_index(drop=True), selected[columns].reset_index(drop=True)


def build_global_preprocessed_data(
    tables: dict[str, pd.DataFrame],
    *,
    negative_threshold: float,
    rns_strategy: str,
    negative_ratio: float,
    seed: int,
    global_gip_weight: float = 0.5,
) -> GlobalPreprocessedData:
    """构建 Global-preprocessing protocol 的一次性数据缓存。"""

    if rns_strategy != RNS_STRATEGY_ADAPTIVE_TOPK:
        raise ValueError("Global-preprocessing protocol 只支持 rns_strategy=adaptive_topk。")
    positives = positive_examples_from_tables(tables)
    global_drug_similarity, global_disease_similarity, similarity_metadata = build_global_gip_similarity_tables(
        tables=tables,
        positives=positives,
        gip_weight=global_gip_weight,
    )
    global_tables = {name: table.copy() for name, table in tables.items()}
    global_tables["drug_similarity"] = global_drug_similarity
    global_tables["disease_similarity"] = global_disease_similarity
    scored_unknown_pairs, reliable_negatives = select_global_reliable_negatives(
        tables=global_tables,
        positives=positives,
        negative_threshold=negative_threshold,
        negative_ratio=negative_ratio,
        seed=seed,
    )
    metadata: dict[str, object] = {
        "evaluation_protocol": EVALUATION_PROTOCOL_GLOBAL_PREPROCESS,
        "protocol_name": GLOBAL_PREPROCESSING_PROTOCOL_NAME,
        "global_positive_count": int(len(positives)),
        "global_unknown_count": int(len(scored_unknown_pairs)),
        "global_reliable_negative_count": int(len(reliable_negatives)),
        "negative_ratio": float(negative_ratio),
        "rns_strategy": RNS_STRATEGY_ADAPTIVE_TOPK,
        **similarity_metadata,
    }
    return GlobalPreprocessedData(
        tables=global_tables,
        positives=positives,
        scored_unknown_pairs=scored_unknown_pairs,
        reliable_negatives=reliable_negatives,
        metadata=metadata,
    )


def create_global_cv_splits(
    positives: pd.DataFrame,
    reliable_negatives: pd.DataFrame,
    *,
    folds: int,
    seed: int,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame, dict[str, object]]]:
    """对固定全局正样本池和可靠负样本池执行同步分层五折划分。"""

    examples = pd.concat([positives, reliable_negatives], ignore_index=True)
    examples = ensure_sample_role_column(ensure_risk_feature_columns(examples))
    examples["label"] = pd.to_numeric(examples["label"], errors="raise").astype(int)
    if folds < 2:
        raise ValueError("folds must be at least 2")
    label_counts = examples["label"].value_counts()
    if len(label_counts) < 2 or int(label_counts.min()) < folds:
        raise ValueError("Global-preprocessing protocol 需要每一类样本数都不少于 folds。")

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    out: list[tuple[int, pd.DataFrame, pd.DataFrame, dict[str, object]]] = []
    for fold_id, (train_indices, test_indices) in enumerate(splitter.split(examples, examples["label"]), start=1):
        train_df = examples.iloc[train_indices].copy().reset_index(drop=True)
        test_df = examples.iloc[test_indices].copy().reset_index(drop=True)
        train_df = train_df.sample(frac=1.0, random_state=seed + fold_id).reset_index(drop=True)
        test_df = test_df.sample(frac=1.0, random_state=seed + 20_000 + fold_id).reset_index(drop=True)
        stats: dict[str, object] = {
            "evaluation_protocol": EVALUATION_PROTOCOL_GLOBAL_PREPROCESS,
            "protocol_name": GLOBAL_PREPROCESSING_PROTOCOL_NAME,
            "negative_sampling_scope": "adaptive_topk_global_preprocess_once",
            "rns_strategy": RNS_STRATEGY_ADAPTIVE_TOPK,
            "train_positive_count": int((train_df["label"] == 1).sum()),
            "train_negative_count": int((train_df["label"] == 0).sum()),
            "test_positive_count": int((test_df["label"] == 1).sum()),
            "test_negative_count": int((test_df["label"] == 0).sum()),
        }
        out.append((fold_id, train_df, test_df, stats))
    return out


def build_p1_pair_universe(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """构造 P1 协议使用的完整药物-疾病 pair 标签空间。"""

    drug_ids, disease_ids = _global_drug_disease_ids(tables)
    positive_pairs = _pair_set(positive_examples_from_tables(tables))
    rows = [
        (drug_id, disease_id, int((drug_id, disease_id) in positive_pairs))
        for drug_id in drug_ids
        for disease_id in disease_ids
    ]
    universe = pd.DataFrame(rows, columns=["drug_id", "disease_id", "label"])
    return ensure_sample_role_column(ensure_risk_feature_columns(universe))


def create_p1_cv_splits(
    tables: dict[str, pd.DataFrame],
    *,
    folds: int,
    seed: int,
    negative_threshold: float,
    rns_strategy: str,
    negative_ratio: float,
    pu_learning: bool,
    pu_unlabeled_ratio: float,
    processed_dir: Path | None = None,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame, dict[str, object]]]:
    """按 P1 划分完整 pair 空间，再在各折训练池内执行当前 RNS/PU。"""

    universe = build_p1_pair_universe(tables)
    label_counts = universe["label"].value_counts()
    if folds < 2 or len(label_counts) < 2 or int(label_counts.min()) < folds:
        raise ValueError("P1 协议要求正、负两类样本数均不少于交叉验证折数。")
    all_positive_pairs = _pair_set(universe.loc[universe["label"] == 1])
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    out: list[tuple[int, pd.DataFrame, pd.DataFrame, dict[str, object]]] = []

    for fold_id, (train_indices, test_indices) in enumerate(
        splitter.split(universe, universe["label"]), start=1
    ):
        train_pool = universe.iloc[train_indices].copy().reset_index(drop=True)
        test_examples = universe.iloc[test_indices].copy().reset_index(drop=True)
        train_positives = train_pool.loc[train_pool["label"] == 1].copy()
        train_unknown = train_pool.loc[train_pool["label"] == 0].copy()
        allowed_unknown_pairs = _pair_set(train_unknown)

        train_negatives = sample_fold_train_negatives(
            tables=tables,
            train_positives=train_positives,
            all_positive_pairs=all_positive_pairs,
            negative_threshold=negative_threshold,
            seed=seed + fold_id,
            rns_strategy=rns_strategy,
            negative_ratio=negative_ratio,
            candidate_pairs=allowed_unknown_pairs,
        )
        pu_unlabeled = (
            make_fold_pu_unlabeled_examples(
                tables=tables,
                train_positives=train_positives,
                train_negatives=train_negatives,
                all_positive_pairs=all_positive_pairs,
                negative_threshold=negative_threshold,
                unlabeled_ratio=pu_unlabeled_ratio,
                seed=seed + 30_000 + fold_id,
                candidate_pairs=allowed_unknown_pairs,
            )
            if pu_learning
            else pd.DataFrame(
                columns=["drug_id", "disease_id", "label", *RISK_FEATURE_COLUMNS, SAMPLE_ROLE_COLUMN]
            )
        )
        train_examples = pd.concat(
            [train_positives, train_negatives, pu_unlabeled], ignore_index=True
        )
        train_examples = ensure_sample_role_column(ensure_risk_feature_columns(train_examples))
        test_examples = ensure_sample_role_column(ensure_risk_feature_columns(test_examples))
        train_examples = train_examples.sample(
            frac=1.0, random_state=seed + fold_id
        ).reset_index(drop=True)
        test_examples = test_examples.sample(
            frac=1.0, random_state=seed + 20_000 + fold_id
        ).reset_index(drop=True)

        stats: dict[str, object] = {
            "evaluation_protocol": EVALUATION_PROTOCOL_P1_TRANSDUCTIVE,
            "protocol_name": P1_TRANSDUCTIVE_PROTOCOL_NAME,
            "negative_sampling_scope": "adaptive_topk_p1_train_pool_only",
            "rns_strategy": rns_strategy,
            "negative_ratio": float(negative_ratio),
            "p1_total_pair_count": int(len(universe)),
            "p1_train_pool_positive_count": int(len(train_positives)),
            "p1_train_pool_unknown_count": int(len(train_unknown)),
            "train_positive_count": int(len(train_positives)),
            "train_negative_count": int(len(train_negatives)),
            "pu_learning_enabled": bool(pu_learning),
            "pu_only_enabled": False,
            "pu_unlabeled_count": int(len(pu_unlabeled)),
            "pu_unlabeled_ratio": float(pu_unlabeled_ratio),
            "test_positive_count": int((test_examples["label"] == 1).sum()),
            "test_negative_count": int((test_examples["label"] == 0).sum()),
            "negative_threshold": float(negative_threshold),
        }
        if processed_dir is not None:
            fold_dir = processed_dir / f"fold_{fold_id}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            train_examples.to_csv(fold_dir / "train_samples.csv", index=False)
            test_examples.to_csv(fold_dir / "test_samples.csv", index=False)
            train_negatives.to_csv(fold_dir / "reliable_train_negatives.csv", index=False)
            pu_unlabeled.to_csv(fold_dir / "pu_unlabeled.csv", index=False)
        out.append((fold_id, train_examples, test_examples, stats))

    return out


def select_global_pu_unlabeled(
    scored_unknown_pairs: pd.DataFrame,
    reliable_negatives: pd.DataFrame,
    positive_count: int,
    unlabeled_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """从未进入固定可靠负样本池的全局未知 pair 中一次性抽取 PU 样本。"""

    if unlabeled_ratio <= 0.0 or scored_unknown_pairs.empty:
        return pd.DataFrame(columns=["drug_id", "disease_id", "label", *RISK_FEATURE_COLUMNS, SAMPLE_ROLE_COLUMN])
    excluded_pairs = _pair_set(reliable_negatives)
    candidates = scored_unknown_pairs.loc[
        [
            (drug_id, disease_id) not in excluded_pairs
            for drug_id, disease_id in scored_unknown_pairs[["drug_id", "disease_id"]].astype(str).itertuples(index=False, name=None)
        ]
    ].copy()
    n_unlabeled = min(len(candidates), int(round(positive_count * unlabeled_ratio)))
    if n_unlabeled <= 0:
        return candidates.iloc[0:0][["drug_id", "disease_id", "label", *RISK_FEATURE_COLUMNS, SAMPLE_ROLE_COLUMN]].copy()
    probabilities = pd.to_numeric(candidates["false_negative_risk"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    probabilities = probabilities + 1e-6
    probabilities = probabilities / probabilities.sum()
    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(candidates.index.to_numpy(), size=n_unlabeled, replace=False, p=probabilities)
    selected = candidates.loc[selected_indices, ["drug_id", "disease_id", "label", *RISK_FEATURE_COLUMNS]].copy()
    selected[SAMPLE_ROLE_COLUMN] = ROLE_PU_UNLABELED
    selected["label"] = 0
    return selected.reset_index(drop=True)


def sample_fold_train_negatives(
    tables: dict[str, pd.DataFrame],
    train_positives: pd.DataFrame,
    all_positive_pairs: set[tuple[str, str]],
    negative_threshold: float,
    seed: int,
    rns_strategy: str = RNS_STRATEGY_THRESHOLD,
    negative_ratio: float = DEFAULT_NEGATIVE_RATIO,
    candidate_pairs: set[tuple[str, str]] | None = None,
    apply_reliability_threshold: bool = False,
) -> pd.DataFrame:
    if candidate_pairs is None:
        train_drugs = sorted(train_positives["drug_id"].astype(str).unique().tolist())
        train_diseases = sorted(train_positives["disease_id"].astype(str).unique().tolist())
    else:
        normalized_candidates = {(str(drug_id), str(disease_id)) for drug_id, disease_id in candidate_pairs}
        train_drugs = sorted({pair[0] for pair in normalized_candidates})
        train_diseases = sorted({pair[1] for pair in normalized_candidates})
    if not train_drugs or not train_diseases:
        return _empty_negative_frame()

    sampler = ReliableNegativeSampler(
        associations=train_positives[["drug_id", "disease_id", "label"]],
        drug_similarity=tables["drug_similarity"],
        disease_similarity=tables["disease_similarity"],
        drug_target=tables["drug_target"],
        target_disease=tables["target_disease"],
        config=NegativeSamplerConfig(
            reliability_threshold=negative_threshold,
            negative_ratio=negative_ratio,
            random_state=seed,
        ),
        all_drugs=train_drugs,
        all_diseases=train_diseases,
        excluded_pairs=all_positive_pairs,
    )
    if rns_strategy == RNS_STRATEGY_ADAPTIVE_TOPK:
        scored = ensure_risk_feature_columns(sampler.score_unknown_pairs())
        if candidate_pairs is not None:
            scored = scored.loc[
                [
                    (str(drug_id), str(disease_id)) in normalized_candidates
                    for drug_id, disease_id in scored[["drug_id", "disease_id"]].itertuples(index=False, name=None)
                ]
            ].copy()
        if apply_reliability_threshold:
            scored = scored.loc[scored["reliability"] >= negative_threshold].copy()
        if scored.empty:
            return _empty_negative_frame()
        n_pos = int(len(train_positives[train_positives["label"] == 1]))
        n_neg = min(len(scored), int(round(n_pos * negative_ratio)))
        if n_neg <= 0:
            return _empty_negative_frame()
        negatives = scored.sort_values(
            ["reliability", "false_negative_risk", "drug_id", "disease_id"],
            ascending=[False, True, True, True],
        ).head(n_neg).copy()
        negatives["label"] = 0
    else:
        negatives = sampler.sample()
        if candidate_pairs is not None and not negatives.empty:
            negatives = negatives.loc[
                [
                    (str(drug_id), str(disease_id)) in normalized_candidates
                    for drug_id, disease_id in negatives[["drug_id", "disease_id"]].itertuples(index=False, name=None)
                ]
            ].copy()
    if negatives.empty:
        return _empty_negative_frame()
    negatives = ensure_risk_feature_columns(negatives)
    negatives = negatives[["drug_id", "disease_id", "label", *RISK_FEATURE_COLUMNS]].copy()
    negatives["drug_id"] = negatives["drug_id"].astype(str)
    negatives["disease_id"] = negatives["disease_id"].astype(str)
    negatives[SAMPLE_ROLE_COLUMN] = ROLE_RELIABLE_NEGATIVE
    return negatives.reset_index(drop=True)


def make_fold_pu_unlabeled_examples(
    tables: dict[str, pd.DataFrame],
    train_positives: pd.DataFrame,
    train_negatives: pd.DataFrame,
    all_positive_pairs: set[tuple[str, str]],
    negative_threshold: float,
    unlabeled_ratio: float,
    seed: int,
    candidate_pairs: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Docstring."""

    if unlabeled_ratio <= 0.0 or train_positives.empty:
        return pd.DataFrame(columns=["drug_id", "disease_id", "label", *RISK_FEATURE_COLUMNS, SAMPLE_ROLE_COLUMN])

    if candidate_pairs is None:
        train_drugs = sorted(train_positives["drug_id"].astype(str).unique().tolist())
        train_diseases = sorted(train_positives["disease_id"].astype(str).unique().tolist())
    else:
        normalized_candidates = {(str(drug_id), str(disease_id)) for drug_id, disease_id in candidate_pairs}
        train_drugs = sorted({pair[0] for pair in normalized_candidates})
        train_diseases = sorted({pair[1] for pair in normalized_candidates})
    if not train_drugs or not train_diseases:
        return pd.DataFrame(columns=["drug_id", "disease_id", "label", *RISK_FEATURE_COLUMNS, SAMPLE_ROLE_COLUMN])

    excluded_pairs = set(all_positive_pairs) | _pair_set(train_negatives)
    sampler = ReliableNegativeSampler(
        associations=train_positives[["drug_id", "disease_id", "label"]],
        drug_similarity=tables["drug_similarity"],
        disease_similarity=tables["disease_similarity"],
        drug_target=tables["drug_target"],
        target_disease=tables["target_disease"],
        config=NegativeSamplerConfig(
            reliability_threshold=negative_threshold,
            random_state=seed,
        ),
        all_drugs=train_drugs,
        all_diseases=train_diseases,
        excluded_pairs=excluded_pairs,
    )
    scored = sampler.score_unknown_pairs()
    if candidate_pairs is not None and not scored.empty:
        scored = scored.loc[
            [
                (str(drug_id), str(disease_id)) in normalized_candidates
                for drug_id, disease_id in scored[["drug_id", "disease_id"]].itertuples(index=False, name=None)
            ]
        ].copy()
    if scored.empty:
        return pd.DataFrame(columns=["drug_id", "disease_id", "label", *RISK_FEATURE_COLUMNS, SAMPLE_ROLE_COLUMN])

    scored = ensure_risk_feature_columns(scored)
    scored["label"] = 0
    scored[SAMPLE_ROLE_COLUMN] = ROLE_PU_UNLABELED
    scored["drug_id"] = scored["drug_id"].astype(str)
    scored["disease_id"] = scored["disease_id"].astype(str)

    n_unlabeled = min(len(scored), int(round(len(train_positives) * unlabeled_ratio)))
    if n_unlabeled <= 0:
        return scored.iloc[0:0][["drug_id", "disease_id", "label", *RISK_FEATURE_COLUMNS, SAMPLE_ROLE_COLUMN]].copy()

    rng = np.random.default_rng(seed)
    probabilities = pd.to_numeric(scored["false_negative_risk"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    probabilities = probabilities + 1e-6
    probabilities = probabilities / probabilities.sum()
    chosen = rng.choice(scored.index.to_numpy(), size=n_unlabeled, replace=False, p=probabilities)
    out = scored.loc[chosen, ["drug_id", "disease_id", "label", *RISK_FEATURE_COLUMNS, SAMPLE_ROLE_COLUMN]].copy()
    return out.reset_index(drop=True)


def rescore_fold_pair_risk_features(
    train_core_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pu_unlabeled_df: pd.DataFrame,
    *,
    tables: dict[str, pd.DataFrame],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """用 train-core 正边为所有阶段的 pair 生成同分布风险特征。"""

    train_core = ensure_sample_role_column(ensure_risk_feature_columns(train_core_df))
    validation = ensure_sample_role_column(ensure_risk_feature_columns(validation_df))
    test = ensure_sample_role_column(ensure_risk_feature_columns(test_df))
    pu_unlabeled = ensure_sample_role_column(ensure_risk_feature_columns(pu_unlabeled_df))
    train_positives = train_core.loc[
        train_core["label"] == 1,
        ["drug_id", "disease_id", "label"],
    ].copy()
    train_positives["drug_id"] = train_positives["drug_id"].astype(str)
    train_positives["disease_id"] = train_positives["disease_id"].astype(str)
    all_drugs, all_diseases = _global_drug_disease_ids(tables)
    sampler = ReliableNegativeSampler(
        associations=train_positives,
        drug_similarity=tables["drug_similarity"],
        disease_similarity=tables["disease_similarity"],
        drug_target=tables["drug_target"],
        target_disease=tables["target_disease"],
        config=NegativeSamplerConfig(random_state=seed),
        all_drugs=all_drugs,
        all_diseases=all_diseases,
    )

    def apply_pair_risk(frame: pd.DataFrame) -> pd.DataFrame:
        rescored = frame.copy().reset_index(drop=True)
        if rescored.empty:
            return rescored
        score_rows = [
            sampler.false_negative_risk(str(drug_id), str(disease_id))
            for drug_id, disease_id in rescored[
                ["drug_id", "disease_id"]
            ].itertuples(index=False, name=None)
        ]
        scores = pd.DataFrame(score_rows, columns=RISK_FEATURE_COLUMNS)
        for column in RISK_FEATURE_COLUMNS:
            rescored[column] = scores[column].to_numpy(dtype=float)
        return rescored

    rescored_train = apply_pair_risk(train_core)
    rescored_validation = apply_pair_risk(validation)
    rescored_test = apply_pair_risk(test)
    rescored_pu = apply_pair_risk(pu_unlabeled)
    stats: dict[str, object] = {
        "fold_pair_risk_rescoring_enabled": True,
        "risk_reference_scope": "train_core_positive_edges_only",
        "risk_reference_positive_count": int(len(train_positives)),
        "train_pair_risk_inference_count": int(len(rescored_train)),
        "validation_pair_risk_inference_count": int(len(rescored_validation)),
        "test_pair_risk_inference_count": int(len(rescored_test)),
        "pu_pair_risk_inference_count": int(len(rescored_pu)),
        "validation_label_ignored_by_risk_scoring": True,
        "test_label_ignored_by_risk_scoring": True,
    }
    return rescored_train, rescored_validation, rescored_test, rescored_pu, stats


def rescore_balanced_train_core(
    train_core_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    tables: dict[str, pd.DataFrame],
    seed: int,
    pu_learning: bool,
    pu_unlabeled_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """用 train-core 正边重算 pair 风险，并构造与留出集隔离的 PU 样本。"""

    train_core = ensure_sample_role_column(ensure_risk_feature_columns(train_core_df))
    validation = ensure_sample_role_column(ensure_risk_feature_columns(validation_df))
    test = ensure_sample_role_column(ensure_risk_feature_columns(test_df))
    train_positives = train_core.loc[train_core["label"] == 1, ["drug_id", "disease_id", "label"]].copy()
    train_positives["drug_id"] = train_positives["drug_id"].astype(str)
    train_positives["disease_id"] = train_positives["disease_id"].astype(str)
    all_drugs, all_diseases = _global_drug_disease_ids(tables)
    sampler = ReliableNegativeSampler(
        associations=train_positives,
        drug_similarity=tables["drug_similarity"],
        disease_similarity=tables["disease_similarity"],
        drug_target=tables["drug_target"],
        target_disease=tables["target_disease"],
        config=NegativeSamplerConfig(random_state=seed),
        all_drugs=all_drugs,
        all_diseases=all_diseases,
    )

    def apply_pair_risk(frame: pd.DataFrame, *, include_positive_labels: bool) -> pd.DataFrame:
        rescored = frame.copy().reset_index(drop=True)
        if rescored.empty:
            return rescored
        mask = (
            pd.Series(True, index=rescored.index)
            if include_positive_labels
            else rescored["label"].to_numpy(dtype=int) == 0
        )
        selected_indices = rescored.index[mask]
        pairs = list(
            rescored.loc[selected_indices, ["drug_id", "disease_id"]]
            .astype(str)
            .itertuples(index=False, name=None)
        )
        if pairs:
            scores = sampler.score_pairs(pairs)
            for column in RISK_FEATURE_COLUMNS:
                rescored.loc[selected_indices, column] = scores[column].to_numpy(dtype=float)
        return rescored

    # 训练正样本是风险评分的参考边；训练负样本和所有留出 pair 均按同一参考图变换。
    rescored_train = apply_pair_risk(train_core, include_positive_labels=False)
    rescored_validation = apply_pair_risk(validation, include_positive_labels=True)
    rescored_test = apply_pair_risk(test, include_positive_labels=True)

    supervised_pairs = _pair_set(train_core)
    validation_pairs = _pair_set(validation)
    test_pairs = _pair_set(test)
    excluded_pairs = supervised_pairs | validation_pairs | test_pairs
    pu_candidates = [
        (drug_id, disease_id)
        for drug_id in all_drugs
        for disease_id in all_diseases
        if (drug_id, disease_id) not in excluded_pairs
    ]
    pu_df = _empty_negative_frame()
    if pu_learning and pu_unlabeled_ratio > 0.0 and pu_candidates:
        scored_candidates = ensure_risk_feature_columns(sampler.score_pairs(pu_candidates))
        target_count = min(
            len(scored_candidates),
            int(round(len(train_positives) * pu_unlabeled_ratio)),
        )
        if target_count > 0:
            probabilities = (
                scored_candidates["false_negative_risk"].to_numpy(dtype=float) + 1e-6
            )
            probabilities = probabilities / probabilities.sum()
            rng = np.random.default_rng(seed)
            chosen = rng.choice(
                scored_candidates.index.to_numpy(),
                size=target_count,
                replace=False,
                p=probabilities,
            )
            pu_df = scored_candidates.loc[
                chosen,
                ["drug_id", "disease_id", *RISK_FEATURE_COLUMNS],
            ].copy()
            pu_df["label"] = 0
            pu_df[SAMPLE_ROLE_COLUMN] = ROLE_PU_UNLABELED
            pu_df = pu_df[
                ["drug_id", "disease_id", "label", *RISK_FEATURE_COLUMNS, SAMPLE_ROLE_COLUMN]
            ].reset_index(drop=True)

    pu_pairs = _pair_set(pu_df)
    train_positive_pairs = _pair_set(train_positives)
    validation_positive_pairs = _pair_set(validation.loc[validation["label"] == 1])
    test_positive_pairs = _pair_set(test.loc[test["label"] == 1])
    stats: dict[str, object] = {
        "risk_reference_scope": "train_core_positive_edges_only",
        "risk_reference_positive_count": int(len(train_positives)),
        "train_negative_rescored_count": int((rescored_train["label"] == 0).sum()),
        "validation_pair_risk_inference_count": int(len(rescored_validation)),
        "test_pair_risk_inference_count": int(len(rescored_test)),
        "test_label_ignored_by_risk_scoring": True,
        "validation_label_ignored_by_risk_scoring": True,
        "pu_unlabeled_count": int(len(pu_df)),
        "validation_pair_excluded_from_pu": pu_pairs.isdisjoint(validation_pairs),
        "test_pair_excluded_from_pu": pu_pairs.isdisjoint(test_pairs),
        "validation_pair_disjoint_from_train": supervised_pairs.isdisjoint(validation_pairs),
        "test_pair_disjoint_from_train": supervised_pairs.isdisjoint(test_pairs),
        "validation_positive_absent_from_association_graph": train_positive_pairs.isdisjoint(
            validation_positive_pairs
        ),
        "test_positive_absent_from_association_graph": train_positive_pairs.isdisjoint(
            test_positive_pairs
        ),
        "global_gip_enabled": False,
        "best_metric_source": "validation",
    }
    return rescored_train, rescored_validation, rescored_test, pu_df, stats


def sample_evaluation_negatives(
    all_positives: pd.DataFrame,
    test_positives: pd.DataFrame,
    train_negatives: pd.DataFrame,
    cv_mode: str,
    seed: int,
    candidate_universe_positives: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    excluded_pairs = _pair_set(all_positives) | _pair_set(train_negatives)

    candidate_universe = candidate_universe_positives if candidate_universe_positives is not None else all_positives
    all_drugs = sorted(candidate_universe["drug_id"].astype(str).unique().tolist())
    all_diseases = sorted(candidate_universe["disease_id"].astype(str).unique().tolist())
    if cv_mode == "cold-drug":
        candidate_drugs = sorted(test_positives["drug_id"].astype(str).unique().tolist())
        candidate_diseases = all_diseases
    elif cv_mode == "cold-disease":
        candidate_drugs = all_drugs
        candidate_diseases = sorted(test_positives["disease_id"].astype(str).unique().tolist())
    elif cv_mode == "double-cold":
        candidate_drugs = sorted(test_positives["drug_id"].astype(str).unique().tolist())
        candidate_diseases = sorted(test_positives["disease_id"].astype(str).unique().tolist())
    else:
        candidate_drugs = all_drugs
        candidate_diseases = all_diseases

    candidates = [
        (drug_id, disease_id)
        for drug_id in candidate_drugs
        for disease_id in candidate_diseases
        if (drug_id, disease_id) not in excluded_pairs
    ]
    if not candidates:
        return _empty_negative_frame()

    target_count = min(len(candidates), len(test_positives))
    chosen_indices = rng.choice(len(candidates), size=target_count, replace=False)
    rows = [candidates[int(index)] for index in chosen_indices]
    negatives = pd.DataFrame(rows, columns=["drug_id", "disease_id"])
    negatives["label"] = 0
    negatives = ensure_risk_feature_columns(negatives)
    negatives[SAMPLE_ROLE_COLUMN] = ROLE_RELIABLE_NEGATIVE
    return negatives.reset_index(drop=True)


def make_fold_aware_kfold_splits(
    tables: dict[str, pd.DataFrame],
    folds: int,
    cv_mode: str,
    negative_threshold: float,
    seed: int,
    processed_dir: Path | None = None,
    pu_learning: bool = False,
    pu_only: bool = False,
    pu_unlabeled_ratio: float = 1.0,
    rns_strategy: str = RNS_STRATEGY_THRESHOLD,
    negative_ratio: float = DEFAULT_NEGATIVE_RATIO,
    validation_ratio: float = 0.1,
    auxiliary_positives: pd.DataFrame | None = None,
    cold_start_rate: float | None = None,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in (0, 1) for fold-wise splitting")
    primary_positives = positive_examples_from_tables(tables)
    primary_pairs = _pair_set(primary_positives)
    if auxiliary_positives is None or auxiliary_positives.empty:
        auxiliary = primary_positives.iloc[0:0].copy()
    else:
        auxiliary = auxiliary_positives[["drug_id", "disease_id", "label"]].copy()
        auxiliary["drug_id"] = auxiliary["drug_id"].astype(str)
        auxiliary["disease_id"] = auxiliary["disease_id"].astype(str)
        auxiliary["label"] = 1
        auxiliary_pair_mask = [
            (drug_id, disease_id) not in primary_pairs
            for drug_id, disease_id in auxiliary[["drug_id", "disease_id"]].itertuples(index=False)
        ]
        auxiliary = auxiliary.loc[auxiliary_pair_mask].copy()
        auxiliary = ensure_sample_role_column(ensure_risk_feature_columns(auxiliary))
    all_known_positives = pd.concat([primary_positives, auxiliary], ignore_index=True)
    all_positive_pairs = _pair_set(all_known_positives)
    if cold_start_rate is None:
        fold_splits = make_kfold_splits(primary_positives, folds=folds, seed=seed, mode=cv_mode)
    else:
        fold_splits = make_repeated_cold_start_splits(
            primary_positives,
            repeats=folds,
            test_ratio=cold_start_rate,
            seed=seed,
            mode=cv_mode,
        )
    out: list[tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]] = []

    for fold_id, train_positives, test_positives in fold_splits:
        train_positives = train_positives.reset_index(drop=True)
        test_positives = test_positives.reset_index(drop=True)
        test_negatives = sample_evaluation_negatives(
            all_positives=all_known_positives,
            test_positives=test_positives,
            train_negatives=_empty_negative_frame(),
            cv_mode=cv_mode,
            seed=seed + 10_000 + fold_id,
            candidate_universe_positives=primary_positives,
        )
        train_core_positives, validation_positives = make_split(
            train_positives,
            mode=cv_mode,
            test_ratio=validation_ratio,
            seed=seed + 50_000 + fold_id,
        )
        if train_core_positives.empty or validation_positives.empty:
            raise ValueError(f"Fold {fold_id} has an empty train core or validation set.")
        validation_negatives = sample_evaluation_negatives(
            all_positives=all_known_positives,
            test_positives=validation_positives,
            train_negatives=test_negatives,
            cv_mode=cv_mode,
            seed=seed + 20_000 + fold_id,
            candidate_universe_positives=primary_positives,
        )
        if len(validation_negatives) != len(validation_positives):
            raise ValueError(f"Fold {fold_id} lacks enough reserved validation negatives.")
        validation_examples = pd.concat(
            [validation_positives, validation_negatives], ignore_index=True
        )
        excluded_training_pairs = (
            all_positive_pairs | _pair_set(test_negatives) | _pair_set(validation_negatives)
        )
        if cv_mode == "cold-drug":
            held_out_drugs = set(test_positives["drug_id"].astype(str)) | set(validation_positives["drug_id"].astype(str))
            fold_auxiliary = auxiliary.loc[~auxiliary["drug_id"].astype(str).isin(held_out_drugs)].copy()
        elif cv_mode == "cold-disease":
            held_out_diseases = set(test_positives["disease_id"].astype(str)) | set(validation_positives["disease_id"].astype(str))
            fold_auxiliary = auxiliary.loc[~auxiliary["disease_id"].astype(str).isin(held_out_diseases)].copy()
        elif cv_mode == "double-cold":
            held_out_drugs = set(test_positives["drug_id"].astype(str)) | set(validation_positives["drug_id"].astype(str))
            held_out_diseases = set(test_positives["disease_id"].astype(str)) | set(validation_positives["disease_id"].astype(str))
            fold_auxiliary = auxiliary.loc[
                ~auxiliary["drug_id"].astype(str).isin(held_out_drugs)
                & ~auxiliary["disease_id"].astype(str).isin(held_out_diseases)
            ].copy()
        else:
            fold_auxiliary = auxiliary.copy()
        train_graph_positives = pd.concat([train_core_positives, fold_auxiliary], ignore_index=True)
        train_negatives = (
            _empty_negative_frame()
            if pu_only
            else sample_fold_train_negatives(
                tables=tables,
                train_positives=train_graph_positives,
                all_positive_pairs=excluded_training_pairs,
                negative_threshold=negative_threshold,
                seed=seed + fold_id,
                rns_strategy=rns_strategy,
                negative_ratio=negative_ratio,
                apply_reliability_threshold=True,
            )
        )
        pu_unlabeled = (
            make_fold_pu_unlabeled_examples(
                tables=tables,
                train_positives=train_graph_positives,
                train_negatives=train_negatives,
                all_positive_pairs=excluded_training_pairs,
                negative_threshold=negative_threshold,
                unlabeled_ratio=pu_unlabeled_ratio,
                seed=seed + 30_000 + fold_id,
            )
            if pu_learning
            else pd.DataFrame(columns=["drug_id", "disease_id", "label", *RISK_FEATURE_COLUMNS, SAMPLE_ROLE_COLUMN])
        )
        train_examples = pd.concat([train_graph_positives, train_negatives, pu_unlabeled], ignore_index=True)
        test_examples = pd.concat([test_positives, test_negatives], ignore_index=True)
        train_examples["label"] = pd.to_numeric(train_examples["label"], errors="raise").astype(int)
        validation_examples["label"] = pd.to_numeric(validation_examples["label"], errors="raise").astype(int)
        test_examples["label"] = pd.to_numeric(test_examples["label"], errors="raise").astype(int)
        train_examples = ensure_sample_role_column(ensure_risk_feature_columns(train_examples))
        validation_examples = ensure_sample_role_column(ensure_risk_feature_columns(validation_examples))
        test_examples = ensure_sample_role_column(ensure_risk_feature_columns(test_examples))
        train_examples = train_examples.sample(frac=1.0, random_state=seed + fold_id).reset_index(drop=True)
        validation_examples = validation_examples.sample(frac=1.0, random_state=seed + 40_000 + fold_id).reset_index(drop=True)
        test_examples = test_examples.sample(frac=1.0, random_state=seed + 20_000 + fold_id).reset_index(drop=True)

        train_pairs = _pair_set(train_examples)
        validation_pairs = _pair_set(validation_examples)
        test_pairs = _pair_set(test_examples)
        pu_pairs = _pair_set(pu_unlabeled)
        train_positive_pairs = _pair_set(train_graph_positives)
        test_positive_pairs = _pair_set(test_positives)
        if not train_pairs.isdisjoint(test_pairs | validation_pairs) or not validation_pairs.isdisjoint(test_pairs):
            raise ValueError(f"Fold {fold_id} has overlapping train, validation or test pairs.")
        if not pu_pairs.isdisjoint(test_pairs | validation_pairs):
            raise ValueError(f"Fold {fold_id} contains a reserved pair in PU data.")
        if not train_positive_pairs.isdisjoint(test_positive_pairs | _pair_set(validation_positives)):
            raise ValueError(f"Fold {fold_id} contains a reserved positive in the training graph.")
        stats: dict[str, object] = {
            "global_preprocessing_scope": "none",
            "strict_leakage_free": True,
            "negative_sampling_scope": negative_sampling_scope_name(rns_strategy, pu_only=pu_only),
            "rns_strategy": rns_strategy,
            "negative_ratio": float(negative_ratio),
            "risk_reference_scope": "fold_train_core_positive_edges_only",
            "risk_reference_positive_count": int(len(train_graph_positives)),
            "train_primary_positive_count": int(len(train_positives)),
            "train_core_positive_count": int(len(train_core_positives)),
            "auxiliary_positive_count": int(len(fold_auxiliary)),
            "train_positive_count": int(len(train_graph_positives)),
            "train_negative_count": int(len(train_negatives)),
            "pu_learning_enabled": bool(pu_learning),
            "pu_only_enabled": bool(pu_only),
            "pu_unlabeled_count": int(len(pu_unlabeled)),
            "pu_unlabeled_ratio": float(pu_unlabeled_ratio),
            "test_positive_count": int(len(test_positives)),
            "test_negative_count": int(len(test_negatives)),
            "validation_positive_count": int(len(validation_positives)),
            "validation_negative_count": int(len(validation_negatives)),
            "negative_threshold": float(negative_threshold),
            "global_gip_enabled": False,
            "test_pair_disjoint_from_train": train_pairs.isdisjoint(test_pairs),
            "test_pair_excluded_from_pu": pu_pairs.isdisjoint(test_pairs),
            "validation_pair_disjoint_from_train": train_pairs.isdisjoint(validation_pairs),
            "validation_pair_excluded_from_pu": pu_pairs.isdisjoint(validation_pairs),
            "validation_positive_absent_from_association_graph": train_positive_pairs.isdisjoint(
                _pair_set(validation_positives)
            ),
            "test_positive_absent_from_association_graph": train_positive_pairs.isdisjoint(
                test_positive_pairs
            ),
            "test_label_ignored_by_risk_scoring": True,
        }
        if cold_start_rate is not None:
            stats["cold_start_rate"] = float(cold_start_rate)
            stats["cold_start_repeat"] = int(fold_id)
        if processed_dir is not None:
            fold_dir = processed_dir / f"fold_{fold_id}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            train_negatives.to_csv(fold_dir / "reliable_train_negatives.csv", index=False)
            pu_unlabeled.to_csv(fold_dir / "pu_unlabeled.csv", index=False)
            validation_examples.to_csv(fold_dir / "validation_samples.csv", index=False)
            test_negatives.to_csv(fold_dir / "evaluation_negatives.csv", index=False)
        out.append((fold_id, train_examples, validation_examples, test_examples, stats))

    return out


def collect_auxiliary_node_ids(
    raw_tables: dict[str, pd.DataFrame] | None,
) -> tuple[list[str], list[str], list[str]]:
    """Docstring."""

    if raw_tables is None:
        return [], [], []

    drugs: set[str] = set()
    diseases: set[str] = set()
    targets: set[str] = set()

    if "drug_similarity" in raw_tables:
        table = raw_tables["drug_similarity"]
        drugs.update(table["drug_id_1"].astype(str))
        drugs.update(table["drug_id_2"].astype(str))
    if "disease_similarity" in raw_tables:
        table = raw_tables["disease_similarity"]
        diseases.update(table["disease_id_1"].astype(str))
        diseases.update(table["disease_id_2"].astype(str))
    if "drug_target" in raw_tables:
        table = raw_tables["drug_target"]
        drugs.update(table["drug_id"].astype(str))
        targets.update(table["target_id"].astype(str))
    if "target_disease" in raw_tables:
        table = raw_tables["target_disease"]
        targets.update(table["target_id"].astype(str))
        diseases.update(table["disease_id"].astype(str))
    if "drug_smiles" in raw_tables:
        table = raw_tables["drug_smiles"]
        if "drug_id" in table.columns:
            drugs.update(table["drug_id"].astype(str))
    if "drug_fingerprint" in raw_tables:
        table = raw_tables["drug_fingerprint"]
        if "drug_id" in table.columns:
            drugs.update(table["drug_id"].astype(str))
    if "disease_mesh" in raw_tables:
        table = raw_tables["disease_mesh"]
        if "disease_id" in table.columns:
            diseases.update(table["disease_id"].astype(str))
    if "disease_gene" in raw_tables:
        table = raw_tables["disease_gene"]
        if "disease_id" in table.columns:
            diseases.update(table["disease_id"].astype(str))

    return sorted(drugs), sorted(diseases), sorted(targets)


def add_node_ids_with_maps(
    examples: pd.DataFrame,
    extra_drug_ids: list[str] | None = None,
    extra_disease_ids: list[str] | None = None,
    target_ids: list[str] | None = None,
) -> tuple[pd.DataFrame, int, dict[str, dict[str, int]]]:
    """Docstring."""

    out = examples.copy()
    drugs = sorted(set(out["drug_id"].astype(str)).union(extra_drug_ids or []))
    diseases = sorted(set(out["disease_id"].astype(str)).union(extra_disease_ids or []))
    targets = sorted(set(target_ids or []))
    drug_to_idx = {drug: idx for idx, drug in enumerate(drugs)}
    disease_to_idx = {disease: len(drugs) + idx for idx, disease in enumerate(diseases)}
    target_to_idx = {target: len(drugs) + len(diseases) + idx for idx, target in enumerate(targets)}
    out["drug_node_id"] = out["drug_id"].map(drug_to_idx).astype(int)
    out["disease_node_id"] = out["disease_id"].map(disease_to_idx).astype(int)
    maps = {
        "drug": drug_to_idx,
        "disease": disease_to_idx,
        "target": target_to_idx,
    }
    return out, len(drugs) + len(diseases) + len(targets), maps


def add_node_ids_from_maps(
    examples: pd.DataFrame,
    node_maps: dict[str, dict[str, int]],
) -> pd.DataFrame:
    """使用预先固定的全局节点映射为样本对编号，防止每折重新编号。"""

    out = examples.copy()
    out["drug_id"] = out["drug_id"].astype(str)
    out["disease_id"] = out["disease_id"].astype(str)
    out["drug_node_id"] = out["drug_id"].map(node_maps["drug"])
    out["disease_node_id"] = out["disease_id"].map(node_maps["disease"])
    if out[["drug_node_id", "disease_node_id"]].isna().any().any():
        missing = out.loc[
            out[["drug_node_id", "disease_node_id"]].isna().any(axis=1),
            ["drug_id", "disease_id"],
        ].head(5)
        raise ValueError(f"全局节点映射缺少样本实体: {missing.to_dict(orient='records')}")
    out["drug_node_id"] = out["drug_node_id"].astype(int)
    out["disease_node_id"] = out["disease_node_id"].astype(int)
    return out


def add_node_ids(examples: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Docstring."""

    out, num_nodes, _ = add_node_ids_with_maps(examples)
    return out, num_nodes


def _first_existing_column(table: pd.DataFrame, candidates: list[str]) -> str | None:
    """Docstring."""

    lower_to_original = {column.lower(): column for column in table.columns}
    for candidate in candidates:
        column = lower_to_original.get(candidate.lower())
        if column is not None:
            return column
    return None


def _hash_token_to_index(token: str, dim: int) -> int:
    """Docstring."""

    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest, 16) % dim


def _mesh_semantic_vector(row: pd.Series, dim: int) -> np.ndarray:
    """Docstring."""

    vector = np.zeros(dim, dtype=np.float32)
    text = " ".join(str(value) for value in row.dropna().tolist())
    tokens = re.findall(r"[A-Za-z0-9_.:-]+", text.lower())
    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        if "." in token:
            pieces = token.split(".")
            for idx in range(1, len(pieces) + 1):
                expanded.append(".".join(pieces[:idx]))
    for token in expanded:
        vector[_hash_token_to_index(token, dim)] += 1.0
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


def _smiles_to_morgan_fingerprint(smiles: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray | None:
    """Docstring."""

    try:
        from rdkit import RDLogger
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except ImportError:
        return None

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    bit_vector = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    array = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bit_vector, array)
    return array


def _parse_fingerprint_value(value: object) -> np.ndarray | None:
    """Docstring."""

    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"[01]+", text):
        return np.array([float(char) for char in text], dtype=np.float32)
    parts = [part for part in re.split(r"[\s,;|]+", text) if part]
    try:
        return np.array([float(part) for part in parts], dtype=np.float32)
    except ValueError:
        return None


def _external_drug_fingerprint_lookup(table: pd.DataFrame) -> tuple[dict[str, np.ndarray], int]:
    """Docstring."""

    if table.empty or "drug_id" not in table.columns:
        return {}, 0
    vector_column = _first_existing_column(table, ["fingerprint", "fp", "morgan", "maccs"])
    lookup: dict[str, np.ndarray] = {}
    if vector_column is not None:
        for drug_id, value in table[["drug_id", vector_column]].itertuples(index=False):
            vector = _parse_fingerprint_value(value)
            if vector is not None and vector.size > 0:
                lookup[str(drug_id)] = vector
    else:
        feature_columns = [column for column in table.columns if column != "drug_id"]
        numeric = table[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        for drug_id, values in zip(table["drug_id"].astype(str), numeric.to_numpy(dtype=np.float32)):
            lookup[str(drug_id)] = values
    if not lookup:
        return {}, 0
    dim = max(vector.size for vector in lookup.values())
    padded = {}
    for drug_id, vector in lookup.items():
        out = np.zeros(dim, dtype=np.float32)
        out[: min(dim, vector.size)] = vector[:dim]
        padded[drug_id] = out
    return padded, dim


def build_node_feature_matrix(
    raw_tables: dict[str, pd.DataFrame] | None,
    node_maps: dict[str, dict[str, int]],
    num_nodes: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, dict[str, int | float | bool]]:
    """Docstring."""

    if raw_tables is None:
        return None, {
            "node_features_enabled": False,
            "node_similarity_profiles_enabled": True,
            "node_feature_dim": 0,
            "node_feature_nonzero": 0,
            "node_feature_rows_with_signal": 0,
        }

    drug_map = node_maps["drug"]
    disease_map = node_maps["disease"]
    target_map = node_maps["target"]
    drug_feature_index = {drug_id: idx for idx, drug_id in enumerate(drug_map)}
    disease_offset = len(drug_feature_index)
    disease_feature_index = {
        disease_id: disease_offset + idx for idx, disease_id in enumerate(disease_map)
    }
    target_offset = disease_offset + len(disease_feature_index)
    target_feature_index = {
        target_id: target_offset + idx for idx, target_id in enumerate(target_map)
    }
    base_feature_dim = target_offset + len(target_feature_index)

    smiles_dim = 2048 if "drug_smiles" in raw_tables else 0
    smiles_offset = base_feature_dim
    external_fp_lookup, external_fp_dim = _external_drug_fingerprint_lookup(
        raw_tables.get("drug_fingerprint", pd.DataFrame())
    )
    external_fp_offset = smiles_offset + smiles_dim
    mesh_dim = 256 if "disease_mesh" in raw_tables else 0
    mesh_offset = external_fp_offset + external_fp_dim
    disease_gene = raw_tables.get("disease_gene", pd.DataFrame())
    gene_column = _first_existing_column(disease_gene, ["gene_id", "gene", "target_id", "entrez_id", "symbol"])
    gene_ids = sorted(disease_gene[gene_column].astype(str).unique().tolist()) if gene_column else []
    gene_offset = mesh_offset + mesh_dim
    gene_feature_index = {gene_id: gene_offset + idx for idx, gene_id in enumerate(gene_ids)}
    feature_dim = gene_offset + len(gene_feature_index)
    if feature_dim == 0:
        return None, {
            "node_features_enabled": False,
            "node_similarity_profiles_enabled": True,
            "node_feature_dim": 0,
            "node_feature_nonzero": 0,
            "node_feature_rows_with_signal": 0,
        }

    features = np.zeros((num_nodes, feature_dim), dtype=np.float32)

    for drug_id, node_id in drug_map.items():
        features[node_id, drug_feature_index[drug_id]] = 1.0
    for disease_id, node_id in disease_map.items():
        features[node_id, disease_feature_index[disease_id]] = 1.0
    for target_id, node_id in target_map.items():
        features[node_id, target_feature_index[target_id]] = 1.0

    drug_similarity = raw_tables.get("drug_similarity", pd.DataFrame())
    if not drug_similarity.empty:
        for left, right, score in drug_similarity[["drug_id_1", "drug_id_2", "score"]].itertuples(index=False):
            left, right, score = str(left), str(right), float(score)
            left_node = drug_map.get(left)
            right_node = drug_map.get(right)
            left_feature = drug_feature_index.get(left)
            right_feature = drug_feature_index.get(right)
            if left_node is not None and right_feature is not None:
                features[left_node, right_feature] = max(features[left_node, right_feature], score)
            if right_node is not None and left_feature is not None:
                features[right_node, left_feature] = max(features[right_node, left_feature], score)

    disease_similarity = raw_tables.get("disease_similarity", pd.DataFrame())
    if not disease_similarity.empty:
        for left, right, score in disease_similarity[["disease_id_1", "disease_id_2", "score"]].itertuples(index=False):
            left, right, score = str(left), str(right), float(score)
            left_node = disease_map.get(left)
            right_node = disease_map.get(right)
            left_feature = disease_feature_index.get(left)
            right_feature = disease_feature_index.get(right)
            if left_node is not None and right_feature is not None:
                features[left_node, right_feature] = max(features[left_node, right_feature], score)
            if right_node is not None and left_feature is not None:
                features[right_node, left_feature] = max(features[right_node, left_feature], score)

    drug_target = raw_tables.get("drug_target", pd.DataFrame())
    if not drug_target.empty:
        for drug_id, target_id in drug_target[["drug_id", "target_id"]].itertuples(index=False):
            drug_node = drug_map.get(str(drug_id))
            target_feature = target_feature_index.get(str(target_id))
            if drug_node is not None and target_feature is not None:
                features[drug_node, target_feature] = 1.0

    target_disease = raw_tables.get("target_disease", pd.DataFrame())
    if not target_disease.empty:
        for target_id, disease_id in target_disease[["target_id", "disease_id"]].itertuples(index=False):
            disease_node = disease_map.get(str(disease_id))
            target_feature = target_feature_index.get(str(target_id))
            if disease_node is not None and target_feature is not None:
                features[disease_node, target_feature] = 1.0

    smiles_valid = 0
    smiles_invalid = 0
    drug_smiles = raw_tables.get("drug_smiles", pd.DataFrame())
    smiles_column = _first_existing_column(drug_smiles, ["smiles", "canonical_smiles", "isomeric_smiles"])
    if smiles_dim and smiles_column is not None and "drug_id" in drug_smiles.columns:
        for drug_id, smiles in drug_smiles[["drug_id", smiles_column]].itertuples(index=False):
            drug_node = drug_map.get(str(drug_id))
            if drug_node is None:
                continue
            fingerprint = _smiles_to_morgan_fingerprint(str(smiles), n_bits=smiles_dim)
            if fingerprint is None:
                smiles_invalid += 1
                continue
            smiles_valid += 1
            features[drug_node, smiles_offset : smiles_offset + smiles_dim] = fingerprint

    external_fp_rows = 0
    for drug_id, vector in external_fp_lookup.items():
        drug_node = drug_map.get(drug_id)
        if drug_node is None:
            continue
        external_fp_rows += 1
        features[drug_node, external_fp_offset : external_fp_offset + external_fp_dim] = vector

    mesh_rows = 0
    disease_mesh = raw_tables.get("disease_mesh", pd.DataFrame())
    if mesh_dim and "disease_id" in disease_mesh.columns:
        semantic_columns = [column for column in disease_mesh.columns if column != "disease_id"]
        for _, row in disease_mesh.iterrows():
            disease_node = disease_map.get(str(row["disease_id"]))
            if disease_node is None:
                continue
            mesh_rows += 1
            vector = _mesh_semantic_vector(row[semantic_columns], mesh_dim)
            features[disease_node, mesh_offset : mesh_offset + mesh_dim] = np.maximum(
                features[disease_node, mesh_offset : mesh_offset + mesh_dim],
                vector,
            )

    disease_gene_rows = 0
    if gene_column is not None and "disease_id" in disease_gene.columns:
        for disease_id, gene_id in disease_gene[["disease_id", gene_column]].drop_duplicates().itertuples(index=False):
            disease_node = disease_map.get(str(disease_id))
            gene_feature = gene_feature_index.get(str(gene_id))
            if disease_node is None or gene_feature is None:
                continue
            disease_gene_rows += 1
            features[disease_node, gene_feature] = 1.0

    row_norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(row_norms, 1.0)
    tensor = torch.tensor(features, dtype=torch.float32, device=device)
    stats = {
        "node_features_enabled": True,
        "node_similarity_profiles_enabled": True,
        "node_feature_dim": int(feature_dim),
        "node_feature_nonzero": int(np.count_nonzero(features)),
        "node_feature_rows_with_signal": int(np.count_nonzero(row_norms.squeeze(-1) > 0)),
        "drug_smiles_rows": int(len(drug_smiles)) if "drug_smiles" in raw_tables else 0,
        "drug_smiles_valid": int(smiles_valid),
        "drug_smiles_invalid": int(smiles_invalid),
        "morgan_fingerprint_dim": int(smiles_dim),
        "external_drug_fingerprint_rows": int(external_fp_rows),
        "external_drug_fingerprint_dim": int(external_fp_dim),
        "disease_mesh_rows": int(mesh_rows),
        "mesh_semantic_dim": int(mesh_dim),
        "disease_gene_rows": int(disease_gene_rows),
        "disease_gene_feature_dim": int(len(gene_feature_index)),
    }
    return tensor, stats


def build_similarity_spectrum_tensors(
    raw_tables: dict[str, pd.DataFrame],
    node_maps: dict[str, dict[str, int]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | bool]]:
    """构建标准二维药物和疾病相似性矩阵，供 SS-Encoder 按行读取。"""

    drug_map = node_maps["drug"]
    disease_map = node_maps["disease"]
    drug_count = len(drug_map)
    disease_count = len(disease_map)
    disease_offset = min(disease_map.values()) if disease_map else 0
    drug_matrix = np.eye(drug_count, dtype=np.float32)
    disease_matrix = np.eye(disease_count, dtype=np.float32)

    drug_similarity = raw_tables.get("drug_similarity", pd.DataFrame())
    for left, right, score in drug_similarity[["drug_id_1", "drug_id_2", "score"]].itertuples(index=False):
        left_index = drug_map.get(str(left))
        right_index = drug_map.get(str(right))
        if left_index is None or right_index is None:
            continue
        value = float(np.clip(pd.to_numeric(score, errors="coerce"), 0.0, 1.0))
        drug_matrix[left_index, right_index] = max(drug_matrix[left_index, right_index], value)
        drug_matrix[right_index, left_index] = max(drug_matrix[right_index, left_index], value)

    disease_similarity = raw_tables.get("disease_similarity", pd.DataFrame())
    for left, right, score in disease_similarity[["disease_id_1", "disease_id_2", "score"]].itertuples(index=False):
        left_node = disease_map.get(str(left))
        right_node = disease_map.get(str(right))
        if left_node is None or right_node is None:
            continue
        left_index = left_node - disease_offset
        right_index = right_node - disease_offset
        value = float(np.clip(pd.to_numeric(score, errors="coerce"), 0.0, 1.0))
        disease_matrix[left_index, right_index] = max(disease_matrix[left_index, right_index], value)
        disease_matrix[right_index, left_index] = max(disease_matrix[right_index, left_index], value)

    stats: dict[str, int | bool] = {
        "ss_encoder_enabled": True,
        "ss_drug_dim": int(drug_count),
        "ss_disease_dim": int(disease_count),
        "ss_input_dim": int(drug_count + disease_count),
    }
    return (
        torch.tensor(drug_matrix, dtype=torch.float32, device=device),
        torch.tensor(disease_matrix, dtype=torch.float32, device=device),
        stats,
    )


def similarity_spectrum_batch(
    drug_node_ids: torch.Tensor,
    disease_node_ids: torch.Tensor,
    drug_spectrum: torch.Tensor | None,
    disease_spectrum: torch.Tensor | None,
    disease_node_offset: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """按当前 pair batch 取出药物和疾病相似性矩阵行。"""

    if drug_spectrum is None or disease_spectrum is None:
        return None
    disease_indices = disease_node_ids - int(disease_node_offset)
    return drug_spectrum[drug_node_ids], disease_spectrum[disease_indices]


def build_association_edge_index(
    train_df: pd.DataFrame,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """Docstring."""

    positives = train_df[train_df["label"] == 1]
    edge_parts = []
    if not positives.empty:
        drug_ids = torch.tensor(positives["drug_node_id"].to_numpy(), dtype=torch.long)
        disease_ids = torch.tensor(positives["disease_node_id"].to_numpy(), dtype=torch.long)
        edge_parts.extend(
            [
                torch.stack([drug_ids, disease_ids], dim=0),
                torch.stack([disease_ids, drug_ids], dim=0),
            ]
        )

    self_loops = torch.arange(num_nodes, dtype=torch.long)
    edge_parts.append(torch.stack([self_loops, self_loops], dim=0))
    edge_index = torch.cat(edge_parts, dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index.to(device)


def _edge_index_from_edges(
    edges: list[tuple[int, int]],
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """Docstring."""

    edge_parts = []
    if edges:
        filtered_edges = [
            (src, dst)
            for src, dst in edges
            if 0 <= src < num_nodes and 0 <= dst < num_nodes
        ]
        if filtered_edges:
            edge_parts.append(torch.tensor(filtered_edges, dtype=torch.long).t().contiguous())

    self_loops = torch.arange(num_nodes, dtype=torch.long)
    edge_parts.append(torch.stack([self_loops, self_loops], dim=0))
    edge_index = torch.cat(edge_parts, dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index.to(device)


def _append_bidirectional_edges(
    edges: list[tuple[int, int]],
    table: pd.DataFrame,
    source_column: str,
    target_column: str,
    source_map: dict[str, int],
    target_map: dict[str, int],
) -> None:
    """Docstring."""

    for source_id, target_id in table[[source_column, target_column]].drop_duplicates().itertuples(index=False, name=None):
        source_node = source_map.get(str(source_id))
        target_node = target_map.get(str(target_id))
        if source_node is None or target_node is None:
            continue
        edges.append((source_node, target_node))
        edges.append((target_node, source_node))


def _top_k_similarity_edges(
    table: pd.DataFrame,
    left_column: str,
    right_column: str,
    node_map: dict[str, int],
    top_k: int | None,
    min_similarity: float,
) -> list[tuple[int, int]]:
    if table.empty:
        return []
    if "score" not in table.columns:
        filtered = table[[left_column, right_column]].copy()
        filtered["score"] = 1.0
    else:
        filtered = table[[left_column, right_column, "score"]].copy()
        filtered["score"] = pd.to_numeric(filtered["score"], errors="coerce")
        filtered = filtered.dropna(subset=["score"])
        filtered = filtered[filtered["score"] > min_similarity]

    directed_rows: list[tuple[int, int, float]] = []
    for left_id, right_id, score in filtered[[left_column, right_column, "score"]].itertuples(index=False, name=None):
        left_node = node_map.get(str(left_id))
        right_node = node_map.get(str(right_id))
        if left_node is None or right_node is None or left_node == right_node:
            continue
        directed_rows.append((left_node, right_node, float(score)))
        directed_rows.append((right_node, left_node, float(score)))

    if not directed_rows:
        return []
    directed = pd.DataFrame(directed_rows, columns=["source", "target", "score"])
    directed = directed.sort_values("score", ascending=False).drop_duplicates(["source", "target"])
    if top_k is not None and top_k > 0:
        directed = directed.sort_values(["source", "score", "target"], ascending=[True, False, True])
        directed = directed.groupby("source", sort=False).head(top_k)
    selected_edges = {(int(row.source), int(row.target)) for row in directed.itertuples(index=False)}
    selected_edges.update({(target, source) for source, target in selected_edges})
    return sorted(selected_edges)


def build_similarity_edge_index(
    raw_tables: dict[str, pd.DataFrame],
    node_maps: dict[str, dict[str, int]],
    num_nodes: int,
    device: torch.device,
    similarity_top_k: int | None = 20,
    drug_similarity_top_k: int | None = None,
    disease_similarity_top_k: int | None = None,
    min_similarity: float = 0.0,
) -> torch.Tensor:
    """Docstring."""

    edges: list[tuple[int, int]] = []
    drug_similarity = raw_tables.get("drug_similarity", pd.DataFrame())
    disease_similarity = raw_tables.get("disease_similarity", pd.DataFrame())
    effective_drug_top_k = similarity_top_k if drug_similarity_top_k is None else drug_similarity_top_k
    effective_disease_top_k = similarity_top_k if disease_similarity_top_k is None else disease_similarity_top_k
    if not drug_similarity.empty:
        edges.extend(
            _top_k_similarity_edges(
                drug_similarity,
                "drug_id_1",
                "drug_id_2",
                node_maps["drug"],
                effective_drug_top_k,
                min_similarity,
            )
        )
    if not disease_similarity.empty:
        edges.extend(
            _top_k_similarity_edges(
                disease_similarity,
                "disease_id_1",
                "disease_id_2",
                node_maps["disease"],
                effective_disease_top_k,
                min_similarity,
            )
    )
    return _edge_index_from_edges(edges, num_nodes, device)


def build_biology_edge_index(
    raw_tables: dict[str, pd.DataFrame],
    node_maps: dict[str, dict[str, int]],
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """Docstring."""

    edges: list[tuple[int, int]] = []
    drug_target = raw_tables.get("drug_target", pd.DataFrame())
    target_disease = raw_tables.get("target_disease", pd.DataFrame())
    if not drug_target.empty:
        _append_bidirectional_edges(
            edges,
            drug_target,
            "drug_id",
            "target_id",
            node_maps["drug"],
            node_maps["target"],
        )
    if not target_disease.empty:
        _append_bidirectional_edges(
            edges,
            target_disease,
            "target_id",
            "disease_id",
            node_maps["target"],
            node_maps["disease"],
        )
    return _edge_index_from_edges(edges, num_nodes, device)


def build_view_edge_indices(
    train_df: pd.DataFrame,
    num_nodes: int,
    device: torch.device,
    raw_tables: dict[str, pd.DataFrame] | None = None,
    node_maps: dict[str, dict[str, int]] | None = None,
    similarity_top_k: int | None = 20,
    drug_similarity_top_k: int | None = None,
    disease_similarity_top_k: int | None = None,
    min_similarity: float = 0.0,
) -> list[torch.Tensor]:
    """Docstring."""

    association = build_association_edge_index(train_df, num_nodes=num_nodes, device=device)
    if raw_tables is None or node_maps is None:
        return [association, association, association]

    similarity = build_similarity_edge_index(
        raw_tables,
        node_maps,
        num_nodes,
        device,
        similarity_top_k=similarity_top_k,
        drug_similarity_top_k=drug_similarity_top_k,
        disease_similarity_top_k=disease_similarity_top_k,
        min_similarity=min_similarity,
    )
    biology = build_biology_edge_index(raw_tables, node_maps, num_nodes, device)
    return [association, similarity, biology]


def _graph_fingerprint(view_edge_indices: list[torch.Tensor]) -> str:
    """为三张固定全局图生成可复现的内容指纹，便于日志核验。"""

    digest = hashlib.sha256()
    for edge_index in view_edge_indices:
        cpu_edges = edge_index.detach().cpu().contiguous().numpy()
        digest.update(str(cpu_edges.shape).encode("ascii"))
        digest.update(cpu_edges.tobytes())
    return digest.hexdigest()


def build_global_graph_views(
    global_data: GlobalPreprocessedData,
    *,
    device: torch.device,
    similarity_top_k: int | None,
    drug_similarity_top_k: int | None,
    disease_similarity_top_k: int | None,
    min_similarity: float,
    use_ss_encoder: bool,
    ss_dim: int,
    ss_channels: int,
) -> GlobalGraphContext:
    """仅在全局预处理阶段构建一次并冻结三视图和相似性谱特征。"""

    global_examples = pd.concat(
        [global_data.positives, global_data.reliable_negatives],
        ignore_index=True,
    )
    extra_drug_ids, extra_disease_ids, target_ids = collect_auxiliary_node_ids(global_data.tables)
    examples_with_ids, num_nodes, node_maps = add_node_ids_with_maps(
        global_examples,
        extra_drug_ids=extra_drug_ids,
        extra_disease_ids=extra_disease_ids,
        target_ids=target_ids,
    )
    global_positive_with_ids = examples_with_ids.iloc[: len(global_data.positives)].copy()
    view_edge_indices = build_view_edge_indices(
        global_positive_with_ids,
        num_nodes=num_nodes,
        device=device,
        raw_tables=global_data.tables,
        node_maps=node_maps,
        similarity_top_k=similarity_top_k,
        drug_similarity_top_k=drug_similarity_top_k,
        disease_similarity_top_k=disease_similarity_top_k,
        min_similarity=min_similarity,
    )
    node_features, node_feature_stats = build_node_feature_matrix(
        raw_tables=global_data.tables,
        node_maps=node_maps,
        num_nodes=num_nodes,
        device=device,
    )
    disease_node_offset = min(node_maps["disease"].values()) if node_maps["disease"] else 0
    ss_feature_stats: dict[str, int | bool] = {
        "ss_encoder_enabled": False,
        "ss_drug_dim": 0,
        "ss_disease_dim": 0,
        "ss_input_dim": 0,
        "ss_output_dim": 0,
        "ss_channels": 0,
    }
    drug_spectrum: torch.Tensor | None = None
    disease_spectrum: torch.Tensor | None = None
    if use_ss_encoder:
        drug_spectrum, disease_spectrum, ss_feature_stats = build_similarity_spectrum_tensors(
            raw_tables=global_data.tables,
            node_maps=node_maps,
            device=device,
        )
        ss_feature_stats["ss_output_dim"] = int(ss_dim)
        ss_feature_stats["ss_channels"] = int(ss_channels)
    return GlobalGraphContext(
        num_nodes=num_nodes,
        node_maps=node_maps,
        view_edge_indices=view_edge_indices,
        node_features=node_features,
        drug_spectrum=drug_spectrum,
        disease_spectrum=disease_spectrum,
        disease_node_offset=disease_node_offset,
        node_feature_stats=node_feature_stats,
        ss_feature_stats=ss_feature_stats,
        graph_fingerprint=_graph_fingerprint(view_edge_indices),
    )


def tensor_batch(df: pd.DataFrame, indices: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Docstring."""

    batch = df.iloc[indices]
    drug_ids = torch.tensor(batch["drug_node_id"].to_numpy(), dtype=torch.long, device=device)
    disease_ids = torch.tensor(batch["disease_node_id"].to_numpy(), dtype=torch.long, device=device)
    labels_np = pd.to_numeric(batch["label"], errors="raise").to_numpy(dtype=np.float32)
    weights_np = pd.to_numeric(batch["reliability"].fillna(1.0), errors="raise").to_numpy(dtype=np.float32)
    risk_columns = []
    for column in RISK_FEATURE_COLUMNS:
        default = DEFAULT_RISK_FEATURE_VALUES[column]
        if column in batch.columns:
            values = pd.to_numeric(batch[column], errors="coerce").fillna(default).to_numpy(dtype=np.float32)
        else:
            values = np.full(len(batch), default, dtype=np.float32)
        risk_columns.append(values)
    reliability_np = np.stack(risk_columns, axis=1).astype(np.float32)
    if SAMPLE_ROLE_COLUMN in batch.columns:
        role_values = batch[SAMPLE_ROLE_COLUMN].fillna(ROLE_RELIABLE_NEGATIVE).astype(str)
    else:
        role_values = pd.Series(
            np.where(labels_np.astype(int) == 1, ROLE_POSITIVE, ROLE_RELIABLE_NEGATIVE),
            index=batch.index,
        )
    role_np = role_values.map(ROLE_CODES).fillna(ROLE_CODES[ROLE_RELIABLE_NEGATIVE]).to_numpy(dtype=np.int64)
    labels = torch.tensor(labels_np, dtype=torch.float32, device=device)
    weights = torch.tensor(weights_np, dtype=torch.float32, device=device)
    reliability_features = torch.tensor(reliability_np, dtype=torch.float32, device=device)
    role_codes = torch.tensor(role_np, dtype=torch.long, device=device)
    return drug_ids, disease_ids, labels, weights, reliability_features, role_codes


def classify_pu_hard_samples(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    role_codes: torch.Tensor,
    reliability_features: torch.Tensor,
    low_threshold: float,
    high_threshold: float,
    potential_risk_threshold: float,
    allow_reliable_negative_potential: bool = False,
) -> dict[str, torch.Tensor]:
    """Docstring."""

    detached_predictions = predictions.detach()
    risk = reliability_features[:, RISK_FEATURE_COLUMNS.index("false_negative_risk")]
    reliable_negative_mask = role_codes == ROLE_CODES[ROLE_RELIABLE_NEGATIVE]
    pu_unlabeled_mask = role_codes == ROLE_CODES[ROLE_PU_UNLABELED]
    positive_mask = labels > 0.5
    hard_reliable = reliable_negative_mask & (detached_predictions >= high_threshold)
    easy_reliable = reliable_negative_mask & (detached_predictions <= low_threshold)
    potential_candidates = pu_unlabeled_mask | (
        reliable_negative_mask if allow_reliable_negative_potential else torch.zeros_like(reliable_negative_mask)
    )
    potential_positive = potential_candidates & (risk >= potential_risk_threshold) & (
        detached_predictions >= high_threshold
    )
    uncertain = pu_unlabeled_mask & ~potential_positive
    return {
        "positive": positive_mask,
        "reliable_negative": easy_reliable,
        "hard_reliable_negative": hard_reliable,
        "uncertain_unlabeled": uncertain,
        "potential_positive": potential_positive,
        "pu_unlabeled": pu_unlabeled_mask,
        "supervised": (~pu_unlabeled_mask) & ~potential_positive,
    }


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Docstring."""

    if values.numel() == 0:
        return values.sum()
    weight_sum = weights.sum()
    if float(weight_sum.detach().cpu()) <= 0.0:
        return values.mean()
    return (values * weights).sum() / weight_sum


def non_negative_pu_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    role_codes: torch.Tensor,
    reliability_features: torch.Tensor,
    class_prior: float,
    high_threshold: float,
    potential_risk_threshold: float,
    uncertain_weight: float,
    hard_weight: float,
) -> torch.Tensor:
    """Docstring."""

    positive_mask = labels > 0.5
    pu_unlabeled_mask = role_codes == ROLE_CODES[ROLE_PU_UNLABELED]
    if not bool(positive_mask.any()) or not bool(pu_unlabeled_mask.any()):
        return predictions.sum() * 0.0

    risk = reliability_features[:, RISK_FEATURE_COLUMNS.index("false_negative_risk")]
    detached_predictions = predictions.detach()
    potential_positive = pu_unlabeled_mask & (risk >= potential_risk_threshold) & (
        detached_predictions >= high_threshold
    )
    active_unlabeled = pu_unlabeled_mask & ~potential_positive
    if not bool(active_unlabeled.any()):
        return predictions.sum() * 0.0

    pos_predictions = predictions[positive_mask]
    unlabeled_predictions = predictions[active_unlabeled]
    pos_positive_loss = torch.nn.functional.binary_cross_entropy(
        pos_predictions,
        torch.ones_like(pos_predictions),
        reduction="none",
    )
    pos_negative_loss = torch.nn.functional.binary_cross_entropy(
        pos_predictions,
        torch.zeros_like(pos_predictions),
        reduction="none",
    )
    unlabeled_negative_loss = torch.nn.functional.binary_cross_entropy(
        unlabeled_predictions,
        torch.zeros_like(unlabeled_predictions),
        reduction="none",
    )

    active_risk = risk[active_unlabeled]
    active_predictions = detached_predictions[active_unlabeled]
    unlabeled_weights = torch.full_like(unlabeled_negative_loss, float(uncertain_weight))
    hard_unlabeled = (active_risk < potential_risk_threshold) & (active_predictions >= high_threshold)
    unlabeled_weights = torch.where(hard_unlabeled, torch.full_like(unlabeled_weights, float(hard_weight)), unlabeled_weights)

    prior = float(np.clip(class_prior, 1e-4, 0.5))
    positive_risk = prior * pos_positive_loss.mean()
    negative_risk = weighted_mean(unlabeled_negative_loss, unlabeled_weights) - prior * pos_negative_loss.mean()
    return positive_risk + torch.clamp(negative_risk, min=0.0)


def safe_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float | None]:
    """Docstring."""

    y_pred = (y_score >= 0.5).astype(int)
    metrics: dict[str, float | None] = {
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Precision": float(np.sum((y_pred == 1) & (y_true == 1)) / max(1, np.sum(y_pred == 1))),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if float(np.std(y_true)) == 0.0 or float(np.std(y_score)) == 0.0:
        metrics["pearson_r"] = None
    else:
        metrics["pearson_r"] = float(np.corrcoef(y_true, y_score)[0, 1])

    if len(np.unique(y_true)) < 2:
        metrics["AUC"] = None
        metrics["AUPR"] = None
    else:
        metrics["AUC"] = float(roc_auc_score(y_true, y_score))
        metrics["AUPR"] = float(average_precision_score(y_true, y_score))

    order = np.argsort(-y_score)[: min(10, len(y_true))]
    metrics["Precision@10"] = float(np.sum(y_true[order]) / max(1, len(order)))
    return metrics


def evaluate_model(
    model: torch.nn.Module,
    test_df: pd.DataFrame,
    device: torch.device,
    view_edge_indices: list[torch.Tensor],
    node_features: torch.Tensor | None = None,
    drug_spectrum: torch.Tensor | None = None,
    disease_spectrum: torch.Tensor | None = None,
    disease_node_offset: int = 0,
) -> tuple[dict[str, float | None], np.ndarray, torch.Tensor]:
    """Docstring."""

    model.eval()
    with torch.no_grad():
        test_indices = np.arange(len(test_df))
        drug_ids, disease_ids, labels, _, reliability_features, _ = tensor_batch(test_df, test_indices, device)
        spectrum = similarity_spectrum_batch(
            drug_ids,
            disease_ids,
            drug_spectrum,
            disease_spectrum,
            disease_node_offset,
        )
        predictions, view_weights, _ = model(
            drug_ids,
            disease_ids,
            view_edge_indices,
            node_features=node_features,
            reliability_features=reliability_features,
            similarity_spectrum=spectrum,
        )

    y_true = labels.cpu().numpy().astype(int)
    y_score = predictions.cpu().numpy()
    return safe_metrics(y_true, y_score), y_score, view_weights


def extract_pair_features(
    model: torch.nn.Module,
    examples: pd.DataFrame,
    device: torch.device,
    view_edge_indices: list[torch.Tensor],
    batch_size: int,
    node_features: torch.Tensor | None = None,
    drug_spectrum: torch.Tensor | None = None,
    disease_spectrum: torch.Tensor | None = None,
    disease_node_offset: int = 0,
) -> np.ndarray:
    """导出融合 pair 表示、视图权重和可靠性特征，供外部分类器使用。"""

    feature_batches: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch_indices = np.arange(start, min(start + batch_size, len(examples)))
            drug_ids, disease_ids, _, _, reliability_features, _ = tensor_batch(
                examples,
                batch_indices,
                device,
            )
            spectrum = similarity_spectrum_batch(
                drug_ids,
                disease_ids,
                drug_spectrum,
                disease_spectrum,
                disease_node_offset,
            )
            fused_pair, view_weights, _ = model.encode_pair_features(
                drug_ids,
                disease_ids,
                view_edge_indices,
                node_features=node_features,
                reliability_features=reliability_features,
                similarity_spectrum=spectrum,
            )
            feature_batches.append(
                torch.cat([fused_pair, view_weights, reliability_features], dim=1).cpu().numpy()
            )

    if not feature_batches:
        raise ValueError("LightGBM 特征导出需要至少一个样本。")
    return np.concatenate(feature_batches, axis=0).astype(np.float32, copy=False)


def extract_fused_pair_features(
    model: torch.nn.Module,
    examples: pd.DataFrame,
    device: torch.device,
    view_edge_indices: list[torch.Tensor],
    batch_size: int,
    node_features: torch.Tensor | None = None,
    drug_spectrum: torch.Tensor | None = None,
    disease_spectrum: torch.Tensor | None = None,
    disease_node_offset: int = 0,
) -> np.ndarray:
    """Export only the learned fused pair embedding for representation analysis."""

    feature_batches: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch_indices = np.arange(start, min(start + batch_size, len(examples)))
            drug_ids, disease_ids, _, _, reliability_features, _ = tensor_batch(
                examples,
                batch_indices,
                device,
            )
            spectrum = similarity_spectrum_batch(
                drug_ids,
                disease_ids,
                drug_spectrum,
                disease_spectrum,
                disease_node_offset,
            )
            fused_pair, _, _ = model.encode_pair_features(
                drug_ids,
                disease_ids,
                view_edge_indices,
                node_features=node_features,
                reliability_features=reliability_features,
                similarity_spectrum=spectrum,
            )
            feature_batches.append(fused_pair.cpu().numpy())

    if not feature_batches:
        raise ValueError("Representation export requires at least one sample.")
    return np.concatenate(feature_batches, axis=0).astype(np.float32, copy=False)


def write_representation_export(
    path: Path,
    features: np.ndarray,
    examples: pd.DataFrame,
) -> None:
    """Persist learned representations with sample identity and labels."""

    feature_array = np.asarray(features, dtype=np.float32)
    if len(feature_array) != len(examples):
        raise ValueError("Representation rows must align with exported samples.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=feature_array,
        labels=examples["label"].to_numpy(dtype=np.int64),
        drug_ids=examples["drug_id"].astype(str).to_numpy(dtype=str),
        disease_ids=examples["disease_id"].astype(str).to_numpy(dtype=str),
    )


def lightgbm_feature_frame(features: np.ndarray) -> pd.DataFrame:
    """为 LightGBM 生成稳定的特征列名，避免训练和推理阶段的列名警告。"""

    feature_array = np.asarray(features, dtype=np.float32)
    return pd.DataFrame(
        feature_array,
        columns=[f"feature_{index}" for index in range(feature_array.shape[1])],
    )


def lightgbm_decoder_parameters(seed: int, n_estimators: int = 300) -> dict[str, int | float | str]:
    """返回写入日志的固定 LightGBM 参数。"""

    return {
        "objective": "binary",
        "n_estimators": int(n_estimators),
        "random_state": int(seed),
        "verbosity": -1,
        **LIGHTGBM_DECODER_DEFAULTS,
    }


def fit_lightgbm_decoder(
    train_features: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    seed: int,
    n_estimators: int = 300,
):
    """在固定图表示上训练 LightGBM 概率分类器。"""

    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise RuntimeError(
            "当前 Conda 虚拟环境缺少 lightgbm，请运行：python -m pip install lightgbm"
        ) from exc

    labels = np.asarray(labels, dtype=np.int64)
    if len(np.unique(labels)) != 2:
        raise ValueError("LightGBM 训练集必须同时包含正样本和可靠负样本。")
    decoder = LGBMClassifier(**lightgbm_decoder_parameters(seed=seed, n_estimators=n_estimators))
    decoder.fit(
        lightgbm_feature_frame(train_features),
        labels,
        sample_weight=np.asarray(sample_weights, dtype=np.float32),
    )
    return decoder


def evaluate_lightgbm_decoder(
    decoder,
    model: MultiViewDrugDiseaseModel,
    examples: pd.DataFrame,
    device: torch.device,
    view_edge_indices: list[torch.Tensor],
    batch_size: int,
    node_features: torch.Tensor | None = None,
    drug_spectrum: torch.Tensor | None = None,
    disease_spectrum: torch.Tensor | None = None,
    disease_node_offset: int = 0,
) -> tuple[dict[str, float | None], np.ndarray]:
    """使用固定的 LightGBM 分类器计算评估指标。"""

    features = extract_pair_features(
        model=model,
        examples=examples,
        device=device,
        view_edge_indices=view_edge_indices,
        batch_size=batch_size,
        node_features=node_features,
        drug_spectrum=drug_spectrum,
        disease_spectrum=disease_spectrum,
        disease_node_offset=disease_node_offset,
    )
    y_true = examples["label"].to_numpy(dtype=int)
    y_score = decoder.predict_proba(lightgbm_feature_frame(features))[:, 1]
    return safe_metrics(y_true, y_score), y_score


def resolve_device(requested: str) -> torch.device:
    """Docstring."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(requested)


def make_validation_split(
    train_df: pd.DataFrame,
    validation_ratio: float,
    seed: int,
    validation_mode: str = "random",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Docstring."""

    if validation_ratio <= 0.0:
        return train_df.reset_index(drop=True), train_df.iloc[0:0].copy().reset_index(drop=True)
    if validation_ratio >= 1.0:
        raise ValueError("validation_ratio must be smaller than 1.0")
    mode = validation_mode if validation_mode in set(SPLIT_MODES) else "random"
    train_core, validation = make_split(train_df, mode=mode, test_ratio=validation_ratio, seed=seed)
    return train_core.reset_index(drop=True), validation.reset_index(drop=True)


def use_reserved_validation(
    supervised_train_df: pd.DataFrame,
    pu_unlabeled_df: pd.DataFrame,
    test_df: pd.DataFrame,
    reserved_validation_df: pd.DataFrame,
) -> pd.DataFrame:
    validation = ensure_sample_role_column(
        ensure_risk_feature_columns(reserved_validation_df)
    ).reset_index(drop=True)
    if validation.empty:
        raise ValueError("The reserved validation set is empty.")
    forbidden_pairs = (
        _pair_set(supervised_train_df) | _pair_set(pu_unlabeled_df) | _pair_set(test_df)
    )
    if not _pair_set(validation).isdisjoint(forbidden_pairs):
        raise ValueError("The reserved validation set overlaps training, PU or test pairs.")
    return validation


def make_matched_random_validation_split(
    supervised_train_df: pd.DataFrame,
    pu_unlabeled_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    tables: dict[str, pd.DataFrame],
    validation_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """构造与 random 测试分布一致的正负平衡 validation。"""

    if validation_ratio <= 0.0 or validation_ratio >= 1.0:
        raise ValueError("matched random validation 要求 validation_ratio 位于 (0, 1)。")
    supervised = ensure_sample_role_column(
        ensure_risk_feature_columns(supervised_train_df)
    )
    positives = supervised.loc[supervised["label"] == 1].copy().reset_index(drop=True)
    negatives = supervised.loc[supervised["label"] == 0].copy().reset_index(drop=True)
    if len(positives) < 2:
        raise ValueError("matched random validation 至少需要 2 个训练正样本。")

    train_positive_indices, validation_positive_indices = train_test_split(
        np.arange(len(positives)),
        test_size=validation_ratio,
        random_state=seed,
        shuffle=True,
    )
    train_positives = positives.iloc[train_positive_indices].copy().reset_index(drop=True)
    validation_positives = positives.iloc[validation_positive_indices].copy().reset_index(drop=True)
    train_core = pd.concat([train_positives, negatives], ignore_index=True)

    known_positives = positive_examples_from_tables(tables)
    excluded_pairs = (
        _pair_set(known_positives)
        | _pair_set(train_core)
        | _pair_set(pu_unlabeled_df)
        | _pair_set(test_df)
    )
    all_drugs, all_diseases = _global_drug_disease_ids(tables)
    candidates = [
        (drug_id, disease_id)
        for drug_id in all_drugs
        for disease_id in all_diseases
        if (drug_id, disease_id) not in excluded_pairs
    ]
    target_count = len(validation_positives)
    if len(candidates) < target_count:
        raise ValueError("可用未知 pair 不足，无法构造等量 random validation 负样本。")
    rng = np.random.default_rng(seed + 1)
    chosen = rng.choice(len(candidates), size=target_count, replace=False)
    validation_negatives = pd.DataFrame(
        [candidates[int(index)] for index in chosen],
        columns=["drug_id", "disease_id"],
    )
    validation_negatives["label"] = 0
    validation_negatives = ensure_sample_role_column(
        ensure_risk_feature_columns(validation_negatives)
    )
    validation = pd.concat(
        [validation_positives, validation_negatives],
        ignore_index=True,
    ).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
    train_core = train_core.sample(frac=1.0, random_state=seed + 3).reset_index(drop=True)
    stats: dict[str, object] = {
        "matched_random_validation_enabled": True,
        "validation_negative_strategy": "uniform_random_unknown_pair",
        "validation_positive_count": int(len(validation_positives)),
        "validation_negative_count": int(len(validation_negatives)),
        "validation_pair_disjoint_from_train": _pair_set(validation).isdisjoint(
            _pair_set(train_core)
        ),
        "validation_pair_excluded_from_pu": _pair_set(validation).isdisjoint(
            _pair_set(pu_unlabeled_df)
        ),
    }
    return train_core, validation, stats


def make_stratified_validation_split(
    train_df: pd.DataFrame,
    validation_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """为平衡 warm-start 协议构造类别分层的 train-core/validation。"""

    if validation_ratio <= 0.0 or validation_ratio >= 1.0:
        raise ValueError("Balanced warm-start protocol 要求 validation_ratio 位于 (0, 1)。")
    labels = train_df["label"].to_numpy(dtype=int)
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2 or int(label_counts.min()) < 2:
        raise ValueError("分层 validation 至少需要每个类别各有 2 个样本。")
    train_indices, validation_indices = train_test_split(
        np.arange(len(train_df)),
        test_size=validation_ratio,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    train_core = train_df.iloc[train_indices].copy().reset_index(drop=True)
    validation = train_df.iloc[validation_indices].copy().reset_index(drop=True)
    return train_core, validation


def clone_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Docstring."""

    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train_predefined_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    run_name: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    embedding_dim: int,
    hidden_dim: int,
    lr: float,
    contrastive_weight: float,
    seed: int,
    device: torch.device,
    raw_tables: dict[str, pd.DataFrame] | None = None,
    similarity_top_k: int | None = 20,
    drug_similarity_top_k: int | None = None,
    disease_similarity_top_k: int | None = None,
    min_similarity: float = 0.0,
    fusion_type: str = "transformer",
    fusion_dim: int = 256,
    fusion_heads: int = 4,
    fusion_layers: int = 1,
    use_ss_encoder: bool = False,
    ss_dim: int = 128,
    ss_channels: int = 16,
    node_contrastive_weight: float = 0.0,
    contrastive_temperature: float = 0.2,
    decoder_type: str = "mlp",
    validation_ratio: float = 0.0,
    validation_metric: str = "AUPR",
    validation_mode: str = "random",
    predefined_validation_df: pd.DataFrame | None = None,
    pu_learning: bool = False,
    pu_unlabeled_ratio: float = 1.0,
    pu_loss_weight: float = 0.1,
    pu_class_prior: float | None = None,
    pu_low_threshold: float = 0.3,
    pu_high_threshold: float = 0.7,
    pu_potential_risk_threshold: float = 0.8,
    pu_uncertain_weight: float = 0.2,
    pu_hard_weight: float = 1.5,
    fixed_graph_context: GlobalGraphContext | None = None,
    evaluation_protocol: str = EVALUATION_PROTOCOL_FOLD_WISE,
    pu_update_reliable_negatives: bool = False,
    fixed_epoch_training: bool = False,
    skip_test_evaluation: bool = False,
    model_variant: str = MODEL_VARIANT_CURRENT,
    spectral_hops: int = 3,
    spectral_pair_dim: int = 256,
    active_views: tuple[str, ...] = ("A", "S", "B"),
    graph_encoder: str = "gatv2",
    representation_export_path: Path | None = None,
) -> dict[str, object]:
    """Docstring."""

    if evaluation_protocol != EVALUATION_PROTOCOL_FOLD_WISE:
        raise ValueError("Only fold_wise evaluation is supported for new training runs.")
    if not fixed_epoch_training and not 0.0 < validation_ratio < 1.0:
        raise ValueError("Strict fold-wise training requires validation_ratio in (0, 1).")
    if decoder_type not in {"mlp", "lightgbm"}:
        raise ValueError("decoder_type 必须为 'mlp' 或 'lightgbm'。")
    if model_variant not in MODEL_VARIANTS:
        raise ValueError(f"model_variant 必须是 {MODEL_VARIANTS} 之一。")
    if spectral_hops < 1 or spectral_pair_dim < 1:
        raise ValueError("spectral_hops 和 spectral_pair_dim 必须为正整数。")
    if model_variant == MODEL_VARIANT_SEMANTIC_PATH_SPECTRAL and decoder_type != "mlp":
        raise ValueError("semantic_path_spectral 首轮实验固定使用 MLP 解码器。")
    # SS-Encoder 仅保留为历史实现，当前正式训练路径不再实例化该模块。
    active_ss_encoder = False
    is_multiview_gatv2_variant = model_variant in {
        MODEL_VARIANT_CURRENT,
        MODEL_VARIANT_NO_SS,
    }
    canonical_views = ("A", "S", "B")
    active_views = tuple(active_views)
    if not active_views or active_views != tuple(v for v in canonical_views if v in active_views):
        raise ValueError("active_views must be a nonempty ordered subset of A, S, B")
    if not is_multiview_gatv2_variant and active_views != canonical_views:
        raise ValueError("View ablation only supports the multi-view GATv2 model")
    active_indices = [canonical_views.index(v) for v in active_views]
    effective_pair_contrastive_weight = (
        float(contrastive_weight) if is_multiview_gatv2_variant and len(active_views) > 1 else 0.0
    )
    effective_node_contrastive_weight = (
        float(node_contrastive_weight) if is_multiview_gatv2_variant and len(active_views) > 1 else 0.0
    )
    if contrastive_temperature <= 0.0:
        raise ValueError("contrastive_temperature must be positive")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = ensure_sample_role_column(ensure_risk_feature_columns(train_df))
    test_df = ensure_sample_role_column(ensure_risk_feature_columns(test_df))
    reserved_validation_len = 0 if predefined_validation_df is None else len(predefined_validation_df)
    if fixed_graph_context is None:
        train_len = len(train_df)
        frames = [train_df]
        if predefined_validation_df is not None:
            frames.append(predefined_validation_df)
        frames.append(test_df)
        combined = pd.concat(frames, ignore_index=True)
        extra_drug_ids, extra_disease_ids, target_ids = collect_auxiliary_node_ids(raw_tables)
        combined_with_ids, num_nodes, node_maps = add_node_ids_with_maps(
            combined,
            extra_drug_ids=extra_drug_ids,
            extra_disease_ids=extra_disease_ids,
            target_ids=target_ids,
        )
        train_df = combined_with_ids.iloc[:train_len].copy().reset_index(drop=True)
        if predefined_validation_df is not None:
            predefined_validation_df = combined_with_ids.iloc[
                train_len : train_len + reserved_validation_len
            ].copy().reset_index(drop=True)
        test_df = combined_with_ids.iloc[train_len + reserved_validation_len:].copy().reset_index(drop=True)
    else:
        num_nodes = fixed_graph_context.num_nodes
        node_maps = fixed_graph_context.node_maps
        train_df = add_node_ids_from_maps(train_df, node_maps).reset_index(drop=True)
        test_df = add_node_ids_from_maps(test_df, node_maps).reset_index(drop=True)
        if predefined_validation_df is not None:
            predefined_validation_df = add_node_ids_from_maps(
                predefined_validation_df, node_maps
            ).reset_index(drop=True)
    pu_unlabeled_df = train_df[train_df[SAMPLE_ROLE_COLUMN] == ROLE_PU_UNLABELED].copy()
    supervised_train_df = train_df[train_df[SAMPLE_ROLE_COLUMN] != ROLE_PU_UNLABELED].copy()
    balanced_integrity_stats: dict[str, object] = {}
    if evaluation_protocol == EVALUATION_PROTOCOL_BALANCED_WARM_START:
        if not pu_unlabeled_df.empty:
            raise ValueError("Balanced warm-start 的 PU 样本必须在 train-core 切分后生成。")
        if fixed_epoch_training:
            train_core_df = supervised_train_df.copy().reset_index(drop=True)
            validation_df = supervised_train_df.iloc[0:0].copy().reset_index(drop=True)
        else:
            train_core_df, validation_df = make_stratified_validation_split(
                train_df=supervised_train_df,
                validation_ratio=validation_ratio,
                seed=seed + 50_000,
            )
        (
            train_core_df,
            validation_df,
            test_df,
            generated_pu_df,
            balanced_integrity_stats,
        ) = rescore_balanced_train_core(
            train_core_df,
            validation_df,
            test_df,
            tables=raw_tables or {},
            seed=seed + 70_000,
            pu_learning=pu_learning,
            pu_unlabeled_ratio=pu_unlabeled_ratio,
        )
        if not generated_pu_df.empty:
            generated_pu_df = add_node_ids_from_maps(generated_pu_df, node_maps)
            pu_unlabeled_df = generated_pu_df.copy()
        train_core_df.to_csv(output_dir / f"train_core_supervised_{run_name}.csv", index=False)
        validation_df.to_csv(output_dir / f"validation_samples_{run_name}.csv", index=False)
        pu_unlabeled_df.to_csv(output_dir / f"pu_unlabeled_{run_name}.csv", index=False)
    else:
        if predefined_validation_df is not None:
            train_core_df = supervised_train_df.reset_index(drop=True)
            validation_df = use_reserved_validation(
                train_core_df, pu_unlabeled_df, test_df, predefined_validation_df
            )
        elif (
            model_variant == MODEL_VARIANT_SEMANTIC_PATH_SPECTRAL
            and evaluation_protocol == EVALUATION_PROTOCOL_FOLD_WISE
            and validation_ratio > 0.0
            and raw_tables is not None
        ):
            (
                train_core_df,
                validation_df,
                matched_validation_stats,
            ) = make_matched_random_validation_split(
                supervised_train_df=supervised_train_df,
                pu_unlabeled_df=pu_unlabeled_df,
                test_df=test_df,
                tables=raw_tables,
                validation_ratio=validation_ratio,
                seed=seed + 50_000,
            )
            train_core_df = add_node_ids_from_maps(train_core_df, node_maps)
            validation_df = add_node_ids_from_maps(validation_df, node_maps)
            test_df = add_node_ids_from_maps(test_df, node_maps)
            if not pu_unlabeled_df.empty:
                pu_unlabeled_df = add_node_ids_from_maps(pu_unlabeled_df, node_maps)
            balanced_integrity_stats.update(matched_validation_stats)
        else:
            train_core_df, validation_df = make_validation_split(
                train_df=supervised_train_df,
                validation_ratio=validation_ratio,
                seed=seed + 50_000,
                validation_mode=validation_mode,
            )
    if (
        model_variant == MODEL_VARIANT_SEMANTIC_PATH_SPECTRAL
        and evaluation_protocol == EVALUATION_PROTOCOL_FOLD_WISE
        and raw_tables is not None
    ):
        (
            train_core_df,
            validation_df,
            test_df,
            pu_unlabeled_df,
            fold_pair_risk_stats,
        ) = rescore_fold_pair_risk_features(
            train_core_df=train_core_df,
            validation_df=validation_df,
            test_df=test_df,
            pu_unlabeled_df=pu_unlabeled_df,
            tables=raw_tables,
            seed=seed + 80_000,
        )
        balanced_integrity_stats.update(fold_pair_risk_stats)
    if not pu_unlabeled_df.empty:
        train_core_df = pd.concat([train_core_df, pu_unlabeled_df], ignore_index=True)
    validation_enabled = not validation_df.empty
    if fixed_graph_context is None:
        view_edge_indices = build_view_edge_indices(
            train_core_df,
            num_nodes=num_nodes,
            device=device,
            raw_tables=raw_tables,
            node_maps=node_maps,
            similarity_top_k=similarity_top_k,
            drug_similarity_top_k=drug_similarity_top_k,
            disease_similarity_top_k=disease_similarity_top_k,
            min_similarity=min_similarity,
        )
        node_features, node_feature_stats = build_node_feature_matrix(
            raw_tables=raw_tables,
            node_maps=node_maps,
            num_nodes=num_nodes,
            device=device,
        )
        drug_spectrum: torch.Tensor | None = None
        disease_spectrum: torch.Tensor | None = None
        disease_node_offset = min(node_maps["disease"].values()) if node_maps["disease"] else 0
        ss_feature_stats: dict[str, int | bool] = {
            "ss_encoder_enabled": False,
            "ss_drug_dim": 0,
            "ss_disease_dim": 0,
            "ss_input_dim": 0,
            "ss_output_dim": 0,
            "ss_channels": 0,
        }
        if active_ss_encoder:
            drug_spectrum, disease_spectrum, ss_feature_stats = build_similarity_spectrum_tensors(
                raw_tables=raw_tables or {},
                node_maps=node_maps,
                device=device,
            )
            ss_feature_stats["ss_output_dim"] = int(ss_dim)
            ss_feature_stats["ss_channels"] = int(ss_channels)
    else:
        view_edge_indices = fixed_graph_context.view_edge_indices
        node_features = fixed_graph_context.node_features
        drug_spectrum = fixed_graph_context.drug_spectrum
        disease_spectrum = fixed_graph_context.disease_spectrum
        disease_node_offset = fixed_graph_context.disease_node_offset
        node_feature_stats = fixed_graph_context.node_feature_stats
        ss_feature_stats = fixed_graph_context.ss_feature_stats
    view_edge_counts = [int(edge_index.size(1)) for edge_index in view_edge_indices]
    # Preserve auxiliary inputs while removing inactive graph encoding branches.
    if is_multiview_gatv2_variant:
        view_edge_indices = [view_edge_indices[i] for i in active_indices]
        view_edge_counts = [n if i in active_indices else 0 for i, n in enumerate(view_edge_counts)]
    train_positive_count = int((train_core_df["label"] == 1).sum())
    train_drug_count = max(1, int(train_core_df["drug_id"].astype(str).nunique()))
    train_disease_count = max(1, int(train_core_df["disease_id"].astype(str).nunique()))
    estimated_class_prior = train_positive_count / max(1, train_drug_count * train_disease_count)
    active_pu_class_prior = (
        float(np.clip(estimated_class_prior, 1e-4, 0.5))
        if pu_class_prior is None
        else float(np.clip(pu_class_prior, 1e-4, 0.5))
    )

    if is_multiview_gatv2_variant:
        model: torch.nn.Module = MultiViewDrugDiseaseModel(
            num_nodes=num_nodes,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=0.2,
            num_views=len(active_views),
            feature_dim=0 if node_features is None else int(node_features.size(1)),
            fusion_type=fusion_type,
            fusion_dim=fusion_dim,
            fusion_heads=fusion_heads,
            fusion_layers=fusion_layers,
            reliability_feature_dim=len(RISK_FEATURE_COLUMNS),
            use_ss_encoder=active_ss_encoder,
            ss_drug_dim=int(ss_feature_stats["ss_drug_dim"]),
            ss_disease_dim=int(ss_feature_stats["ss_disease_dim"]),
            ss_dim=ss_dim,
            ss_channels=ss_channels,
            graph_encoder=graph_encoder,
        ).to(device)
        active_fusion_type = fusion_type
        encoder_name = model.encoders[0].__class__.__name__
        fusion_band_names = [f"view_{index}" for index in active_indices]
        reported_active_views = list(active_views)
    else:
        model = SemanticPathSpectralDDAModel(
            num_nodes=num_nodes,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=0.2,
            num_views=3,
            spectral_hops=spectral_hops,
            spectral_pair_dim=spectral_pair_dim,
            gat_heads=4,
            feature_dim=0 if node_features is None else int(node_features.size(1)),
            reliability_feature_dim=len(RISK_FEATURE_COLUMNS),
        ).to(device)
        active_fusion_type = "reliability_guided_semantic_band_gate"
        encoder_name = "SharedGATv2MultiScaleSemanticGraphEncoder"
        fusion_band_names = list(model.band_names)
        reported_active_views = list(active_views)
    total_parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable_parameter_count = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    train_losses: list[float] = []
    epoch_logs: list[dict[str, float | int | None | str]] = []
    train_indices = np.arange(len(train_core_df))
    selection_df = validation_df if validation_enabled else test_df
    selection_split_name = (
        "fixed_epoch_no_selection"
        if fixed_epoch_training
        else ("validation" if validation_enabled else "test")
    )
    selection_metric = validation_metric if validation_metric in {"AUC", "AUPR"} else "AUPR"
    best_selection_score = -np.inf
    best_selection_epoch = 0
    best_selection_metrics: dict[str, float | None] = {}
    best_model_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, epochs + 1):
        rng = np.random.default_rng(seed + epoch)
        rng.shuffle(train_indices)
        batch_losses = []
        batch_bce_losses = []
        batch_contrastive_losses = []
        batch_pair_contrastive_losses = []
        batch_node_contrastive_losses = []
        batch_pu_losses = []
        epoch_pu_unlabeled_count = 0
        epoch_hard_reliable_count = 0
        epoch_uncertain_count = 0
        epoch_potential_positive_count = 0
        model.train()
        for start in range(0, len(train_indices), batch_size):
            batch_indices = train_indices[start : start + batch_size]
            drug_ids, disease_ids, labels, weights, reliability_features, role_codes = tensor_batch(
                train_core_df,
                batch_indices,
                device,
            )
            optimizer.zero_grad()
            spectrum = similarity_spectrum_batch(
                drug_ids,
                disease_ids,
                drug_spectrum,
                disease_spectrum,
                disease_node_offset,
            )
            if effective_node_contrastive_weight > 0.0:
                predictions, _, view_pairs, view_nodes = model(
                    drug_ids,
                    disease_ids,
                    view_edge_indices,
                    node_features=node_features,
                    reliability_features=reliability_features,
                    similarity_spectrum=spectrum,
                    return_view_nodes=True,
                )
                contrastive_node_ids = torch.unique(torch.cat([drug_ids, disease_ids], dim=0))
                node_contrastive = node_contrastive_loss(
                    [view_nodes[index][contrastive_node_ids] for index in range(len(view_nodes))],
                    temperature=contrastive_temperature,
                )
            else:
                predictions, _, view_pairs = model(
                    drug_ids,
                    disease_ids,
                    view_edge_indices,
                    node_features=node_features,
                    reliability_features=reliability_features,
                    similarity_spectrum=spectrum,
                )
                node_contrastive = predictions.sum() * 0.0
            pu_masks = classify_pu_hard_samples(
                predictions=predictions,
                labels=labels,
                role_codes=role_codes,
                reliability_features=reliability_features,
                low_threshold=pu_low_threshold,
                high_threshold=pu_high_threshold,
                potential_risk_threshold=pu_potential_risk_threshold,
                allow_reliable_negative_potential=pu_update_reliable_negatives,
            )
            dynamic_weights = weights.clone()
            dynamic_weights = torch.where(
                pu_masks["hard_reliable_negative"],
                dynamic_weights * float(pu_hard_weight),
                dynamic_weights,
            )
            supervised_mask = pu_masks["supervised"]
            if bool(supervised_mask.any()):
                bce = weighted_bce_loss(
                    predictions[supervised_mask],
                    labels[supervised_mask],
                    dynamic_weights[supervised_mask],
                )
            else:
                bce = predictions.sum() * 0.0
            pu_loss = (
                non_negative_pu_loss(
                    predictions=predictions,
                    labels=labels,
                    role_codes=role_codes,
                    reliability_features=reliability_features,
                    class_prior=active_pu_class_prior,
                    high_threshold=pu_high_threshold,
                    potential_risk_threshold=pu_potential_risk_threshold,
                    uncertain_weight=pu_uncertain_weight,
                    hard_weight=pu_hard_weight,
                )
                if pu_learning
                else predictions.sum() * 0.0
            )
            pair_contrastive = (
                pair_contrastive_loss(view_pairs, temperature=contrastive_temperature)
                if effective_pair_contrastive_weight > 0.0
                else predictions.sum() * 0.0
            )
            contrastive = (
                pair_contrastive
                + effective_node_contrastive_weight * node_contrastive
            )
            loss = (
                bce
                + float(pu_loss_weight) * pu_loss
                + effective_pair_contrastive_weight * pair_contrastive
                + effective_node_contrastive_weight * node_contrastive
            )
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
            batch_bce_losses.append(float(bce.detach().cpu()))
            batch_pu_losses.append(float(pu_loss.detach().cpu()))
            batch_contrastive_losses.append(float(contrastive.detach().cpu()))
            batch_pair_contrastive_losses.append(float(pair_contrastive.detach().cpu()))
            batch_node_contrastive_losses.append(float(node_contrastive.detach().cpu()))
            epoch_pu_unlabeled_count += int(pu_masks["pu_unlabeled"].sum().detach().cpu())
            epoch_hard_reliable_count += int(pu_masks["hard_reliable_negative"].sum().detach().cpu())
            epoch_uncertain_count += int(pu_masks["uncertain_unlabeled"].sum().detach().cpu())
            epoch_potential_positive_count += int(pu_masks["potential_positive"].sum().detach().cpu())
        train_losses.append(float(np.mean(batch_losses)))

        if fixed_epoch_training:
            metrics = {
                "AUC": None,
                "AUPR": None,
                "F1": None,
                "Precision": None,
                "Recall": None,
                "pearson_r": None,
            }
            band_count = len(fusion_band_names)
            view_weights = torch.full((1, band_count), 1.0 / float(band_count))
        else:
            metrics, _, view_weights = evaluate_model(
                model,
                selection_df,
                device,
                view_edge_indices,
                node_features=node_features,
                drug_spectrum=drug_spectrum,
                disease_spectrum=disease_spectrum,
                disease_node_offset=disease_node_offset,
            )
            selection_score = metrics.get(selection_metric)
            numeric_selection_score = -np.inf if selection_score is None else float(selection_score)
            if validation_enabled and (
                best_model_state is None or numeric_selection_score > best_selection_score
            ):
                best_selection_score = numeric_selection_score
                best_selection_epoch = epoch
                best_selection_metrics = metrics.copy()
                best_model_state = clone_model_state(model)
        mean_view_weights = view_weights.mean(dim=0).cpu().numpy().round(6).tolist()
        row: dict[str, float | int | None | str] = {
            "split": run_name,
            "epoch": epoch,
            "train_loss": float(np.mean(batch_losses)),
            "bce_loss": float(np.mean(batch_bce_losses)),
            "pu_loss": float(np.mean(batch_pu_losses)),
            "pu_loss_weight": float(pu_loss_weight if pu_learning else 0.0),
            "pu_learning_enabled": bool(pu_learning),
            "pu_class_prior": float(active_pu_class_prior),
            "pu_low_threshold": float(pu_low_threshold),
            "pu_high_threshold": float(pu_high_threshold),
            "pu_potential_risk_threshold": float(pu_potential_risk_threshold),
            "pu_uncertain_weight": float(pu_uncertain_weight),
            "pu_hard_weight": float(pu_hard_weight),
            "pu_update_reliable_negatives": bool(pu_update_reliable_negatives),
            "evaluation_protocol": evaluation_protocol,
            "pu_unlabeled_seen": int(epoch_pu_unlabeled_count),
            "pu_hard_reliable_seen": int(epoch_hard_reliable_count),
            "pu_uncertain_seen": int(epoch_uncertain_count),
            "pu_potential_positive_seen": int(epoch_potential_positive_count),
            "contrastive_loss": float(np.mean(batch_contrastive_losses)),
            "pair_contrastive_loss": float(np.mean(batch_pair_contrastive_losses)),
            "node_contrastive_loss": float(np.mean(batch_node_contrastive_losses)),
            "pair_contrastive_weight": float(effective_pair_contrastive_weight),
            "node_contrastive_weight": float(effective_node_contrastive_weight),
            "contrastive_temperature": float(contrastive_temperature),
            "train_rows": int(len(train_core_df)),
            "outer_train_rows": int(len(train_df)),
            "train_core_rows": int(len(train_core_df)),
            "validation_rows": int(len(validation_df)),
            "test_rows": int(len(test_df)),
            "evaluation_split": selection_split_name,
            "selection_metric": selection_metric,
            "device": str(device),
            "model_variant": model_variant,
            "graph_encoder": graph_encoder,
            "encoder": encoder_name,
            "fusion_type": active_fusion_type,
            "fusion_dim": int(fusion_dim),
            "fusion_heads": int(fusion_heads),
            "fusion_layers": int(fusion_layers),
            "spectral_hops": int(spectral_hops if model_variant == MODEL_VARIANT_SEMANTIC_PATH_SPECTRAL else 0),
            "spectral_pair_dim": int(
                spectral_pair_dim if model_variant == MODEL_VARIANT_SEMANTIC_PATH_SPECTRAL else 0
            ),
            "semantic_band_count": int(len(fusion_band_names)),
            "total_parameter_count": total_parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "gpu_peak_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024**2))
                if device.type == "cuda"
                else 0.0
            ),
            "edge_count": view_edge_counts[0],
            "association_edge_count": view_edge_counts[0],
            "similarity_edge_count": view_edge_counts[1],
            "biology_edge_count": view_edge_counts[2],
            "similarity_top_k": -1 if similarity_top_k is None else int(similarity_top_k),
            "drug_similarity_top_k": (
                -1
                if (similarity_top_k if drug_similarity_top_k is None else drug_similarity_top_k) is None
                else int(similarity_top_k if drug_similarity_top_k is None else drug_similarity_top_k)
            ),
            "disease_similarity_top_k": (
                -1
                if (similarity_top_k if disease_similarity_top_k is None else disease_similarity_top_k) is None
                else int(similarity_top_k if disease_similarity_top_k is None else disease_similarity_top_k)
            ),
            "min_similarity": float(min_similarity),
            "similarity_view_enabled": view_edge_counts[1] > 0,
            "active_views": "+".join(reported_active_views),
            "active_view_count": len(reported_active_views),
        }
        row.update(node_feature_stats)
        row.update(ss_feature_stats)
        row.update(balanced_integrity_stats)
        row.update(metrics)
        if validation_enabled:
            for metric_name, metric_value in metrics.items():
                row[f"validation_{metric_name}"] = metric_value
        for idx, weight in enumerate(mean_view_weights):
            canonical_index = active_indices[idx] if is_multiview_gatv2_variant else idx
            row[f"view_weight_{canonical_index}"] = float(weight)
            row[f"band_weight_{fusion_band_names[idx]}"] = float(weight)
        if is_multiview_gatv2_variant:
            for idx in set(range(3)) - set(active_indices):
                row[f"view_weight_{idx}"] = 0.0
        epoch_logs.append(row)
        pd.DataFrame(epoch_logs).to_csv(output_dir / f"training_log_{run_name}.csv", index=False)

    if validation_enabled and best_model_state is not None:
        model.load_state_dict(best_model_state)
    final_evaluation_df = validation_df if skip_test_evaluation else test_df
    final_evaluation_split = "validation" if skip_test_evaluation else "test"
    mlp_metrics, mlp_y_score, view_weights = evaluate_model(
        model,
        final_evaluation_df,
        device,
        view_edge_indices,
        node_features=node_features,
        drug_spectrum=drug_spectrum,
        disease_spectrum=disease_spectrum,
        disease_node_offset=disease_node_offset,
    )
    decoder_metadata: dict[str, object] = {
        "decoder_type": decoder_type,
        "encoder_training_head": "MLP auxiliary head" if decoder_type == "lightgbm" else "MLP",
        "lightgbm_feature_dim": 0,
        "lightgbm_train_rows": 0,
        "lightgbm_train_positive_count": 0,
        "lightgbm_train_negative_count": 0,
        "lightgbm_parameters": {},
    }
    if decoder_type == "lightgbm":
        # PU 未标注样本不作为 LightGBM 的确定负标签，避免引入额外假负样本噪声。
        lightgbm_train_df = train_core_df[
            train_core_df[SAMPLE_ROLE_COLUMN] != ROLE_PU_UNLABELED
        ].copy()
        lightgbm_features = extract_pair_features(
            model=model,
            examples=lightgbm_train_df,
            device=device,
            view_edge_indices=view_edge_indices,
            batch_size=batch_size,
            node_features=node_features,
            drug_spectrum=drug_spectrum,
            disease_spectrum=disease_spectrum,
            disease_node_offset=disease_node_offset,
        )
        lightgbm_decoder = fit_lightgbm_decoder(
            train_features=lightgbm_features,
            labels=lightgbm_train_df["label"].to_numpy(dtype=int),
            sample_weights=lightgbm_train_df["reliability"].to_numpy(dtype=np.float32),
            seed=seed,
        )
        metrics, y_score = evaluate_lightgbm_decoder(
            decoder=lightgbm_decoder,
            model=model,
            examples=final_evaluation_df,
            device=device,
            view_edge_indices=view_edge_indices,
            batch_size=batch_size,
            node_features=node_features,
            drug_spectrum=drug_spectrum,
            disease_spectrum=disease_spectrum,
            disease_node_offset=disease_node_offset,
        )
        decoder_metadata.update(
            {
                "lightgbm_feature_dim": int(lightgbm_features.shape[1]),
                "lightgbm_train_rows": int(len(lightgbm_train_df)),
                "lightgbm_train_positive_count": int((lightgbm_train_df["label"] == 1).sum()),
                "lightgbm_train_negative_count": int((lightgbm_train_df["label"] == 0).sum()),
                "lightgbm_parameters": lightgbm_decoder_parameters(seed=seed),
            }
        )
        with (output_dir / f"lightgbm_decoder_{run_name}.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    **decoder_metadata,
                    "metrics": metrics,
                    "pretraining_mlp_metrics": mlp_metrics,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    else:
        metrics, y_score = mlp_metrics, mlp_y_score
    y_true = final_evaluation_df["label"].to_numpy(dtype=int)
    best_auc_row = max(
        epoch_logs,
        key=lambda item: -1.0 if item["AUC"] is None else float(item["AUC"]),
    )
    best_aupr_row = max(
        epoch_logs,
        key=lambda item: -1.0 if item["AUPR"] is None else float(item["AUPR"]),
    )
    result = {
        "split": run_name,
        "epochs": epochs,
        "train_rows": int(len(train_core_df)),
        "outer_train_rows": int(len(train_df)),
        "train_core_rows": int(len(train_core_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "validation_enabled": bool(validation_enabled),
        "validation_ratio": float(validation_ratio),
        "validation_mode": validation_mode,
        "selection_metric": selection_metric,
        "selected_epoch": int(best_selection_epoch if validation_enabled else epochs),
        "best_metric_source": selection_split_name,
        "fixed_epoch_training": bool(fixed_epoch_training),
        "final_evaluation_split": final_evaluation_split,
        "test_evaluation_count": int(not skip_test_evaluation),
        "pu_learning_enabled": bool(pu_learning),
        "pu_loss_weight": float(pu_loss_weight if pu_learning else 0.0),
        "pu_class_prior": float(active_pu_class_prior),
        "pu_low_threshold": float(pu_low_threshold),
        "pu_high_threshold": float(pu_high_threshold),
        "pu_potential_risk_threshold": float(pu_potential_risk_threshold),
        "pu_uncertain_weight": float(pu_uncertain_weight),
        "pu_hard_weight": float(pu_hard_weight),
        "pu_update_reliable_negatives": bool(pu_update_reliable_negatives),
        "evaluation_protocol": evaluation_protocol,
        "model_variant": model_variant,
        "graph_encoder": graph_encoder,
        "pair_contrastive_weight": float(effective_pair_contrastive_weight),
        "node_contrastive_weight": float(effective_node_contrastive_weight),
        "contrastive_temperature": float(contrastive_temperature),
        "num_nodes": int(num_nodes),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "encoder": encoder_name,
        "spectral_hops": int(spectral_hops if model_variant == MODEL_VARIANT_SEMANTIC_PATH_SPECTRAL else 0),
        "spectral_pair_dim": int(
            spectral_pair_dim if model_variant == MODEL_VARIANT_SEMANTIC_PATH_SPECTRAL else 0
        ),
        "semantic_band_names": fusion_band_names,
        "total_parameter_count": total_parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "gpu_peak_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if device.type == "cuda"
            else 0.0
        ),
        **decoder_metadata,
        "fusion_type": active_fusion_type,
        "fusion_dim": int(fusion_dim),
        "fusion_heads": int(fusion_heads),
        "fusion_layers": int(fusion_layers),
        "edge_count": view_edge_counts[0],
        "association_edge_count": view_edge_counts[0],
        "similarity_edge_count": view_edge_counts[1],
        "biology_edge_count": view_edge_counts[2],
        "view_edge_counts": view_edge_counts,
        "similarity_top_k": None if similarity_top_k is None else int(similarity_top_k),
        "drug_similarity_top_k": (
            None
            if (similarity_top_k if drug_similarity_top_k is None else drug_similarity_top_k) is None
            else int(similarity_top_k if drug_similarity_top_k is None else drug_similarity_top_k)
        ),
        "disease_similarity_top_k": (
            None
            if (similarity_top_k if disease_similarity_top_k is None else disease_similarity_top_k) is None
            else int(similarity_top_k if disease_similarity_top_k is None else disease_similarity_top_k)
        ),
        "min_similarity": float(min_similarity),
        "similarity_view_enabled": view_edge_counts[1] > 0,
        "active_views": reported_active_views,
        "active_view_count": len(reported_active_views),
        "node_feature_stats": node_feature_stats,
        "ss_feature_stats": ss_feature_stats,
        "last_train_loss": train_losses[-1] if train_losses else None,
        "metrics": metrics,
        "best_auc": best_auc_row["AUC"],
        "best_auc_epoch": best_auc_row["epoch"],
        "best_aupr": best_aupr_row["AUPR"],
        "best_aupr_epoch": best_aupr_row["epoch"],
        "validation_best_metrics": best_selection_metrics if validation_enabled else {},
        "validation_best_auc": best_selection_metrics.get("AUC") if validation_enabled else None,
        "validation_best_aupr": best_selection_metrics.get("AUPR") if validation_enabled else None,
        "mean_view_weights": view_weights.mean(dim=0).cpu().numpy().round(4).tolist(),
        "mean_band_weights": {
            band_name: float(weight)
            for band_name, weight in zip(
                fusion_band_names,
                view_weights.mean(dim=0).cpu().numpy().tolist(),
            )
        },
    }
    result.update(balanced_integrity_stats)
    if is_multiview_gatv2_variant:
        weights = result["mean_view_weights"]
        result["mean_view_weights"] = [
            weights[active_indices.index(i)] if i in active_indices else 0.0
            for i in range(3)
        ]

    pd.DataFrame(
        {
            "drug_id": final_evaluation_df["drug_id"].to_numpy(),
            "disease_id": final_evaluation_df["disease_id"].to_numpy(),
            "label": y_true,
            "score": y_score,
            "decoder_type": decoder_type,
        }
    ).to_csv(
        output_dir
        / (
            f"validation_predictions_{run_name}.csv"
            if skip_test_evaluation
            else f"predictions_{run_name}.csv"
        ),
        index=False,
    )
    if representation_export_path is not None:
        exported_features = extract_fused_pair_features(
            model=model,
            examples=final_evaluation_df,
            device=device,
            view_edge_indices=view_edge_indices,
            batch_size=batch_size,
            node_features=node_features,
            drug_spectrum=drug_spectrum,
            disease_spectrum=disease_spectrum,
            disease_node_offset=disease_node_offset,
        )
        write_representation_export(
            representation_export_path,
            exported_features,
            final_evaluation_df,
        )
        result["representation_export_path"] = str(representation_export_path)
        result["representation_feature_dim"] = int(exported_features.shape[1])
    return result


def train_one_split(
    examples: pd.DataFrame,
    split: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    embedding_dim: int,
    hidden_dim: int,
    lr: float,
    contrastive_weight: float,
    test_ratio: float,
    seed: int,
    device_name: str = "auto",
    raw_tables: dict[str, pd.DataFrame] | None = None,
    similarity_top_k: int | None = 20,
    drug_similarity_top_k: int | None = None,
    disease_similarity_top_k: int | None = None,
    min_similarity: float = 0.0,
    fusion_type: str = "transformer",
    fusion_dim: int = 256,
    fusion_heads: int = 4,
    fusion_layers: int = 1,
    use_ss_encoder: bool = False,
    ss_dim: int = 128,
    ss_channels: int = 16,
    node_contrastive_weight: float = 0.0,
    contrastive_temperature: float = 0.2,
    decoder_type: str = "mlp",
    validation_ratio: float = 0.1,
    validation_metric: str = "AUPR",
    model_variant: str = MODEL_VARIANT_CURRENT,
    spectral_hops: int = 3,
    spectral_pair_dim: int = 256,
    graph_encoder: str = "gatv2",
) -> dict[str, object]:
    """Docstring."""

    examples_with_ids, _ = add_node_ids(examples)
    train_df, test_df = make_split(examples_with_ids, split, test_ratio, seed)
    return train_predefined_split(
        train_df=train_df,
        test_df=test_df,
        run_name=split,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        contrastive_weight=contrastive_weight,
        seed=seed,
        device=resolve_device(device_name),
        raw_tables=raw_tables,
        similarity_top_k=similarity_top_k,
        drug_similarity_top_k=drug_similarity_top_k,
        disease_similarity_top_k=disease_similarity_top_k,
        min_similarity=min_similarity,
        fusion_type=fusion_type,
        fusion_dim=fusion_dim,
        fusion_heads=fusion_heads,
        fusion_layers=fusion_layers,
        use_ss_encoder=use_ss_encoder,
        ss_dim=ss_dim,
        ss_channels=ss_channels,
        node_contrastive_weight=node_contrastive_weight,
        contrastive_temperature=contrastive_temperature,
        decoder_type=decoder_type,
        validation_ratio=validation_ratio,
        validation_metric=validation_metric,
        validation_mode=split if split in set(SPLIT_MODES) else "random",
        model_variant=model_variant,
        spectral_hops=spectral_hops,
        spectral_pair_dim=spectral_pair_dim,
        graph_encoder=graph_encoder,
    )


def train_kfold(
    examples: pd.DataFrame,
    folds: int,
    cv_mode: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    embedding_dim: int,
    hidden_dim: int,
    lr: float,
    contrastive_weight: float,
    seed: int,
    device_name: str,
    raw_tables: dict[str, pd.DataFrame] | None = None,
    similarity_top_k: int | None = 20,
    drug_similarity_top_k: int | None = None,
    disease_similarity_top_k: int | None = None,
    min_similarity: float = 0.0,
    fusion_type: str = "transformer",
    fusion_dim: int = 256,
    fusion_heads: int = 4,
    fusion_layers: int = 1,
    use_ss_encoder: bool = False,
    ss_dim: int = 128,
    ss_channels: int = 16,
    node_contrastive_weight: float = 0.0,
    contrastive_temperature: float = 0.2,
    decoder_type: str = "mlp",
    validation_ratio: float = 0.0,
    validation_metric: str = "AUPR",
    pu_learning: bool = False,
    pu_only: bool = False,
    pu_unlabeled_ratio: float = 1.0,
    pu_loss_weight: float = 0.1,
    pu_class_prior: float | None = None,
    pu_low_threshold: float = 0.3,
    pu_high_threshold: float = 0.7,
    pu_potential_risk_threshold: float = 0.8,
    pu_uncertain_weight: float = 0.2,
    pu_hard_weight: float = 1.5,
    model_variant: str = MODEL_VARIANT_CURRENT,
    spectral_hops: int = 3,
    spectral_pair_dim: int = 256,
    graph_encoder: str = "gatv2",
) -> list[dict[str, object]]:
    """Docstring."""

    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    results = []
    logs = []
    for fold_id, train_df, test_df in make_kfold_splits(examples, folds=folds, seed=seed, mode=cv_mode):
        run_name = f"fold_{fold_id}"
        result = train_predefined_split(
            train_df=train_df,
            test_df=test_df,
            run_name=run_name,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            contrastive_weight=contrastive_weight,
            seed=seed + fold_id,
            device=device,
            raw_tables=raw_tables,
            similarity_top_k=similarity_top_k,
            drug_similarity_top_k=drug_similarity_top_k,
            disease_similarity_top_k=disease_similarity_top_k,
            min_similarity=min_similarity,
            fusion_type=fusion_type,
            fusion_dim=fusion_dim,
            fusion_heads=fusion_heads,
            fusion_layers=fusion_layers,
            use_ss_encoder=use_ss_encoder,
            ss_dim=ss_dim,
            ss_channels=ss_channels,
            node_contrastive_weight=node_contrastive_weight,
            contrastive_temperature=contrastive_temperature,
            decoder_type=decoder_type,
            validation_ratio=validation_ratio,
            validation_metric=validation_metric,
            validation_mode=cv_mode,
            model_variant=model_variant,
            spectral_hops=spectral_hops,
            spectral_pair_dim=spectral_pair_dim,
            graph_encoder=graph_encoder,
        )
        result["fold"] = fold_id
        result["cv_mode"] = cv_mode
        results.append(result)
        fold_log = pd.read_csv(output_dir / f"training_log_{run_name}.csv")
        fold_log.insert(0, "fold", fold_id)
        fold_log.insert(1, "cv_mode", cv_mode)
        logs.append(fold_log)

    all_logs = pd.concat(logs, ignore_index=True)
    all_logs.to_csv(output_dir / "training_log_5fold.csv", index=False)

    summary_rows = []
    for result in results:
        row = {
            "fold": result["fold"],
            "cv_mode": result["cv_mode"],
            "final_auc": result["metrics"]["AUC"],
            "final_aupr": result["metrics"]["AUPR"],
            "final_f1": result["metrics"]["F1"],
            "final_recall": result["metrics"]["Recall"],
            "final_pearson_r": result["metrics"]["pearson_r"],
            "best_auc": result["best_auc"],
            "best_auc_epoch": result["best_auc_epoch"],
            "best_aupr": result["best_aupr"],
            "best_aupr_epoch": result["best_aupr_epoch"],
            "view_weight_0": result["mean_view_weights"][0],
            "view_weight_1": result["mean_view_weights"][1],
            "view_weight_2": result["mean_view_weights"][2],
            "association_edge_count": result["association_edge_count"],
            "similarity_edge_count": result["similarity_edge_count"],
            "biology_edge_count": result["biology_edge_count"],
            "similarity_top_k": result["similarity_top_k"],
            "drug_similarity_top_k": result["drug_similarity_top_k"],
            "disease_similarity_top_k": result["disease_similarity_top_k"],
            "min_similarity": result["min_similarity"],
            "similarity_view_enabled": result["similarity_view_enabled"],
            "decoder_type": result["decoder_type"],
            "encoder_training_head": result["encoder_training_head"],
            "lightgbm_feature_dim": result["lightgbm_feature_dim"],
            "lightgbm_train_rows": result["lightgbm_train_rows"],
            "lightgbm_train_positive_count": result["lightgbm_train_positive_count"],
            "lightgbm_train_negative_count": result["lightgbm_train_negative_count"],
            "fusion_type": result["fusion_type"],
            "fusion_dim": result["fusion_dim"],
            "fusion_heads": result["fusion_heads"],
            "fusion_layers": result["fusion_layers"],
            "validation_enabled": result["validation_enabled"],
            "validation_ratio": result["validation_ratio"],
            "validation_mode": result["validation_mode"],
            "validation_rows": result["validation_rows"],
            "train_core_rows": result["train_core_rows"],
            "selection_metric": result["selection_metric"],
            "selected_epoch": result["selected_epoch"],
            "best_metric_source": result["best_metric_source"],
            "validation_best_auc": result["validation_best_auc"],
            "validation_best_aupr": result["validation_best_aupr"],
            "device": result["device"],
            "gpu_name": result["gpu_name"],
            "model_variant": result["model_variant"],
            "total_parameter_count": result["total_parameter_count"],
            "trainable_parameter_count": result["trainable_parameter_count"],
            "gpu_peak_memory_mb": result["gpu_peak_memory_mb"],
        }
        row.update(
            {
                f"band_weight_{name}": weight
                for name, weight in result["mean_band_weights"].items()
            }
        )
        row.update(result["node_feature_stats"])
        row.update(result["ss_feature_stats"])
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    numeric_cols = [
        "final_auc",
        "final_aupr",
        "final_f1",
        "final_recall",
        "final_pearson_r",
        "lightgbm_feature_dim",
        "lightgbm_train_rows",
        "lightgbm_train_positive_count",
        "lightgbm_train_negative_count",
        "best_auc",
        "best_aupr",
        "selected_epoch",
        "validation_rows",
        "train_core_rows",
        "validation_best_auc",
        "validation_best_aupr",
        "ss_drug_dim",
        "ss_disease_dim",
        "ss_input_dim",
        "ss_output_dim",
        "ss_channels",
        "view_weight_0",
        "view_weight_1",
        "view_weight_2",
        "total_parameter_count",
        "trainable_parameter_count",
        "gpu_peak_memory_mb",
    ]
    numeric_cols.extend(
        sorted(column for column in summary.columns if column.startswith("band_weight_"))
    )
    mean_row = {
        "fold": "mean",
        "cv_mode": cv_mode,
        "device": str(device),
        "gpu_name": results[0]["gpu_name"],
    }
    std_row = {
        "fold": "std",
        "cv_mode": cv_mode,
        "device": str(device),
        "gpu_name": results[0]["gpu_name"],
    }
    for col in numeric_cols:
        mean_row[col] = float(summary[col].mean())
        std_row[col] = float(summary[col].std(ddof=1))
    summary = pd.concat([summary, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    summary.to_csv(output_dir / "summary_metrics_5fold.csv", index=False)
    return results


def train_kfold_global_protocol(
    tables: dict[str, pd.DataFrame],
    processed_dir: Path,
    folds: int,
    cv_mode: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    embedding_dim: int,
    hidden_dim: int,
    lr: float,
    contrastive_weight: float,
    node_contrastive_weight: float,
    negative_threshold: float,
    seed: int,
    device_name: str,
    contrastive_temperature: float = 0.2,
    similarity_top_k: int | None = 20,
    drug_similarity_top_k: int | None = None,
    disease_similarity_top_k: int | None = None,
    min_similarity: float = 0.0,
    fusion_type: str = "transformer",
    fusion_dim: int = 256,
    fusion_heads: int = 4,
    fusion_layers: int = 1,
    use_ss_encoder: bool = False,
    ss_dim: int = 128,
    ss_channels: int = 16,
    decoder_type: str = "mlp",
    validation_ratio: float = 0.0,
    validation_metric: str = "AUPR",
    pu_learning: bool = False,
    pu_unlabeled_ratio: float = 1.0,
    pu_loss_weight: float = 0.1,
    pu_class_prior: float | None = None,
    pu_low_threshold: float = 0.3,
    pu_high_threshold: float = 0.7,
    pu_potential_risk_threshold: float = 0.8,
    pu_uncertain_weight: float = 0.2,
    pu_hard_weight: float = 1.5,
    rns_strategy: str = RNS_STRATEGY_ADAPTIVE_TOPK,
    negative_ratio: float = DEFAULT_NEGATIVE_RATIO,
    global_gip_weight: float = 0.5,
    model_variant: str = MODEL_VARIANT_CURRENT,
    spectral_hops: int = 3,
    spectral_pair_dim: int = 256,
    graph_encoder: str = "gatv2",
) -> list[dict[str, object]]:
    """运行传统 Global-preprocessing protocol，不改变严格折内训练器。"""

    raise ValueError("global_preprocess is retired; use train_kfold_fold_aware with fold_wise.")

    if cv_mode != "random":
        raise ValueError("Global-preprocessing protocol 当前仅支持 cv_mode=random 的分层五折比较。")
    if rns_strategy != RNS_STRATEGY_ADAPTIVE_TOPK:
        raise ValueError("Global-preprocessing protocol 必须使用 rns_strategy=adaptive_topk。")
    run_started_at = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    global_data = build_global_preprocessed_data(
        tables=tables,
        negative_threshold=negative_threshold,
        rns_strategy=rns_strategy,
        negative_ratio=negative_ratio,
        seed=seed,
        global_gip_weight=global_gip_weight,
    )
    global_data.scored_unknown_pairs.to_csv(processed_dir / "global_scored_unknown_pairs.csv", index=False)
    global_data.reliable_negatives.to_csv(processed_dir / "global_reliable_negatives.csv", index=False)
    global_data.tables["drug_similarity"].to_csv(processed_dir / "global_drug_similarity.csv", index=False)
    global_data.tables["disease_similarity"].to_csv(processed_dir / "global_disease_similarity.csv", index=False)
    graph_context = build_global_graph_views(
        global_data=global_data,
        device=device,
        similarity_top_k=similarity_top_k,
        drug_similarity_top_k=drug_similarity_top_k,
        disease_similarity_top_k=disease_similarity_top_k,
        min_similarity=min_similarity,
        use_ss_encoder=use_ss_encoder,
        ss_dim=ss_dim,
        ss_channels=ss_channels,
    )
    pu_unlabeled = (
        select_global_pu_unlabeled(
            scored_unknown_pairs=global_data.scored_unknown_pairs,
            reliable_negatives=global_data.reliable_negatives,
            positive_count=len(global_data.positives),
            unlabeled_ratio=pu_unlabeled_ratio,
            seed=seed + 30_000,
        )
        if pu_learning
        else _empty_negative_frame().assign(**{SAMPLE_ROLE_COLUMN: ROLE_PU_UNLABELED})
    )
    pu_unlabeled.to_csv(processed_dir / "global_pu_unlabeled.csv", index=False)
    global_metadata = {
        **global_data.metadata,
        "cv_mode": cv_mode,
        "folds": int(folds),
        "global_pu_unlabeled_count": int(len(pu_unlabeled)),
        "global_graph_fingerprint": graph_context.graph_fingerprint,
        "global_graph_build_count": int(graph_context.build_count),
        "global_association_edge_count": int(graph_context.view_edge_indices[0].size(1)),
        "global_similarity_edge_count": int(graph_context.view_edge_indices[1].size(1)),
        "global_biology_edge_count": int(graph_context.view_edge_indices[2].size(1)),
    }
    with (processed_dir / "global_preprocess_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(global_metadata, f, indent=2, ensure_ascii=False)

    fold_splits = create_global_cv_splits(
        positives=global_data.positives,
        reliable_negatives=global_data.reliable_negatives,
        folds=folds,
        seed=seed,
    )
    results: list[dict[str, object]] = []
    logs: list[pd.DataFrame] = []
    for fold_id, supervised_train_df, test_df, fold_stats in fold_splits:
        fold_dir = processed_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_df = pd.concat([supervised_train_df, pu_unlabeled], ignore_index=True)
        train_pairs = _pair_set(train_df)
        test_pairs = _pair_set(test_df)
        if train_pairs & test_pairs:
            raise RuntimeError("Global-preprocessing protocol 的训练样本与测试样本发生 pair 重叠。")
        supervised_train_df.to_csv(fold_dir / "supervised_train_samples.csv", index=False)
        train_df.to_csv(fold_dir / "train_samples.csv", index=False)
        test_df.to_csv(fold_dir / "test_samples.csv", index=False)
        fold_started_at = time.perf_counter()
        result = train_predefined_split(
            train_df=train_df,
            test_df=test_df,
            run_name=f"fold_{fold_id}",
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            contrastive_weight=contrastive_weight,
            seed=seed + fold_id,
            device=device,
            raw_tables=global_data.tables,
            similarity_top_k=similarity_top_k,
            drug_similarity_top_k=drug_similarity_top_k,
            disease_similarity_top_k=disease_similarity_top_k,
            min_similarity=min_similarity,
            fusion_type=fusion_type,
            fusion_dim=fusion_dim,
            fusion_heads=fusion_heads,
            fusion_layers=fusion_layers,
            use_ss_encoder=use_ss_encoder,
            ss_dim=ss_dim,
            ss_channels=ss_channels,
            node_contrastive_weight=node_contrastive_weight,
            contrastive_temperature=contrastive_temperature,
            decoder_type=decoder_type,
            validation_ratio=validation_ratio,
            validation_metric=validation_metric,
            validation_mode="random",
            pu_learning=pu_learning,
            pu_loss_weight=pu_loss_weight,
            pu_class_prior=pu_class_prior,
            pu_low_threshold=pu_low_threshold,
            pu_high_threshold=pu_high_threshold,
            pu_potential_risk_threshold=pu_potential_risk_threshold,
            pu_uncertain_weight=pu_uncertain_weight,
            pu_hard_weight=pu_hard_weight,
            fixed_graph_context=graph_context,
            evaluation_protocol=EVALUATION_PROTOCOL_GLOBAL_PREPROCESS,
            pu_update_reliable_negatives=True,
            model_variant=model_variant,
            spectral_hops=spectral_hops,
            spectral_pair_dim=spectral_pair_dim,
            graph_encoder=graph_encoder,
        )
        fold_train_seconds = float(time.perf_counter() - fold_started_at)
        result.update(
            {
                "fold": fold_id,
                "cv_mode": cv_mode,
                "fold_train_seconds": fold_train_seconds,
                "global_graph_fingerprint": graph_context.graph_fingerprint,
                "global_graph_build_count": int(graph_context.build_count),
                "global_positive_count": int(global_data.metadata["global_positive_count"]),
                "global_unknown_count": int(global_data.metadata["global_unknown_count"]),
                "global_reliable_negative_count": int(global_data.metadata["global_reliable_negative_count"]),
                "global_pu_unlabeled_count": int(len(pu_unlabeled)),
                **fold_stats,
            }
        )
        results.append(result)
        fold_log = pd.read_csv(output_dir / f"training_log_fold_{fold_id}.csv")
        excluded_log_values = {
            "metrics",
            "node_feature_stats",
            "ss_feature_stats",
            "validation_best_metrics",
            "view_edge_counts",
            "mean_view_weights",
            "mean_band_weights",
            "semantic_band_names",
        }
        fold_metadata = {
            "fold": fold_id,
            "cv_mode": cv_mode,
            "evaluation_protocol": EVALUATION_PROTOCOL_GLOBAL_PREPROCESS,
            "protocol_name": GLOBAL_PREPROCESSING_PROTOCOL_NAME,
            "global_graph_fingerprint": graph_context.graph_fingerprint,
            **{
                key: value
                for key, value in result.items()
                if key not in excluded_log_values
            },
        }
        fold_log = fold_log.drop(
            columns=[column for column in fold_metadata if column in fold_log.columns]
        )
        fold_log = pd.concat(
            [
                pd.DataFrame(
                    {key: [value] * len(fold_log) for key, value in fold_metadata.items()}
                ),
                fold_log.reset_index(drop=True),
            ],
            axis=1,
        )
        fold_log.to_csv(output_dir / f"training_log_fold_{fold_id}.csv", index=False)
        logs.append(fold_log)

    all_logs = pd.concat(logs, ignore_index=True)
    summary_rows: list[dict[str, object]] = []
    for result in results:
        metrics = result["metrics"]
        row = {
            "fold": result["fold"],
            "evaluation_protocol": EVALUATION_PROTOCOL_GLOBAL_PREPROCESS,
            "protocol_name": GLOBAL_PREPROCESSING_PROTOCOL_NAME,
            "cv_mode": result["cv_mode"],
            "negative_sampling_scope": result["negative_sampling_scope"],
            "rns_strategy": result["rns_strategy"],
            "negative_ratio": global_data.metadata["negative_ratio"],
            "global_positive_count": result["global_positive_count"],
            "global_unknown_count": result["global_unknown_count"],
            "global_reliable_negative_count": result["global_reliable_negative_count"],
            "global_pu_unlabeled_count": result["global_pu_unlabeled_count"],
            "train_positive_count": result["train_positive_count"],
            "train_core_positive_count": result.get("train_core_positive_count", np.nan),
            "train_negative_count": result["train_negative_count"],
            "pu_unlabeled_count": result["global_pu_unlabeled_count"],
            "test_positive_count": result["test_positive_count"],
            "test_negative_count": result["test_negative_count"],
            "validation_positive_count": result.get("validation_positive_count", np.nan),
            "validation_negative_count": result.get("validation_negative_count", np.nan),
            "train_rows": result["train_rows"],
            "outer_train_rows": result["outer_train_rows"],
            "test_rows": result["test_rows"],
            "final_auc": metrics["AUC"],
            "final_aupr": metrics["AUPR"],
            "final_f1": metrics["F1"],
            "final_precision": metrics["Precision"],
            "final_recall": metrics["Recall"],
            "final_pearson_r": metrics["pearson_r"],
            "fold_train_seconds": result["fold_train_seconds"],
            "association_edge_count": result["association_edge_count"],
            "similarity_edge_count": result["similarity_edge_count"],
            "biology_edge_count": result["biology_edge_count"],
            "global_graph_fingerprint": result["global_graph_fingerprint"],
            "global_graph_build_count": result["global_graph_build_count"],
            "device": result["device"],
            "gpu_name": result["gpu_name"],
            "model_variant": result["model_variant"],
            "total_parameter_count": result["total_parameter_count"],
            "trainable_parameter_count": result["trainable_parameter_count"],
            "gpu_peak_memory_mb": result["gpu_peak_memory_mb"],
        }
        row.update(
            {
                f"band_weight_{name}": weight
                for name, weight in result["mean_band_weights"].items()
            }
        )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    numeric_columns = [column for column in summary.columns if pd.api.types.is_numeric_dtype(summary[column])]
    mean_row = {column: (float(summary[column].mean()) if column in numeric_columns else None) for column in summary.columns}
    std_row = {column: (float(summary[column].std(ddof=1)) if column in numeric_columns else None) for column in summary.columns}
    mean_row.update({
        "fold": "mean",
        "evaluation_protocol": EVALUATION_PROTOCOL_GLOBAL_PREPROCESS,
        "protocol_name": GLOBAL_PREPROCESSING_PROTOCOL_NAME,
        "cv_mode": cv_mode,
        "global_graph_fingerprint": graph_context.graph_fingerprint,
        "global_graph_build_count": int(graph_context.build_count),
        "total_run_seconds": float(time.perf_counter() - run_started_at),
    })
    std_row.update({
        "fold": "std",
        "evaluation_protocol": EVALUATION_PROTOCOL_GLOBAL_PREPROCESS,
        "protocol_name": GLOBAL_PREPROCESSING_PROTOCOL_NAME,
        "cv_mode": cv_mode,
        "global_graph_fingerprint": graph_context.graph_fingerprint,
        "global_graph_build_count": int(graph_context.build_count),
        "total_run_seconds": 0.0,
    })
    summary = pd.concat([summary, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    summary.to_csv(output_dir / "summary_metrics_5fold.csv", index=False)
    all_logs = all_logs.drop(columns=["total_run_seconds"], errors="ignore")
    all_logs = pd.concat(
        [
            all_logs.reset_index(drop=True),
            pd.DataFrame(
                {
                    "total_run_seconds": [
                        float(mean_row["total_run_seconds"])
                    ]
                    * len(all_logs)
                }
            ),
        ],
        axis=1,
    )
    all_logs.to_csv(output_dir / "training_log_5fold.csv", index=False)
    add_total_runtime_to_fold_logs(output_dir, float(mean_row["total_run_seconds"]))
    run_metadata = {
        **global_metadata,
        "total_run_seconds": float(mean_row["total_run_seconds"]),
        "device": str(device),
        "gpu_name": results[0]["gpu_name"] if results else None,
        "decoder_type": decoder_type,
        "mean_metrics": {
            "AUC": mean_row["final_auc"],
            "AUPR": mean_row["final_aupr"],
            "F1": mean_row["final_f1"],
            "Precision": mean_row["final_precision"],
            "Recall": mean_row["final_recall"],
        },
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2, ensure_ascii=False)
    return results


def resolve_run_protocol_name(
    evaluation_protocol: str,
    results: list[dict[str, object]],
) -> str:
    """返回实际执行的协议名称，优先采用折级元数据。"""

    if results and results[0].get("protocol_name"):
        return str(results[0]["protocol_name"])
    if evaluation_protocol == EVALUATION_PROTOCOL_BALANCED_WARM_START:
        return BALANCED_WARM_START_PROTOCOL_NAME
    if evaluation_protocol == EVALUATION_PROTOCOL_P1_TRANSDUCTIVE:
        return P1_TRANSDUCTIVE_PROTOCOL_NAME
    return "Fold-wise reconstruction protocol"


def train_kfold_fold_aware(
    tables: dict[str, pd.DataFrame],
    processed_dir: Path,
    folds: int,
    cv_mode: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    embedding_dim: int,
    hidden_dim: int,
    lr: float,
    contrastive_weight: float,
    node_contrastive_weight: float,
    negative_threshold: float,
    seed: int,
    device_name: str,
    contrastive_temperature: float = 0.2,
    similarity_top_k: int | None = 20,
    drug_similarity_top_k: int | None = None,
    disease_similarity_top_k: int | None = None,
    min_similarity: float = 0.0,
    fusion_type: str = "transformer",
    fusion_dim: int = 256,
    fusion_heads: int = 4,
    fusion_layers: int = 1,
    use_ss_encoder: bool = False,
    ss_dim: int = 128,
    ss_channels: int = 16,
    decoder_type: str = "mlp",
    validation_ratio: float = 0.1,
    validation_metric: str = "AUPR",
    pu_learning: bool = False,
    pu_only: bool = False,
    pu_unlabeled_ratio: float = 1.0,
    pu_loss_weight: float = 0.1,
    pu_class_prior: float | None = None,
    pu_low_threshold: float = 0.3,
    pu_high_threshold: float = 0.7,
    pu_potential_risk_threshold: float = 0.8,
    pu_uncertain_weight: float = 0.2,
    pu_hard_weight: float = 1.5,
    rns_strategy: str = RNS_STRATEGY_THRESHOLD,
    negative_ratio: float = DEFAULT_NEGATIVE_RATIO,
    balanced_negative_pool: str = BALANCED_NEGATIVE_POOL_RANDOM,
    auxiliary_positives: pd.DataFrame | None = None,
    evaluation_protocol: str = EVALUATION_PROTOCOL_FOLD_WISE,
    model_variant: str = MODEL_VARIANT_CURRENT,
    spectral_hops: int = 3,
    spectral_pair_dim: int = 256,
    active_views: tuple[str, ...] = ("A", "S", "B"),
    graph_encoder: str = "gatv2",
    cold_start_rate: float | None = None,
    positive_retention_rate: float | None = None,
    label_noise_rate: float | None = None,
) -> list[dict[str, object]]:
    if evaluation_protocol != EVALUATION_PROTOCOL_FOLD_WISE:
        raise ValueError("Only fold_wise evaluation is supported for new training runs.")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("Strict fold-wise training requires validation_ratio in (0, 1).")
    if folds < 2:
        raise ValueError("Strict fold-wise training requires at least two folds.")
    if positive_retention_rate is not None and label_noise_rate is not None:
        raise ValueError("Sparsity and label-noise perturbations must be run separately.")
    run_started_at = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    if evaluation_protocol == EVALUATION_PROTOCOL_P1_TRANSDUCTIVE:
        if cv_mode != "random":
            raise ValueError("P1 transductive split protocol 只支持 random 模式。")
        fold_splits = create_p1_cv_splits(
            tables=tables,
            folds=folds,
            seed=seed,
            negative_threshold=negative_threshold,
            rns_strategy=rns_strategy,
            negative_ratio=negative_ratio,
            pu_learning=pu_learning,
            pu_unlabeled_ratio=pu_unlabeled_ratio,
            processed_dir=processed_dir,
        )
    elif evaluation_protocol == EVALUATION_PROTOCOL_BALANCED_WARM_START:
        if cv_mode != "random":
            raise ValueError("Balanced warm-start protocol 只支持 random 模式。")
        if validation_ratio <= 0.0:
            raise ValueError("Balanced warm-start protocol 必须启用 validation 选择 epoch。")
        fold_splits = create_balanced_warm_start_cv_splits(
            tables=tables,
            folds=folds,
            negative_ratio=negative_ratio,
            seed=seed,
            processed_dir=processed_dir,
            negative_pool_strategy=balanced_negative_pool,
            negative_threshold=negative_threshold,
        )
    else:
        fold_splits = make_fold_aware_kfold_splits(
            tables=tables,
            folds=folds,
            cv_mode=cv_mode,
            negative_threshold=negative_threshold,
            seed=seed,
            processed_dir=processed_dir,
            pu_learning=pu_learning,
            pu_only=pu_only,
            pu_unlabeled_ratio=pu_unlabeled_ratio,
            rns_strategy=rns_strategy,
            negative_ratio=negative_ratio,
            validation_ratio=validation_ratio,
            auxiliary_positives=auxiliary_positives,
            cold_start_rate=cold_start_rate,
        )

    results = []
    logs = []
    for fold_split in fold_splits:
        if evaluation_protocol == EVALUATION_PROTOCOL_FOLD_WISE:
            fold_id, train_df, validation_df, test_df, fold_stats = fold_split
        else:
            fold_id, train_df, test_df, fold_stats = fold_split
            validation_df = None
        if positive_retention_rate is not None:
            train_df, sparsity_stats = subsample_balanced_outer_train(
                train_df,
                positive_retention_rate=positive_retention_rate,
                seed=seed + fold_id + int(round(positive_retention_rate * 10_000)),
            )
            fold_stats.update(sparsity_stats)
            fold_dir = processed_dir / f"fold_{fold_id}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            train_df.to_csv(fold_dir / "sparse_outer_train_samples.csv", index=False)
        if label_noise_rate is not None:
            train_df, noise_stats = inject_balanced_label_noise(
                train_df,
                label_noise_rate=label_noise_rate,
                seed=seed + fold_id + int(round(label_noise_rate * 100_000)),
            )
            fold_stats.update(noise_stats)
            fold_dir = processed_dir / f"fold_{fold_id}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            train_df.to_csv(fold_dir / "noisy_outer_train_samples.csv", index=False)
        print(f"Starting fold {fold_id}/{folds}, views={'+'.join(active_views)}", flush=True)
        if evaluation_protocol == EVALUATION_PROTOCOL_BALANCED_WARM_START:
            fold_stats.update(
                {
                    "negative_threshold": float(negative_threshold),
                    "pu_learning_enabled": bool(pu_learning),
                    "pu_only_enabled": False,
                    "pu_unlabeled_ratio": float(pu_unlabeled_ratio if pu_learning else 0.0),
                }
            )
        run_name = f"fold_{fold_id}"
        fold_started_at = time.perf_counter()
        common_train_kwargs = dict(
            train_df=train_df,
            test_df=test_df,
            output_dir=output_dir,
            batch_size=batch_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            contrastive_weight=contrastive_weight,
            node_contrastive_weight=node_contrastive_weight,
            contrastive_temperature=contrastive_temperature,
            seed=seed + fold_id,
            device=device,
            raw_tables=tables,
            similarity_top_k=similarity_top_k,
            drug_similarity_top_k=drug_similarity_top_k,
            disease_similarity_top_k=disease_similarity_top_k,
            min_similarity=min_similarity,
            fusion_type=fusion_type,
            fusion_dim=fusion_dim,
            fusion_heads=fusion_heads,
            fusion_layers=fusion_layers,
            use_ss_encoder=use_ss_encoder,
            ss_dim=ss_dim,
            ss_channels=ss_channels,
            decoder_type=decoder_type,
            validation_metric=validation_metric,
            validation_mode=cv_mode,
            predefined_validation_df=validation_df,
            pu_learning=pu_learning,
            pu_unlabeled_ratio=pu_unlabeled_ratio,
            pu_loss_weight=pu_loss_weight,
            pu_class_prior=pu_class_prior,
            pu_low_threshold=pu_low_threshold,
            pu_high_threshold=pu_high_threshold,
            pu_potential_risk_threshold=pu_potential_risk_threshold,
            pu_uncertain_weight=pu_uncertain_weight,
            pu_hard_weight=pu_hard_weight,
            evaluation_protocol=evaluation_protocol,
            pu_update_reliable_negatives=(
                evaluation_protocol == EVALUATION_PROTOCOL_BALANCED_WARM_START
            ),
            model_variant=model_variant,
            spectral_hops=spectral_hops,
            spectral_pair_dim=spectral_pair_dim,
            active_views=active_views,
            graph_encoder=graph_encoder,
        )
        if evaluation_protocol == EVALUATION_PROTOCOL_BALANCED_WARM_START:
            selection_result = train_predefined_split(
                **common_train_kwargs,
                run_name=f"selection_{run_name}",
                epochs=epochs,
                validation_ratio=validation_ratio,
                skip_test_evaluation=True,
            )
            selected_epoch = max(1, int(selection_result["selected_epoch"]))
            result = train_predefined_split(
                **common_train_kwargs,
                run_name=run_name,
                epochs=selected_epoch,
                validation_ratio=0.0,
                fixed_epoch_training=True,
            )
            result.update(
                {
                    "validation_enabled": True,
                    "validation_ratio": float(validation_ratio),
                    "validation_mode": cv_mode,
                    "validation_rows": int(selection_result["validation_rows"]),
                    "selection_metric": selection_result["selection_metric"],
                    "selected_epoch": selected_epoch,
                    "best_metric_source": "validation",
                    "validation_best_metrics": selection_result["validation_best_metrics"],
                    "validation_best_auc": selection_result["validation_best_auc"],
                    "validation_best_aupr": selection_result["validation_best_aupr"],
                    "best_auc": selection_result["best_auc"],
                    "best_auc_epoch": selection_result["best_auc_epoch"],
                    "best_aupr": selection_result["best_aupr"],
                    "best_aupr_epoch": selection_result["best_aupr_epoch"],
                    "refit_on_outer_train": True,
                    "refit_epochs": selected_epoch,
                    "selection_phase_test_evaluation_count": int(
                        selection_result["test_evaluation_count"]
                    ),
                    "test_evaluation_count": 1,
                }
            )
        else:
            result = train_predefined_split(
                **common_train_kwargs,
                run_name=run_name,
                epochs=epochs,
                validation_ratio=validation_ratio,
            )
            result.update(
                {
                    "refit_on_outer_train": False,
                    "refit_epochs": 0,
                    "selection_phase_test_evaluation_count": 0,
                }
            )
        result["fold_train_seconds"] = float(time.perf_counter() - fold_started_at)
        result["fold"] = fold_id
        result["cv_mode"] = cv_mode
        result.update(fold_stats)
        strict_checks = (
            result.get("strict_leakage_free"),
            result.get("test_pair_disjoint_from_train"),
            result.get("test_pair_excluded_from_pu"),
            result.get("test_positive_absent_from_association_graph"),
            result.get("validation_pair_disjoint_from_train"),
            result.get("validation_pair_excluded_from_pu"),
            result.get("validation_positive_absent_from_association_graph"),
            result.get("best_metric_source") == "validation",
            result.get("test_evaluation_count") == 1,
        )
        if not all(strict_checks):
            raise RuntimeError(f"Fold {fold_id} failed strict fold-wise audit.")
        results.append(result)

        # Persist completed folds so an interrupted run retains auditable results.
        with (output_dir / "completed_folds.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Completed fold {fold_id}/{folds}: {result['metrics']}", flush=True)

        fold_log = pd.read_csv(output_dir / f"training_log_{run_name}.csv")
        fold_metadata = {
            "fold": fold_id,
            "cv_mode": cv_mode,
            "fold_train_seconds": result["fold_train_seconds"],
            "refit_on_outer_train": result["refit_on_outer_train"],
            "refit_epochs": result["refit_epochs"],
            "selected_epoch": result["selected_epoch"],
            "best_metric_source": result["best_metric_source"],
            "selection_phase_test_evaluation_count": result[
                "selection_phase_test_evaluation_count"
            ],
            "test_evaluation_count": result["test_evaluation_count"],
            **fold_stats,
        }
        fold_log = fold_log.drop(
            columns=[column for column in fold_metadata if column in fold_log.columns]
        )
        fold_log = pd.concat(
            [
                pd.DataFrame(
                    {key: [value] * len(fold_log) for key, value in fold_metadata.items()}
                ),
                fold_log.reset_index(drop=True),
            ],
            axis=1,
        )
        fold_log.to_csv(output_dir / f"training_log_{run_name}.csv", index=False)
        logs.append(fold_log)

    all_logs = pd.concat(logs, ignore_index=True)
    artifact_fold_count = folds
    all_logs.to_csv(output_dir / f"training_log_{artifact_fold_count}fold.csv", index=False)

    summary_rows = []
    for result in results:
        row = {
            "fold": result["fold"],
            "evaluation_protocol": result["evaluation_protocol"],
            "protocol_name": result.get("protocol_name", ""),
            "cv_mode": result["cv_mode"],
            "negative_sampling_scope": result["negative_sampling_scope"],
            "negative_pool_strategy": result.get("negative_pool_strategy", ""),
            "rns_strategy": result["rns_strategy"],
            "negative_ratio": result["negative_ratio"],
            "global_positive_count": result.get("global_positive_count", np.nan),
            "global_random_negative_count": result.get("global_random_negative_count", np.nan),
            "global_reliable_negative_count": result.get("global_reliable_negative_count", np.nan),
            "global_preprocessing_scope": result.get("global_preprocessing_scope", ""),
            "strict_leakage_free": result.get("strict_leakage_free", False),
            "global_unlabeled_pair_count": result.get("global_unlabeled_pair_count", np.nan),
            "global_gip_enabled": result.get("global_gip_enabled", False),
            "risk_reference_scope": result.get("risk_reference_scope", ""),
            "risk_reference_positive_count": result.get("risk_reference_positive_count", np.nan),
            "positive_retention_rate": result.get("positive_retention_rate", 1.0),
            "realized_positive_retention_rate": result.get(
                "realized_positive_retention_rate", 1.0
            ),
            "original_positive_count": result.get(
                "original_positive_count", result["train_positive_count"]
            ),
            "original_negative_count": result.get(
                "original_negative_count", result["train_negative_count"]
            ),
            "retained_positive_count": result.get(
                "retained_positive_count", result["train_positive_count"]
            ),
            "retained_negative_count": result.get(
                "retained_negative_count", result["train_negative_count"]
            ),
            "sparsity_sampling_seed": result.get("sparsity_sampling_seed", np.nan),
            "label_noise_rate": result.get("label_noise_rate", 0.0),
            "realized_label_noise_rate": result.get("realized_label_noise_rate", 0.0),
            "flipped_positive_to_negative_count": result.get(
                "flipped_positive_to_negative_count", 0
            ),
            "flipped_negative_to_positive_count": result.get(
                "flipped_negative_to_positive_count", 0
            ),
            "total_flipped_label_count": result.get("total_flipped_label_count", 0),
            "label_noise_sampling_seed": result.get("label_noise_sampling_seed", np.nan),
            "train_positive_count": result["train_positive_count"],
            "train_negative_count": result["train_negative_count"],
            "pu_learning_enabled": result.get("pu_learning_enabled", False),
            "pu_only_enabled": result.get("pu_only_enabled", False),
            "pu_unlabeled_count": result.get("pu_unlabeled_count", 0),
            "pu_unlabeled_ratio": result.get("pu_unlabeled_ratio", 0.0),
            "pu_loss_weight": result["pu_loss_weight"],
            "pu_class_prior": result["pu_class_prior"],
            "pu_low_threshold": result["pu_low_threshold"],
            "pu_high_threshold": result["pu_high_threshold"],
            "pu_potential_risk_threshold": result["pu_potential_risk_threshold"],
            "pu_uncertain_weight": result["pu_uncertain_weight"],
            "pu_hard_weight": result["pu_hard_weight"],
            "test_positive_count": result["test_positive_count"],
            "test_negative_count": result["test_negative_count"],
            "p1_total_pair_count": result.get("p1_total_pair_count", np.nan),
            "p1_train_pool_positive_count": result.get("p1_train_pool_positive_count", np.nan),
            "p1_train_pool_unknown_count": result.get("p1_train_pool_unknown_count", np.nan),
            "negative_threshold": result["negative_threshold"],
            "final_auc": result["metrics"]["AUC"],
            "final_aupr": result["metrics"]["AUPR"],
            "final_f1": result["metrics"]["F1"],
            "final_precision": result["metrics"]["Precision"],
            "final_recall": result["metrics"]["Recall"],
            "final_pearson_r": result["metrics"]["pearson_r"],
            "fold_train_seconds": result["fold_train_seconds"],
            "best_auc": result["best_auc"],
            "best_auc_epoch": result["best_auc_epoch"],
            "best_aupr": result["best_aupr"],
            "best_aupr_epoch": result["best_aupr_epoch"],
            "view_weight_0": result["mean_view_weights"][0],
            "view_weight_1": result["mean_view_weights"][1],
            "view_weight_2": result["mean_view_weights"][2],
            "association_edge_count": result["association_edge_count"],
            "similarity_edge_count": result["similarity_edge_count"],
            "biology_edge_count": result["biology_edge_count"],
            "similarity_top_k": result["similarity_top_k"],
            "drug_similarity_top_k": result["drug_similarity_top_k"],
            "disease_similarity_top_k": result["disease_similarity_top_k"],
            "min_similarity": result["min_similarity"],
            "similarity_view_enabled": result["similarity_view_enabled"],
            "decoder_type": result["decoder_type"],
            "encoder_training_head": result["encoder_training_head"],
            "lightgbm_feature_dim": result["lightgbm_feature_dim"],
            "lightgbm_train_rows": result["lightgbm_train_rows"],
            "lightgbm_train_positive_count": result["lightgbm_train_positive_count"],
            "lightgbm_train_negative_count": result["lightgbm_train_negative_count"],
            "fusion_type": result["fusion_type"],
            "fusion_dim": result["fusion_dim"],
            "fusion_heads": result["fusion_heads"],
            "fusion_layers": result["fusion_layers"],
            "validation_enabled": result["validation_enabled"],
            "validation_ratio": result["validation_ratio"],
            "validation_mode": result["validation_mode"],
            "validation_rows": result["validation_rows"],
            "train_core_rows": result["train_core_rows"],
            "selection_metric": result["selection_metric"],
            "selected_epoch": result["selected_epoch"],
            "best_metric_source": result["best_metric_source"],
            "refit_on_outer_train": result.get("refit_on_outer_train", False),
            "refit_epochs": result.get("refit_epochs", 0),
            "selection_phase_test_evaluation_count": result.get(
                "selection_phase_test_evaluation_count", 0
            ),
            "test_evaluation_count": result.get("test_evaluation_count", 1),
            "test_pair_disjoint_from_train": result.get("test_pair_disjoint_from_train", False),
            "validation_pair_disjoint_from_train": result.get(
                "validation_pair_disjoint_from_train", False
            ),
            "test_pair_excluded_from_pu": result.get("test_pair_excluded_from_pu", False),
            "validation_pair_excluded_from_pu": result.get(
                "validation_pair_excluded_from_pu", False
            ),
            "test_positive_absent_from_association_graph": result.get(
                "test_positive_absent_from_association_graph", False
            ),
            "validation_positive_absent_from_association_graph": result.get(
                "validation_positive_absent_from_association_graph", False
            ),
            "test_label_ignored_by_risk_scoring": result.get(
                "test_label_ignored_by_risk_scoring", False
            ),
            "validation_best_auc": result["validation_best_auc"],
            "validation_best_aupr": result["validation_best_aupr"],
            "device": result["device"],
            "gpu_name": result["gpu_name"],
            "model_variant": result["model_variant"],
            "spectral_hops": result["spectral_hops"],
            "spectral_pair_dim": result["spectral_pair_dim"],
            "total_parameter_count": result["total_parameter_count"],
            "trainable_parameter_count": result["trainable_parameter_count"],
            "gpu_peak_memory_mb": result["gpu_peak_memory_mb"],
        }
        row.update(
            {
                f"band_weight_{name}": weight
                for name, weight in result["mean_band_weights"].items()
            }
        )
        row.update(result["node_feature_stats"])
        row.update(result["ss_feature_stats"])
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    numeric_cols = [
        "train_positive_count",
        "train_negative_count",
        "global_positive_count",
        "global_random_negative_count",
        "global_reliable_negative_count",
        "global_unlabeled_pair_count",
        "risk_reference_positive_count",
        "positive_retention_rate",
        "realized_positive_retention_rate",
        "original_positive_count",
        "original_negative_count",
        "retained_positive_count",
        "retained_negative_count",
        "sparsity_sampling_seed",
        "label_noise_rate",
        "realized_label_noise_rate",
        "flipped_positive_to_negative_count",
        "flipped_negative_to_positive_count",
        "total_flipped_label_count",
        "label_noise_sampling_seed",
        "pu_unlabeled_count",
        "pu_unlabeled_ratio",
        "pu_loss_weight",
        "pu_class_prior",
        "pu_low_threshold",
        "pu_high_threshold",
        "pu_potential_risk_threshold",
        "pu_uncertain_weight",
        "pu_hard_weight",
        "negative_ratio",
        "test_positive_count",
        "test_negative_count",
        "p1_total_pair_count",
        "p1_train_pool_positive_count",
        "p1_train_pool_unknown_count",
        "final_auc",
        "final_aupr",
        "final_f1",
        "final_precision",
        "final_recall",
        "final_pearson_r",
        "fold_train_seconds",
        "lightgbm_feature_dim",
        "lightgbm_train_rows",
        "lightgbm_train_positive_count",
        "lightgbm_train_negative_count",
        "best_auc",
        "best_aupr",
        "selected_epoch",
        "refit_epochs",
        "selection_phase_test_evaluation_count",
        "test_evaluation_count",
        "validation_rows",
        "train_core_rows",
        "validation_best_auc",
        "validation_best_aupr",
        "ss_drug_dim",
        "ss_disease_dim",
        "ss_input_dim",
        "ss_output_dim",
        "ss_channels",
        "view_weight_0",
        "view_weight_1",
        "view_weight_2",
        "spectral_hops",
        "spectral_pair_dim",
        "total_parameter_count",
        "trainable_parameter_count",
        "gpu_peak_memory_mb",
    ]
    numeric_cols.extend(
        sorted(column for column in summary.columns if column.startswith("band_weight_"))
    )
    mean_row = {
        "fold": "mean",
        "evaluation_protocol": evaluation_protocol,
        "cv_mode": cv_mode,
        "negative_sampling_scope": str(summary["negative_sampling_scope"].iloc[0]),
        "rns_strategy": rns_strategy,
        "device": str(device),
        "gpu_name": results[0]["gpu_name"],
    }
    std_row = {
        "fold": "std",
        "evaluation_protocol": evaluation_protocol,
        "cv_mode": cv_mode,
        "negative_sampling_scope": str(summary["negative_sampling_scope"].iloc[0]),
        "rns_strategy": rns_strategy,
        "device": str(device),
        "gpu_name": results[0]["gpu_name"],
    }
    for col in numeric_cols:
        mean_row[col] = float(summary[col].mean())
        std_row[col] = float(summary[col].std(ddof=1))
    total_run_seconds = float(time.perf_counter() - run_started_at)
    mean_row["total_run_seconds"] = total_run_seconds
    std_row["total_run_seconds"] = 0.0
    summary = pd.concat([summary, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    summary.to_csv(output_dir / f"summary_metrics_{artifact_fold_count}fold.csv", index=False)
    all_logs = all_logs.drop(columns=["total_run_seconds"], errors="ignore")
    all_logs = pd.concat(
        [
            all_logs.reset_index(drop=True),
            pd.DataFrame({"total_run_seconds": [total_run_seconds] * len(all_logs)}),
        ],
        axis=1,
    )
    all_logs.to_csv(output_dir / f"training_log_{artifact_fold_count}fold.csv", index=False)
    add_total_runtime_to_fold_logs(output_dir, total_run_seconds)
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "total_run_seconds": total_run_seconds,
                "evaluation_protocol": evaluation_protocol,
                "protocol_name": resolve_run_protocol_name(evaluation_protocol, results),
                "strict_leakage_free": all(
                    bool(result.get("strict_leakage_free", False)) for result in results
                ),
                "cv_mode": cv_mode,
                "folds": int(folds),
                "device": str(device),
                "gpu_name": results[0]["gpu_name"] if results else None,
                "decoder_type": decoder_type,
                "model_variant": model_variant,
                "spectral_hops": int(
                    spectral_hops
                    if model_variant == MODEL_VARIANT_SEMANTIC_PATH_SPECTRAL
                    else 0
                ),
                "spectral_pair_dim": int(
                    spectral_pair_dim
                    if model_variant == MODEL_VARIANT_SEMANTIC_PATH_SPECTRAL
                    else 0
                ),
                "total_parameter_count_mean": mean_row["total_parameter_count"],
                "trainable_parameter_count_mean": mean_row["trainable_parameter_count"],
                "gpu_peak_memory_mb_mean": mean_row["gpu_peak_memory_mb"],
                "mean_band_weights": {
                    column.removeprefix("band_weight_"): mean_row[column]
                    for column in numeric_cols
                    if column.startswith("band_weight_")
                },
                "refit_on_outer_train": (
                    evaluation_protocol == EVALUATION_PROTOCOL_BALANCED_WARM_START
                ),
                "mean_metrics": {
                    "AUC": mean_row["final_auc"],
                    "AUPR": mean_row["final_aupr"],
                    "F1": mean_row["final_f1"],
                    "Precision": mean_row["final_precision"],
                    "Recall": mean_row["final_recall"],
                },
                "integrity_checks": {
                    "global_gip_enabled": bool(
                        summary.loc[
                            summary["fold"].astype(str).isin(
                                [str(index) for index in range(1, folds + 1)]
                            ),
                            "global_gip_enabled",
                        ].any()
                    ),
                    "all_test_pairs_disjoint_from_train": bool(
                        summary.loc[
                            summary["fold"].astype(str).isin(
                                [str(index) for index in range(1, folds + 1)]
                            ),
                            "test_pair_disjoint_from_train",
                        ].all()
                    ),
                    "all_test_pairs_excluded_from_pu": bool(
                        summary.loc[
                            summary["fold"].astype(str).isin(
                                [str(index) for index in range(1, folds + 1)]
                            ),
                            "test_pair_excluded_from_pu",
                        ].all()
                    ),
                    "all_test_positives_absent_from_association_graph": bool(
                        summary.loc[
                            summary["fold"].astype(str).isin(
                                [str(index) for index in range(1, folds + 1)]
                            ),
                            "test_positive_absent_from_association_graph",
                        ].all()
                    ),
                    "best_metric_source": "validation"
                    if evaluation_protocol == EVALUATION_PROTOCOL_BALANCED_WARM_START
                    else str(summary["best_metric_source"].iloc[0]),
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return results


def build_training_argument_parser() -> argparse.ArgumentParser:
    """构建训练命令行参数，便于测试默认协议且保持主入口简洁。"""

    parser = argparse.ArgumentParser(description="Quickly train the multi-view GATv2 model.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--feature-dir", default=None)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="outputs/quick_train")
    parser.add_argument("--cv-folds", type=int, default=10)
    parser.add_argument("--cv-mode", choices=list(SPLIT_MODES), default="random")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--model-variant", choices=list(MODEL_VARIANTS), default=MODEL_VARIANT_CURRENT)
    parser.add_argument("--graph-encoder", choices=list(GRAPH_ENCODERS), default="gatv2")
    parser.add_argument("--spectral-hops", type=int, default=3)
    parser.add_argument("--spectral-pair-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--contrastive-weight", type=float, default=0.75)
    parser.add_argument("--node-contrastive-weight", type=float, default=0.75)
    parser.add_argument("--contrastive-temperature", type=float, default=0.2)
    parser.add_argument(
        "--use-ss-encoder",
        action="store_true",
        help="历史兼容参数；当前正式模型始终禁用 SS-Encoder。",
    )
    parser.add_argument("--ss-dim", type=int, default=128)
    parser.add_argument("--ss-channels", type=int, default=16)
    parser.add_argument("--decoder", choices=["mlp", "lightgbm"], default="mlp")
    parser.add_argument(
        "--cold-start-rate",
        type=float,
        default=None,
        help="固定比例实体冷启动；与 --cv-folds 配合时，folds 表示独立重复次数。",
    )
    parser.add_argument("--negative-threshold", type=float, default=0.75)
    parser.add_argument("--rns-strategy", choices=list(RNS_STRATEGIES), default=RNS_STRATEGY_ADAPTIVE_TOPK)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--auxiliary-positive-csv", default=None)
    parser.add_argument("--similarity-top-k", type=int, default=20)
    parser.add_argument("--drug-similarity-top-k", type=int, default=None)
    parser.add_argument("--disease-similarity-top-k", type=int, default=None)
    parser.add_argument("--min-similarity", type=float, default=0.0)
    parser.add_argument(
        "--fusion-type",
        choices=["mean", "attention", "transformer", "reliability_gate"],
        default="reliability_gate",
    )
    parser.add_argument("--fusion-dim", type=int, default=256)
    parser.add_argument("--fusion-heads", type=int, default=4)
    parser.add_argument("--fusion-layers", type=int, default=1)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--validation-metric", choices=["AUC", "AUPR"], default="AUPR")
    parser.add_argument("--pu-learning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pu-only", action="store_true")
    parser.add_argument("--pu-unlabeled-ratio", type=float, default=1.0)
    parser.add_argument("--pu-loss-weight", type=float, default=0.1)
    parser.add_argument("--pu-class-prior", type=float, default=None)
    parser.add_argument("--pu-low-threshold", type=float, default=0.3)
    parser.add_argument("--pu-high-threshold", type=float, default=0.7)
    parser.add_argument("--pu-potential-risk-threshold", type=float, default=0.8)
    parser.add_argument("--pu-uncertain-weight", type=float, default=0.2)
    parser.add_argument("--pu-hard-weight", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--evaluation-protocol",
        "--protocol",
        choices=[EVALUATION_PROTOCOL_FOLD_WISE],
        default=EVALUATION_PROTOCOL_FOLD_WISE,
        help="New runs use fold_wise strict reconstruction only.",
    )
    return parser


def normalize_evaluation_protocol_args(args: argparse.Namespace) -> argparse.Namespace:
    """对协议专属参数做显式规范化，避免全局协议静默退回旧流程。"""

    if args.evaluation_protocol != EVALUATION_PROTOCOL_FOLD_WISE:
        raise ValueError("Only fold_wise evaluation is supported for new training runs.")
    if args.cv_folds < 2:
        raise ValueError("Strict fold-wise training requires --cv-folds >= 2.")
    if not 0.0 < args.validation_ratio < 1.0:
        raise ValueError("Strict fold-wise training requires --validation-ratio in (0, 1).")
    if args.spectral_hops < 1:
        raise ValueError("--spectral-hops 必须至少为 1。")
    if args.spectral_pair_dim < 1:
        raise ValueError("--spectral-pair-dim 必须为正整数。")
    if args.cold_start_rate is not None:
        if not 0.0 < args.cold_start_rate < 1.0:
            raise ValueError("--cold-start-rate 必须在 0 和 1 之间。")
        if args.cv_mode not in {"cold-drug", "cold-disease"}:
            raise ValueError("--cold-start-rate 仅支持 cold-drug 或 cold-disease。")
        if not args.cv_folds:
            raise ValueError("--cold-start-rate 必须同时设置 --cv-folds 作为重复次数。")
    if args.model_variant == MODEL_VARIANT_SEMANTIC_PATH_SPECTRAL:
        if args.decoder != "mlp":
            raise ValueError("semantic_path_spectral 首轮实验固定使用 MLP 解码器。")
        args.contrastive_weight = 0.0
        args.node_contrastive_weight = 0.0
        args.use_ss_encoder = False
    elif args.model_variant in {MODEL_VARIANT_CURRENT, MODEL_VARIANT_NO_SS}:
        args.use_ss_encoder = False

    return args


def main() -> None:
    """运行多视图 GATv2 药物-疾病关联预测训练。"""

    parser = build_training_argument_parser()
    args = parser.parse_args()
    args = normalize_evaluation_protocol_args(args)
    if args.pu_only:
        args.pu_learning = True

    raw_dir = Path(args.raw_dir)
    feature_dir = Path(args.feature_dir) if args.feature_dir else None
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    tables = load_raw_tables(raw_dir, feature_dir=feature_dir)
    auxiliary_positives = (
        load_auxiliary_positive_examples(Path(args.auxiliary_positive_csv))
        if args.auxiliary_positive_csv
        else None
    )
    if args.cv_folds:
        common_kwargs = {
            "tables": tables,
            "processed_dir": processed_dir,
            "folds": args.cv_folds,
            "cv_mode": args.cv_mode,
            "output_dir": output_dir,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "model_variant": args.model_variant,
            "graph_encoder": args.graph_encoder,
            "spectral_hops": args.spectral_hops,
            "spectral_pair_dim": args.spectral_pair_dim,
            "lr": args.lr,
            "contrastive_weight": args.contrastive_weight,
            "node_contrastive_weight": args.node_contrastive_weight,
            "contrastive_temperature": args.contrastive_temperature,
            "negative_threshold": args.negative_threshold,
            "seed": args.seed,
            "device_name": args.device,
            "similarity_top_k": args.similarity_top_k,
            "drug_similarity_top_k": args.drug_similarity_top_k,
            "disease_similarity_top_k": args.disease_similarity_top_k,
            "min_similarity": args.min_similarity,
            "fusion_type": args.fusion_type,
            "fusion_dim": args.fusion_dim,
            "fusion_heads": args.fusion_heads,
            "fusion_layers": args.fusion_layers,
            "use_ss_encoder": args.use_ss_encoder,
            "ss_dim": args.ss_dim,
            "ss_channels": args.ss_channels,
            "decoder_type": args.decoder,
            "validation_ratio": args.validation_ratio,
            "validation_metric": args.validation_metric,
            "pu_learning": args.pu_learning,
            "pu_unlabeled_ratio": args.pu_unlabeled_ratio,
            "pu_loss_weight": args.pu_loss_weight,
            "pu_class_prior": args.pu_class_prior,
            "pu_low_threshold": args.pu_low_threshold,
            "pu_high_threshold": args.pu_high_threshold,
            "pu_potential_risk_threshold": args.pu_potential_risk_threshold,
            "pu_uncertain_weight": args.pu_uncertain_weight,
            "pu_hard_weight": args.pu_hard_weight,
            "rns_strategy": args.rns_strategy,
            "negative_ratio": args.negative_ratio,
            "cold_start_rate": args.cold_start_rate,
        }
        results = train_kfold_fold_aware(
            **common_kwargs,
            pu_only=args.pu_only,
            auxiliary_positives=auxiliary_positives,
            evaluation_protocol=args.evaluation_protocol,
        )
    else:
        raise AssertionError("Cross-validation validation should require at least two folds.")

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
