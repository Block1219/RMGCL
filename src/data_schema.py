from __future__ import annotations

"""药物-疾病关联预测项目的 CSV 数据格式校验模块。

模型需要五个原始 CSV 表。本模块负责检查必需列、规范化 ID 字符串、
校验标签和相似度分数，并返回可用于负采样和训练的干净 DataFrame。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class TableSpec:
    """单个输入表的必需字段定义。"""

    name: str
    required_columns: tuple[str, ...]


TABLE_SPECS = {
    "drug_disease": TableSpec("drug_disease", ("drug_id", "disease_id", "label")),
    "drug_similarity": TableSpec("drug_similarity", ("drug_id_1", "drug_id_2", "score")),
    "disease_similarity": TableSpec("disease_similarity", ("disease_id_1", "disease_id_2", "score")),
    "drug_target": TableSpec("drug_target", ("drug_id", "target_id")),
    "target_disease": TableSpec("target_disease", ("target_id", "disease_id")),
}


def read_csv(path: str | Path) -> pd.DataFrame:
    """读取 CSV 文件；如果文件不存在，则给出清晰路径。"""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def require_columns(df: pd.DataFrame, required: Iterable[str], table_name: str) -> None:
    """确保表中包含所有必需列。"""

    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def normalize_string_ids(df: pd.DataFrame) -> pd.DataFrame:
    """规范化 ID 列，保证后续表连接时使用一致的字符串。"""

    out = df.copy()
    for col in out.columns:
        if col.endswith("_id") or col.endswith("_id_1") or col.endswith("_id_2"):
            out[col] = out[col].astype(str).str.strip()
    return out


def validate_similarity_scores(df: pd.DataFrame, table_name: str) -> None:
    """校验相似度分数，要求不存在缺失且范围在 [0, 1] 内。"""

    if "score" not in df.columns:
        return
    if df["score"].isna().any():
        raise ValueError(f"{table_name} contains missing similarity scores.")
    if ((df["score"] < 0) | (df["score"] > 1)).any():
        raise ValueError(f"{table_name} contains scores outside [0, 1].")


def validate_binary_labels(df: pd.DataFrame, table_name: str) -> None:
    """如果存在 label 列，则关联标签必须是二分类取值 0/1。"""

    if "label" not in df.columns:
        return
    labels = set(df["label"].dropna().unique().tolist())
    if not labels.issubset({0, 1}):
        raise ValueError(f"{table_name} labels must be 0 or 1, got: {sorted(labels)}")


def load_and_validate(path: str | Path, spec: TableSpec) -> pd.DataFrame:
    """读取单个表，完成校验、ID 规范化和去重。"""

    df = read_csv(path)
    require_columns(df, spec.required_columns, spec.name)
    df = normalize_string_ids(df)
    validate_similarity_scores(df, spec.name)
    validate_binary_labels(df, spec.name)
    return df.drop_duplicates()


def validate_project_tables(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    """加载模型所需的全部原始表。"""

    raw_dir = Path(raw_dir)
    files = {
        "drug_disease": raw_dir / "drug_disease.csv",
        "drug_similarity": raw_dir / "drug_similarity.csv",
        "disease_similarity": raw_dir / "disease_similarity.csv",
        "drug_target": raw_dir / "drug_target.csv",
        "target_disease": raw_dir / "target_disease.csv",
    }
    return {
        name: load_and_validate(files[name], TABLE_SPECS[name])
        for name in files
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    args = parser.parse_args()

    tables = validate_project_tables(args.raw_dir)
    for name, df in tables.items():
        print(f"{name}: {df.shape[0]} rows, {df.shape[1]} columns")
