from __future__ import annotations

"""将 LRSSL 官方矩阵转换为当前项目使用的五表格式。"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


REQUIRED_FILES = {
    "association": "drug_dis_mat.txt",
    "disease_similarity": "disease_similarity.txt",
    "pubchem": "drug_pubchem_mat.txt",
    "target_domain": "drug_target_domain_mat.txt",
    "target_go": "drug_target_go_mat.txt",
}

# LRSSL 官方 PubChem 表中唯一一个与关联矩阵写法不同的药名。
OFFICIAL_DRUG_ALIASES = {
    "ergoloid mesylates": "ergoloid mesylates, usp",
}


def sha256_file(path: Path) -> str:
    """分块计算源文件 SHA256，避免将大矩阵一次性读入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_r_table(path: Path) -> pd.DataFrame:
    """读取 LRSSL R 代码使用的制表符矩阵，并统一实体 ID 类型。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少 LRSSL 官方文件: {path}")

    frame = pd.read_csv(path, sep="\t", header=0, index_col=0)
    if frame.empty or frame.shape[1] == 0:
        raise ValueError(f"矩阵为空: {path}")

    frame.index = frame.index.astype(str).str.strip()
    frame.columns = frame.columns.astype(str).str.strip()
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError(f"矩阵存在重复行或列 ID: {path}")

    numeric = frame.apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"矩阵包含 NaN 或无穷值: {path}")
    return numeric


def cosine_similarity_rows(matrix: np.ndarray) -> np.ndarray:
    """计算行向量余弦相似度；零向量仅与自身保持相似度 1。"""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("余弦相似度输入必须是二维矩阵")

    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = np.divide(
        values,
        norms,
        out=np.zeros_like(values, dtype=np.float64),
        where=norms > 0,
    )
    similarity = normalized @ normalized.T
    similarity = np.clip(similarity, 0.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    return similarity


def fuse_drug_profiles(
    profiles: Sequence[pd.DataFrame], drug_ids: Sequence[str]
) -> np.ndarray:
    """对三类官方药物特征的余弦相似度进行等权融合。"""

    if not profiles:
        raise ValueError("至少需要一个药物特征矩阵")

    expected_ids = list(map(str, drug_ids))
    similarities: list[np.ndarray] = []
    for profile in profiles:
        aligned_profile = profile.rename(index=OFFICIAL_DRUG_ALIASES)
        if not aligned_profile.index.is_unique:
            raise ValueError("药物别名归一化后出现重复 ID")
        actual_ids = aligned_profile.index.astype(str).tolist()
        if set(actual_ids) != set(expected_ids):
            missing = sorted(set(expected_ids) - set(actual_ids))[:5]
            extra = sorted(set(actual_ids) - set(expected_ids))[:5]
            raise ValueError(
                "药物 ID 与关联矩阵不一致: "
                f"missing={missing}, extra={extra}"
            )

        # 官方特征表为“药物 x 特征”，每行直接对应一个药物。
        aligned = aligned_profile.loc[expected_ids, :].to_numpy(dtype=np.float64)
        similarities.append(cosine_similarity_rows(aligned))

    fused = np.mean(np.stack(similarities, axis=0), axis=0)
    fused = (fused + fused.T) / 2.0
    fused = np.clip(fused, 0.0, 1.0)
    np.fill_diagonal(fused, 1.0)
    return fused


def matrix_to_similarity_table(
    matrix: np.ndarray,
    entity_ids: Sequence[str],
    left_column: str,
    right_column: str,
) -> pd.DataFrame:
    """将对称相似性矩阵上三角转换为项目的长表格式。"""

    values = np.asarray(matrix, dtype=np.float64)
    ids = list(map(str, entity_ids))
    if values.shape != (len(ids), len(ids)):
        raise ValueError("相似性矩阵尺寸与实体 ID 数量不一致")
    if not np.isfinite(values).all():
        raise ValueError("相似性矩阵包含 NaN 或无穷值")

    left, right = np.triu_indices(len(ids), k=1)
    scores = values[left, right]
    keep = scores > 0
    return pd.DataFrame(
        {
            left_column: np.asarray(ids, dtype=object)[left[keep]],
            right_column: np.asarray(ids, dtype=object)[right[keep]],
            "score": np.round(scores[keep], 8),
        }
    )


def _validate_square_matrix(
    frame: pd.DataFrame, expected_ids: Sequence[str], matrix_name: str
) -> tuple[np.ndarray, float, str]:
    """检查方阵及 ID 对齐，并返回轻微数值对称化后的矩阵。"""

    ids = list(map(str, expected_ids))
    if frame.shape[0] != frame.shape[1]:
        raise ValueError(f"{matrix_name} 必须是方阵，实际尺寸为 {frame.shape}")
    if set(frame.index) == set(ids) and set(frame.columns) == set(ids):
        values = frame.loc[ids, ids].to_numpy(dtype=np.float64)
        alignment = "matched_by_id"
    else:
        # LRSSL 的关联矩阵使用疾病名称，相似性矩阵使用 MeSH ID；
        # 官方 R 代码按完全相同的列顺序进行逻辑索引，因此这里显式复现该规则。
        if frame.shape != (len(ids), len(ids)):
            raise ValueError(f"{matrix_name} 的疾病数量与关联矩阵不一致")
        if frame.index.tolist() != frame.columns.tolist():
            raise ValueError(f"{matrix_name} 的行列 MeSH ID 顺序不一致")
        values = frame.to_numpy(dtype=np.float64)
        alignment = "official_positional_order"
    max_asymmetry = float(np.max(np.abs(values - values.T)))
    if max_asymmetry > 1e-4:
        raise ValueError(
            f"{matrix_name} 非对称程度过大: max_abs_diff={max_asymmetry:.6g}"
        )

    values = np.clip((values + values.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(values, 1.0)
    return values, max_asymmetry, alignment


def convert_lrssl_dataset(
    source_dir: str | Path, output_dir: str | Path
) -> dict[str, object]:
    """转换 LRSSL 官方数据，且不构造任何代理药物-靶点-疾病边。"""

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        name: read_r_table(source_dir / filename)
        for name, filename in REQUIRED_FILES.items()
    }
    association = tables["association"]
    association_values = association.to_numpy(dtype=np.float64)
    if not np.isin(association_values, [0.0, 1.0]).all():
        raise ValueError("drug_dis_mat.txt 必须是只包含 0/1 的关联矩阵")

    drug_ids = association.index.astype(str).tolist()
    disease_ids = association.columns.astype(str).tolist()
    (
        disease_similarity,
        disease_max_asymmetry,
        disease_similarity_alignment,
    ) = _validate_square_matrix(
        tables["disease_similarity"], disease_ids, "疾病相似性矩阵"
    )

    profile_names = ("pubchem", "target_domain", "target_go")
    aliases_used = {
        name: {
            source: target
            for source, target in OFFICIAL_DRUG_ALIASES.items()
            if source in tables[name].index and target in drug_ids
        }
        for name in profile_names
    }
    drug_similarity = fuse_drug_profiles(
        [tables[name] for name in profile_names], drug_ids
    )

    positive_drugs, positive_diseases = np.where(association_values > 0)
    drug_disease = pd.DataFrame(
        {
            "drug_id": np.asarray(drug_ids, dtype=object)[positive_drugs],
            "disease_id": np.asarray(disease_ids, dtype=object)[positive_diseases],
            "label": np.ones(len(positive_drugs), dtype=np.int64),
        }
    )
    drug_similarity_table = matrix_to_similarity_table(
        drug_similarity,
        drug_ids,
        "drug_id_1",
        "drug_id_2",
    )
    disease_similarity_table = matrix_to_similarity_table(
        disease_similarity,
        disease_ids,
        "disease_id_1",
        "disease_id_2",
    )

    # LRSSL 官方仓库没有可直接还原为标准边表的靶点关系，明确留空。
    drug_target = pd.DataFrame(columns=["drug_id", "target_id"])
    target_disease = pd.DataFrame(columns=["target_id", "disease_id"])

    drug_disease.to_csv(output_dir / "drug_disease.csv", index=False)
    drug_similarity_table.to_csv(output_dir / "drug_similarity.csv", index=False)
    disease_similarity_table.to_csv(
        output_dir / "disease_similarity.csv", index=False
    )
    drug_target.to_csv(output_dir / "drug_target.csv", index=False)
    target_disease.to_csv(output_dir / "target_disease.csv", index=False)

    summary: dict[str, object] = {
        "dataset": "LRSSL",
        "source_repository": "https://github.com/LiangXujun/LRSSL",
        "source_file_sha256": {
            filename: sha256_file(source_dir / filename)
            for filename in REQUIRED_FILES.values()
        },
        "conversion_policy": "official_features_only_no_proxy_biology_edges",
        "num_drugs": int(len(drug_ids)),
        "num_diseases": int(len(disease_ids)),
        "num_positive_pairs": int(len(drug_disease)),
        "num_unlabeled_pairs": int(association_values.size - len(drug_disease)),
        "num_drug_similarity_edges": int(len(drug_similarity_table)),
        "num_disease_similarity_edges": int(len(disease_similarity_table)),
        "drug_feature_dimensions": {
            name: int(tables[name].shape[1]) for name in profile_names
        },
        "drug_similarity_fusion": "equal_weight_cosine_mean",
        "drug_profile_aliases_used": aliases_used,
        "disease_similarity_alignment": disease_similarity_alignment,
        "disease_similarity_max_asymmetry_before_cleanup": disease_max_asymmetry,
        "biology_view": "self_loops_only",
        "num_drug_target_edges": 0,
        "num_target_disease_edges": 0,
    }
    (output_dir / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(
        description="将 LRSSL 官方矩阵转换为项目使用的五个 CSV 文件。"
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    summary = convert_lrssl_dataset(args.source_dir, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
