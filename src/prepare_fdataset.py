from __future__ import annotations

"""将 F-dataset 转换为项目统一的药物-疾病预测数据格式。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read_node_ids(source_dir: Path) -> list[str]:
    """读取 F-dataset 的全局节点索引到实体 ID 的映射。"""

    nodes = pd.read_csv(source_dir / "AllNode.csv")
    return nodes["id"].astype(str).tolist()


def _read_indexed_matrix(path: Path) -> pd.DataFrame:
    """读取第一列为实体 ID 或行索引、后续列为矩阵值的 CSV。"""

    matrix = pd.read_csv(path)
    return matrix.drop(columns=[matrix.columns[0]]).apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _matrix_to_similarity_edges(
    matrix: pd.DataFrame,
    ids: list[str],
    left_col: str,
    right_col: str,
) -> pd.DataFrame:
    """把相似性矩阵上三角转换为边表。"""

    values = matrix.to_numpy(dtype=float)
    rows = []
    for left in range(values.shape[0]):
        for right in range(left + 1, values.shape[1]):
            score = float(values[left, right])
            if score > 0:
                rows.append((ids[left], ids[right], round(score, 6)))
    return pd.DataFrame(rows, columns=[left_col, right_col, "score"])


def _numeric_feature_table(matrix_path: Path, id_col: str, id_values: list[str]) -> pd.DataFrame:
    """把矩阵式特征文件转换为 id + feature columns 的宽表。"""

    matrix = _read_indexed_matrix(matrix_path)
    feature_columns = [f"feature_{idx}" for idx in range(matrix.shape[1])]
    out = matrix.copy()
    out.columns = feature_columns
    out.insert(0, id_col, id_values[: len(out)])
    return out


def convert_fdataset(source_dir: str | Path, raw_dir: str | Path) -> dict[str, int]:
    """转换 F-dataset，并返回转换摘要。"""

    source_dir = Path(source_dir)
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    node_ids = _read_node_ids(source_dir)
    drug_information = pd.read_csv(source_dir / "DrugInformation.csv").drop_duplicates("id", keep="first")
    protein_information = pd.read_csv(source_dir / "ProteinInformation.csv")
    disease_feature = pd.read_csv(source_dir / "DiseaseFeature.csv", header=None)

    num_drugs = len(_read_indexed_matrix(source_dir / "DrugFingerprint.csv"))
    num_diseases = len(_read_indexed_matrix(source_dir / "DiseasePS.csv"))
    protein_ids = protein_information["id"].astype(str).tolist()
    num_proteins = len(protein_ids)
    drug_ids = node_ids[:num_drugs]
    disease_ids = node_ids[num_drugs : num_drugs + num_diseases]

    expected_total = num_drugs + num_diseases + num_proteins
    if expected_total > len(node_ids):
        raise ValueError("AllNode.csv does not contain enough nodes for drug/disease/protein offsets.")

    def drug_id(index: int) -> str:
        return node_ids[index]

    def disease_id(index: int) -> str:
        return node_ids[num_drugs + index]

    def protein_id(index: int) -> str:
        return node_ids[num_drugs + num_diseases + index]

    drug_disease_number = pd.read_csv(source_dir / "DrugDiseaseAssociationNumber.csv")
    drug_disease = pd.DataFrame(
        {
            "drug_id": [drug_id(int(idx)) for idx in drug_disease_number["drug"]],
            "disease_id": [disease_id(int(idx)) for idx in drug_disease_number["disease"]],
            "label": 1,
        }
    ).drop_duplicates()

    drug_protein_number = pd.read_csv(source_dir / "DrugProteinAssociationNumber.csv")
    drug_target = pd.DataFrame(
        {
            "drug_id": [drug_id(int(idx)) for idx in drug_protein_number["drug"]],
            "target_id": [protein_id(int(idx)) for idx in drug_protein_number["protein"]],
        }
    ).drop_duplicates()

    protein_disease_number = pd.read_csv(source_dir / "ProteinDiseaseAssociationNumber.csv")
    target_disease = pd.DataFrame(
        {
            "target_id": [protein_id(int(idx)) for idx in protein_disease_number["protein"]],
            "disease_id": [disease_id(int(idx)) for idx in protein_disease_number["disease"]],
        }
    ).drop_duplicates()

    drug_similarity = _matrix_to_similarity_edges(
        _read_indexed_matrix(source_dir / "DrugFingerprint.csv"),
        drug_ids,
        "drug_id_1",
        "drug_id_2",
    )
    disease_similarity = _matrix_to_similarity_edges(
        _read_indexed_matrix(source_dir / "DiseasePS.csv"),
        disease_ids,
        "disease_id_1",
        "disease_id_2",
    )

    drug_smiles = (
        pd.DataFrame({"drug_id": drug_ids})
        .merge(drug_information[["id", "smiles"]], left_on="drug_id", right_on="id", how="left")
        .drop(columns=["id"])
        .dropna()
    )
    drug_fingerprint = _numeric_feature_table(source_dir / "DrugFingerprint.csv", "drug_id", drug_ids)
    drug_mol2vec = _numeric_feature_table(source_dir / "Drug_mol2vec.csv", "drug_id", drug_ids)
    disease_feature_out = disease_feature.copy()
    disease_feature_out.columns = ["disease_id"] + [f"feature_{idx}" for idx in range(disease_feature.shape[1] - 1)]
    disease_mesh = pd.DataFrame(
        {
            "disease_id": disease_ids,
            "mesh_id": disease_ids,
            "tree_number": "",
            "description": "",
        }
    )
    disease_gene = target_disease.rename(columns={"target_id": "gene_id"})[["disease_id", "gene_id"]].drop_duplicates()

    drug_disease.to_csv(raw_dir / "drug_disease.csv", index=False)
    drug_similarity.to_csv(raw_dir / "drug_similarity.csv", index=False)
    disease_similarity.to_csv(raw_dir / "disease_similarity.csv", index=False)
    drug_target.to_csv(raw_dir / "drug_target.csv", index=False)
    target_disease.to_csv(raw_dir / "target_disease.csv", index=False)
    drug_smiles.to_csv(raw_dir / "drug_smiles.csv", index=False)
    drug_fingerprint.to_csv(raw_dir / "drug_fingerprint.csv", index=False)
    drug_mol2vec.to_csv(raw_dir / "drug_mol2vec.csv", index=False)
    disease_feature_out.to_csv(raw_dir / "disease_feature.csv", index=False)
    disease_mesh.to_csv(raw_dir / "disease_mesh.csv", index=False)
    disease_gene.to_csv(raw_dir / "disease_gene.csv", index=False)

    summary = {
        "num_drugs": int(num_drugs),
        "num_diseases": int(num_diseases),
        "num_proteins": int(num_proteins),
        "num_positive_pairs": int(len(drug_disease)),
        "num_drug_similarity_edges": int(len(drug_similarity)),
        "num_disease_similarity_edges": int(len(disease_similarity)),
        "num_drug_target_edges": int(len(drug_target)),
        "num_target_disease_edges": int(len(target_disease)),
        "num_drug_smiles": int(len(drug_smiles)),
        "num_drug_fingerprint_rows": int(len(drug_fingerprint)),
        "num_disease_mesh_rows": int(len(disease_mesh)),
        "num_disease_gene_edges": int(len(disease_gene)),
    }
    (raw_dir / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert F-dataset into the project raw schema.")
    parser.add_argument("--source-dir", default="data/fdataset_original")
    parser.add_argument("--raw-dir", default="data/fdataset_raw")
    args = parser.parse_args()

    summary = convert_fdataset(args.source_dir, args.raw_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
