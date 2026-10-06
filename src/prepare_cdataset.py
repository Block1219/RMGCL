from __future__ import annotations

"""Convert the AMDGT C-dataset matrices to RMGCL's input tables."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _similarity_edges(
    path: Path, prefix: str, left_column: str, right_column: str
) -> tuple[pd.DataFrame, int]:
    matrix = pd.read_csv(path, index_col=0).to_numpy(dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square similarity matrix: {path}")
    left, right = np.triu_indices(matrix.shape[0], k=1)
    scores = matrix[left, right]
    keep = np.isfinite(scores) & (scores > 0)
    edges = pd.DataFrame(
        {
            left_column: [f"{prefix}{index:03d}" for index in left[keep]],
            right_column: [f"{prefix}{index:03d}" for index in right[keep]],
            "score": np.round(scores[keep], 6),
        }
    )
    return edges, matrix.shape[0]


def convert_cdataset(source_dir: str | Path, output_dir: str | Path) -> dict[str, int | str]:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    drug_similarity, num_drugs = _similarity_edges(
        source_dir / "DrugFingerprint.csv", "D", "drug_id_1", "drug_id_2"
    )
    disease_similarity, num_diseases = _similarity_edges(
        source_dir / "DiseasePS.csv", "C", "disease_id_1", "disease_id_2"
    )

    associations = pd.read_csv(source_dir / "DrugDiseaseAssociationNumber.csv")
    drug_target_source = pd.read_csv(source_dir / "DrugProteinAssociationNumber.csv")
    target_disease_source = pd.read_csv(source_dir / "ProteinDiseaseAssociationNumber.csv")
    if associations["drug"].max() >= num_drugs or associations["disease"].max() >= num_diseases:
        raise ValueError("Association indices exceed the similarity matrices")

    drug_disease = pd.DataFrame(
        {
            "drug_id": associations["drug"].map(lambda index: f"D{index:03d}"),
            "disease_id": associations["disease"].map(lambda index: f"C{index:03d}"),
            "label": 1,
        }
    ).drop_duplicates()
    drug_target = pd.DataFrame(
        {
            "drug_id": drug_target_source["drug"].map(lambda index: f"D{index:03d}"),
            "target_id": drug_target_source["protein"].map(lambda index: f"P{index:04d}"),
        }
    ).drop_duplicates()
    target_disease = pd.DataFrame(
        {
            "target_id": target_disease_source["protein"].map(lambda index: f"P{index:04d}"),
            "disease_id": target_disease_source["disease"].map(lambda index: f"C{index:03d}"),
        }
    ).drop_duplicates()

    tables = {
        "drug_disease": drug_disease,
        "drug_similarity": drug_similarity,
        "disease_similarity": disease_similarity,
        "drug_target": drug_target,
        "target_disease": target_disease,
    }
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    summary: dict[str, int | str] = {
        "source_repository": "https://github.com/JK-Liu7/AMDGT/tree/main/data/C-dataset",
        "drug_similarity_source": "DrugFingerprint.csv",
        "disease_similarity_source": "DiseasePS.csv",
        "num_drugs": num_drugs,
        "num_diseases": num_diseases,
        "num_positive_pairs": len(drug_disease),
        "num_drug_similarity_edges": len(drug_similarity),
        "num_disease_similarity_edges": len(disease_similarity),
        "num_drug_target_edges": len(drug_target),
        "num_target_disease_edges": len(target_disease),
    }
    (output_dir / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="data/cdataset_original")
    parser.add_argument("--output-dir", default="data/cdataset_raw")
    args = parser.parse_args()
    print(json.dumps(convert_cdataset(args.source_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
