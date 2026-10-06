from __future__ import annotations

"""用于药物-疾病关联预测的可靠负样本选择模块。

未观测到的药物-疾病 pair 不能直接视为真实负样本。本模块从相似性、
拓扑结构、生物路径和低度数不确定性等角度评估假负样本风险，并选择
低风险 pair 作为训练负样本。
"""

from dataclasses import dataclass
from math import exp
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NegativeSamplingWeights:
    """假负样本风险评分中五类风险信号的权重。"""

    drug_similarity: float = 0.25
    disease_similarity: float = 0.25
    biological_path: float = 0.20
    topology: float = 0.10
    cold_start_uncertainty: float = 0.20


@dataclass(frozen=True)
class NegativeSamplerConfig:
    """可靠负样本选择过程的配置。"""

    reliability_threshold: float = 0.75
    tau_cold_start: float = 5.0
    negative_ratio: float = 0.5
    weights: NegativeSamplingWeights = NegativeSamplingWeights()
    random_state: int = 42


def sigmoid(x: float) -> float:
    """将风险 logit 映射到 [0, 1] 区间。"""

    return 1.0 / (1.0 + exp(-x))


def build_similarity_lookup(
    df: pd.DataFrame,
    left_col: str,
    right_col: str,
    score_col: str = "score",
) -> dict[tuple[str, str], float]:
    """构建成对相似度的对称查找表。"""

    lookup: dict[tuple[str, str], float] = {}
    for row in df[[left_col, right_col, score_col]].itertuples(index=False):
        a, b, score = str(row[0]), str(row[1]), float(row[2])
        lookup[(a, b)] = score
        lookup[(b, a)] = score
    return lookup


def max_similarity_to_known(
    query: str,
    known_entities: Iterable[str],
    sim_lookup: dict[tuple[str, str], float],
) -> float:
    """返回查询实体与已知正相关实体之间的最大相似度。"""

    best = 0.0
    for entity in known_entities:
        best = max(best, sim_lookup.get((query, entity), 0.0))
    return best


class ReliableNegativeSampler:
    """面向未标记 pair 噪声控制的可靠负样本选择器。

    假负样本风险越低，说明候选未知 pair 越适合作为负样本使用。（P1, P2)
    """

    def __init__(
        self,
        associations: pd.DataFrame,
        drug_similarity: pd.DataFrame,
        disease_similarity: pd.DataFrame,
        drug_target: pd.DataFrame | None = None,
        target_disease: pd.DataFrame | None = None,
        config: NegativeSamplerConfig | None = None,
        all_drugs: Iterable[str] | None = None,
        all_diseases: Iterable[str] | None = None,
        excluded_pairs: Iterable[tuple[str, str]] | None = None,
    ) -> None:
        self.config = config or NegativeSamplerConfig()
        self.associations = associations.copy()
        self.positive = self.associations[self.associations["label"] == 1].copy()

        self.drugs = (
            sorted({str(drug_id) for drug_id in all_drugs})
            if all_drugs is not None
            else sorted(self.associations["drug_id"].astype(str).unique().tolist())
        )

        self.diseases = (
            sorted({str(disease_id) for disease_id in all_diseases})
            if all_diseases is not None
            else sorted(self.associations["disease_id"].astype(str).unique().tolist())
        )
        self.positive_pairs = set(
            zip(self.positive["drug_id"].astype(str), self.positive["disease_id"].astype(str))
        )
        self.excluded_pairs = set(self.positive_pairs)
        if excluded_pairs is not None:
            self.excluded_pairs.update((str(drug_id), str(disease_id)) for drug_id, disease_id in excluded_pairs)

        self.drug_sim = build_similarity_lookup(drug_similarity, "drug_id_1", "drug_id_2")
        self.disease_sim = build_similarity_lookup(disease_similarity, "disease_id_1", "disease_id_2")

        self.drugs_by_disease = (
            self.positive.groupby("disease_id")["drug_id"].apply(lambda x: set(map(str, x))).to_dict()
        )
        self.diseases_by_drug = (
            self.positive.groupby("drug_id")["disease_id"].apply(lambda x: set(map(str, x))).to_dict()
        )
        self.drug_degree = self.positive.groupby("drug_id").size().to_dict()
        self.disease_degree = self.positive.groupby("disease_id").size().to_dict()

        self.drug_targets = self._group_edges(drug_target, "drug_id", "target_id")
        self.target_diseases = self._group_edges(target_disease, "target_id", "disease_id")

    @staticmethod
    def _group_edges(df: pd.DataFrame | None, src: str, dst: str) -> dict[str, set[str]]:
        """将边表按起点聚合为邻接集合。"""

        if df is None or df.empty:
            return {}
        return df.groupby(src)[dst].apply(lambda x: set(map(str, x))).to_dict()

    def biological_path_risk(self, drug_id: str, disease_id: str) -> float:
        """根据 drug-target-disease 生物路径估计风险。"""

        targets = self.drug_targets.get(drug_id, set())
        if not targets:
            return 0.0
        hit_count = 0
        for target in targets:
            if disease_id in self.target_diseases.get(target, set()):
                hit_count += 1
        return min(1.0, hit_count / max(1, len(targets)))

    def topology_risk(self, drug_id: str, disease_id: str) -> float:
        # 使用共同邻域密度作为轻量级拓扑风险近似。
        similar_known_drugs = self.drugs_by_disease.get(disease_id, set())
        similar_known_diseases = self.diseases_by_drug.get(drug_id, set())
        if not similar_known_drugs and not similar_known_diseases:
            return 0.0
        return 0.5 * min(1.0, len(similar_known_drugs) / 20.0) + 0.5 * min(1.0, len(similar_known_diseases) / 20.0)

    def cold_start_uncertainty(self, drug_id: str, disease_id: str) -> float:
        """为低度数药物或疾病赋予更高的冷启动不确定性。"""

        d_deg = int(self.drug_degree.get(drug_id, 0))
        c_deg = int(self.disease_degree.get(disease_id, 0))
        return exp(-min(d_deg, c_deg) / self.config.tau_cold_start)

    def false_negative_risk(self, drug_id: str, disease_id: str) -> dict[str, float]:
        """计算所有风险信号以及最终可靠性分数。"""

        known_drugs_for_disease = self.drugs_by_disease.get(disease_id, set())
        known_diseases_for_drug = self.diseases_by_drug.get(drug_id, set())

        r_drug = max_similarity_to_known(drug_id, known_drugs_for_disease, self.drug_sim)
        r_disease = max_similarity_to_known(disease_id, known_diseases_for_drug, self.disease_sim)
        r_bio = self.biological_path_risk(drug_id, disease_id)
        r_topo = self.topology_risk(drug_id, disease_id)
        r_cs = self.cold_start_uncertainty(drug_id, disease_id)

        w = self.config.weights
        weighted_sum = (
            w.drug_similarity * r_drug
            + w.disease_similarity * r_disease
            + w.biological_path * r_bio
            + w.topology * r_topo
            + w.cold_start_uncertainty * r_cs
        )
        risk = sigmoid(4.0 * (weighted_sum - 0.5))
        reliability = 1.0 - risk
        return {
            "drug_similarity_risk": r_drug,
            "disease_similarity_risk": r_disease,
            "biological_path_risk": r_bio,
            "topology_risk": r_topo,
            "cold_start_uncertainty": r_cs,
            "false_negative_risk": risk,
            "reliability": reliability,
        }

    def candidate_unknown_pairs(self) -> list[tuple[str, str]]:
        """枚举未观测 pair，作为候选负样本。"""

        return [
            (drug_id, disease_id)
            for drug_id in self.drugs
            for disease_id in self.diseases
            if (drug_id, disease_id) not in self.excluded_pairs
        ]

    def score_pairs(self, pairs: Iterable[tuple[str, str]]) -> pd.DataFrame:
        """仅计算指定未知 pair 的风险，并保持调用方给出的顺序。"""

        rows: list[dict[str, str | float]] = []
        for drug_id, disease_id in pairs:
            normalized_drug_id = str(drug_id)
            normalized_disease_id = str(disease_id)
            if (normalized_drug_id, normalized_disease_id) in self.positive_pairs:
                raise ValueError("不能对训练集中已知正样本计算负样本风险。")
            rows.append(
                {
                    "drug_id": normalized_drug_id,
                    "disease_id": normalized_disease_id,
                    **self.false_negative_risk(normalized_drug_id, normalized_disease_id),
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "drug_id",
                "disease_id",
                "drug_similarity_risk",
                "disease_similarity_risk",
                "biological_path_risk",
                "topology_risk",
                "cold_start_uncertainty",
                "false_negative_risk",
                "reliability",
            ],
        )

    def score_unknown_pairs(self) -> pd.DataFrame:
        """为每个候选未知 pair 打分，并划分 easy/medium/hard 难度。"""

        rows = []
        for drug_id, disease_id in self.candidate_unknown_pairs():
            scores = self.false_negative_risk(drug_id, disease_id)
            rows.append({"drug_id": drug_id, "disease_id": disease_id, **scores})
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        try:
            difficulty_codes = pd.qcut(
                df["false_negative_risk"],
                q=3,
                labels=False,
                duplicates="drop",
            )
            df["difficulty"] = difficulty_codes.map({0: "easy", 1: "medium", 2: "hard"}).fillna("easy")
        except ValueError:
            df["difficulty"] = "easy"
        return df.sort_values(["reliability", "false_negative_risk"], ascending=[False, True])

    def sample(self) -> pd.DataFrame:
        """按可靠性加权抽样可靠负样本。"""

        rng = np.random.default_rng(self.config.random_state)
        scored = self.score_unknown_pairs()
        if scored.empty:
            return pd.DataFrame(
                columns=[
                    "drug_id",
                    "disease_id",
                    "drug_similarity_risk",
                    "disease_similarity_risk",
                    "biological_path_risk",
                    "topology_risk",
                    "cold_start_uncertainty",
                    "false_negative_risk",
                    "reliability",
                    "difficulty",
                    "label",
                ]
            )
        reliable = scored[scored["reliability"] >= self.config.reliability_threshold].copy()
        n_pos = len(self.positive_pairs)
        n_neg = min(len(reliable), int(round(n_pos * self.config.negative_ratio)))
        if n_neg == 0:
            return reliable
        probabilities = reliable["reliability"].to_numpy(dtype=float)
        probabilities = probabilities / probabilities.sum()
        chosen = rng.choice(reliable.index.to_numpy(), size=n_neg, replace=False, p=probabilities)
        out = reliable.loc[chosen].copy()
        out["label"] = 0
        return out.sort_values("reliability", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output", default="reliable_negatives.csv")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    sampler = ReliableNegativeSampler(
        associations=pd.read_csv(raw_dir / "drug_disease.csv"),
        drug_similarity=pd.read_csv(raw_dir / "drug_similarity.csv"),
        disease_similarity=pd.read_csv(raw_dir / "disease_similarity.csv"),
        drug_target=pd.read_csv(raw_dir / "drug_target.csv"),
        target_disease=pd.read_csv(raw_dir / "target_disease.csv"),
    )
    negatives = sampler.sample()
    negatives.to_csv(args.output, index=False)
    print(f"Saved {len(negatives)} reliable negative samples to {args.output}")
