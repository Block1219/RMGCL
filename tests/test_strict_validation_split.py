"""Regression check for isolation of the inner validation partition.

The splitter is loaded from quick_train.py without importing the neural-network
runtime, so this small data-partition test can run on a CPU-only machine.
"""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


SOURCE = Path(__file__).resolve().parents[1] / "src" / "quick_train.py"
FUNCTIONS = {
    "make_split",
    "make_kfold_splits",
    "positive_examples_from_tables",
    "_pair_set",
    "_empty_negative_frame",
    "negative_sampling_scope_name",
    "ensure_risk_feature_columns",
    "ensure_sample_role_column",
    "sample_evaluation_negatives",
    "make_fold_aware_kfold_splits",
    "use_reserved_validation",
}


def load_split_functions() -> dict[str, object]:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"), filename=str(SOURCE))
    selected = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in FUNCTIONS
    ]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    risk_columns = [
        "reliability", "false_negative_risk", "drug_similarity_risk",
        "disease_similarity_risk", "biological_path_risk", "topology_risk",
        "cold_start_uncertainty",
    ]
    namespace = {
        "np": np,
        "pd": pd,
        "Path": Path,
        "SPLIT_MODES": ("random", "cold-drug", "cold-disease", "double-cold"),
        "RISK_FEATURE_COLUMNS": risk_columns,
        "DEFAULT_RISK_FEATURE_VALUES": {name: 1.0 if name == "reliability" else 0.0 for name in risk_columns},
        "SAMPLE_ROLE_COLUMN": "sample_role",
        "ROLE_POSITIVE": "positive",
        "ROLE_RELIABLE_NEGATIVE": "reliable_negative",
        "ROLE_PU_UNLABELED": "pu_unlabeled",
        "RNS_STRATEGY_THRESHOLD": "threshold",
        "RNS_STRATEGY_ADAPTIVE_TOPK": "adaptive_topk",
        "DEFAULT_NEGATIVE_RATIO": 0.5,
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace


class StrictValidationSplitTest(unittest.TestCase):
    def test_reserved_validation_cannot_overlap_train_pu_or_test(self):
        functions = load_split_functions()
        frame = lambda pairs: pd.DataFrame(
            [(drug, disease, label) for drug, disease, label in pairs],
            columns=["drug_id", "disease_id", "label"],
        )
        train = frame([("d1", "s1", 1)])
        pu = frame([("d2", "s2", 0)])
        test = frame([("d3", "s3", 1)])
        validation = frame([("d4", "s4", 1), ("d5", "s5", 0)])
        selected = functions["use_reserved_validation"](train, pu, test, validation)
        self.assertEqual(functions["_pair_set"](selected), {("d4", "s4"), ("d5", "s5")})
        for forbidden in [train, pu, test]:
            with self.assertRaises(ValueError):
                functions["use_reserved_validation"](train, pu, test, forbidden)

    def test_validation_is_reserved_before_risk_sampling(self):
        functions = load_split_functions()
        positives = [(f"d{i}", f"s{j}") for i in range(6) for j in range(6) if (i + j) % 2 == 0]
        positives += [("d0", "s1"), ("d1", "s0")]
        source = pd.DataFrame(positives, columns=["drug_id", "disease_id"])
        source["label"] = 1
        calls = []

        def record_training_sampler(**kwargs):
            calls.append(("negative", kwargs))
            return functions["_empty_negative_frame"]()

        def record_pu_sampler(**kwargs):
            calls.append(("pu", kwargs))
            return functions["_empty_negative_frame"]()

        functions["sample_fold_train_negatives"] = record_training_sampler
        functions["make_fold_pu_unlabeled_examples"] = record_pu_sampler
        fold_splits = functions["make_fold_aware_kfold_splits"](
            tables={"drug_disease": source},
            folds=10,
            cv_mode="random",
            negative_threshold=0.75,
            seed=42,
            validation_ratio=0.1,
            pu_learning=True,
            rns_strategy="adaptive_topk",
            negative_ratio=1.0,
        )

        self.assertEqual(len(fold_splits), 10)
        self.assertEqual(len(calls), 20)
        test_positive_occurrences = []
        for index, (fold_id, train, validation, test, stats) in enumerate(fold_splits):
            train_pairs = functions["_pair_set"](train)
            validation_pairs = functions["_pair_set"](validation)
            test_pairs = functions["_pair_set"](test)
            self.assertTrue(train_pairs.isdisjoint(validation_pairs))
            self.assertTrue(train_pairs.isdisjoint(test_pairs))
            self.assertTrue(validation_pairs.isdisjoint(test_pairs))
            self.assertEqual(int((validation["label"] == 1).sum()), int((validation["label"] == 0).sum()))
            self.assertEqual(int((test["label"] == 1).sum()), int((test["label"] == 0).sum()))
            self.assertTrue(stats["validation_pair_disjoint_from_train"])
            self.assertTrue(stats["validation_pair_excluded_from_pu"])
            test_positive_occurrences.extend(
                functions["_pair_set"](test.loc[test["label"] == 1])
            )
            for kind, kwargs in calls[2 * index:2 * index + 2]:
                self.assertEqual(
                    functions["_pair_set"](kwargs["train_positives"]),
                    functions["_pair_set"](train.loc[train["label"] == 1]),
                    kind,
                )
                self.assertTrue(validation_pairs <= kwargs["all_positive_pairs"])
                self.assertTrue(test_pairs <= kwargs["all_positive_pairs"])

        self.assertEqual(len(test_positive_occurrences), len(positives))
        self.assertEqual(set(test_positive_occurrences), set(positives))


if __name__ == "__main__":
    unittest.main()
