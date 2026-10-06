from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.prepare_cdataset import convert_cdataset


ROOT = Path(__file__).resolve().parents[1]


class CdatasetConversionTest(unittest.TestCase):
    def test_original_similarity_and_associations(self):
        with TemporaryDirectory() as output_dir:
            summary = convert_cdataset(ROOT / "data/cdataset_original", output_dir)
            output = Path(output_dir)
            drug_similarity = pd.read_csv(output / "drug_similarity.csv")
            disease_similarity = pd.read_csv(output / "disease_similarity.csv")

            self.assertEqual(summary["num_drugs"], 663)
            self.assertEqual(summary["num_diseases"], 409)
            self.assertEqual(summary["num_positive_pairs"], 2532)
            self.assertEqual(summary["num_drug_target_edges"], 3672)
            self.assertEqual(summary["num_target_disease_edges"], 10691)
            self.assertEqual(
                drug_similarity.loc[
                    (drug_similarity.drug_id_1 == "D000")
                    & (drug_similarity.drug_id_2 == "D002"),
                    "score",
                ].iloc[0],
                0.2,
            )
            self.assertEqual(
                disease_similarity.loc[
                    (disease_similarity.disease_id_1 == "C000")
                    & (disease_similarity.disease_id_2 == "C001"),
                    "score",
                ].iloc[0],
                0.103951,
            )


if __name__ == "__main__":
    unittest.main()
