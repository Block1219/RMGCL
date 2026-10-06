# RMGCL

Code and input data for reliability-guided multi-view graph contrastive learning
on Cdataset, Fdataset and LRSSL.

## Contents

```text
src/                         RMGCL model, sampling, training and data converters
tools/run_paper_strict_cv.py  Ten-fold training entry point
data/*_original/             Source dataset files used by the converters
data/*_raw/                  Model input associations, similarities and biology edges
data/cdataset_disease_semantic_features/  Cdataset disease annotations
tests/                       Cdataset conversion check
```



`*_original` contains the source files used for association and similarity
conversion. `*_raw` contains the tables read by the training code. Cdataset similarities
are generated from the source structural and phenotypic matrices, not the
association-derived GIP matrices. The original positive associations and
biological edges are preserved. Cdataset's disease annotation mapping is in
`data/cdataset_disease_semantic_features/source_mapping.csv`. LRSSL has no
drug-target or target-disease edges in the supplied source, so its biological
edge tables are empty.

## Environment

Use Python 3.11 and install the packages in `requirements.txt`. The paper's
training configuration uses PyTorch 2.11, Adam, 100 epochs and seed 42. CUDA
is recommended for training.

## Rebuild input tables

Run these commands from the archive root:

```bash
python -m src.prepare_cdataset
python -m src.prepare_fdataset
python -m src.prepare_lrssl_dataset --source-dir data/lrssl_original --output-dir data/lrssl_raw
```

The first command regenerates Cdataset's structural/phenotypic similarities.
The second and third regenerate the supplied Fdataset and LRSSL input tables.

## Train

```bash
python tools/run_paper_strict_cv.py --dataset Cdataset --strategy full --output-root outputs/paper_cv --device cuda
```

`--dataset` accepts `Cdataset`, `Fdataset`, `LRSSL` or `all`.
`--strategy` accepts `full`, `wo_cl`, `pair_cl`, `node_cl` or `all`.
The runner uses ten positive folds. Within each outer fold, it reserves test
positives and an equal number of test negatives. It then holds out 10% of the
remaining positives for validation and reserves an equal number of validation
negatives. Training association graphs, reliability-based negative selection
and PU sampling use only the remaining training-core positives; both reserved
partitions are excluded. Validation AUPR selects the epoch, followed by one
test evaluation. Fold-specific validation samples are saved under
`processed/fold_*/validation_samples.csv`. Outputs are written under
`--output-root`. This stricter validation isolation changes the training
samples and may change the reported metrics; rerun experiments before citing
results from this version.

## Check the converter

```bash
python -m unittest discover -s tests -p test_prepare_cdataset.py -v
```
