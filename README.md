# RMGCL

Code and input data for reliability-guided multi-view graph contrastive learning
on Cdataset, Fdataset, and LRSSL.

## Repository layout

```text
src/                         Model, sampling, training, and data converters
tools/run_paper_strict_cv.py  Ten-fold experiment entry point
data/*_original/             Supplied source dataset files
data/*_raw/                  Model-ready input tables
data/cdataset_disease_semantic_features/  Cdataset disease annotations
tests/                       Data conversion and validation split checks
requirements.txt             Python dependencies
```

## Input data

`*_original` contains the source files used for association and similarity
conversion. `*_raw` contains the tables read by the training code. Cdataset similarities
are generated from the source structural and phenotypic matrices, not the
association-derived GIP matrices. The original positive associations and
biological edges are preserved. Cdataset's disease annotation mapping is in
`data/cdataset_disease_semantic_features/source_mapping.csv`. 

Dataset files may have their own citation and reuse requirements. Check the
original dataset publications before reusing them outside this experiment.

## Setup

Python 3.11 is recommended. From the repository root, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

CUDA is recommended for full training. Use `--device cpu` when CUDA is unavailable.

## Rebuild input tables

Run these commands from the repository root:

```bash
python -m src.prepare_cdataset
python -m src.prepare_fdataset
python -m src.prepare_lrssl_dataset --source-dir data/lrssl_original --output-dir data/lrssl_raw
```

The first command regenerates Cdataset's structural/phenotypic similarities.
The second and third regenerate the supplied Fdataset and LRSSL input tables.

## Ten-fold experiment

```bash
python tools/run_paper_strict_cv.py --dataset Cdataset --strategy full --output-root outputs/paper_cv --device cuda
```

`--dataset` accepts `Cdataset`, `Fdataset`, `LRSSL` or `all`.
`--strategy` accepts `full`, `wo_cl`, `pair_cl`, `node_cl` or `all`.
The default paper configuration uses 100 epochs and seed 42. Each
dataset/strategy run writes its configuration and metrics under
`--output-root/<dataset>/<strategy>/`. The runner refuses to overwrite a
nonempty experiment directory.

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

## Checks

Run the data conversion and validation split checks with:

```bash
python -m unittest discover -s tests -v
```
