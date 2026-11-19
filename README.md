# General progress substitutes for specific training data in language models

Paper: [arXiv:2611.24844](https://arxiv.org/abs/2611.24844)

Developers withhold data during training to keep language models from learning capabilities that could be
misused. The paper measures how long this delays a capability and how much data the model needs to acquire it
later. Its result is that the progress a model makes on the rest of its training data substitutes for the data
that was withheld.

The code trains decoder-only language models while a chosen fraction of one skill's next-token targets is kept
in the loss (target thinning), returns the withheld targets later, and measures how many tokens the skill then
costs. It covers chess, Othello, SMILES strings, Dyck-4 and web text, in models of 0.76M to 7B parameters.

## Findings

- **Thinning delays a skill by a predictable amount.** At the same background loss, a model trained with a
  fraction *f* of a skill's targets has the skill's log odds lower than the fully supervised model by
  ln[(*f* + τ)/(1 + τ)]. The corpus support τ is what the rest of the corpus contributes, in units of what the
  skill's own targets add. The response measured in small models predicts the delay in larger ones.
- **A never-supervised model holds much of the skill.** Chess models trained on games without a check rank the
  legal replies to check nearly as fully supervised models do.
- **Returned targets cost a small fraction of the data, and less as the model progresses.** The acquisition
  cost follows *P* + 10^(*a* + *b* min(*x*, 0)), with logit distance *x* and progress term *P*. It falls about
  tenfold per logit of surplus in chess and web text and fourfold in SMILES.
- **The saving is specific to the withheld skill.** From the same checkpoints, a permuted rule of equal entropy
  needs 2.4-16 times the tokens of the true rule.
- **Whole domains return in the same way.** A 7B model trained with Rust documents filtered to 3% of their share
  closes half of its gap in Rust term accuracy with 2.1% of the tokens the unfiltered model needed from
  initialization.

## Quick start

The commands below run the whole method on a CPU in about five minutes (Python 3.12, eight threads). They
build a small chess corpus, train a 0.76M-parameter model with none, a tenth and all of the in-check targets,
compare the thinned model with the thinning relation, and return the targets to the never-supervised model.

```
pip install -e .

# corpus: one month of Lichess games (18 MB download), and the label of every target
python -m thinning.domains.chess_data 2013-01 --out data/chess --workers 4
python -m thinning.domains label --domain chess --data data/chess/train --n-tokens 3.4e9 \
    --out data/chess/labels.npy --workers 4

# three models: never supervised (f = 0), thinned (f = 0.1), fully supervised (f = 1)
for f in 0 0.1 1; do
  python -m thinning.train --data data/chess/train --labels data/chess/labels.npy --skill-classes 1 --keep $f \
      --size 0.76M --seed 0 --tokens-per-param 1 --snapshots 0.25,0.5,1 --batch-seqs 8 --out runs/f$f
  python -m thinning.evaluate --domain chess --run runs/f$f --eval-data data/chess/eval --eval-offset 0 \
      --n-eval 100 --out eval/f$f.csv --per-position eval/f$f
done

# derived and measured offsets of the thinned model
python -m thinning.analysis.thinning_relation --full eval/f1 --never eval/f0 --thinned 0.1=eval/f0.1 \
    --grid 5.2:6.2:0.2

# return all in-check targets to the never-supervised model
python -m thinning.returns --preset chess --source runs/f0/snap_000000755328.pt --share 1 \
    --data data/chess/train --labels data/chess/labels.npy --eval-data data/chess/eval --eval-offset 0 \
    --n-eval 100 --batch-seqs 8 --max-tokens 1e6 --out returns/f0-s1.json
```

`eval/f*.csv` holds the competence and the background loss of every snapshot, and `returns/f0-s1.json` the
same at every evaluation after the return. The thinning-relation command prints, at each background loss, the
corpus support and the offset of the thinned model, derived from the other two models and measured (the last
digits vary between machines):

```
f background_loss tau derived measured
0.1 5.200 0.5754 -0.6443 -0.7148
0.1 5.400 0.9843 -0.4974 -0.5269
0.1 5.600 1.644 -0.3523 -0.3562
0.1 5.800 3.188 -0.2052 -0.1951
0.1 6.000 3.991 -0.1764 -0.1710
0.1 6.200 4.701 -0.1569 -0.1543
r.m.s. error 0.0316 logits over 6 cells
```

Models of this size and length show the mechanics, not the paper's numbers; the settings of the paper follow
each command under [Experiments](#experiments).

## Contents

- [Layout](#layout), [Installation](#installation), [Terms](#terms)
- [Data](#data): building each corpus
- [Experiments](#experiments): one section per experiment, each with a small command and the paper's settings
- [Commands behind the main figures](#commands-behind-the-main-figures)
- [Not included](#not-included), [Equivalence with the paper's runs](#equivalence-with-the-papers-runs),
  [Differences that a rerun will meet](#differences-that-a-rerun-will-meet)
- [Citation](#citation), [Licence](#licence)

## Layout

```
src/thinning/
  sizes.py, model.py        model sizes by parameter count; CompleteP decoder, initialization, AdamW groups
  data.py, masks.py         token streams and the fixed-order row reader; position hash, thinning, permuted rule
  train.py                  training with target thinning
  evaluate.py               competence and background loss of every snapshot of a run
  returns.py                returned targets (presets per domain), runs from initialization
  removal.py                withdrawn targets, gradient ascent, misdirection; returns after each
  domains/                  chess (+ chess_data builder), othello, smiles, dyck: corpora, labels, held-out measures
  webtext/fineweb.py        FineWeb-Edu corpus, agreement labels and measure (models up to 163M)
  webtext/                  Dolma models of 20M-7B: dolma, labels, entities, evalsets, train, returns, docfilter
  analysis/                 trajectories, thinning_relation, cost, cost_relation
tests/                      CPU tests (under a minute)
```

The repository holds code only. Every corpus is rebuilt from public sources or generated from a seed with the
commands below; trained models, evaluation outputs and figures are not included.

## Installation

Python 3.12. The pinned versions are those of the paper's runs.

```
pip install -e .            # torch 2.12.0, numpy, scipy, numba, chess, zstandard, pyarrow, tokenizers
pip install -e ".[webtext]" # adds transformers, safetensors and spaCy with en_core_web_sm (web text)
pip install -e ".[test]" && pytest         # WEBTEXT_TOKENIZER=tokenizer.json enables the tokenizer tests
```

Install the PyTorch build for your CUDA version first if the default wheel does not match it. numba writes its
compilation cache next to the installed sources unless `NUMBA_CACHE_DIR` is set.

Every command runs on a CPU with small settings. On a GPU the models use variable-length attention and bfloat16
autocast, as in the paper; the tests and the checks behind this release exercise the CPU path only (dense
attention mask, float32).

## Terms

| Term | Meaning |
|---|---|
| target thinning | the next-token loss at a skill's targets is kept iff `hash(stream position) < f`; inputs are never changed |
| retained fraction *f* | the fraction of the skill's targets a model is trained with |
| competence | the skill's score on held-out documents (chess: legal / (legal + king-unsafe) probability mass at in-check boards) |
| background loss | held-out cross-entropy, in nats, at the targets outside the skill |
| source model | a model trained with a retained fraction (0 for a never-supervised source) to which the targets are later returned |
| returned share *s* | the fraction of the skill's targets in the loss after the return |
| acquisition level, acquisition cost | the score a return must reach, and the tokens it needs, divided by the tokens a same-size model needs from initialization |
| acquisition loss *L\*(s)* | background loss at which a model trained with share *s* from the start reaches the acquisition level |
| logit distance *x* | log odds of the acquisition level minus the log odds of the same-size model trained with share *s*, read at the background loss of the return |
| permuted rule | the returned chess targets are replaced by their images under a fixed permutation of the move tokens |
| corpus support τ | the weight of the rest of the corpus in the thinning relation, offset(*f*) = κ ln[(*f* + τ)/(1 + τ)] |

## Data

A corpus is a directory with `train/` and `eval/` token streams (`meta.json` and `shard_*.bin`). A training
stream is read once, in order, from row 0, so the batch schedule fixes the data order and a stream position
identifies one prediction in every run. Label arrays give the skill class of every position of a training stream.

**Chess.** Rated Lichess games (database.lichess.org, CC0), one move per token. The paper uses the months 2024-01
to 2024-08 (about 240 GB of downloads; 299,144,492 training games, 20.4B tokens).

```
python -m thinning.domains.chess_data 2024-01 2024-02 2024-03 2024-04 2024-05 2024-06 2024-07 2024-08 \
    --out data/chess --workers 32
python -m thinning.domains label --domain chess --data data/chess/train --n-tokens 3.4e9 \
    --out data/chess/labels.npy --workers 32
```

The split and the order of the games depend only on the set of games (keyed hashes of the game id), so the same
months give the same streams. For a small corpus use one early month, for example `2013-01` (18 MB, 44,184
training games); `--n-tokens` larger than the stream labels all of it.

**Othello and Dyck-4.** Generated from a seed; game or document *i* is a function of (seed, split, *i*).

```
python -m thinning.domains.othello build --out data/othello --n-train 700000000 --n-eval 1000000 --seed 0
python -m thinning.domains.dyck build --out data/dyck --n-train 40000000 --n-eval 50000 --seed 0
python -m thinning.domains label --domain othello --data data/othello/train --n-tokens 2.7e9 --out data/othello/labels.npy
python -m thinning.domains label --domain dyck --data data/dyck/train --n-tokens 1e9 --out data/dyck/labels.npy
```

**SMILES.** The Hugging Face dataset `chandar-lab/ZINC_22` (revision `f13c8b41`, pinned in the code), split
`train`, column `SMILES`, in file order: the first 300,000,000 molecules are the training stream and the last
3,000,000 of the split the held-out stream. The dataset card states no licence; ZINC-22 is the source. The two
commands fetch about 250 of the 1,212 parquet files, those that hold these molecules (about 50 GB).

```
python -m thinning.domains.smiles download --out raw/zinc22
python -m thinning.domains.smiles build --parquet raw/zinc22 --out data/smiles
python -m thinning.domains label --domain smiles --data data/smiles/train --n-tokens 2.1e9 --out data/smiles/labels.npy
```

`build --smiles FILE ...` builds a corpus from any text files with one canonical SMILES string per line.

**FineWeb-Edu** (web-text models up to 163M). Five parquet files of `HuggingFaceFW/fineweb-edu` (8.6 GB) and the
tokenizer of `EleutherAI/pythia-14m`, both at pinned revisions. `prepare` tokenizes, labels number-marked verbs
with spaCy and packs rows of 2,049 tokens; it writes `labels.npy` next to each stream.

```
python -m thinning.webtext.fineweb download --out raw/fineweb
python -m thinning.webtext.fineweb prepare --raw raw/fineweb --tokenizer raw/fineweb/tokenizer.json --out data/fineweb
```

**Dolma 1.7** (web-text models of 20M-7B). The tokenized files of `allenai/DataDecide-data-recipes` at a pinned
revision, read with HTTP range requests (about 50 GB staged for the three pools).

```
python -m thinning.webtext.dolma stage --mixes named_data_mixes.py --out data/dolma
python -m thinning.webtext.evalsets build --data data/dolma --tokenizer tokenizer.json --out data/dolma-eval
```

`--mixes` is `olmo/data/named_data_mixes.py` from the `DataDecide` branch of github.com/allenai/OLMo, which
lists the files of the 15 sources; `tokenizer.json` is that of `EleutherAI/pythia-14m`. Staging is
deterministic: with this file, the file sizes of the pinned revision and the seeds in the code, the plan
reproduces every chunk of the paper's three pools (held-out 33.5M tokens, training 6.6B, extra 18B), and a seed
then gives the data order of the paper's run with that seed. The plan and the data orders match those stored for
the paper's runs.

## Experiments

Each section gives a small command for a CPU and then the settings of the paper. The small commands assume the
corpus and the runs of the [quick start](#quick-start) and take minutes; the web-text commands at 20M-7B need
the staged Dolma pools and take longer, and document filtering has no small command.

### Training with target thinning

```
python -m thinning.train --data data/chess/train --labels data/chess/labels.npy --skill-classes 1 --keep 0.1 \
    --size 0.76M --seed 0 --tokens-per-param 1 --snapshots 0.25,0.5,1 --batch-seqs 8 --out runs/f0.1
```

`--keep f` is the retained fraction (0: never supervised, 1: fully supervised); `--salt` selects an independent
hash. Paper: sizes 0.76M to 163M (chess to 3.0B) with each size's batch, sequences of 2,048 tokens, AdamW with
peak learning rate 0.004 and weight decay 0.1, a warm-up over 2 tokens per parameter with a batch ramp over 16,
then a constant rate (the defaults). The constant-rate chess runs add `--warmup-tokens 15e6
--ramp-tokens-per-param 0 --batch-seqs 64`; FineWeb-Edu runs use `--lr 0.008 --skill-classes 1,2,3
--half-snapshots`. Skill classes: chess in-check `1`; Othello forced pass `2`; SMILES ring closure `2`; Dyck
closing brackets at match distance 2-7 `2,6`. Two variants of the loss are options: `--target-weight w` keeps
every target and weights the skill's targets by *w*, and `--eligible-denominator` divides by the number of
targets before thinning. `--drop-random p` removes a fraction of all targets whatever their class.

### Competence, background loss and output classes

```
python -m thinning.evaluate --domain chess --run runs/f0.1 --eval-data data/chess/eval --eval-offset 0 \
    --n-eval 100 --out eval/f0.1.csv --per-position eval/f0.1 --output-classes
```

Paper: 1,000 held-out chess games from token offset 80,000,000 of the held-out stream (the default), 8,000
Othello games, 8,000 molecules, 2,000 Dyck rows, 384 FineWeb-Edu sequences. `--output-classes` gives the mean
probability of each class of output token at the skill's positions (chess and SMILES).

### Thinning relation

```
python -m thinning.analysis.thinning_relation --full eval/f1 --never eval/f0 --thinned 0.1=eval/f0.1 \
    --grid 5.2:6.2:0.2
```

reads the per-position files of a fully supervised, a never-supervised and any thinned runs (trained and
evaluated as above with `--keep 1`, `0` and `0.1`) and prints the derived and the measured offsets on a grid of
background loss. The grid of the paper's chess models is `1.85:2.80:0.05`. `--interpolate logit` is the form
for the domains whose competence is a mean over positions (Othello, SMILES, Dyck, web text).
`analysis.trajectories` gives onsets and offsets at matched background loss, `fit_relation` the pooled fit of
κ and τ.

### Returned targets and acquisition cost

```
python -m thinning.returns --preset chess --source runs/f0/snap_000000755328.pt --share 1 \
    --data data/chess/train --labels data/chess/labels.npy --eval-data data/chess/eval --eval-offset 0 \
    --n-eval 100 --batch-seqs 8 --max-tokens 1e6 --out returns/f0-s1.json
```

The presets hold the paper's settings; the small command overrides the batch and the held-out set.

| Preset | Sequences per step | Before the return | At the return | Held-out set | Stops at |
|---|---|---|---|---|---|
| `chess` | 64 | 20 moment steps, 30 warm-up steps | full rate | 1,000 games | 0.80 twice |
| `chess-warmup-at-return` | 64 | 20 moment steps | 20-step warm-up | 1,000 games | 0.80 twice |
| `smiles` | 16 | 20 moment steps | 20-step warm-up | 8,000 molecules | 0.75 twice |
| `othello` | 64 | 20 moment steps | 20-step warm-up | 2,000 games | 0.6 twice |
| `webtext` (FineWeb-Edu) | 64 | 20 moment steps on fixed rows, first moment reset, 30 steps at the source rate | full rate | 384 sequences | 0.90, or twice the steps of the first 0.85 |

`--rule permuted` returns the chess targets under the permuted rule and scores it. `--size S --seed N` in
place of `--source` trains from initialization with the same loop, which gives the denominator of the cost.
`--lr`, `--batch-seqs`, `--warmup-before` and `--warmup-at` vary the return.

### Cost relation

`thinning.analysis.cost` turns a return into a row: `acquisition_tokens`, `logit_distance` against the reference
trajectory (the same-size model trained with share *s*), `progress_term` from the source run's loss curve, and
`smoothed_crossing` for the 7B runs. `thinning.analysis.cost_relation` fits a table of such rows:

```
python -m thinning.analysis.cost_relation table.csv   # columns domain, source, x, P, cost, censored, n_params
```

The fit of the paper takes the returns of chess, web text and SMILES that reached the acquisition level (312
returns of 27 source runs); rows marked `censored` are left out.

The cost relation of the paper reads the acquisition loss *L\*(s)*, which sets the progress term and separates
returns below the acquisition loss from returns above it, on one curve per domain fitted across reference runs
(`fit_share_curve` over returned shares and sizes for chess and over returned shares at 9.7M for SMILES,
`fit_size_curve` over sizes for web text). `acquisition_loss` reads it on the same-size reference instead; the two
readings agree within 0.04 nats in chess and SMILES and within 0.08 nats in web text.

### Removal routes

```
python -m thinning.removal --route gradient-ascent --source runs/f1/snap_000000755328.pt --batch-seqs 8 \
    --max-rows 8 --data data/chess/train --labels data/chess/labels.npy --eval-data data/chess/eval \
    --eval-offset 0 --n-eval 100 --removal-max-steps 20 --shares 1,0 --budgets 3e5,1e5 --out removal/ga.json
```

Routes: `withdrawn` (`--withdraw-tokens 8e6` or `64e6`), `gradient-ascent` (weight 0.3, learning rate 1e-4),
`misdirection` (block at 0.7 of the depth, direction scale 4, weight 100, learning rate 3e-4) and `none` for the
never-supervised model of a case. Paper: 64 sequences per step (SMILES: `--domain smiles --batch-seqs 16 --n-eval
8000`), `--competence-target` set to the competence of the never-supervised model of the case, shares 1, 0.1
and 0.

### Web text at 20M-7B

The models of 20M to 7B parameters follow the Dolma 1.7 recipe of DataDecide and have their own trainer:
Llama-architecture models from `transformers`, AdamW with weight decay 0.05, a 32-step warm-up, batches of 128
rows of 2,048 tokens and a constant learning rate. Here the seed sets the initialization and the data order.
`--keep` gives the retained fraction of each structural class (closing brackets `close`, number-marked
auxiliaries `agr`, counting sequences `succ`, repeated 4-gram completions `copy`) and `--fkeep` that of the two
factual relations (country capitals `cap`, city countries `ctry`), whose answer mentions are kept or removed
whole.

```
python -m thinning.webtext.train --size 2x64x2 --seed 0 --keep agr=0,copy=0.3 --fkeep cap=0 --batch 8 --micro 4 \
    --cap-tokens 5e5 --horizon-tokens 1e7 --save-every-log2 1 \
    --data data/dolma --eval data/dolma-eval --tokenizer tokenizer.json --out runs/web-small
python -m thinning.webtext.returns --size 2x64x2 --batch 8 --micro 4 --init-from runs/web-small/step16 \
    --src-keep agr=0,copy=0.3 --src-fkeep cap=0 --keep agr=0,copy=0.3 --fkeep cap=1 --mom-steps 4 \
    --prewarm-steps 4 --cap-tokens 2e5 --data data/dolma --eval data/dolma-eval --tokenizer tokenizer.json \
    --out returns/web-small
```

On a CPU one evaluation of the full held-out sets takes about two minutes and a run evaluates at many steps, so
these two commands take about an hour; without `--eval` a run trains without evaluating. The small return
shortens the moment estimation and the warm-up to 4 steps each (the paper's 20 and 30 are the defaults), so that
9 of its 13 steps train with the capital targets returned. The small commands need the held-out and the
training pool (13 GB).

Paper: sizes 20M, 60M, 150M, 300M (learning rate 2.12e-3, the default), 1B (`--lr 0.00106`) and 7B
(`--lr 0.000265`). The 7B source model of the returns is

```
torchrun --nproc_per_node 4 -m thinning.webtext.train --size 7B --lr 0.000265 --horizon-tokens 1.2e10 \
    --micro 8 --grad-ckpt --keep agr=0,copy=0 --fkeep cap=0 --cap-tokens 3359899648 --save-every-log2 2 ...
```

and its reference keeps the capital targets (`--fkeep cap=1`). A return (`thinning.webtext.returns --init-from
RUN/step<S>`) estimates the AdamW moments over 20 steps at learning rate 0 on the 20 batches before the
snapshot, warms up over 30 steps under the source model's masks and then applies the masks of the return.
`--fact-rule cap=perm` is the permuted assignment of the capitals and `cap=first` its control, which supervises
the same positions with the true first token. `thinning.webtext.evalsets score-capitals` writes the per-prompt
masses of a snapshot for `thinning.analysis.thinning_relation --interpolate logit`.

### Document filtering in web text

Three domains of the Dolma stream are thinned by removing whole documents instead of targets: astronomy, Rust
and Chinese, which hold 0.53%, 0.22% and 0.15% of the stream's tokens by their reference labels (functions of the
decoded text: lexicon density, Rust-only syntax, share of CJK characters). The filter is one linear classifier
per domain over the hashed unigrams and bigrams of a document's first 512 tokens, fitted to the reference labels
of documents that the runs do not read. A share *f* of a domain removes every document that scores at or above
the largest threshold that leaves at most *f* of the domain's reference-labelled tokens in the stream; the
remaining tokens are packed into rows of 2,048 and each keeps its position in the unfiltered stream.

```
python -m thinning.webtext.docfilter index --data data/dolma --tokenizer tokenizer.json --out data/docfilter --workers 32
python -m thinning.webtext.docfilter fit   --data data/dolma --out data/docfilter
python -m thinning.webtext.docfilter terms --data data/dolma --tokenizer tokenizer.json --out data/docfilter
python -m thinning.webtext.train --size 20M --seed 0 --filter astro=1,rust=0.1,zh=0.01 \
    --filter-dir data/docfilter --terms data/docfilter/terms.npz --cap-tokens 1e9 --state-every 500 \
    --data data/dolma --eval data/dolma-eval --tokenizer tokenizer.json --out runs/20M-s0-r0
python -m thinning.webtext.returns --size 20M --resume-from runs/20M-s0-r0/state.pt \
    --order runs/20M-s0-r0/order.npy --data-offset B --terms data/docfilter/terms.npz --cap-tokens 2.5e8 \
    --data data/dolma --eval data/dolma-eval --tokenizer tokenizer.json --out returns/20M-s0-r0
```

`index` decodes all 36.4M documents of the training and extra pools and labels them (CPU-hours); `fit` trains
the three classifiers (14 minutes on four CPU threads; held-out average precision 0.877, 0.980 and 0.980),
scores every document and writes the thresholds; `terms` writes 96 rows per domain of unread reference-labelled
documents with the targets that require the domain: tokens of astronomy terms, Rust idiom tokens, and tokens at
least 20 times as frequent in Chinese as in English documents. With `--terms`, eval.jsonl gains
`competence_astronomy`, `competence_rust` and `competence_chinese`, the top-1 accuracy at those targets. The
return continues the filtered model on the unfiltered pools from batch *B* of its data order, with *B* at least
the last `rows_read` of train.jsonl divided by the batch size, so that it reads no row twice. There is no small
command for this section: the index, the classifiers and the term rows are built from the staged pools.

Paper: models of 20M, 60M, 150M and 300M parameters (1.0, 1.5, 2.0 and 3.0B tokens), six runs per size and
seed (0 and 1) that form a Latin square over the shares 1, 0.3, 0.1, 0.03, 0.01 and 0: run *r* = 0..5 gives
astronomy the *r*-th share of that list, Rust the share two places on (four with seed 1) and Chinese the share
four places on (two with seed 1), cyclically. The 1B model has all three domains at 0.03 (`--lr 0.00106`,
4.5B tokens) and the 7B model Rust at 0.03 (`--filter rust=0.03 --fkeep ctry=0.1 --lr 0.000265
--horizon-tokens 1.2e10`, return from `--data-offset 15872`); each has an unfiltered reference with the same
seed. Three details of the filter:

- Below the lowest share that the classifier sets alone (astronomy 0.1, Rust 0.03, Chinese 0.01), a run removes
  what that threshold removes and also every reference-labelled document, except a hashed fraction *f* of them;
  at *f* = 0 none is kept.
- Thresholds are stored rounded to six decimals, as the runs received them.
- The documents of terms.npz lie outside the rows that the data orders of seeds 0 and 1 read within 4.7B tokens
  when drawn for the default `--horizon-tokens 8e9`. A data order drawn for another horizon is a different order
  and reads some of them: the 7B order (`1.2e10`) reads rows that hold 55 of the 290 Rust documents of terms.npz
  within its first 17,780 batches (52 within the first 15,872).

## Commands behind the main figures

| Figure | Content | Commands |
|---|---|---|
| 1 | delay from thinning; derived thinning relation | `train` at several `--keep`, `evaluate --per-position`, `analysis.thinning_relation`, `analysis.trajectories` |
| 2 | output classes of a never-supervised model; what returned targets add | `evaluate --output-classes`, `returns` |
| 3 | returns of withheld skills; permuted rule; returns at 7B | `returns` (all presets, `--rule permuted`, from initialization), `webtext.train`, `webtext.returns` |
| 4 | cost relation | `analysis.cost`, `analysis.cost_relation` |
| 5 | removal routes | `removal` |

## Not included

- Data, trained models, evaluation outputs, figures and figure code.
- The following analyses of the paper: Rust function tasks, the representation edit, related king-safety
  positions, earlier supervision, transplanted weight changes, restricted returns and stitched models, the
  prediction of the web-text onset shift, the chess game filter, removal routes on web text, probes, and the
  two controls of document filtering (thinned domain targets in place of removed documents; a random residual
  of reference-labelled documents at *f* = 0.1).
- The compute. A 7B model is trained with four GPU processes under `torchrun`, 8 rows per forward pass and
  activation checkpointing in every layer; the source model of the 7B returns alone reads 3.4B tokens. The
  trainer of the other domains runs on one GPU.

## Equivalence with the paper's runs

Each part of the release was compared with the code of the paper's runs on a CPU, on small inputs and with the
same seeds:

- model, initialization and optimizer groups: equal weights and logits; row reader and position hashes: equal;
- training with thinning, loss weights, the other denominator and random removal: bit-identical weights after
  39 steps of a 0.76M model; FineWeb-Edu training: bit-identical bfloat16 snapshot;
- returns (both warm-up conditions, permuted rule, from initialization, FineWeb-Edu protocol) and the three
  removal routes with their returns: background loss equal and competence equal to 1e-7 at every evaluation;
- corpora: chess streams byte-identical for the month 2013-01; Dyck-4 byte-identical for the paper's corpus;
  Othello identical for the first 100,000 games of the paper's corpus and for a 20M-game build; SMILES identical
  for the first 1.2M molecules apart from the excluded molecule noted below; FineWeb-Edu held-out stream and
  the first rows of every training file identical; Dolma plan and data orders identical to the stored ones;
- labels and held-out measures of every domain: equal to the paper's label arrays and scorers;
- document filter: reference labels equal on the stored text features of all 36.4M documents and on 30,000
  decoded documents; classifier weights bit-identical after a full refit on a CPU; thresholds equal at six
  decimals; removed documents equal for every run of the paper, and the packed stream equal on 24 batches of
  five of them; term rows and term accuracy equal;
- analysis: the cost relation (single line, rates per domain, ordering near the floor), the derived thinning
  relation (0.039 and 0.089 logits in chess, 0.130 and 0.142 at 7B) and the 7B crossing rule reproduce the
  paper's numbers from its stored evaluations.

Not done: no corpus was rebuilt at full size, nothing was downloaded at full size, and no model of the paper
was retrained. Nothing was run on a GPU, so the variable-length attention path, bfloat16 autocast,
`torch.compile` and multi-GPU training are ported but unexecuted; numbers computed in float32 on a CPU differ
from the paper's bfloat16 evaluations in about the fourth decimal.

## Differences that a rerun will meet

- **SMILES stream.** The paper's training stream lacks 830 of the first 300,000,000 molecules, which a
  round-trip check through another string format excluded; the release keeps them. The first is row 419,362, so
  later stream positions, and with them the thinned sets, are shifted relative to the paper's. The paper's
  held-out stream lacks a further 39 molecules; the first 8,000 held-out molecules differ from the paper's in one
  molecule.
- **Schedule.** Only the warm-up and the constant phase are implemented, which is what the thinning, return and
  removal runs use.
- **Source loss curve of the progress term.** `LossCurve` takes the source run's evaluated snapshots;
  `extend_with_training_loss` adds its training loss past the last snapshot, as in the paper. Assembling the table
  of runs (which reference serves which return) is left to the reader.
- **Permuted-rule returns.** The moment-estimation steps use the batch of the return; in the paper's
  permuted-rule returns they used the batch of the source run, and every step up to step 20 was evaluated
  (`--eval-every-step-until 20`).
- **FineWeb-Edu returns inside the source warm-up** continue the schedule from the tokens the snapshot was
  trained on; the paper's runs read the nominal token count of the snapshot, less than one batch away.
- **7B initialization** is drawn on the training device and is reproducible on CUDA only.
- **Document-filtering runs.** The paper's runs at 20M-300M were evaluated on another grid of steps (spaced by
  2^(1/4) from step 16, and every 400 steps), and the background loss of the document-filtering runs left out
  the held-out targets in documents that resemble the three domains (about 2.6% of the targets); the release
  reports the background loss of the other web-text runs. The loss on the domain rows is not computed.
- **Statistics.** Bootstrap intervals and likelihood-ratio tests are not included; `fit_mixed` returns the
  negative log-likelihood from which the tests follow.

## Citation

```
@misc{marosi2026general,
  title         = {General progress substitutes for specific training data in language models},
  author        = {Marosi, Mark and Antal, Peter},
  year          = {2026},
  eprint        = {2611.24844},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2611.24844}
}
```

## Licence

The code is released under the MIT licence (see `LICENSE`).
