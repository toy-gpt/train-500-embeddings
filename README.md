# train-500-embeddings

[![Docs](https://img.shields.io/badge/docs-live-blue)](https://toy-gpt.github.io/train-500-embeddings/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/MIT)
[![CI](https://github.com/toy-gpt/train-500-embeddings/actions/workflows/ci-shared.yml/badge.svg?branch=main)](https://github.com/toy-gpt/train-500-embeddings/actions/workflows/ci-shared.yml)
[![Deploy-Docs](https://github.com/toy-gpt/train-500-embeddings/actions/workflows/deploy-docs-shared.yml/badge.svg?branch=main)](https://github.com/toy-gpt/train-500-embeddings/actions/workflows/deploy-docs-shared.yml)
[![Check Links](https://github.com/toy-gpt/train-500-embeddings/actions/workflows/links.yml/badge.svg)](https://github.com/toy-gpt/train-500-embeddings/actions/workflows/links.yml)
[![Dependabot](https://img.shields.io/badge/Dependabot-enabled-brightgreen.svg)](https://github.com/toy-gpt/train-500-embeddings/security)

> Demonstrates, at very small scale, how a language model learns vector representations of tokens.

This repository is part of a series of toy training repositories plus a companion client repository:

- [**Training repositories**](https://github.com/toy-gpt) produce pretrained artifacts (vocabulary, weights, metadata).
- A [**web app**](https://toy-gpt.github.io/toy-gpt-chat/) loads the artifacts and provides an interactive prompt.

## What is different about this model (500 vs 400)

Earlier repositories (100-400) predict the next token by
looking up a table indexed by the exact sequence of preceding tokens.
The table has one row per observed context
(unigram: 1 token, bigram: 2 tokens, context-3: 3 tokens).
Any context the model never saw during training gets no signal at all.

This repository introduces **learned embeddings**:
instead of a lookup table indexed by exact token sequences,
the model learns a dense numeric vector for each token.
Prediction is computed from a small linear layer
applied to the concatenated context vectors.

The key consequence:
**token representations are shared across all contexts.**
Two tokens that appear in **similar positions**
will end up with **similar vectors**
even if no identical context pair was seen.
This is the foundation of generalization in neural language models.

## Concepts

<details>
<summary>Token</summary>

A discrete unit of text used as input to the model.
In this project, tokens are words produced by whitespace splitting.
Real language models use subword tokenizers
that can split words into smaller pieces,
but the principle is the same.

</details>

<details>
<summary>Vocabulary</summary>

The complete set of unique tokens observed in the training corpus,
each assigned an integer ID.
The vocabulary size determines
the number of **rows in the embedding matrix** and
the number of **output columns in the weight matrix**.

Example from this repo:
228 tokens in the corpus yields
112 unique tokens in the vocabulary.

</details>

<details>
<summary>Embedding</summary>

A learned **dense numeric vector** assigned to each token.
During training, the embedding vectors are adjusted
so tokens appearing in similar contexts
end up close together in the vector space.

In this repo: each of the
112 vocabulary tokens gets a vector of
16 numbers (`embedding_dim=16`).
The full embedding matrix has shape
`112 × 16` = **1,792 parameters**.

Contrast with context-3:
the context-3 weight table has
`vocab^4 = 112^4` ≈ **157 million**
entries for the same vocabulary,
most of which are zero and never updated.

</details>

<details>
<summary>Context window</summary>

The number of preceding tokens used as input
when predicting the next token.
In this model, `context_size=2`:
the two tokens immediately before the target
are used as context.

</details>

<details>
<summary>Concatenation</summary>

To form a single input vector,
the embedding vectors for all context tokens are
joined end-to-end.
For `context_size=2` and `embedding_dim=16`,
the concatenated vector has length `2 × 16 = 32`.

</details>

<details>
<summary>Linear layer</summary>

A weight matrix that projects the
concatenated context embedding (length 32) to a
score for each vocabulary token (length 112).
Shape: `32 × 112` = **3,584 parameters**.
A **bias vector** of length 112 is also learned.

</details>

<details>
<summary>Softmax</summary>

A function that converts raw scores (any real numbers)
into a probability distribution that sums to 1.0.
The highest score gets the highest probability.
Used as the final step of the forward pass
to produce next-token probabilities.

</details>

<details>
<summary>Cross-entropy loss</summary>

A measure of how **surprised** the model is by the correct next token.
If the model assigns probability 1.0 to the correct token,
loss is 0.
If the model assigns probability close to 0,
loss is very large.
**Training minimizes cross-entropy loss**.

Initial loss ≈ `log(vocab_size) = log(112) ≈ 4.72`,
matching the observed epoch-1 loss of 4.69;
confirming the model starts at near-uniform predictions.

</details>

<details>
<summary>Gradient descent</summary>

An iterative optimization process.
After each training example, the model computes
how much each parameter contributed to the prediction error,
then nudges each parameter slightly in the direction
that reduces error.
The step size is controlled by `learning_rate`.

In this model, gradient descent updates **three** parameter groups per example:

1. Linear weights `W`
2. Bias `b`
3. Embedding vectors for each token in the context window

This is the key departure from the 100-400 models,
where only a single weight row
(the one corresponding to the exact observed context)
was updated.

</details>

<details>
<summary>Backpropagation through embeddings</summary>

Gradient flow from the linear layer back into the embedding vectors.
For context position `p` and embedding dimension `k`:

```
d_E[token_id][k] = sum over vocab( W[p*emb_dim + k][j] * d_scores[j] )
```

This means every training example updates
the embeddings of the tokens that appeared
in the context window,
even for token combinations the model has never seen before.

</details>

<details>
<summary>Epoch</summary>

One complete pass through all training pairs.
With 228 tokens and `context_size=2`,
there are 226 training pairs per epoch.
**After 50 epochs the model has seen each pair 50 times.**

</details>

<details>
<summary>Accuracy (training)</summary>

The fraction of training examples where the model's
top prediction matches the correct next token.
A high training accuracy on a small corpus from an n-gram model
is often a sign of **overfitting** (memorizing exact sequences)
rather than generalization.
Embeddings trade **lower training accuracy**
for **better generalization to unseen contexts**.

</details>

## Training observations (this corpus, 50 epochs)

| Epoch | Avg loss | Accuracy |
| ----: | -------: | -------: | ------------------------------------------------- |
|     1 |    4.695 |    0.142 |
|    10 |    4.355 |    0.150 |
|    31 |    4.067 |    0.150 |
|    32 |    4.034 |    0.177 | ← embeddings differentiate; top predictions shift |
|    50 |    3.661 |    0.181 |

Loss is still declining at epoch 50;
more epochs would continue to improve.
Accuracy is stuck at 0.150 for the first 31 epochs
because the embedding vectors have not yet
differentiated enough to change which token ranks first.
Around epoch 32 they cross a threshold and
several predictions flip, jumping accuracy to ~0.177.

After training: `['data', 'analytics'] -> 'a'`.
Not semantically rich yet:
`'a'` is a high-frequency token in the corpus.
More epochs or a larger corpus
would produce more meaningful predictions.

After training with config.toml:

(seed: `['a', 'model']`):

- Top prediction: `'has'` (0.1277) appears in corpus as "a model has limitations"
- `'variable'` ranks #2 (0.0970) the embedding for `'a'` generalizes across
  all "a \_\_\_" patterns in the corpus, not just "a model"
- Greedy generation loops: `a model has a pipeline has a pipeline...`
  - Expected on a small corpus with greedy decoding.
  - Temperature sampling (introduced in train-600-attention) breaks the loop.

## Parameter count vs earlier models

| Model          |   Vocab |  Parameters | Notes                 |
| -------------- | ------: | ----------: | --------------------- |
| Unigram        |     112 |         112 | one score per token   |
| Bigram         |     112 |      12,544 | vocab²                |
| Context-2      |     112 |   1,404,928 | vocab³                |
| Context-3      |     112 | 157,351,936 | vocab⁴ — mostly zeros |
| **Embeddings** | **112** |   **5,488** | 1,792 + 3,584 + 112   |

The embeddings model uses fewer parameters than
the bigram model while operating with richer representation.

## Artifacts

Training produces the following files under `artifacts/`:

| File                      | Contents                                                                         |
| ------------------------- | -------------------------------------------------------------------------------- |
| `00_meta.json`            | Corpus hash, model kind, training settings, concept glossary                     |
| `01_vocabulary.csv`       | token_id, token, frequency                                                       |
| `02_model_weights.csv`    | Linear layer weights: input_dim rows × vocab_size columns                        |
| `03_token_embeddings.csv` | Learned embedding vectors: one row per vocabulary token, `embedding_dim` columns |

Note: `03_token_embeddings.csv` in this model (500+)
contains **learned embedding vectors**,
one per vocabulary token.
In earlier models (100-400) it
contained a 2D projection of model weights
for visualization only. These are different things.

Training logs are written to `outputs/train_log.csv` (epoch, avg_loss, accuracy).

## Contents

- `corpus/` — declared training corpus (`030_analytics.txt`)
- `src/toy_gpt_train/` — tokenizer, vocabulary, model, training loop, inference, I/O utilities
- `artifacts/` — committed pretrained artifacts for downstream use
- `outputs/` — training logs (not committed)

## Scope

This is an educational, inspectable training pipeline:
a next-token predictor trained on an explicit corpus.
It is not a production system, a full Transformer,
a chat interface, or a claim of semantic understanding.

## Quick start

```shell
uv run python src/toy_gpt_train/d_train.py
```

Run individual pipeline steps:

```shell
uv run python src/toy_gpt_train/a_tokenizer.py
uv run python src/toy_gpt_train/b_vocab.py
uv run python src/toy_gpt_train/c_model.py
uv run python src/toy_gpt_train/d_train.py
uv run python src/toy_gpt_train/e_infer.py
```

<details>
<summary>Command reference</summary>

### In a machine terminal (open in your `Repos` folder)

```shell
# Replace username with YOUR GitHub username.
git clone https://github.com/username/train-500-embeddings
cd train-500-embeddings
code .
```

### In a VS Code terminal

```shell
uv self update
uv python pin 3.14
uv sync --extra dev --extra docs --upgrade

uvx pre-commit install
git add -A
uvx pre-commit run --all-files

uv run python src/toy_gpt_train/a_tokenizer.py
uv run python src/toy_gpt_train/b_vocab.py
uv run python src/toy_gpt_train/c_model.py
uv run python src/toy_gpt_train/d_train.py
uv run python src/toy_gpt_train/e_infer.py

uv run ruff format .
uv run ruff check . --fix
uv run zensical build

git add -A
git commit -m "update"
git push -u origin main
```

</details>

## Provenance and Purpose

The primary corpus used for training is declared in `SE_MANIFEST.toml`.
This repository commits pretrained artifacts so the client can run without retraining.

## Resources

- [Toy GPT organization](https://github.com/toy-gpt) — all training repositories
- [ANNOTATIONS.md](./ANNOTATIONS.md) — REQ/WHY/OBS annotations used
- [SE_MANIFEST.toml](./SE_MANIFEST.toml) — project intent, scope, and declared corpus

## Citation

[CITATION.cff](./CITATION.cff)

## License

[MIT](./LICENSE)
