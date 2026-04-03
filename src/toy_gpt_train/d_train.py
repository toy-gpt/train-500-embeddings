"""d_train.py - Training loop for the embeddings-based next-token model.

Trains EmbeddingNextTokenModel on a token corpus using a sliding context window.

Responsibilities:
- Create (context_ids -> next_token_id) training pairs
- Run gradient descent updating both embeddings and linear weights
- Track loss and accuracy per epoch
- Write a CSV log of training progress
- Write inspectable training artifacts

Concepts:
- backpropagation through embeddings: gradients flow from the linear layer
  back into the embedding vectors themselves, updating token representations.
- embedding update: each context token's embedding vector is nudged in the
  direction that reduces prediction error — this is how tokens learn to
  occupy meaningful positions in vector space.

Key difference from context-3 (d_train.py in train-400-context-3):
- Context-3 updates a single weight row per training example.
  Only the row for the exact (t-2, t-1, t) trigram is touched.
- Embeddings update the linear weights AND the embedding vectors
  for every context token on every example.
  Unseen token combinations still get gradient signal because
  the embedding vectors are shared across all contexts.
"""

import logging
from pathlib import Path
from typing import Final

from datafun_toolkit.logger import get_logger, log_header, log_path

from toy_gpt_train.c_model import (
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_EMBEDDING_DIM,
    EmbeddingNextTokenModel,
)
from toy_gpt_train.io_artifacts import (
    VocabularyLike,
    find_single_corpus_file,
    write_artifacts,
    write_training_log,
)
from toy_gpt_train.math_training import argmax, cross_entropy_loss, softmax

LOG: logging.Logger = get_logger("TRAIN", level="INFO")

BASE_DIR: Final[Path] = Path(__file__).resolve().parents[2]
OUTPUTS_DIR: Final[Path] = BASE_DIR / "outputs"
TRAIN_LOG_PATH: Final[Path] = OUTPUTS_DIR / "train_log.csv"

type ContextPair = tuple[list[int], int]


def make_training_pairs(
    token_ids: list[int],
    context_size: int,
) -> list[ContextPair]:
    """Convert token IDs into (context_ids, next_id) training pairs.

    Example (context_size=2):
        ids = [3, 1, 2, 4, 5]
        pairs = [([3, 1], 2), ([1, 2], 4), ([2, 4], 5)]

    Args:
        token_ids:    Integer token IDs from the corpus.
        context_size: Number of preceding tokens used as context.

    Returns:
        List of (context_ids, next_id) pairs.
    """
    pairs: list[ContextPair] = []
    for i in range(len(token_ids) - context_size):
        context_ids: list[int] = token_ids[i : i + context_size]
        next_id: int = token_ids[i + context_size]
        pairs.append((context_ids, next_id))
    return pairs


def row_labeler_embeddings(vocab: VocabularyLike, context_size: int):  # type: ignore[return]
    """Map a linear weight row index to a label for artifact inspection.

    Each row of the linear weight matrix corresponds to one input dimension:
        row_idx = token_position * embedding_dim + embedding_dimension

    Label format: 'ctx{position}_dim{dimension}'
    """

    def label(row_idx: int) -> str:
        # row_idx encodes which position in context and which embedding dimension
        # This is for the linear weight matrix rows, not vocab rows
        return f"input_dim_{row_idx}"

    return label


def train_model(
    model: EmbeddingNextTokenModel,
    pairs: list[ContextPair],
    learning_rate: float,
    epochs: int,
) -> list[dict[str, float]]:
    """Train the model with gradient descent on softmax cross-entropy.

    Two parameter groups are updated per example:
    1. Linear weights (W): project context embedding -> next-token scores.
    2. Embeddings (E): one vector per token in the context window.

    Gradient derivation:
        Forward:  h = concat(E[c_0], ..., E[c_{k-1}])
                  scores = h @ W + b
                  probs = softmax(scores)

        Loss gradient w.r.t. scores:
                  d_scores[j] = probs[j] - y[j]   (y = one-hot(target))

        Linear weight gradient:
                  d_W[i][j] = h[i] * d_scores[j]

        Bias gradient:
                  d_b[j] = d_scores[j]

        Embedding gradient for context position p:
                  d_E[c_p][k] = sum_j( W[p*emb_dim + k][j] * d_scores[j] )

    Args:
        model:         The EmbeddingNextTokenModel to train.
        pairs:         Training pairs (context_ids, next_id).
        learning_rate: Step size for gradient updates.
        epochs:        Number of full passes through the training data.

    Returns:
        List of per-epoch metric dictionaries (epoch, avg_loss, accuracy).
    """
    history: list[dict[str, float]] = []
    vocab_size = model.vocab_size
    embedding_dim = model.embedding_dim
    context_size = model.context_size
    LOG.info(
        f"Context size: {context_size}, embedding dim: {embedding_dim}, vocab size: {vocab_size}   "
    )

    for epoch in range(1, epochs + 1):
        total_loss: float = 0.0
        correct: int = 0

        for context_ids, target_id in pairs:
            # === FORWARD PASS ===

            # 1. Look up embedding for each context token.
            context_vecs: list[list[float]] = [
                model.embeddings[tid] for tid in context_ids
            ]

            # 2. Concatenate into a single input vector h.
            h: list[float] = []
            for vec in context_vecs:
                h.extend(vec)

            # 3. Linear layer: scores = h @ W + b.
            scores: list[float] = list(model.bias)
            for i, val in enumerate(h):
                for j in range(vocab_size):
                    scores[j] += val * model.weights[i][j]

            # 4. Softmax -> probabilities.
            probs: list[float] = softmax(scores)

            # === LOSS AND ACCURACY ===
            total_loss += cross_entropy_loss(probs, target_id)
            if argmax(probs) == target_id:
                correct += 1

            # === BACKWARD PASS ===

            # Gradient of loss w.r.t. scores (softmax cross-entropy shortcut).
            d_scores: list[float] = [
                probs[j] - (1.0 if j == target_id else 0.0) for j in range(vocab_size)
            ]

            # Update linear weights: d_W[i][j] = h[i] * d_scores[j]
            for i, h_val in enumerate(h):
                for j in range(vocab_size):
                    model.weights[i][j] -= learning_rate * h_val * d_scores[j]

            # Update bias: d_b[j] = d_scores[j]
            for j in range(vocab_size):
                model.bias[j] -= learning_rate * d_scores[j]

            # Update embeddings: gradient flows back through the linear layer.
            # For context position p, embedding dimension k:
            #   d_E[c_p][k] = sum_j( W[p*emb_dim + k][j] * d_scores[j] )
            for p, token_id in enumerate(context_ids):
                for k in range(embedding_dim):
                    row = p * embedding_dim + k
                    grad_e: float = sum(
                        model.weights[row][j] * d_scores[j] for j in range(vocab_size)
                    )
                    model.embeddings[token_id][k] -= learning_rate * grad_e

        avg_loss = total_loss / len(pairs) if pairs else float("nan")
        accuracy = correct / len(pairs) if pairs else 0.0

        history.append(
            {"epoch": float(epoch), "avg_loss": avg_loss, "accuracy": accuracy}
        )
        LOG.info(
            f"Epoch {epoch}/{epochs} | avg_loss={avg_loss:.6f} | accuracy={accuracy:.3f}"
        )

    return history


def main() -> None:
    """Run embeddings training end-to-end."""
    from toy_gpt_train.a_tokenizer import CORPUS_DIR, SimpleTokenizer
    from toy_gpt_train.b_vocab import Vocabulary

    log_header(LOG, "Training Demo: Next-Token Prediction with Embeddings")
    log_path(LOG, "BASE_DIR", BASE_DIR)
    log_path(LOG, "OUTPUTS_DIR", OUTPUTS_DIR)

    corpus_path: Path = find_single_corpus_file(CORPUS_DIR)

    tokenizer: SimpleTokenizer = SimpleTokenizer(corpus_path=corpus_path)
    tokens: list[str] = tokenizer.get_tokens()

    if len(tokens) < DEFAULT_CONTEXT_SIZE + 1:
        LOG.error(f"Need at least {DEFAULT_CONTEXT_SIZE + 1} tokens for training.")
        return

    vocab: Vocabulary = Vocabulary(tokens)
    vocab_size: int = vocab.vocab_size()

    token_ids: list[int] = []
    for tok in tokens:
        tok_id = vocab.get_token_id(tok)
        if tok_id is None:
            LOG.error("Token not found in vocabulary: %r", tok)
            return
        token_ids.append(tok_id)

    pairs: list[ContextPair] = make_training_pairs(
        token_ids, context_size=DEFAULT_CONTEXT_SIZE
    )
    LOG.info(
        f"Created {len(pairs)} training pairs (context_size={DEFAULT_CONTEXT_SIZE})."
    )

    model: EmbeddingNextTokenModel = EmbeddingNextTokenModel(
        vocab_size=vocab_size,
        embedding_dim=DEFAULT_EMBEDDING_DIM,
        context_size=DEFAULT_CONTEXT_SIZE,
    )

    learning_rate: float = 0.01
    epochs: int = 50

    history = train_model(
        model=model, pairs=pairs, learning_rate=learning_rate, epochs=epochs
    )

    write_training_log(TRAIN_LOG_PATH, history)

    write_artifacts(
        base_dir=BASE_DIR,
        corpus_path=corpus_path,
        vocab=vocab,
        model=model,
        model_kind="embeddings",
        learning_rate=learning_rate,
        epochs=epochs,
        row_labeler=row_labeler_embeddings(vocab, DEFAULT_CONTEXT_SIZE),
    )

    # Qualitative check: top prediction after first context window.
    context_tokens = tokens[:DEFAULT_CONTEXT_SIZE]
    context_ids: list[int] = []
    for tok in context_tokens:
        tid = vocab.get_token_id(tok)
        if tid is None:
            LOG.error("Token not found: %r", tok)
            return
        context_ids.append(tid)

    probs = model.forward(context_ids)
    best_id = argmax(probs)
    best_tok = vocab.get_id_token(best_id)

    LOG.info(
        f"After training, most likely next token after "
        f"{context_tokens} is {best_tok!r} (ID {best_id})."
    )


if __name__ == "__main__":
    main()
