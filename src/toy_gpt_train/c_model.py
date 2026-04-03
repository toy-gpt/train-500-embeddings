"""c_model.py - Embeddings-based next-token prediction model.

Defines a next-token prediction model that uses learned vector representations
(embeddings) rather than direct lookup tables indexed by token ID tuples.

Concepts:
- embedding: a dense numeric vector assigned to each token, learned during training.
  Related tokens end up with similar vectors after training.
- embedding matrix: a table of shape (vocab_size x embedding_dim) storing one
  vector per token.
- context window: the number of preceding tokens used as input.
- linear layer: a weight matrix that projects the concatenated context embeddings
  (shape: context_size * embedding_dim) to a score for each vocabulary token.

Architecture:
    token IDs -> embedding lookup -> concatenate -> linear -> softmax -> probs

This is the key departure from context-3:
- Context-3 uses a direct lookup: (prev2, prev1, current) -> next scores.
  The table has vocab^3 rows; unseen trigrams get no signal.
- Embeddings use two learned tables: one maps tokens to vectors,
  one maps concatenated vectors to next-token scores.
  Similar tokens share similar vectors; unseen combinations generalize.

Training is handled in d_train.py.
"""

import logging
import random
from typing import Final

from datafun_toolkit.logger import get_logger, log_header

from toy_gpt_train.math_training import softmax

LOG: logging.Logger = get_logger("MODEL", level="INFO")

DEFAULT_EMBEDDING_DIM: Final[int] = 16
DEFAULT_CONTEXT_SIZE: Final[int] = 2


class EmbeddingNextTokenModel:
    """Next-token prediction model using learned token embeddings.

    Parameters:
        vocab_size:     Number of unique tokens in the vocabulary.
        embedding_dim:  Size of each token's embedding vector.
        context_size:   Number of preceding tokens used as context.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        context_size: int = DEFAULT_CONTEXT_SIZE,
    ) -> None:
        """Initialize model parameters with small random values."""
        self.vocab_size: Final[int] = vocab_size
        self.embedding_dim: Final[int] = embedding_dim
        self.context_size: Final[int] = context_size

        # Embedding matrix: one row per token, each row is a vector of length embedding_dim.
        # Shape: vocab_size x embedding_dim
        self.embeddings: list[list[float]] = [
            [random.gauss(0.0, 0.01) for _ in range(embedding_dim)]
            for _ in range(vocab_size)
        ]

        # Linear (output) layer: projects concatenated context embeddings to vocab scores.
        # Input size: context_size * embedding_dim
        # Output size: vocab_size
        input_dim = context_size * embedding_dim
        self.weights: list[list[float]] = [
            [random.gauss(0.0, 0.01) for _ in range(vocab_size)]
            for _ in range(input_dim)
        ]

        # Bias: one value per output token.
        self.bias: list[float] = [0.0] * vocab_size

        LOG.info(
            f"EmbeddingNextTokenModel initialized: "
            f"vocab_size={vocab_size}, "
            f"embedding_dim={embedding_dim}, "
            f"context_size={context_size}."
        )
        LOG.info(
            f"Parameters: "
            f"embeddings={vocab_size * embedding_dim:,}, "
            f"weights={input_dim * vocab_size:,}, "
            f"bias={vocab_size} "
            f"(total={vocab_size * embedding_dim + input_dim * vocab_size + vocab_size:,})."
        )

    def _lookup_embedding(self, token_id: int) -> list[float]:
        """Return the embedding vector for a token ID."""
        return self.embeddings[token_id]

    def _concatenate(self, context_ids: list[int]) -> list[float]:
        """Concatenate embedding vectors for a list of context token IDs.

        Args:
            context_ids: Token IDs for the context window.

        Returns:
            A single flat vector of length context_size * embedding_dim.
        """
        result: list[float] = []
        for token_id in context_ids:
            result.extend(self._lookup_embedding(token_id))
        return result

    def _linear(self, input_vec: list[float]) -> list[float]:
        """Apply the linear layer: input_vec @ weights + bias.

        Args:
            input_vec: Concatenated context embedding vector.

        Returns:
            Raw scores for each token in the vocabulary.
        """
        scores: list[float] = list(self.bias)
        for i, val in enumerate(input_vec):
            for j in range(self.vocab_size):
                scores[j] += val * self.weights[i][j]
        return scores

    def forward(self, context_ids: list[int]) -> list[float]:
        """Compute next-token probabilities from context token IDs.

        Args:
            context_ids: List of token IDs of length context_size.

        Returns:
            Probability distribution over the vocabulary.

        Raises:
            ValueError: If context_ids length does not match context_size.
        """
        if len(context_ids) != self.context_size:
            msg = f"Expected {self.context_size} context token IDs, got {len(context_ids)}."
            raise ValueError(msg)

        context_vec = self._concatenate(context_ids)
        scores = self._linear(context_vec)
        return softmax(scores)


def main() -> None:
    """Demonstrate a forward pass of the embeddings model."""
    from toy_gpt_train.a_tokenizer import SimpleTokenizer
    from toy_gpt_train.b_vocab import Vocabulary

    log_header(LOG, "Embedding Next-Token Model Demo")

    tokenizer: SimpleTokenizer = SimpleTokenizer()
    tokens: list[str] = tokenizer.get_tokens()

    if len(tokens) < DEFAULT_CONTEXT_SIZE + 1:
        LOG.info(f"Need at least {DEFAULT_CONTEXT_SIZE + 1} tokens for demonstration.")
        return

    vocab: Vocabulary = Vocabulary(tokens)
    model: EmbeddingNextTokenModel = EmbeddingNextTokenModel(
        vocab_size=vocab.vocab_size(),
        embedding_dim=DEFAULT_EMBEDDING_DIM,
        context_size=DEFAULT_CONTEXT_SIZE,
    )

    # Select context window from the start of the corpus.
    context_tokens = tokens[:DEFAULT_CONTEXT_SIZE]
    context_ids: list[int] = []
    for tok in context_tokens:
        tid = vocab.get_token_id(tok)
        if tid is None:
            LOG.info(f"Token {tok!r} not found in vocabulary.")
            return
        context_ids.append(tid)

    probs: list[float] = model.forward(context_ids)

    LOG.info(f"Context tokens: {context_tokens}")
    LOG.info(f"Context IDs:    {context_ids}")
    LOG.info("Top 5 predicted next tokens (untrained, random weights):")

    indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    for rank, (idx, prob) in enumerate(indexed[:5], 1):
        tok = vocab.get_id_token(idx)
        LOG.info(f"  {rank}. {tok!r} (ID {idx}) -> {prob:.4f}")


if __name__ == "__main__":
    main()
