"""
N-gram draft model.
Used as a no-neural baseline to isolate the cost of verification
from the cost of draft generation.
"""

from collections import defaultdict
import torch
import random


class NgramDraftModel:
    """
    Simple n-gram language model that acts as a drop-in draft model.
    Builds a lookup table from the prompt context and proposes the
    most likely continuation based on n-gram statistics.
    Falls back to random token sampling when the n-gram is unseen.
    """

    def __init__(self, n: int = 3, vocab_size: int = 32000):
        self.n = n
        self.vocab_size = vocab_size
        self.ngram_counts = defaultdict(lambda: defaultdict(int))

    def fit(self, token_ids: list[int]):
        """Build n-gram table from a token sequence (e.g. the prompt)."""
        for i in range(len(token_ids) - self.n):
            context = tuple(token_ids[i: i + self.n - 1])
            next_tok = token_ids[i + self.n - 1]
            self.ngram_counts[context][next_tok] += 1

    def propose(self, context_ids: torch.Tensor, gamma: int) -> tuple[torch.Tensor, list]:
        """
        Propose `gamma` draft tokens given the current context.
        Returns (proposed_ids, draft_probs) where draft_probs are uniform
        placeholders (acceptance will be driven by target model).
        """
        ids = context_ids[0].tolist()
        proposed = []
        draft_probs = []

        for _ in range(gamma):
            context = tuple(ids[-(self.n - 1):])
            counts = self.ngram_counts.get(context, {})

            if counts:
                next_tok = max(counts, key=counts.get)
            else:
                next_tok = random.randint(0, self.vocab_size - 1)

            proposed.append(next_tok)
            ids.append(next_tok)

            # Uniform placeholder prob (target model drives acceptance)
            prob = torch.zeros(1, self.vocab_size)
            prob[0, next_tok] = 1.0
            draft_probs.append(prob)

        proposed_tensor = torch.tensor(proposed, dtype=torch.long).unsqueeze(0)
        return proposed_tensor, draft_probs
