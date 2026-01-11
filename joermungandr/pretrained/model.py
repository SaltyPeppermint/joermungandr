import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from .config import RerankerConfig
from .data import format_messages
from .download import load_model, load_tokenizer


class Qwen3Reranker:
    """Wrapper for Qwen3-Reranker with binary classification and ranking.

    This class provides a convenient interface for:
    - Scoring individual (start, goal, guide) triples
    - Binary classification (is this guide valid?)
    - Ranking multiple guides by relevance
    """

    config: RerankerConfig
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    token_yes_id: int
    token_no_id: int
    suffix: str
    device: torch.device

    def __init__(
        self,
        config: RerankerConfig | None = None,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
    ):
        """Initialize the reranker.

        Args:
            config: Reranker configuration. Required if model/tokenizer not provided.
            model: Optional pre-loaded model.
            tokenizer: Optional pre-loaded tokenizer.
        """
        if config is None:
            config = RerankerConfig()
        self.config = config

        if model is None:
            model = load_model(config.model_id, config)
        if tokenizer is None:
            tokenizer = load_tokenizer(config.model_id)

        self.model = model
        self.tokenizer = tokenizer

        # Get yes/no token IDs for scoring
        # convert_tokens_to_ids can return int or list[int], but for single token it's int
        token_yes = self.tokenizer.convert_tokens_to_ids("yes")
        token_no = self.tokenizer.convert_tokens_to_ids("no")
        assert isinstance(token_yes, int), "Expected single token ID for 'yes'"
        assert isinstance(token_no, int), "Expected single token ID for 'no'"
        self.token_yes_id = token_yes
        self.token_no_id = token_no

        # Suffix for the assistant turn
        self.suffix = "<|im_start|>assistant\n<think>\n\n</think>\n\n"

        # Move model to appropriate device
        self.device = next(self.model.parameters()).device

    def _prepare_input(
        self, start: str, goal: str, guide: str, instruction: str | None = None
    ) -> str:
        """Prepare input text for the model."""
        messages = format_messages(start, goal, guide, instruction)
        text = self.tokenizer.apply_chat_template(  # pyright: ignore[reportUnknownMemberType]
            messages, tokenize=False, add_generation_prompt=False
        )
        assert isinstance(text, str), "Expected string from apply_chat_template with tokenize=False"
        return text + self.suffix

    def _compute_scores(self, input_texts: list[str]) -> list[float]:
        """Compute relevance scores for a batch of inputs.

        Args:
            input_texts: List of formatted input strings.

        Returns:
            List of scores in [0, 1] representing P(valid).
        """
        # Tokenize with left-padding for batch inference
        encoded = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        encoded_dict: dict[str, torch.Tensor] = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            # Model forward pass - outputs has .logits attribute
            outputs = self.model(**encoded_dict)
            logits: torch.Tensor = outputs.logits[:, -1, :]

            # Extract yes/no logits
            yes_logits = logits[:, self.token_yes_id]
            no_logits = logits[:, self.token_no_id]

            # Compute P(yes) via softmax
            scores = torch.stack([no_logits, yes_logits], dim=1)
            probs = F.softmax(scores, dim=1)
            yes_probs = probs[:, 1]

        result: list[float] = yes_probs.cpu().tolist()
        return result

    def score(self, start: str, goal: str, guide: str, instruction: str | None = None) -> float:
        """Score a single (start, goal, guide) triple.

        Args:
            start: Start state/sentence.
            goal: Goal state/sentence.
            guide: Guide/middle sentence to evaluate.
            instruction: Optional custom instruction.

        Returns:
            Score in [0, 1] representing P(valid guide).
        """
        input_text = self._prepare_input(start, goal, guide, instruction)
        return self._compute_scores([input_text])[0]

    def score_batch(
        self,
        starts: list[str],
        goals: list[str],
        guides: list[str],
        instruction: str | None = None,
    ) -> list[float]:
        """Score multiple (start, goal, guide) triples efficiently.

        Args:
            starts: List of start states.
            goals: List of goal states.
            guides: List of guides to evaluate.
            instruction: Optional custom instruction (applied to all).

        Returns:
            List of scores in [0, 1].
        """
        if not (len(starts) == len(goals) == len(guides)):
            raise ValueError("starts, goals, and guides must have the same length")

        input_texts = [
            self._prepare_input(s, g, d, instruction) for s, g, d in zip(starts, goals, guides)
        ]
        return self._compute_scores(input_texts)

    def classify(
        self,
        start: str,
        goal: str,
        guide: str,
        threshold: float = 0.5,
        instruction: str | None = None,
    ) -> bool:
        """Binary classification: is this guide valid?

        Args:
            start: Start state/sentence.
            goal: Goal state/sentence.
            guide: Guide/middle sentence to evaluate.
            threshold: Classification threshold (default 0.5).
            instruction: Optional custom instruction.

        Returns:
            True if the guide is predicted to be valid.
        """
        return self.score(start, goal, guide, instruction) >= threshold

    def rank(
        self,
        start: str,
        goal: str,
        guides: list[str],
        instruction: str | None = None,
    ) -> list[tuple[int, float]]:
        """Rank multiple guides by relevance to (start, goal).

        Args:
            start: Start state/sentence.
            goal: Goal state/sentence.
            guides: List of guide candidates to rank.
            instruction: Optional custom instruction.

        Returns:
            List of (original_index, score) tuples, sorted by score descending.
        """
        if not guides:
            return []

        # Score all guides
        starts = [start] * len(guides)
        goals = [goal] * len(guides)
        scores = self.score_batch(starts, goals, guides, instruction)

        # Create (index, score) pairs and sort by score descending
        ranked = [(i, s) for i, s in enumerate(scores)]
        ranked.sort(key=lambda x: x[1], reverse=True)

        return ranked

    def to(self, device: str | torch.device) -> "Qwen3Reranker":
        """Move the model to a device.

        Args:
            device: Target device (e.g., "cuda", "cpu").

        Returns:
            Self for chaining.
        """
        # HF model.to() returns self but type stubs are incomplete
        self.model = self.model.to(device)  # pyright: ignore[reportArgumentType]
        self.device = next(self.model.parameters()).device
        return self

    def eval(self) -> "Qwen3Reranker":
        """Set the model to evaluation mode.

        Returns:
            Self for chaining.
        """
        self.model.eval()
        return self

    def train(self) -> "Qwen3Reranker":
        """Set the model to training mode.

        Returns:
            Self for chaining.
        """
        self.model.train()
        return self
