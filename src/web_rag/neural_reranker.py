from __future__ import annotations

from web_rag.config import get_settings
from web_rag.models import Snippet
from web_rag.text_utils import clean_text


class MedCPTReranker:
    """
    Biomedical semantic reranker using ncbi/MedCPT-Cross-Encoder.

    The model scores query-snippet pairs directly. This is more expensive than
    lexical scoring, but it is more suitable for biomedical semantic relevance.
    """

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> None:
        settings = get_settings()

        self.model_name = model_name or settings.medcpt_model_name
        self.batch_size = batch_size or settings.medcpt_batch_size
        self.max_length = max_length or settings.medcpt_max_length

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "MedCPT reranking requires torch and transformers. "
                "Install them with: pip install torch transformers"
            ) from error

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def build_candidate_text(snippet: Snippet) -> str:
        """
        Build the text given to the cross-encoder.

        We combine title and evidence text because the title often contains the
        biomedical entity, while the snippet contains the stability evidence.
        """
        title = clean_text(snippet.paper.title)
        evidence = clean_text(snippet.text)

        if title and evidence:
            return f"{title}. {evidence}"

        return title or evidence

    def score_snippets(self, question: str, snippets: list[Snippet]) -> list[float]:
        """
        Score snippets against a question using MedCPT.

        Higher score means higher estimated relevance.
        """
        if not snippets:
            return []

        all_scores: list[float] = []

        with self.torch.no_grad():
            for start in range(0, len(snippets), self.batch_size):
                batch = snippets[start:start + self.batch_size]

                pairs = [
                    [question, self.build_candidate_text(snippet)]
                    for snippet in batch
                ]

                encoded = self.tokenizer(
                    pairs,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=self.max_length,
                )

                encoded = {
                    key: value.to(self.device)
                    for key, value in encoded.items()
                }

                outputs = self.model(**encoded)
                logits = outputs.logits

                if logits.ndim == 2 and logits.shape[1] == 1:
                    logits = logits.squeeze(dim=-1)

                if logits.ndim == 0:
                    batch_scores = [float(logits.detach().cpu().item())]
                else:
                    batch_scores = [
                        float(value)
                        for value in logits.detach().cpu().tolist()
                    ]

                all_scores.extend(batch_scores)

        return all_scores