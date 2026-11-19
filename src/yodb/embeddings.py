"""A library-shipped embedding provider: an open model run locally through fastembed (ONNX).

    from yodb.embeddings import FastEmbedProvider
    embedder = FastEmbedProvider()                  # BAAI/bge-small-en-v1.5, 384 dimensions
    with yodb.connect("catalog/", embedder=embedder) as db: ...

The same provider serves the similarity search (``{"similar": ...}``) and the
semantic filter's vector shortlist, and ``embed_texts`` fills a pgvector column
when the data is loaded.  The catalog's ``embeddings.model`` must equal the model
name given here, and its ``dimensions`` the model's, or the planner refuses the
search rather than compare vectors from different spaces.  Needs the optional
``semantic`` extra (``pip install "yodb[semantic]"``).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
import time

from .semantic.contracts import EmbeddingRequest, EmbeddingResult, ProviderCost, ProviderInfo

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedProvider:
    """An :class:`EmbeddingProvider` over a fastembed model, with usage counters."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        cache_dir: str | None = None,
        threads: int | None = None,
        version: str = "1",
    ) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as error:
            raise RuntimeError('the embedding provider needs the "fastembed" package: pip install "yodb[semantic]"') from error
        self._model = TextEmbedding(model_name, cache_dir=cache_dir, threads=threads)
        probe = next(iter(self._model.embed(["probe"])))
        self.dimensions = len(probe)
        self.info = ProviderInfo("fastembed", model_name, version)
        self.cost_hint = ProviderCost(money_per_call=0.0, latency_ms_per_call=30.0)
        self.calls = 0
        self.texts_embedded = 0
        self.seconds = 0.0

    @property
    def model_name(self) -> str:
        return self.info.model

    def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        started = time.perf_counter()
        vectors = tuple(self.embed_texts(request.texts))
        self.calls += 1
        self.seconds += time.perf_counter() - started
        return EmbeddingResult(vectors, self.info, self.dimensions)

    def embed_texts(self, texts: Iterable[str], batch_size: int = 256) -> Iterator[tuple[float, ...]]:
        """Vectors for many texts, in order (for loading a vector column)."""

        for vector in self._model.embed(list(texts), batch_size=batch_size):
            self.texts_embedded += 1
            yield tuple(float(x) for x in vector)
