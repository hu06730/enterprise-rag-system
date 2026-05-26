import re
from app.core.interfaces import BaseSplitter


class RecursiveSplitter(BaseSplitter):
    SEPARATORS = ["\n\n", "\n", "。", ". ", ".", "；", ";", " ", ""]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str, **kwargs) -> list[str]:
        return self._split_text(text, self.SEPARATORS)

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        if not text.strip():
            return []
        sep = separators[0]
        remaining_seps = separators[1:]
        if sep == "":
            return self._split_by_length(text)
        if sep in text:
            parts = text.split(sep)
        else:
            return self._split_text(text, remaining_seps)
        chunks = []
        current = ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                if len(part) > self.chunk_size:
                    sub = self._split_text(part, remaining_seps)
                    if current and self.chunk_overlap > 0 and sub:
                        sub[0] = current[-self.chunk_overlap:] + sub[0]
                    chunks.extend(sub)
                    current = ""
                else:
                    current = part
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def _split_by_length(self, text: str) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap
        return chunks


class MarkdownSplitter(BaseSplitter):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str, **kwargs) -> list[str]:
        chunks = []
        sections = re.split(r'(?=^#{1,3} )', text, flags=re.MULTILINE)
        for section in sections:
            if not section.strip():
                continue
            content = section.strip()
            if len(content) <= self.chunk_size:
                chunks.append(content)
            else:
                paragraphs = content.split("\n\n")
                current = ""
                for p in paragraphs:
                    candidate = current + ("\n\n" if current else "") + p
                    if len(candidate) <= self.chunk_size:
                        current = candidate
                    else:
                        if current:
                            chunks.append(current)
                        current = p
                if current:
                    chunks.append(current)
        return chunks


class SemanticSplitter(BaseSplitter):
    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        min_chunk_size: int = 100,
        similarity_threshold: float = 0.5,
    ):
        self.model_name = model_name
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def split(self, text: str, **kwargs) -> list[str]:
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [text] if text.strip() else []
        embeddings = self.model.encode(sentences)
        import numpy as np
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]) + 1e-8
            )
            similarities.append(float(sim))
        split_points = [0]
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                split_points.append(i + 1)
        split_points.append(len(sentences))
        chunks = []
        for i in range(len(split_points) - 1):
            chunk = "".join(sentences[split_points[i]:split_points[i + 1]])
            if chunk.strip() and len(chunk) >= self.min_chunk_size:
                chunks.append(chunk.strip())
        if not chunks:
            return [text.strip()]
        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        return re.split(r'(?<=[。！？.!?\n])\s*', text)


SPLITTER_REGISTRY: dict[str, type[BaseSplitter]] = {
    "recursive": RecursiveSplitter,
    "markdown": MarkdownSplitter,
    "semantic": SemanticSplitter,
}


def get_splitter(strategy: str, **kwargs) -> BaseSplitter:
    cls = SPLITTER_REGISTRY.get(strategy)
    if cls is None:
        raise ValueError(f"Unknown splitter strategy: {strategy}")
    return cls(**kwargs)
