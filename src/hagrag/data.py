from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pdfplumber
from tqdm import tqdm

from .errors import DataError
from .io_utils import ensure_dir, read_json, write_json

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Document:
    document_id: str
    filename: str
    text: str


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    document_id: str
    filename: str
    index: int
    text: str


class PDFProcessor:
    def __init__(self, pdf_dir: str | Path, checkpoint_dir: str | Path):
        self.pdf_dir = Path(pdf_dir)
        self.checkpoint_dir = ensure_dir(checkpoint_dir)
        self.documents_file = self.checkpoint_dir / "documents.json"
        self.chunks_file = self.checkpoint_dir / "chunks.json"
        self.state_file = self.checkpoint_dir / "pdf_state.json"

    def _pdf_files(self) -> list[Path]:
        if not self.pdf_dir.exists():
            raise DataError(f"PDF directory not found: {self.pdf_dir}")
        files = sorted(self.pdf_dir.glob("*.pdf"))
        if not files:
            raise DataError(f"No PDF files found in {self.pdf_dir}")
        return files

    def _state_hash(self) -> str:
        digest = hashlib.sha256()
        for path in self._pdf_files():
            stat = path.stat()
            digest.update(path.name.encode("utf-8"))
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
        return digest.hexdigest()

    def extract_documents(self, force: bool = False) -> list[Document]:
        state_hash = self._state_hash()
        if not force and self.documents_file.exists() and self.state_file.exists():
            state = read_json(self.state_file)
            if state.get("pdf_state_hash") == state_hash:
                return [Document(**item) for item in read_json(self.documents_file)]

        documents: list[Document] = []
        for path in tqdm(self._pdf_files(), desc="Extracting PDFs"):
            pages: list[str] = []
            try:
                with pdfplumber.open(path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text() or ""
                        if text.strip():
                            pages.append(text)
            except Exception as exc:
                LOGGER.warning("Skipping unreadable PDF %s: %s", path.name, exc)
                continue
            text = "\n\n".join(pages).strip()
            if text:
                documents.append(Document(path.stem, path.name, text))

        if not documents:
            raise DataError("PDF extraction completed without any readable text")
        write_json(self.documents_file, [asdict(item) for item in documents])
        write_json(self.state_file, {"pdf_state_hash": state_hash, "documents": len(documents)})
        return documents

    @staticmethod
    def _split_text(text: str, chunk_size: int, overlap: int, separators: tuple[str, ...]) -> list[str]:
        if len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []

        chunks: list[str] = []
        start = 0
        while start < len(text):
            hard_end = min(len(text), start + chunk_size)
            end = hard_end
            if hard_end < len(text):
                window = text[start:hard_end]
                min_break = max(1, int(chunk_size * 0.55))
                best = -1
                for separator in separators:
                    if not separator:
                        continue
                    pos = window.rfind(separator, min_break)
                    if pos > best:
                        best = pos + len(separator)
                if best > 0:
                    end = start + best
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
        return chunks

    def create_chunks(
        self,
        documents: list[Document],
        chunk_size: int = 1024,
        chunk_overlap: int = 20,
        separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", ""),
        force: bool = False,
    ) -> list[TextChunk]:
        if not force and self.chunks_file.exists():
            return [TextChunk(**item) for item in read_json(self.chunks_file)]

        chunks: list[TextChunk] = []
        for document in documents:
            pieces = self._split_text(document.text, chunk_size, chunk_overlap, separators)
            for index, text in enumerate(pieces):
                chunks.append(
                    TextChunk(
                        chunk_id=f"{document.document_id}:{index}",
                        document_id=document.document_id,
                        filename=document.filename,
                        index=index,
                        text=text,
                    )
                )
        write_json(self.chunks_file, [asdict(item) for item in chunks])
        return chunks


def load_qa_dataset(path: str | Path, max_queries: int | None = None) -> list[dict]:
    source = Path(path)
    if not source.exists():
        raise DataError(f"QA dataset not found: {source}")
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        raise DataError(f"QA dataset is empty: {source}")
    if text.startswith("["):
        data = json.loads(text)
    else:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(data, list):
        raise DataError("QA dataset must be a JSON array or JSONL file")
    return data[:max_queries] if max_queries else data
