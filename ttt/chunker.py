from dataclasses import dataclass, field
from pathlib import Path
from ttt.config import encoding


@dataclass
class Chunker:
    text: str
    template_file: str
    chunk_size: int = 3500
    summary_size: int = 500
    token_chunks: list = field(default_factory=list)
    chunks: list = field(default_factory=list)
    processed_chunks: list = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.tokens = encoding.encode(self.text)
        template_size = len(encoding.encode(Path(self.template_file).read_text()))
        self.chunk_size = self.chunk_size - template_size

    def chunk(self):
        self.token_chunks = [self.tokens[i:i + self.chunk_size] for i in range(0, len(self.tokens), self.chunk_size)]
        self.chunks = [encoding.decode(chunk) for chunk in self.token_chunks]
        return self.chunks
