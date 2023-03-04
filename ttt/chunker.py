from dataclasses import dataclass, field
from pathlib import Path

import click
from tttp.prompter import Prompter

from ttt.config import arg2dict, encoding


@dataclass
class Chunker:
    text: str
    token_chunks: list = field(default_factory=list)
    chunks: list = field(default_factory=list)
    processed_chunks: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    template_size: int = 0

    def __post_init__(self):
        if self.params.get("template_file", None):
            file = Prompter.find_file(self.params["template_file"])
            self.template_size = len(encoding.encode(Path(file).read_text()))
            self.prompter = Prompter(file)

        self.summary_size: int = self.params.get("summary_size", 500)
        self.chunk_size = self.params.get("chunk_size", self.params.get("token_limit") - self.summary_size)

        self.tokens = encoding.encode(self.text)
        self.tokens_size = len(self.tokens)
        self.in_size = min(self.chunk_size, self.tokens_size) + self.template_size
        self.total_size = self.in_size + self.summary_size

    def needs_chunking(self):
        if self.total_size > self.params.get("token_limit") or self.tokens_size > self.chunk_size:
            click.echo(
                "Prompt is too long. "
                f"Prompt/template/summary: {self.tokens_size}/{self.template_size}/{self.summary_size} "
                f"Token limit tokens: {self.params.get('token_limit')}",
                err=True,
            )
            return True
        return False

    def chunk(self):
        prompt_args = arg2dict(self.params["template_args"])
        self.token_chunks = [self.tokens[i : i + self.chunk_size] for i in range(0, len(self.tokens), self.chunk_size)]
        self.chunks = [encoding.decode(chunk) for chunk in self.token_chunks]
        self.chunks = [self.prompter.prompt(chunk, prompt_args) for chunk in self.chunks]
        return self.chunks
