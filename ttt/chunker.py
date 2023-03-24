from dataclasses import dataclass, field
from pathlib import Path

import click
from tttp.prompter import Prompter

from ttt.config import arg2dict, get_encoding


@dataclass
class Chunker:
    text: str
    token_chunks: list = field(default_factory=list)
    chunks: list = field(default_factory=list)
    processed_chunks: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    template_size: int = 0
    model_cost_per_token = {
        "gpt-4": {"prompt": 0.03 / 1000, "completion": 0.06 / 1000},
        "gpt-4-32k": {"prompt": 0.06 / 1000, "completion": 0.012 / 1000},
        "gpt-3.5-turbo": {"prompt": 0.002 / 1000, "completion": 0.002 / 1000},
    }
    verbose: bool = False

    def __post_init__(self):
        encoding = get_encoding(self.params["model"])
        if self.params.get("template_file", None):
            file = Prompter.find_file(self.params["template_file"])
            self.template_size = len(encoding.encode(Path(file).read_text()))
            self.prompter = Prompter(file)

        self.summary_size: int = self.params.get("summary_size", 500)
        self.chunk_size = self.params.get("chunk_size", self.params.get("token_limit") - self.summary_size)

        self.tokens = encoding.encode(self.text)
        self.tokens_size = len(self.tokens)
        self.in_size = min(self.chunk_size, self.tokens_size) + self.template_size
        if self.verbose:
            print(f"Tokens used for the initial request: {self.in_size}")
            model_cost = 0.02
            if self.params.get("model") in self.model_cost_per_token:
                model_cost = self.model_cost_per_token[self.params.get("model")]["prompt"]
            cost = self.in_size * model_cost
            print(f"Cost of the initial request: {cost:.2f} USD")
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
        encoding = get_encoding(self.params["model"])
        prompt_args = arg2dict(self.params["template_args"])
        self.token_chunks = [self.tokens[i : i + self.chunk_size] for i in range(0, len(self.tokens), self.chunk_size)]
        self.chunks = [encoding.decode(chunk) for chunk in self.token_chunks]
        self.chunks = [self.prompter.prompt(chunk, prompt_args) for chunk in self.chunks]
        return self.chunks
