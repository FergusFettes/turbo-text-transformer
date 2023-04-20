from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import click
from gpt_index import Document, GPTMultiverseIndex


@dataclass
class DummyTree:
    params: Optional[dict] = None
    prompt: str = ""

    def input(self, prompt):
        self.prompt = prompt

    def output(self, *_):
        pass

    def __len__(self):
        return 0


@dataclass
class Tree:
    file: Optional[str] = None
    in_str: str = "In: "
    out_str: str = "Out: "
    index: Optional[GPTMultiverseIndex] = None
    params: Optional[dict] = None
    name: Optional[str] = None

    def __post_init__(self):
        self.file = Path(self.file)

        if self.file and self.file.exists():
            self.index = GPTMultiverseIndex.load_from_disk(str(self.file))
            return

        self.index = GPTMultiverseIndex(documents=[])

    @property
    def prompt(self):
        prompt = self.index.__str__()
        prompt += f"\n"
        return prompt

    def input(self, prompt):
        self.extend(prompt, preface=self.in_str)

    def output(self, prompt):
        self.extend(prompt, preface=self.out_str, save=True)

    def extend(self, response, preface=None, save=False):
        if not response.startswith(preface):
            response = f"{preface}{response}"
        self.index.extend(Document(response))
        if save:
            self.save()

    def save(self):
        self.index.save_to_disk(self.file)

    def __len__(self):
        return len(self.index.index_struct.all_nodes)
