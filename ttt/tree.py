from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click
from gpt_index import Document, GPTMultiverseIndex


@dataclass
class Tree:
    file: str
    in_str: str = "In: "
    out_str: str = "Out: "
    index: Optional[GPTMultiverseIndex] = None
    params: Optional[dict] = None

    def __post_init__(self):
        self.index = self.get_or_create(self.file)

    def get_or_create(self, file):
        if file and Path(file).exists():
            return GPTMultiverseIndex.load_from_disk(file)
        return GPTMultiverseIndex([])

    def get_and_run_commands(self, prompt: str):
        """Commands can be the following:
        tag:tag_name
        checkout:[int or tag_name]
        display
        """
        if not prompt.startswith("::"):
            return prompt

        commands = prompt.split("::")[1]
        prompt = prompt.split("::")[2]

        parsed_commands = commands.split(",")
        for command in parsed_commands:
            if command.startswith("tag:"):
                tag = command.split(":")[1]
                self.index.tag(tag)
            elif command.startswith("checkout:"):
                tag = command.split(":")[1]
                if tag.isdigit():
                    self.index.checkout(int(tag))
                else:
                    self.index.checkout(tag)
            elif command == "display":
                click.echo(self.index.index_struct)

        return prompt

    @property
    def prompt(self):
        prompt = self.index.__str__()
        prompt += f"\n{self.out_str}"
        return prompt

    def input(self, prompt):
        self.extend(prompt, preface=self.in_str)

    def output(self, prompt):
        self.extend(prompt, preface=self.out_str, save=True)

    def extend(self, response, preface=None, save=False):
        # Prepend self.out_str if it's not already there
        if not response.startswith(preface):
            response = f"{preface}{response}"
        self.index.extend(Document(response))
        if save:
            self.save_to_disk(self.file)

    def __len__(self):
        return len(self.index.index_struct.all_nodes)


@dataclass
class DummyTree:
    params: Optional[dict] = None
    prompt: str = ""

    def input(self, prompt):
        self.prompt = prompt

    def output(self, *_):
        pass

    def get_and_run_commands(self, prompt):
        return prompt

    def __len__(self):
        return 0
