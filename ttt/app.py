import os
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Union

import openai
from langchain.llms import OpenAI
from rich.live import Live
from rich.table import Table

from ttt.config import Config, click, rich, shell
from ttt.io import IO
from ttt.store import Store
from ttt.templater import Templater


@dataclass
class App:
    echo_prompt: bool = False
    append: bool = False

    def __post_init__(self):
        self.config = Config.check_config()
        self.store = Store(config=self.config)
        self.templater = Templater(config=self.config)
        self.io = IO
        self.tree = self.store.load_file()
        self.params = Config.load_openai_config()["engine_params"]
        self.tree.params = self.params

    @staticmethod
    def simple_gen(params):
        if params["model"] == "test":
            return prompt
        App.max_tokens(params)
        if params["model"] == "code-davinci-002":
            params["openai_api_base"] = os.environ.get("CD2_URL")
            params["openai_api_key"] = os.environ.get("CD2_KEY")
            llm = OpenAI(**params)
            responses = llm.generate([params["prompt"]])
            return [generation.text for generation in responses[0]]
        generations, choice = OAIGen.gen(params)

        for i, gen in generations.items():
            if gen.startswith("\n"):
                generations[i] = gen[1:]
        choice = choice - 1 if choice != -1 else None
        return generations, choice

    @staticmethod
    def max_tokens(params):
        model_max = Config.model_tokens.get(params["model"], 2048)

        encoding = Config.get_encoding(params.get("model", "gpt-3.5-turbo"))
        request_total = params["max_tokens"] + len(encoding.encode(params["prompt"]))

        if request_total > model_max:
            params["max_tokens"] = model_max - len(encoding.encode(params["prompt"]))

    def output(self, response):
        self.io.return_prompt(
            response, self.prompt if self.echo_prompt else None, self.prompt_file if self.append else None
        )


class OAIGen:
    @staticmethod
    def gen(params):
        if params["model"].startswith("gpt-3.5") or params["model"].startswith("gpt-4"):
            return OAIGen._chat(params)
        return OAIGen._gen(params)

    @staticmethod
    def _gen(params):
        resp = openai.Completion.create(**params)
        if not params["stream"]:
            resp = [resp]

        choice = -1
        with Live(screen=True) as live:
            completions = defaultdict(str)
            for part in resp:
                choices = part["choices"]
                for chunk in sorted(choices, key=lambda s: s["index"]):
                    c_idx = chunk["index"]
                    if not chunk["text"]:
                        continue
                    completions[c_idx] += chunk["text"]
                    OAIGen.richprint(params["prompt"], completions, live)
            if len(completions):
                OAIGen.richprint(params["prompt"], completions, live, final=True)
                choice = click.prompt("Choose a completion", default=-1, type=int)
        return completions, choice

    @staticmethod
    def _chat(params):
        params["messages"] = [{"role": "user", "content": params["prompt"]}]
        if "prompt" in params:
            prompt = params["prompt"]
            del params["prompt"]
        if "logprobs" in params:
            del params["logprobs"]

        resp = openai.ChatCompletion.create(**params)

        if not params["stream"]:
            resp = [resp]

        choice = -1
        with Live(screen=True) as live:
            completions = defaultdict(str)
            for part in resp:
                choices = part["choices"]
                for chunk in sorted(choices, key=lambda s: s["index"]):
                    c_idx = chunk["index"]
                    delta = chunk["delta"]
                    if "content" not in delta:
                        continue
                    content = chunk["delta"]["content"]
                    if not content:
                        break
                    completions[c_idx] += content
                    OAIGen.richprint(prompt, completions, live)
            if len(completions):
                OAIGen.richprint(prompt, completions, live, final=True)
                choice = click.prompt("Choose a completion", default=-1, type=int)

        for i, gen in completions.items():
            # If the completion starts with a letter, prepend a space
            if re.match(r"^[a-zA-Z]", gen):
                completions[i] = " " + gen
        return completions, choice

    @staticmethod
    def richprint(prompt, messages, live, final=False):
        messages = {k: v for k, v in sorted(messages.items(), key=lambda item: item[0])}
        choice_msg = ""
        if final:
            choice_msg = "Choose a completion (optional). [Enter] to continue."
        table = Table(
            box=rich.box.MINIMAL_DOUBLE_HEAD,
            width=shutil.get_terminal_size().columns,
            show_lines=True,
            show_header=False,
            title=prompt[-1000:],
            title_justify="left",
            caption=choice_msg + ", ".join([str(i + 1) for i in messages.keys()]),
            style="bold blue",
            highlight=True,
            title_style="bold blue",
            caption_style="bold blue",
        )
        for i, message in messages.items():
            table.add_row(str(i + 1), f"[bold]{message}[/bold]")
        live.update(table)


@shell(prompt="params> ")
@click.pass_context
def cli(ctx):
    """Manage model params."""
    rich.print(ctx.obj.tree.params)


@cli.command(hidden=True)
@click.pass_context
def _print(ctx):
    "Print the current config."
    rich.print(ctx.obj.tree.params)


cli.add_command(_print, "print")
cli.add_command(_print, "p")


@cli.command()
@click.pass_context
def save(ctx):
    "Save the current config to the config file."
    config = Config.load_openai_config()
    config["engine_params"] = ctx.obj.tree.params
    Config.save_openai_config(config)


cli.add_command(save, "s")


@cli.command()
@click.argument("name", required=False)
@click.argument("value", required=False)
@click.argument("kv", required=False)
@click.pass_context
def update(ctx, name, value, kv):
    "Update a config value, or set of values. (kv in the form of 'name1=value1,name2=value2')"
    if kv:
        updates = kv.split(",")
        for kv in updates:
            name, value = kv.split("=")
            _update(name, value, ctx.obj.tree.params)
        return
    _update(name, value, ctx.obj.tree.params)
    rich.print(ctx.obj.tree.params)


cli.add_command(update, "u")


def _update(key, value, dict):
    if value in ["True", "False", "true", "false"]:
        value = value in ["True", "true"]
    elif value in ["None"]:
        value = None
    elif value.isdigit():
        value = int(value)
    elif value.replace(".", "").isdigit():
        value = float(value)
    dict.update({key: value})
