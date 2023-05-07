import os
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass

import openai
from langchain.llms import OpenAI

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
    def simple_gen(prompt, tree):
        params = deepcopy(tree.params)
        params["prompt"] = prompt
        if params["model"] == "test":
            return prompt
        if tree.params["model"] == "code-davinci-002":
            params["openai_api_base"] = os.environ.get("CD2_URL")
            params["openai_api_key"] = os.environ.get("CD2_KEY")
            llm = OpenAI(**tree.params)
            responses = llm.generate([params["prompt"]])
            return [generation.text for generation in responses[0]]
        generations = OAIGen.gen(params)
        for i, gen in generations.items():
            if gen.startswith("\n"):
                generations[i] = gen[1:]
        return generations

    @staticmethod
    def max_tokens(params):
        model_max = Config.model_tokens.get(params["model"], 2048)

        encoding = Config.get_encoding(params.get("model", "gpt-3.5-turbo"))
        request_total = params["max_tokens"] + len(encoding.encode(prompt))

        if request_total > model_max:
            params["max_tokens"] = model_max - len(encoding.encode(prompt))
        else:
            params["max_tokens"] = request_total

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

        completions = defaultdict(str)
        for part in resp:
            choices = part["choices"]
            for chunk in sorted(choices, key=lambda s: s["index"]):
                c_idx = chunk["index"]
                if not chunk["text"]:
                    continue
                completions[c_idx] += chunk["text"]
                OAIGen.printlines(completions)

        return completions

    @staticmethod
    def _chat(params):
        params["messages"] = [{"role": "user", "content": params["prompt"]}]
        if "prompt" in params:
            del params["prompt"]
        if "logprobs" in params:
            del params["logprobs"]

        resp = openai.ChatCompletion.create(**params)

        if not params["stream"]:
            resp = [resp]

        completions = defaultdict(str)
        for part in resp:
            choices = part["choices"]
            for chunk in sorted(choices, key=lambda s: s["index"]):
                c_idx = chunk["index"]
                if len(choices) > 1:
                    sys.stdout.write("===== Chat Completion {} =====\n".format(c_idx))
                delta = chunk["delta"]
                if "content" not in delta:
                    continue
                content = chunk["delta"]["content"]
                if not content:
                    break
                completions[c_idx] += content
                OAIGen.printlines(completions)

        return completions

    @staticmethod
    def printlines(messages):
        lines = 0
        for message in messages.values():
            lines += OAIGen.get_lines(message)  # get number of lines for message
        OAIGen.clear_console(lines)  # clear those lines
        for i, message in messages.items():
            print(f"{i}: {message}")

    @staticmethod
    def clear_console(n):
        print("\033[{}A".format(n), end="")  # move cursor up by n lines
        print("\033[J", end="")  # clear from cursor to end of screen

    @staticmethod
    def get_lines(message):
        newlines = message.count("\n")  # get number of newlines in message
        term_width = os.get_terminal_size().columns  # get terminal width in characters
        message_length = len(message)  # get message length in characters
        return (message_length // term_width) + 1 + newlines  # get number of lines occupied by message


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
