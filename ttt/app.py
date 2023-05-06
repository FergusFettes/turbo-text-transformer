import sys
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
        if tree.params["model"] == "test":
            return prompt
        encoding = Config.get_encoding(tree.params.get("model", "gpt-3.5-turbo"))
        tree.params["max_tokens"] -= len(encoding.encode(tree.prompt))
        if tree.params["model"] == "code-davinci-002":
            tree.params["openai_api_base"] = os.environ.get("CD2_URL")
            tree.params["openai_api_key"] = os.environ.get("CD2_KEY")
        # llm = OpenAI(**tree.params)
        # responses = llm.generate([prompt])
        # return responses.generations[0][0].text
        return OAIGen.gen(prompt, tree.params)

    def output(self, response):
        self.io.return_prompt(
            response, self.prompt if self.echo_prompt else None, self.prompt_file if self.append else None
        )


class OAIGen:
    @staticmethod
    def gen(prompt, params):
        if params["n"] is not None and params["n"] > 1 and params["stream"]:
            print("Can't stream completions with n>1 with the current CLI")
            params["stream"] = False

        if params["model"].startswith("gpt-3.5") or params["model"].startswith("gpt-4"):
            return OAIGen._chat(prompt, params)
        return OAIGen._gen(prompt, params)

    @staticmethod
    def _gen(prompt, params):
        params["prompt"] = prompt

        resp = openai.Completion.create(**params)
        if not params["stream"]:
            resp = [resp]

        for part in resp:
            choices = part["choices"]
            for c_idx, c in enumerate(sorted(choices, key=lambda s: s["index"])):
                if len(choices) > 1:
                    sys.stdout.write("===== Completion {} =====\n".format(c_idx))
                sys.stdout.write(c["text"])
                if len(choices) > 1:
                    sys.stdout.write("\n")
                sys.stdout.flush()

        return resp

    @staticmethod
    def _chat(prompt, params):
        params["messages"] = [{"role": "user", "content": prompt}]
        if "prompt" in params:
            del params["prompt"]
        if "logprobs" in params:
            del params["logprobs"]

        resp = openai.ChatCompletion.create(**params)

        if not params["stream"]:
            resp = [resp]

        text = ""
        for part in resp:
            choices = part["choices"]
            for c_idx, chunk in enumerate(sorted(choices, key=lambda s: s["index"])):
                if len(choices) > 1:
                    sys.stdout.write("===== Chat Completion {} =====\n".format(c_idx))
                delta = chunk["delta"]
                if "content" not in delta:
                    continue
                content = chunk["delta"]["content"]
                if not content:
                    break
                text += content
                sys.stdout.write(content)
                if len(choices) > 1:
                    sys.stdout.write("\n")
                sys.stdout.flush()

        return text

        # for c in resp["choices"]:
        #     c["text"] = c["message"]["content"]
        #     del c["message"]
        # resp["params"]["prompt"] = resp["params"]["messages"][0]["content"]
        # del resp["params"]["messages"]
        #
        # return resp


@shell(prompt="params> ")
@click.pass_context
def cli(ctx):
    """Manage model params."""
    rich.print(ctx.obj.tree.params)


@cli.command()
@click.pass_context
def print(ctx):
    "Print the current config."
    rich.print(ctx.obj.tree.params)


cli.add_command(print, "p")


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
    if value in ["True", "False"]:
        value = value == "True"
    elif value in ["None"]:
        value = None
    elif value.isdigit():
        value = int(value)
    elif value.replace(".", "").isdigit():
        value = float(value)
    dict.update({key: value})
