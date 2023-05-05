from dataclasses import dataclass

from ttt.config import Config, click, rich, shell
from ttt.io import IO
from ttt.store import Store


@dataclass
class App:
    echo_prompt: bool = False
    append: bool = False

    def __post_init__(self):
        self.config = Config.check_config()
        self.store = Store(config=self.config)
        self.io = IO
        self.prompter = self.store.get_prompter()
        self.tree = self.store.load_file()
        self.params = Config.load_openai_config()["engine_params"]
        self.tree.params = self.params
        self.cli = cli

    @staticmethod
    def simple_gen(tree):
        if tree.params["model"] == "test":
            return tree.prompt
        encoding = Config.get_encoding(tree.params.get("model", "gpt-3.5-turbo"))
        tree.params["max_tokens"] -= len(encoding.encode(tree.prompt))
        if tree.params["model"] == "code-davinci-002":
            tree.params["openai_api_base"] = os.environ.get("CD2_URL")
            tree.params["openai_api_key"] = os.environ.get("CD2_KEY")
        llm = OpenAI(**tree.params)
        responses = llm.generate([tree.prompt])
        return responses.generations[0][0].text

    def print(self, response):
        self.io.return_prompt(
            response, self.prompt if self.echo_prompt else None, self.prompt_file if self.append else None
        )


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
