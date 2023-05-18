import datetime
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import tiktoken
import tttp
import typer
import yaml
from rich import print
from typer import Argument, Context
from typing_extensions import Annotated

from .typer_shell import make_typer_shell

file_path = Path("/tmp/ttt/")
file_path.mkdir(parents=True, exist_ok=True)

# log_path = file_path / "logs"
# log_path.mkdir(parents=True, exist_ok=True)

timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
# logfile = log_path / f"{timestamp}.log"

logging.basicConfig(
    # filename=logfile,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger()


@dataclass
class Config:
    model_tokens = {
        "gpt-4": 4096 - 8,
        "gpt-3.5-turbo": 4096 - 8,
        "text-davinci-003": 4096 - 8,
        "text-davinci-002": 4096 - 8,
        "code-davinci-002": 4096 - 8,
    }
    TURBO_TEXT_TRANSFORMER_DEFAULT_PARAMS = {
        "file": True,
        "chat_path": "~/.config/ttt/chats",
        "chat_name": "default",
        "echo_prompt": False,
        "append": False,
        "backup_path": "/tmp/ttt/",
        "templater": {
            "in_postfix": "\n",
            "out_postfix": "\n",
            "in_prefix": "Human: ",
            "out_prefix": "GPT:",
            "template": True,
            "template_path": "~/.config/ttt/templates",
            "template_file": "assist.j2",
        },
    }
    OPENAI_DEFAULT_PARAMS = {
        "frequency_penalty": 0,
        "logprobs": 1,
        "max_tokens": 200,
        "model": "gpt-3.5-turbo",
        "n": 1,
        "presence_penalty": 0,
        "stop": None,
        "temperature": 0.9,
        "top_p": 1,
        "stream": True,
    }
    config_dir = Path().home() / ".config/ttt"
    _dict: dict = field(default_factory=dict)

    def __post_init__(self):
        self.config_path = self.config_dir / "config.yaml"

    @classmethod
    def load(cls):
        instance = cls()
        if not Config().config_path.exists():
            return {}
        instance._dict = yaml.load(Config().config_path.read_text(), Loader=yaml.Loader)
        return instance

    def save(self):
        self.config_path.write_text(yaml.dump(self._dict))

    @classmethod
    def create(cls):
        instance = cls()
        instance.config_dir.mkdir(parents=True, exist_ok=True)
        instance.config_path.write_text(yaml.dump(Config.TURBO_TEXT_TRANSFORMER_DEFAULT_PARAMS))
        config = Config.load_config()

        # Find the templates, and make sure they are in the right place
        tttp_dir = Path(tttp.__file__).parent
        new_templates = tttp_dir.parent / "templates"
        templates = instance.config_dir / "templates"
        templates.mkdir(parents=True, exist_ok=True)
        for template in new_templates.glob("*.j2"):
            if not (templates / template.name).exists():
                (templates / template.name).write_text(template.read_text())

        Path(config._dict.get("chat_path", "~/.config/ttt/chats")).expanduser().mkdir(parents=True, exist_ok=True)
        return instance

    @staticmethod
    def create_openai_config(api_key):
        path = Config.config_dir / "openai.yaml"

        oai_config = {
            "engine_params": Config.OPENAI_DEFAULT_PARAMS,
            "api_key": api_key,
            "models": [],
        }
        path.write_text(yaml.dump(oai_config))

    @staticmethod
    def load_openai_config():
        path = Config.config_dir / "openai.yaml"
        if not path.exists():
            return {}
        oaiconfig = yaml.load(path.read_text(), Loader=yaml.Loader)
        return oaiconfig

    @staticmethod
    def save_openai_config(config):
        path = Config.config_dir / "openai.yaml"
        path.write_text(yaml.dump(config))

    @staticmethod
    def check_config(reinit=False):
        """Check that the config file exists."""
        if not reinit and Config().config_path.exists():
            return Config.load()

        print("Config file not found. Creating one for you...", err=True, color="red")
        config = Config.create()
        openai_api_key = typer.prompt("OpenAI API Key", type=str)
        if openai_api_key:
            config.create_openai_config(openai_api_key)

        return config

    @staticmethod
    def get_encoding(model):
        try:
            encoding = tiktoken.encoding_for_model(model)
        except (KeyError, ValueError):
            encoding = tiktoken.get_encoding("gpt2")
        return encoding

    @staticmethod
    def arg2dict(args):
        d = {}
        if not args:
            return d
        if "=" not in args:
            return d
        for arg in args.split(","):
            k, v = arg.split("=")
            d[k] = v
        return d

    @staticmethod
    def _update(key, value, dict):
        if value in ["True", "False", "true", "false"]:
            value = value in ["True", "true"]
        elif value in ["None"]:
            value = None
        elif value.isdigit():
            value = int(value)
        elif value[1:].replace(".", "").isdigit():
            value = float(value)
        dict.update({key: value})

    @staticmethod
    def prepare_engine_params(params):
        """Prepare options for the OpenAI API."""
        params = {k: v for k, v in params.items() if v is not None}

        params["max_tokens"] = Config.model_tokens.get(params["model"], 2048)
        if "number" in params:
            params["n"] = params.pop("number")
        return params

    @staticmethod
    def check_file(toggle, default, config):
        if toggle:
            config._dict["file"] = not config._dict["file"]

        if default:
            config._dict["chat_name"] = default

        if toggle or config._dict["chat_name"]:
            config.save()

        if config._dict["file"]:
            print(f"File mode is on. Using {config._dict['chat_name']} as the chat history file.")
        else:
            print("File mode is off.")


cli = make_typer_shell(prompt="📜: ", intro="Welcome to the Config! Type help or ? to list commands.")


@cli.command()
def reinit(ctx: Context):
    "Recreate the config file from defaults."
    ctx.obj.config = Config.check_config(reinit)


@cli.command(name="print")
@cli.command(name="p", hidden=True)
def _print(ctx: Context):
    "Print the current config."
    print(ctx.obj.config._dict)


@cli.command()
@cli.command(name="s", hidden=True)
def save(ctx: Context):
    "Save the current config to the config file."
    ctx.obj.config.save()


@cli.command()
@cli.command(name="u", hidden=True)
def update(
    ctx: Context,
    name: Annotated[Optional[str], Argument()] = None,
    value: Annotated[Optional[str], Argument()] = None,
    kv: Annotated[Optional[str], Argument()] = None,
):
    "Update a config value, or set of values. (kv in the form of 'name1=value1,name2=value2')"
    if kv:
        updates = kv.split(",")
        for kv in updates:
            name, value = kv.split("=")
            ctx.obj.config._update(name, value, ctx.obj.config._dict)
        return
    ctx.obj.config._update(name, value, ctx.obj.config._dict)
