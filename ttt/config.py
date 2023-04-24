import datetime
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import click
import tiktoken
import tttp
import yaml

file_path = Path("/tmp/ttt/")
file_path.mkdir(parents=True, exist_ok=True)

log_path = file_path / "logs"
log_path.mkdir(parents=True, exist_ok=True)

timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
logfile = log_path / f"{timestamp}.log"

logging.basicConfig(
    filename=logfile,
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
    TURBO_TEXT_TRANSFORMER_DEFAULT_PARAMS = {"format": "clean", "echo_prompt": False, "backup_path": "/tmp/ttt/"}
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
    }
    config_dir = Path().home() / ".config/ttt"

    def __post_init__(self):
        self.config_path = self.config_dir / "config.yaml"

    @staticmethod
    def load_config():
        global config
        if not Config().config_path.exists():
            return {}
        config = yaml.load(Config().config_path.read_text(), Loader=yaml.Loader)
        return config

    @staticmethod
    def save_config(config):
        Config().config_path.write_text(yaml.dump(config))

    @staticmethod
    def create_config():
        Config.config_dir.mkdir(parents=True, exist_ok=True)
        Config().config_path.write_text(yaml.dump(Config.TURBO_TEXT_TRANSFORMER_DEFAULT_PARAMS))

        # Find the templates, and make sure they are in the right place
        tttp_dir = Path(tttp.__file__).parent
        new_templates = tttp_dir.parent / "templates"
        templates = Config.config_dir / "templates"
        templates.mkdir(parents=True, exist_ok=True)
        for template in new_templates.glob("*.j2"):
            if not (templates / template.name).exists():
                (templates / template.name).write_text(template.read_text())

    @staticmethod
    def create_openai_config(api_key):
        path = Config.config_dir / "openai.yaml"

        config = {
            "engine_params": Config.OPENAI_DEFAULT_PARAMS,
            "api_key": api_key,
            "backup_path": str(config.get("backup_path", "/tmp/ttt/")),
            "models": [],
        }
        path.write_text(yaml.dump(config))

    @staticmethod
    def load_openai_config():
        path = Config.config_dir / "openai.yaml"
        return yaml.load(path.read_text(), Loader=yaml.Loader)

    @staticmethod
    def save_openai_config(config):
        path = Config.config_dir / "openai.yaml"
        path.write_text(yaml.dump(config))

    @staticmethod
    def check_config(reinit=False):
        """Check that the config file exists."""
        openai_config = Config.load_openai_config()
        os.environ["OPENAI_API_KEY"] = openai_config["api_key"]
        if not reinit and Config().config_path.exists():
            return config

        click.echo("Config file not found. Creating one for you...", err=True, color="red")
        Config.create_config()
        openai_api_key = click.prompt("OpenAI API Key", type=str)
        if openai_api_key:
            Config.create_openai_config(openai_api_key)

        return Config.load_config()

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
            config["file"] = not config["file"]

        if default:
            config["chat_name"] = default

        if toggle or config["chat_name"]:
            Config.save_config(config)

        if config["file"]:
            click.echo(f"File mode is on. Using {config['chat_name']} as the chat history file.", err=True)
        else:
            click.echo(f"File mode is off.", err=True)

    @staticmethod
    def check_template(toggle, default, config):
        if toggle:
            config["template"] = not config["template"]

        if default:
            config["template_file"] = default

        if toggle or config["template_file"]:
            Config.save_config(config)

        if config["template"]:
            click.echo(f"Template mode is on. Using {config['template_file']} as the template file.", err=True)
        else:
            click.echo(f"Template mode is off.", err=True)


Config.load_config()
