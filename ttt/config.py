import datetime
import logging
from pathlib import Path

import click
import tiktoken
import tttp
import yaml
from dotenv import load_dotenv

load_dotenv()

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
    config_path = config_dir / "config.yaml"

    @staticmethod
    def load_config():
        global config
        if not Config.config_path.exists():
            config = {}
            return
        config = yaml.load(Config.config_path.read_text(), Loader=yaml.Loader)
        return config

    @staticmethod
    def save_config(config):
        Config.config_path.write_text(yaml.dump(config))

    @staticmethod
    def create_config():
        Config.config_dir.mkdir(parents=True, exist_ok=True)
        Config.config_path.write_text(yaml.dump(Config.TURBO_TEXT_TRANSFORMER_DEFAULT_PARAMS))

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
    def check_config(reinit):
        """Check that the config file exists."""
        if not reinit and Config.config_path.exists():
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
    def check_chat(toggle, default, config):
        if toggle:
            config["chat"] = not config["chat"]
            click.echo(f"Chat mode is now {'on' if config['chat'] else 'off'}.", err=True)

        if default:
            config["chat_file"] = default

        if config["chat"] or config["chat_file"]:
            Config.save_config(config)

        if config["chat"]:
            click.echo(f"Chat mode is on. Using {config['chat_file']} as the chat history file.", err=True)


Config.load_config()
