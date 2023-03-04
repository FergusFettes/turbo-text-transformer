import datetime
import logging
from pathlib import Path

import tiktoken
import tttp
import yaml

encoding = tiktoken.get_encoding("gpt2")

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

config_dir = Path().home() / ".config/ttt"
config_path = config_dir / "config.yaml"


def load_config():
    global config
    if not config_path.exists():
        config = {}
        return
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)


def create_config():
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.dump(TURBO_TEXT_TRANSFORMER_DEFAULT_PARAMS))

    # Find the templates, and make sure they are in the right place
    tttp_dir = Path(tttp.__file__).parent
    new_templates = tttp_dir.parent / "templates"
    templates = config_dir / "templates"
    templates.mkdir(parents=True, exist_ok=True)
    for template in new_templates.glob("*.j2"):
        if not (templates / template.name).exists():
            (templates / template.name).write_text(template.read_text())


TURBO_TEXT_TRANSFORMER_DEFAULT_PARAMS = {"format": "clean", "echo_prompt": False, "backup_path": "/tmp/ttt/"}

OPENAI_DEFAULT_PARAMS = {
    "frequency_penalty": 0,
    "logprobs": 1,
    "max_tokens": 50,
    "model": "gpt-3.5-turbo",
    "n": 4,
    "presence_penalty": 0,
    "stop": None,
    "temperature": 0.9,
    "top_p": 1,
}


load_config()


def arg2dict(args):
    d = {}
    if "=" not in args:
        return d
    for arg in args.split(","):
        k, v = arg.split("=")
        d[k] = v
    return d
