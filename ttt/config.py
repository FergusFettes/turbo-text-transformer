import datetime
import logging
from pathlib import Path

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

config_dir = Path().home() / ".config/ttt"
config_path = config_dir / "config.yaml"
config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
