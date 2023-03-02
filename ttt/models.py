import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import openai
import yaml

from ttt.config import config_path


@dataclass
class BaseModel:
    model: str = ""
    completion_url: str = ""
    operator: str = ""
    config_base: Path = config_path
    backup_path: Path = Path("/tmp/ttt/")
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.backup_path.mkdir(parents=True, exist_ok=True)
        self.n = self.params.get("n", 1)

    @staticmethod
    def token_position(token, text_offset):
        return {"start": text_offset, "end": text_offset + len(token)}

    def gen(self, prompt):
        return [prompt] * self.n


@dataclass
class OpenAIModel(BaseModel):
    operator: str = "OpenAI"
    completion_url: str = "https://api.openai.com/v1/completions"

    model: Optional[str] = None
    path: Path = config_path / "openai.yaml"

    _list: list = field(default_factory=list)
    _params: dict = field(default_factory=dict)
    _config: dict = field(default_factory=dict)

    chat_models: list = field(default_factory=lambda: ["gpt-3.5-turbo-0301", "gpt-3.5-turbo"])

    def __post_init__(self):
        self._config = yaml.load(self.path.read_text(), Loader=yaml.FullLoader)
        if self._config.get("backup_path", None):
            self.backup_path = Path(self._config["backup_path"])

        self._params = self._config["engine_params"]
        self._params.update(self.params)
        if self.model:
            self._params.update({"model": self.model})
        self._list = self._config["models"]
        self.api_key = self._config["api_key"]

        openai.api_key = self.api_key

        super().__post_init__()

    @property
    def list(self):
        return self._list

    def update_list(self):
        full_list = openai.Model.list()
        self._list = [m["id"] for m in full_list["data"]]
        self._config["models"] = self._list

    def save(self):
        self._config["engine_params"] = self._params
        self.path.write_text(yaml.dump(self._config))

    def gen(self, prompt):
        self._params["prompt"] = prompt
        if self._params["model"] in self.chat_models:
            return self._chat(prompt)
        return self._gen(prompt)

    def _gen(self, prompt):
        self._params["prompt"] = prompt

        response = openai.Completion.create(**self._params)
        self.backup(response)

        return [c["text"] for c in response["choices"]]

    def _chat(self, prompt):
        params = self._params
        params["messages"] = [{"role": "user", "content": prompt}]
        del params["prompt"]
        del params["logprobs"]

        response = openai.ChatCompletion.create(**params)
        self.backup(response)

        return [c["message"]["content"] for c in response["choices"]]

    def backup(self, response):
        timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
        response["params"] = self._params

        backup = self.backup_path / f"{timestamp}.json"
        backup.write_text(json.dumps(response, indent=4))


@dataclass
class GooseAIModel(OpenAIModel):
    operator: str = "GooseAI"
    completion_url: str = "https://api.goose.ai/v1/completions"


@dataclass
class AI21Model(BaseModel):
    operator: str = "AI21"
    completion_url: str = "https://api.ai21.com/studio/v1/completions"

    # def _gen(prompt):
    #     response = requests.post(
    #         f"https://api.ai21.com/studio/v1/{engine}/complete",
    #         headers={"Authorization": f"Bearer {api_key}"},
    #         json=request_json,
    #     )
