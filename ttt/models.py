import datetime
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import click
import openai
import yaml

from ttt.config import config, config_dir, get_encoding
from ttt.formatter import Formatter

# from langchain import LLMChain
# from langchain.chat_models import ChatOpenAI
# from langchain.llms import OpenAI
# from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate


@dataclass
class BaseModel:
    operator: str = ""
    completion_url: str = ""
    config_base: Path = config_dir
    backup_path: Path = Path(config.get("backup_path", "/tmp/ttt/"))
    params: dict = field(default_factory=dict)
    format: str = config.get("format", "clean")
    echo_prompt: bool = config.get("echo_prompt", False)

    def __post_init__(self):
        self.backup_path.mkdir(parents=True, exist_ok=True)
        self.n = self.params.get("n", 1)

        self.formatter = Formatter(operator=self.operator, format=self.format, echo_prompt=self.echo_prompt)

    @staticmethod
    def token_position(token, text_offset):
        return {"start": text_offset, "end": text_offset + len(token)}

    def gen(self, prompt):
        return self.formatter.format_response([prompt] * self.n)


@dataclass
class OpenAIModel(BaseModel):
    operator: str = "OpenAI"
    completion_url: str = "https://api.openai.com/v1/completions"

    path: Path = config_dir / "openai.yaml"
    format: Optional[str] = None

    _model_list: list = field(default_factory=list)
    _params: dict = field(default_factory=dict)
    _config: dict = field(default_factory=dict)

    chat_models: list = field(default_factory=lambda: ["gpt-3.5-turbo-0301", "gpt-3.5-turbo", "gpt-4"])
    large_models: list = field(
        default_factory=lambda: [
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0301",
            "text-davinci-003",
            "text-davinci-002",
            "code-davinci-002",
        ]
    )

    def __post_init__(self):
        self._config = yaml.load(self.path.read_text(), Loader=yaml.FullLoader)

        self._params = self._config.get("engine_params", {})
        for k, _ in self._params.items():
            if k in self.params:
                self._params[k] = self.params[k]

        self._model_list = self._config.get("models", [])

        openai.api_key = self._config.get("api_key", "")

        super().__post_init__()

    @property
    def list(self):
        return self._model_list

    def update_list(self):
        full_list = openai.Model.list()
        self._model_list = [m["id"] for m in full_list["data"]]
        self._config["models"] = self._model_list

    def save(self):
        self._config["engine_params"] = self._params
        self.path.write_text(yaml.dump(self._config))

    def gen(self, prompt):
        encoding = get_encoding(self._params["model"])
        prompt_tokens = encoding.encode(prompt)
        max_tokens = 4000 if self._params["model"] in OpenAIModel().large_models else 2048
        if len(prompt_tokens) + int(self._params["max_tokens"]) > max_tokens:
            self._params["max_tokens"] = max_tokens - len(prompt_tokens)
            click.echo(
                "Prompt is too long. " f"Max tokens adjusted: {max_tokens}. " f"Prompt tokens: {len(prompt_tokens)}",
                err=True,
            )

        if self._params["model"] in self.chat_models:
            return self.formatter.format_response(self._chat(prompt))
        return self.formatter.format_response(self._gen(prompt))

    def _gen(self, prompt):
        self._params["prompt"] = prompt

        response = openai.Completion.create(**self._params).to_dict_recursive()
        self.backup(response)

        return response

    def _chat(self, prompt):
        params = self._params
        params["messages"] = [{"role": "user", "content": prompt}]
        if "prompt" in params:
            del params["prompt"]
        if "logprobs" in params:
            del params["logprobs"]

        response = openai.ChatCompletion.create(**params).to_dict_recursive()
        self.backup(response)

        for c in response["choices"]:
            c["text"] = c["message"]["content"]
            del c["message"]
        response["params"]["prompt"] = response["params"]["messages"][0]["content"]
        del response["params"]["messages"]

        return response

    def backup(self, response):
        timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
        response["params"] = self._params

        backup = self.backup_path / f"{timestamp}.json"
        backup.write_text(json.dumps(response, indent=4))

    @staticmethod
    def create_config(api_key):
        path = config_dir / "openai.yaml"
        from ttt.config import OPENAI_DEFAULT_PARAMS

        config = {
            "engine_params": OPENAI_DEFAULT_PARAMS,
            "api_key": api_key,
            "backup_path": str(BaseModel().backup_path),
            "models": [],
        }
        path.write_text(yaml.dump(config))

        model = OpenAIModel()
        model.update_list()
        model.save()
