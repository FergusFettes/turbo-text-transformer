import datetime
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import click
import openai
import yaml
from colored import attr, bg, fg

from ttt.config import config, config_dir, encoding


@dataclass
class ProbColors:
    # Colors of foreground and background for different probabilities
    prob_1 = (239, 49)
    prob_2 = (239, 78)
    prob_3 = (195, 145)
    prob_4 = (195, 173)
    prob_5 = (195, 209)
    prob_6 = (195, 203)

    @staticmethod
    def choose_color(logprob):
        prob = math.exp(logprob)
        if prob >= 0.8:
            return ProbColors.prob_1
        if prob >= 0.6:
            return ProbColors.prob_2
        if prob >= 0.4:
            return ProbColors.prob_3
        if prob >= 0.2:
            return ProbColors.prob_4
        if prob >= 0.05:
            return ProbColors.prob_5
        return ProbColors.prob_6


@dataclass
class Formatter:
    operator: str
    format: str = config.get("format", "clean")
    echo_prompt: bool = config.get("echo_prompt", False)

    def format_response(self, response):
        if self.operator == "OpenAI":
            return self._openai(response)
        return self._base(response)

    def _openai(self, response):
        if self.format == "json":
            response = self._clean_json(response)
            return json.dumps(response, indent=4)
        if self.format == "logprobs":
            response = self._logprobs(response)
        if self.echo_prompt:
            return "\n".join([response["params"]["prompt"] + c["text"] for c in response["choices"]])
        return "\n".join([c["text"] for c in response["choices"]])

    def _clean_json(self, response):
        for choice in response["choices"]:
            if "logprobs" not in choice:
                continue
            if choice.get("logprobs", None) is None:
                del choice["logprobs"]
                del choice["index"]
                continue

            choice["token_logprobs"] = choice.get("logprobs", {}).get("token_logprobs", None)
            choice["logprob_offset"] = choice.get("logprobs", {}).get("text_offset", None)
            if choice.get("logprobs", None):
                del choice["logprobs"]
            if choice.get("index", None):
                del choice["index"]

            # # If the first token is a newline, remove it
            # if len(choice["text"]) == 0:
            #     continue
            # while choice["text"][0] == "\n":
            #     choice["text"] = choice["text"][1:]
            #     # And drop the first token logprob
            #     if choice.get("token_logprobs", None):
            #         choice["token_logprobs"] = choice["token_logprobs"][1:]
            #         # And update the offsets
            #         choice["logprob_offset"] = [offset - 1 for offset in choice["logprob_offset"]]
            #         choice["logprob_offset"] = choice["logprob_offset"][1:]

        return response

    def _logprobs(self, response):
        response = self._clean_json(response)
        prompt_offset = len(response["params"]["prompt"])
        if response["choices"][0].get("token_logprobs", None) is None:
            return response

        for c in response["choices"]:
            colorized = self._colorize(
                c["text"], c["token_logprobs"], [offset - prompt_offset for offset in c["logprob_offset"]]
            )
            c["text"] = colorized

        return response

    def _colorize(self, text, token_logprobs, offset):
        colorized_string = ""
        for i, logprob in enumerate(token_logprobs):
            start = offset[i]
            end = offset[i + 1] if i < len(offset) - 1 else len(text)
            # Start with a set of colors
            fg_, bg_ = ProbColors.choose_color(logprob)
            colorized_string += f"{bg(bg_)}{fg(fg_)}{text[start:end]}{attr(0)}"
        return colorized_string

    def _base(self, response):
        if self.format == "json":
            response_dict = {"choices": [{"text": c} for c in response]}
            return json.dumps(response_dict, indent=4)
        return "\n".join(response)


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

    _list: list = field(default_factory=list)
    _params: dict = field(default_factory=dict)
    _config: dict = field(default_factory=dict)

    chat_models: list = field(default_factory=lambda: ["gpt-3.5-turbo-0301", "gpt-3.5-turbo"])
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

        self._list = self._config.get("models", [])

        openai.api_key = self._config.get("api_key", "")

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
