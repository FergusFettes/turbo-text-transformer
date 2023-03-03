import datetime
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import openai
import yaml
from colored import attr, bg, fg

from ttt.config import config, config_dir, encoding


class ProbColors(Enum):
    # Colors of foreground and background for different probabilities
    prob_0 = (195, 196)
    prob_1 = (195, 202)
    prob_2 = (195, 208)
    prob_3 = (239, 220)
    prob_4 = (239, 226)
    prob_5 = (239, 76)
    prob_6 = (239, 82)


@dataclass
class Formatter:
    operator: str
    format: str = config.get("format", "clean")

    def format_response(self, response):
        if self.operator == "OpenAI":
            return self._openai(response)
        return self._base(response)

    def _openai(self, response):
        if self.format == "json":
            response = self._clean_json(response)
            return json.dumps(response, indent=4)
        if self.format == "logprobs":
            return self._logprobs(response)
        return "\n".join([c["text"] for c in response["choices"]])

    def _clean_json(self, response):
        for c in response["choices"]:
            c["token_logprobs"] = c.get("logprobs", {}).get("token_logprobs", None)
            c["logprob_offset"] = c.get("logprobs", {}).get("text_offset", None)
            if c.get("logprobs", None):
                del c["logprobs"]
            if c.get("index", None):
                del c["index"]

            # If the first token is a newline, remove it
            if len(c["text"]) == 0:
                continue
            while c["text"][0] == "\n":
                c["text"] = c["text"][1:]
                # And drop the first token logprob
                if c.get("token_logprobs", None):
                    c["token_logprobs"] = c["token_logprobs"][1:]
                    # And update the offsets
                    c["logprob_offset"] = [offset - 1 for offset in c["logprob_offset"]]
                    c["logprob_offset"] = c["logprob_offset"][1:]

        return response

    def _logprobs(self, response):
        response = self._clean_json(response)
        prompt_offset = len(response["params"]["prompt"])
        for c in response["choices"]:
            c["colorized_tokens"] = self._colorize(
                c["text"], c["token_logprobs"], [offset - prompt_offset for offset in c["logprob_offset"]]
            )
        return "\n".join([c["colorized_tokens"] for c in response["choices"]])

    def _colorize(self, text, token_logprobs, offset):
        colorized_string = ""
        for i, logprob in enumerate(token_logprobs):
            start = offset[i]
            end = offset[i + 1] if i < len(offset) - 1 else len(text)
            # Start with a set of colors
            fg_, bg_ = self.choose_color(logprob)
            colorized_string += f"{bg(bg_)}{fg(fg_)}{text[start:end]}{attr(0)}"
        return colorized_string

    def choose_color(self, logprob):
        if logprob < -0.5:
            return ProbColors.prob_0.value
        if logprob < -0.4:
            return ProbColors.prob_1.value
        if logprob < -0.3:
            return ProbColors.prob_2.value
        if logprob < -0.2:
            return ProbColors.prob_3.value
        if logprob < -0.1:
            return ProbColors.prob_4.value
        if logprob < 0.01:
            return ProbColors.prob_5.value
        return ProbColors.prob_6.value

    def _base(self, response):
        if self.format == "json":
            response_dict = {"choices": [{"text": c} for c in response]}
            return json.dumps(response_dict, indent=4)
        return "\n".join(response)


@dataclass
class BaseModel:
    model: str = ""
    completion_url: str = ""
    operator: str = ""
    config_base: Path = config_dir
    backup_path: Path = Path("/tmp/ttt/")
    params: dict = field(default_factory=dict)
    format: str = config.get("format", "clean")

    def __post_init__(self):
        self.backup_path.mkdir(parents=True, exist_ok=True)
        self.n = self.params.get("n", 1)

        self.formatter = Formatter(operator=self.operator, format=self.format)

    @staticmethod
    def token_position(token, text_offset):
        return {"start": text_offset, "end": text_offset + len(token)}

    def gen(self, prompt):
        return self.formatter.format_response([prompt] * self.n)


@dataclass
class OpenAIModel(BaseModel):
    operator: str = "OpenAI"
    completion_url: str = "https://api.openai.com/v1/completions"

    model: Optional[str] = None
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
        self.check_max_tokens(prompt)
        self._params["prompt"] = prompt
        if self._params["model"] in self.chat_models:
            return self.formatter.format_response(self._chat(prompt))
        return self.formatter.format_response(self._gen(prompt))

    def check_max_tokens(self, prompt):
        prompt_tokens = encoding.encode(prompt)
        max_tokens = 4000 if self._params["model"] in self.large_models else 2048
        if len(prompt_tokens) > max_tokens:
            raise ValueError(f"Prompt is too long. Max tokens: {max_tokens}")
        if len(prompt_tokens) + self._params["max_tokens"] > max_tokens:
            self._params["max_tokens"] = max_tokens - len(prompt_tokens)

    def _gen(self, prompt):
        self._params["prompt"] = prompt

        response = openai.Completion.create(**self._params).to_dict_recursive()
        self.backup(response)

        return response

    def _chat(self, prompt):
        params = self._params
        params["messages"] = [{"role": "user", "content": prompt}]
        del params["prompt"]
        del params["logprobs"]

        response = openai.ChatCompletion.create(**params).to_dict_recursive()
        self.backup(response)

        # Set the 'text' field to the 'message'.'content' field
        for c in response["choices"]:
            c["text"] = c["message"]["content"]
            del c["message"]

        return response

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
