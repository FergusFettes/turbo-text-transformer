import json
import math
from dataclasses import dataclass

from colored import attr, bg, fg

from ttt.config import config


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
