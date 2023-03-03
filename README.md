# Turbo Text Transformer

Turbo Text Transformer is a Python command-line tool for generating text using OpenAI's GPT-3 and other models. It includes a modular model system that allows for easy integration of new models and customization of existing ones.

Best used in combination with the [Turbo Text Transformer Prompts](https://github.com/fergusfettes/turbo-text-transformer-prompts) repository!

## Configuration

Configs are in the `.config` folder, put your api key in there

```~/.config/ttt/openai.yaml
api_key: sk-<your api key here>
engine_params:
  frequency_penalty: 0
  logprobs: null
  max_tokens: 1000
  model: davinci
  n: 4
  presence_penalty: 0
  stop: null
  temperature: 0.9
  top_p: 1
models:
- babbage
- davinci
- gpt-3.5-turbo-0301
- text-davinci-003
etc.
```

## Installation

To install Turbo Text Transformer, you can use pip:

```sh
pip install turbo-text-transformer
```

or clone the repository and install it manually:

```sh
git clone https://github.com/fergusfettes/turbo-text-transformer.git
cd turbo-text-transformer
poetry install
```

## Usage

You can use Turbo Text Transformer by running the `ttt` command in your terminal:

```sh
ttt --model davinci --prompt "Hello, GPT-3!"
```

The above example will generate text using the davinci model and the prompt "Hello, GPT-3!".

### Options

There are several options you can use with the `ttt` command:

* `--model` or `-m`: The name of the model to use. Default is "davinci".
* `--prompt` or `-p`: The prompt to use for text generation.
* `--list_models` or `-l`: List available models.
- `--echo_prompt, -e`: Whether to echo the prompt in the output.
- `--format, -f FORMAT`: The format of the output. Can be "clean", "json", or "logprobs". Defaults to "clean".
- `--number, -n NUMBER`: The number of completions to generate. Defaults to 1.
- `--logprobs, -L LOGPROBS`: Whether to show logprobs for each completion. Defaults to False.
- `--max_tokens, -M MAX_TOKENS`: The maximum number of tokens to return. Defaults to None.

## Configuration

Before using Turbo Text Transformer, you need to set up a configuration file. This should happen automatically when you run the `ttt` command for the first time:
This will create a configuration file in your home directory. You'll also be prompted to enter API keys for the transformer models you want to use. See the documentation for each model to learn how to obtain an API key.

## Examples

Here are some examples of how to use Turbo Text Transformer:

```
# Generate text with the default model
ttt -p "Once upon a time, there was a"

# Generate text with a specific model
ttt -m gpt-2-medium "The meaning of life is"

# Generate multiple completions
ttt -n 5 "I like to eat"

# Show logprobs
ttt -L 1 "I like to eat"

# Use the JSON format
ttt -f json "I like to eat"
```

You can also tell it to output a formatted json file with the `-f json` flag. This is useful for piping into other programs.

```
ttt -f json "The cat sat on the"
```

### Models

Turbo Text Transformer includes support for text generation with all the openai models. Have a look at the model list with `ttt -l`.

## Contributing

If you find a bug or would like to contribute to Turbo Text Transformer, please create a new GitHub issue or pull request.

## License

Turbo Text Transformer is released under the MIT License. See `LICENSE` for more information.
