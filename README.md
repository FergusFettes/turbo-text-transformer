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
* `--number` or `-n`: The number of completions to generate. Default is 1.
* `--list_models` or `-l`: List available models.

### Models

Turbo Text Transformer includes support for several models:

* `davinci`: The default model, provides highly coherent long-form text.
* `babbage`: The best model for code completion and generation.
* `ada`: The most capable model for natural language processing and generating highly coherent short-form text.
* `curie`: A highly capable model for natural language processing and generating highly coherent short-form text.

### Customization

You can customize the behavior of Turbo Text Transformer by creating your own models. To do so, you can create a new Python file in the `ttt` directory and define it according to the following template:

```python
from ttt.models import BaseModel


class MyModel(BaseModel):
    model = "my_model"
    completion_url = "https://api.openai.com/v1/completions"
    operator = "OpenAI"
    params = {
        "model": model,
        "max_tokens": 50,
        "temperature": 0.5,
    }

    def gen(self, prompt):
        # Your text generation logic here
        return ["generated text"]
```

Replace `MyModel` with the name of your model and implement the `gen` method with your text generation logic. You can then use your model with the `--model` option.

## Contributing

If you find a bug or would like to contribute to Turbo Text Transformer, please create a new GitHub issue or pull request.

## License

Turbo Text Transformer is released under the MIT License. See `LICENSE` for more information.
