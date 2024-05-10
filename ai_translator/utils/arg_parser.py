import argparse


class ArgParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Translate English PDF file to Chinese.")
        self.parser.add_argument(
            "--config", type=str, default="config.yaml", help="Configuration file with model and API settings."
        )
        self.parser.add_argument(
            "--model_type", type=str, required=True, choices=["GLMModel", "OpenAIModel"],
            help="The type of translation model to use. Choose between 'GLMModel' and 'OpenAIModel'."
        )
        self.parser.add_argument("--glm_model_url", type=str, help="The URL of the ChatGLM model URL.")
        self.parser.add_argument("--timeout", type=int, help="Timeout for the API request in seconds.")
        self.parser.add_argument(
            "--openai_model", type=str, help="The model name of OpenAI Model. Required if model_type is 'OpenAiModel'."
        )
        self.parser.add_argument(
            "--open_ai_key", type=str, help="The API Key for OpenAI Model. Required if model_type is 'OpenAIModel'."
        )
        self.parser.add_argument("--file", type=str, help="PDF file to translate.")
        self.parser.add_argument(
            "--file_format", type=str, help="The file format of translated file. Now supporting PDF and Markdown."
        )

    def parse_args(self):
        args = self.parser.parse_args()
        if args.model_type == "OpenAIModel" and not args.openai_model and not args.openai_api_key:
            self.parser.error("--openai_model and --openai_api_key is required when using OpenAI Model.")
        return args

