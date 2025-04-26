import logging
from openai import OpenAI
from google import genai
from .constants import APIProvider, API_CONFIGS

class APIClientFactory:
    @staticmethod
    def create_client(provider: APIProvider):
        config = API_CONFIGS[provider]
        if not config.api_key:
            raise EnvironmentError(f"Missing API key for {provider.value}")

        if provider in [APIProvider.DEEPSEEK, APIProvider.QWEN]:
            return OpenAI(api_key=config.api_key, base_url=config.base_url)
        elif provider == APIProvider.GEMINI:
            return genai.Client(api_key=config.api_key, http_options=config.http_options)
        elif provider == APIProvider.OPENAI:
            return OpenAI(api_key=config.api_key)
        else:
            raise ValueError(f"Unsupported API provider: {provider}")

class ModelInterface:
    def __init__(self, provider: APIProvider):
        self.provider = provider
        self.client = APIClientFactory.create_client(provider)
        self.config = API_CONFIGS[provider]

    def generate_completion(self, prompt: str) -> str:
        try:
            if self.provider in [APIProvider.DEEPSEEK, APIProvider.QWEN, APIProvider.OPENAI]:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False
                )
                return response.choices[0].message.content

            elif self.provider == APIProvider.GEMINI:
                response = self.client.models.generate_content(
                    model=self.config.model_name,
                    contents=prompt,
                    config={'thinking_config': {'include_thoughts': True}}
                )
                return response.text

        except Exception as e:
            logging.error(f"Error generating completion with {self.provider}: {str(e)}")
            raise

