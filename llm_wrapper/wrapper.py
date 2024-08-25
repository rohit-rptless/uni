import os
import requests

class LLMWrapper:
    def __init__(self, config):
        """
        Initialize the LLMWrapper with a given configuration.
        
        :param config: Dictionary containing model configurations.
                       Example: {"model": "openai:gpt4-o", "endpoint": "", "api_key_env_var": ""}
        """
        self.config = config

    def chat_completions(self, prompt):
        """
        Make a chat completion request using the configured model.
        
        :param prompt: The input prompt for the model.
        :return: The model's response.
        """
        model = self.config.get('model')
        
        if not model:
            raise ValueError("Model configuration is missing.")

        if model.startswith("openai:"):
            return self._openai_request(prompt)
        elif model.startswith("anthropic:"):
            return self._anthropic_request(prompt)
        else:
            raise ValueError(f"Unsupported model: {model}")
        # TODO: Add more models.

    def _openai_request(self, prompt):
        headers = {
            'Authorization': f"Bearer {os.getenv(self.config['api_key_env_var'])}",
            'Content-Type': 'application/json'
        }
        data = {
            'model': 'gpt-4',
            'messages': [{'role': 'user', 'content': prompt}],
        }
        response = requests.post(self.config['endpoint'], headers=headers, json=data)
        return response.json()['choices'][0]['message']['content']

    def _anthropic_request(self, prompt):
        headers = {
            'Authorization': f"Bearer {os.getenv(self.config['api_key_env_var'])}",
            'Content-Type': 'application/json'
        }
        data = {
            'model': 'claude-3',
            'prompt': prompt,
            'max_tokens': 100  # TODO: Make this configurable.
        }
        response = requests.post(self.config['endpoint'], headers=headers, json=data)
        return response.json()['completion']