import os
import unittest
from unittest.mock import patch
from llm_wrapper.wrapper import LLMWrapper

class TestLLMWrapper(unittest.TestCase):

    def setUp(self):
        self.wrapper_openai = LLMWrapper({
            "model": "openai:gpt4-o",
            "endpoint": "https://api.openai.com/v1/completions",
            "api_key_env_var": "OPENAI_API_KEY"
        })

        self.wrapper_anthropic = LLMWrapper({
            "model": "anthropic:claude3",
            "endpoint": "https://api.anthropic.com/v1/complete",
            "api_key_env_var": "ANTHROPIC_API_KEY"
        })

    @patch('requests.post')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-openai-key'})
    def test_openai_request_format(self, mock_post):
        self.wrapper_openai.chat_completions('Test prompt')
        expected_headers = {
            'Authorization': 'Bearer test-openai-key',
            'Content-Type': 'application/json'
        }
        expected_data = {
            'model': 'gpt-4',
            'messages': [{'role': 'user', 'content': 'Test prompt'}]
        }
        mock_post.assert_called_once_with(
            'https://api.openai.com/v1/completions',
            headers=expected_headers,
            json=expected_data
        )

    @patch('requests.post')
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-anthropic-key'})
    def test_anthropic_request_format(self, mock_post):
        self.wrapper_anthropic.chat_completions('Test prompt')
        expected_headers = {
            'Authorization': 'Bearer test-anthropic-key',
            'Content-Type': 'application/json'
        }
        expected_data = {
            'model': 'claude-3',
            'prompt': 'Test prompt',
            'max_tokens': 100
        }
        mock_post.assert_called_once_with(
            'https://api.anthropic.com/v1/complete',
            headers=expected_headers,
            json=expected_data
        )

if __name__ == '__main__':
    unittest.main()
