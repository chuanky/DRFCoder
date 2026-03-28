import os

from openai import OpenAI
from transformers import AutoTokenizer
from openai.types import Completion
from openai.types.chat import ChatCompletion

from src import utils

logger = utils.logger


class ApiClient:
    def __init__(self):
        api_key = os.environ["DEEPSEEK_API"]
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.serve_model_name = "deepseek-chat"

    def chat(self, prompt):
        response = self.client.chat.completions.create(
            model=self.serve_model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        res = response.choices[0].message.content
        return res

    def complete(self, messages, max_tokens=128, temperature=0.7, stop=["## Task"]):
        logger.debug(messages)
        response: ChatCompletion = self.client.chat.completions.create(
            model=self.serve_model_name,
            messages=messages,
            stream=False,
            stop=stop,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )

        return {
            "output": response.choices[0].message.content,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }


class LocalClient:
    def __init__(self, model_path="/sda/models/Qwen/Qwen3-4B-Instruct-2507", serve_model_name="Qwen3-4B", think=False):
        self.base_url = "http://localhost:8000/v1"
        self.model_path = model_path
        self.serve_model_name = serve_model_name
        self.client = OpenAI(base_url=self.base_url, api_key="EMPTY")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.think = think

        logger.info(f"vLLM should be running on {self.base_url} with model {self.model_path}")

    def complete(self, messages, max_tokens=128, temperature=0.7, stop=["## Task"]):
        chat_prompt = utils.get_chat_prompt(messages, self.tokenizer)
        logger.debug(chat_prompt)

        if self.serve_model_name == "Qwen3-9B":
            response: Completion = self.client.chat.completions.create(
                model=self.serve_model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": self.think},
                },
            )

            return {
                "output": response.choices[0].message.content,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
        else:
            response: Completion = self.client.completions.create(
                model=self.serve_model_name,
                prompt=chat_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )

            return {
                "output": response.choices[0].text,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }


if __name__ == "__main__":
    from src import utils

    # client = LocalClient()
    client = LocalClient("/sda/models/Qwen3.5-9B", "Qwen3-9B", False)

    prompt = 'hello'
    messages = utils.get_messages(prompt)

    response = client.complete(messages, max_tokens=1024)
    print(response['output'])
    print(response['input_tokens'])
    print(response['output_tokens'])
