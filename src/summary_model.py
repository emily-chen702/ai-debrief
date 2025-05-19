from google import genai
from jinja2 import Environment, FileSystemLoader
import re 
import json
import os

class SummaryModel:
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name
        api_key = api_key if api_key else os.environ['GEMINI_API_KEY']
        if api_key:
            self.client = self.get_client(api_key)
        else:
            raise ValueError("Please set an environment key as an environment variable or give as an argument.")
    
    def get_client(self, api_key: str):
        return genai.Client(api_key=api_key)
    
    def apply_prompt_template(self, sentence: str) -> str:
        env = Environment(loader=FileSystemLoader(""))
        template = env.get_template("resources/summary_prompt.txt")
        output = template.render({"example_sentence": sentence})
        return output

    def format_result(self, response: str) -> dict:
        print(response)
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
        json_str = match.group(1)
        return json.loads(json_str)

    def compute(self, sentence: str) -> dict:
        prompt = self.apply_prompt_template(sentence)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return self.format_result(response.text)