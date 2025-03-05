from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com/v1")

models = client.models.list()
print(models)