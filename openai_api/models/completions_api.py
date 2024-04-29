from openai import OpenAI
from pprint import pprint

client = OpenAI()

messages = [
    {"role": "system", "content": "你是一个经验丰富的程序开发者"},
    {"role": "user", "content": "Linux上如何设置定时任务？"}
]

data = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)

pprint(data.choices[0].message.content)
