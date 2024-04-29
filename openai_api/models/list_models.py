from pprint import pprint
from openai import OpenAI

client = OpenAI()

models = client.models.list()
pprint(models.data)
