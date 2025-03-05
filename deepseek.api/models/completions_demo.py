from openai import OpenAI
import os

client = OpenAI(
  api_key=os.getenv("DEEPSEEK_API_KEY"),
  base_url="https://api.deepseek.com/v1"
)
COMPLETION_MODEL = "deepseek-chat"

prompt = """
Consideration proudct : 工厂现货PVC充气青蛙夜市地摊热卖充气玩具发光蛙儿童水上玩具

1. Compose human readale product title used on Amazon in english within 20 words.
2. Write 5 selling points for the products in Amazon.
3. Evaluate a price range for this product in U.S.

Output the result in json format with three properties called title, selling_points and price_range
"""

def get_response(prompt):
    completions = client.chat.completions.create (
        model=COMPLETION_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.7,        
        stream=False
    )
    return completions.choices[0].message.content

print(get_response(prompt)) 