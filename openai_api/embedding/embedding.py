import pandas as pd
import tiktoken
from openai import OpenAI

embedding_model = "text-embedding-3-small"
embedding_encoding = "cl100k_base"
max_tokens = 8000

in_data_path = "input/fine_food_reviews_1k.csv"

df = pd.read_csv(in_data_path, index_col=0)
df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]
df = df.dropna()

df["combined"] = (
    "Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip()
)

# 讲样本减少到最近的 1,000 个，并删除过长样本。
top_n = 1000

df = df.sort_values("Time").tail(top_n * 2)
df.drop("Time", axis=1, inplace=True)

encoding = tiktoken.get_encoding(embedding_encoding)

df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))

# 生成 embeddings 并保存
client = OpenAI()


def get_embedding(text, model=embedding_model):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


df["embedding"] = df.combined.apply(get_embedding)
df.to_csv("output/embedded_fine_food_reviews_1k.csv", index=False)
