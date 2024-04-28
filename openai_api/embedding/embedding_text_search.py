import numpy as np
import pandas as pd
import ast

from openai import OpenAI

client = OpenAI()


def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


# 计算两个嵌入向量之间的余弦相似度
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search_reviews(df, product_description, n=3, pprint=True):
    product_embedding = get_embedding(product_description)

    df["similarity"] = df.embedding_vec.apply(lambda x: cosine_similarity(x, product_embedding))

    rs = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .combined.str.replace("Title", "")
        .str.replace("; Content:", ": ")
    )

    if pprint:
        for r in rs:
            print(r[:200])
            print()

    return rs


df_embedded = pd.read_csv("output/embedded_fine_food_reviews_1k.csv", index_col=0)
df_embedded["embedding_vec"] = df_embedded["embedding"].apply(ast.literal_eval)

res = search_reviews(df_embedded, "awful", n=5, pprint=True)
