import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import ast

from sklearn.manifold import TSNE

df_embedded = pd.read_csv("output/embedded_fine_food_reviews_1k.csv", index_col=0)
df_embedded["embedding_vec"] = df_embedded["embedding"].apply(ast.literal_eval)

# 使用 t-SNE 可视化 1536 纬 embedding
# 确保嵌入向量都是等长的
assert df_embedded["embedding_vec"].apply(len).nunique() == 1

# 将嵌入向量列表转换为二维 numpy 数组
matrix = np.vstack(df_embedded["embedding_vec"].values)

# 创建 t-SNE 模型（t-SNE 是一个非线性降维方法，常用于高维数据可视化）
# n_components 表示降维后的纬度
# perplexity 可以被理解为近邻的数量
# random_state 是随机数生成器的种子
# init 设置初始化方式
# learning_rate 是学习率
tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)

# 使用 t-SNE 对数据降维
vis_dims = tsne.fit_transform(matrix)
colors = ["red", "darkorange", "gold", "turquoise", "darkgreen"]

# 从降维后的坐标中获得所有数据点的横坐标和纵坐标
x = vis_dims[:, 0]
y = vis_dims[:, 1]

# 根据评分获取对应颜色的索引
color_indices = df_embedded.Score.values - 1

assert len(vis_dims) == len(df_embedded.Score.values)

colormap = matplotlib.colors.ListedColormap(colors)
plt.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.3)

plt.title("Amazon ratings visualized in language using t-SNE")
plt.show()
