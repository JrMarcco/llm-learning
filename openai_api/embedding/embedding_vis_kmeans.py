import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import ast

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

df_embedded = pd.read_csv("output/embedded_fine_food_reviews_1k.csv", index_col=0)
df_embedded["embedding_vec"] = df_embedded["embedding"].apply(ast.literal_eval)

# 使用 t-SNE 可视化 1536 纬 embedding
# 确保嵌入向量都是等长的
assert df_embedded["embedding_vec"].apply(len).nunique() == 1

# 将嵌入向量列表转换为二维 numpy 数组
matrix = np.vstack(df_embedded["embedding_vec"].values)

# 定义要生成的聚类数
n_clusters = 4
# 创建一个 KMeans 对象，用于进行 K-Means 聚类。
# n_clusters 参数指定了要创建的聚类的数量；
# init 参数指定了初始化方法（在这种情况下是 'k-means++'）；
# random_state 参数为随机数生成器设定了种子值，用于生成初始聚类中心。
# n_init=10 消除警告 'FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4'
kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42, n_init=10)
kmeans.fit(matrix)

df_embedded["Cluster"] = kmeans.labels_

colors = ["red", "green", "blue", "black"]

tsne = TSNE(n_components=2, random_state=42)
vis_dims = tsne.fit_transform(matrix)

# 从降维后的坐标中获得所有数据点的横坐标和纵坐标
x = vis_dims[:, 0]
y = vis_dims[:, 1]

# 根据评分获取对应颜色的索引
color_indices = df_embedded["Cluster"].values

assert len(vis_dims) == len(df_embedded["Cluster"].values)

colormap = matplotlib.colors.ListedColormap(colors)
plt.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.3)


plt.title("Clustering visualized in 2D using t-SNE")
plt.show()
