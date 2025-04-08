import os
import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from experiments.advanced_experiments.graph_basics.graph_builder import build_propagation_graph


def simple_node_embedding(G: nx.Graph, embedding_dim: int = 16) -> dict:
    """生成随机节点嵌入（后续可替换为DeepWalk）"""
    embeddings = {}
    for node in G.nodes():
        embeddings[node] = np.random.randn(embedding_dim)
    return embeddings


def main():
    # 构建图 - 修正路径计算方式
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    data_path = os.path.join(project_root, 'data', 'raw', 'social_graph.csv')

    # 打印路径用于调试
    # print(f"尝试加载数据路径: {data_path}")
    G = build_propagation_graph(data_path)

    # 生成节点嵌入（示例：随机初始化）
    embeddings = simple_node_embedding(G)

    # 模拟节点标签（假设已有标签数据）
    nodes = list(G.nodes())
    labels = np.random.randint(0, 2, size=len(nodes))  # 假设二分类标签

    # 训练分类器
    X = np.array([embeddings[node] for node in nodes])
    clf = LogisticRegression()
    clf.fit(X, labels)
    print(f"图分类准确率: {clf.score(X, labels):.2f}")


if __name__ == "__main__":
    main()