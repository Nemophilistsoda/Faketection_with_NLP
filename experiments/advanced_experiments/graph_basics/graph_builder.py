import pandas as pd
import networkx as nx

def build_propagation_graph(data_path: str) -> nx.Graph:
    """构建用户传播图"""
    df = pd.read_csv(data_path)
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['user_id'], row['source_user'], weight=row['interaction_count'])
    return G