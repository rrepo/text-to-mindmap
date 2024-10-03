import spacy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage

# 1. NLPのモデルをロード（文単位で処理）
nlp = spacy.load("en_core_web_md")  # 'md'モデルを使用してベクトルを扱う

# 2. 文単位でテキストを解析し、意味に基づく階層構造を作成する関数
def create_semantic_mind_map(text):
    doc = nlp(text)
    
    # 文章単位での分割
    sentences = list(doc.sents)
    
    # 各文のベクトルを取得
    vectors = np.array([sent.vector for sent in sentences])

    # 文同士の類似性（コサイン類似度）に基づく階層クラスタリング
    Z = linkage(vectors, method='ward')

    return Z, sentences

# 3. 階層クラスタリング結果を元にグラフを作成
def linkage_to_graph(Z, sentences):
    G = nx.Graph()
    
    # 文をノードとして追加
    for i, sentence in enumerate(sentences):
        G.add_node(i, label=sentence.text.strip())
    
    # クラスタ結合をエッジとして追加
    n = len(sentences)
    for i, (node1, node2, _, _) in enumerate(Z):
        new_node = n + i  # 新しいクラスタとしてのノード
        G.add_node(new_node, label=f"Cluster {new_node}")
        G.add_edge(new_node, int(node1))
        G.add_edge(new_node, int(node2))
    
    return G

# 4. Mind Mapをプロットする関数
def plot_mind_map(G):
    pos = nx.spring_layout(G, k=1.5, seed=42)
    labels = nx.get_node_attributes(G, 'label')
    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, with_labels=True, labels=labels, node_color='lightblue', font_size=10, font_weight='bold', node_size=3000, edge_color='gray')
    plt.title("Hierarchical Mind Map")
    plt.show()

# 5. テスト用の文章（意味に基づく階層化を確認）
text = """
Artificial intelligence is transforming many industries.
One of the key areas of AI is natural language processing.
It allows machines to understand and respond to human languages.
In business, AI is used for automating repetitive tasks.
It can also be used to make more informed decisions based on data.
Machine learning is another key area of AI that focuses on patterns in data.
"""

# 6. 階層クラスタリングの生成とグラフへの変換
Z, sentences = create_semantic_mind_map(text)
G = linkage_to_graph(Z, sentences)

# 7. Mind Mapのプロット
plot_mind_map(G)
