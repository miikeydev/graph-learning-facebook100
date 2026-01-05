import os
import csv
import argparse
import networkx as nx
import community.community_louvain as louvain
from sklearn.metrics import normalized_mutual_info_score

def load_lcc(path):
    G = nx.read_gml(path)
    if nx.is_empty(G): return None
    lcc = max(nx.connected_components(G), key=len)
    return G.subgraph(lcc).copy()

def get_labels(G, attr):
    node_attr = nx.get_node_attributes(G, attr)
    nodes, labels = [], []
    for n in G.nodes():
        val = node_attr.get(n, 0)
        if val != 0:
            nodes.append(n)
            labels.append(val)
    return nodes, labels

def run_q6(data_dir, graphs):
    print(f"{'Network':<15} | {'Attribute':<12} | {'NMI Score':<10}")
    print("-" * 45)
    
    attributes = ["dorm", "year", "major", "gender"]
    results = []

    for g_name in graphs:
        path = os.path.join(data_dir, f"{g_name}.gml")
        if not os.path.exists(path): continue
            
        G = load_lcc(path)
        
        partition = louvain.best_partition(G)
        
        for attr in attributes:
            valid_nodes, true_labels = get_labels(G, attr)
            if len(valid_nodes) < 50: continue

            pred_labels = [partition[n] for n in valid_nodes]
            
            nmi = normalized_mutual_info_score(true_labels, pred_labels)
            
            print(f"{g_name:<15} | {attr:<12} | {nmi:.4f}")
            results.append([g_name, attr, nmi])

    os.makedirs("results/q6", exist_ok=True)
    with open("results/q6/community_nmi.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["network", "attribute", "nmi"])
        writer.writerows(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--graphs", nargs="*", default=["Caltech36", "Reed98", "Rice31", "Smith60"])
    args = parser.parse_args()
    run_q6(args.data_dir, args.graphs)
    