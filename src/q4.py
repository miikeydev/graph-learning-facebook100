import os
import csv
import argparse
import random
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from q4_link_prediction import CommonNeighbors, Jaccard, AdamicAdar


METRICS = {
    "cn": CommonNeighbors,
    "jaccard": Jaccard,
    "aa": AdamicAdar,
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_lcc(path):
    G = nx.read_gml(path)
    lcc = max(nx.connected_components(G), key=len)
    return G.subgraph(lcc).copy()


def edge_key(u, v):
    return (u, v) if u <= v else (v, u)


def remove_edges_random(G, fraction, seed):
    rng = random.Random(seed)
    edges = list(G.edges())
    m = len(edges)
    r = int(round(fraction * m))
    removed = rng.sample(edges, r)
    Gt = G.copy()
    Gt.remove_edges_from(removed)
    removed_set = {edge_key(u, v) for (u, v) in removed}
    return Gt, removed_set


def build_existing_edge_set(G):
    return {edge_key(u, v) for (u, v) in G.edges()}


def generate_candidate_pairs(G):
    adj = {n: list(G.neighbors(n)) for n in G.nodes()}
    existing = build_existing_edge_set(G)
    cand = set()

    for w, neigh in adj.items():
        d = len(neigh)
        if d < 2:
            continue
        neigh.sort()
        for i in range(d):
            u = neigh[i]
            for j in range(i + 1, d):
                v = neigh[j]
                ek = edge_key(u, v)
                if ek in existing:
                    continue
                cand.add(ek)

    return cand


def evaluate_one_metric(G_train, removed_set, metric_name, ks):
    Predictor = METRICS[metric_name]
    pred = Predictor(G_train).fit()

    candidates = generate_candidate_pairs(G_train)
    scored = []
    for (u, v) in candidates:
        s = pred.score(u, v)
        if s > 0:
            scored.append((u, v, s))

    scored.sort(key=lambda x: x[2], reverse=True)

    max_k = max(ks)
    top_edges = [edge_key(u, v) for (u, v, _) in scored[:max_k]]
    top_prefix = []
    res = []

    top_set = set()
    for e in top_edges:
        top_prefix.append(e)

    for k in ks:
        topk = set(top_prefix[:k])
        tp = len(topk & removed_set)
        precision = tp / k if k > 0 else 0.0
        recall = tp / len(removed_set) if len(removed_set) > 0 else 0.0
        res.append((k, tp, precision, recall, len(removed_set), len(candidates), len(scored)))

    return res


def run_for_graph(graph_path, fractions, metrics, ks, seed):
    G = load_lcc(graph_path)
    name = os.path.splitext(os.path.basename(graph_path))[0]

    all_rows = []
    for f in fractions:
        Gt, removed = remove_edges_random(G, f, seed)
        for metric in metrics:
            out = evaluate_one_metric(Gt, removed, metric, ks)
            for (k, tp, prec, rec, n_removed, n_cand, n_scored) in out:
                all_rows.append({
                    "network": name,
                    "fraction_removed": f,
                    "metric": metric,
                    "k": k,
                    "tp": tp,
                    "precision_at_k": prec,
                    "recall_at_k": rec,
                    "n_removed": n_removed,
                    "n_candidates": n_cand,
                    "n_scored_positive": n_scored,
                    "n_nodes_lcc": G.number_of_nodes(),
                    "n_edges_lcc": G.number_of_edges(),
                })

    return name, all_rows


def plot_curves(rows, out_dir):
    ensure_dir(out_dir)
    grouped_data = {}
    for r in rows:
        key = (r["network"], r["fraction_removed"])
        grouped_data.setdefault(key, []).append(r)

    styles = {
        "cn": {"color": "blue", "label": "Common Neighbors", "marker": "o"},
        "jaccard": {"color": "green", "label": "Jaccard", "marker": "s"},
        "aa": {"color": "red", "label": "Adamic Adar", "marker": "^"},
    }

    for (net, frac), data_list in grouped_data.items():
        metrics_data = {}
        for r in data_list:
            metrics_data.setdefault(r["metric"], []).append(r)

        plt.figure(figsize=(10, 6))
        for metric, points in metrics_data.items():
            points.sort(key=lambda x: x["k"])
            ks = [x["k"] for x in points]
            precs = [x["precision_at_k"] for x in points]
            
            style = styles.get(metric, {"color": "black", "label": metric, "marker": "x"})
            plt.plot(ks, precs, label=style["label"], color=style["color"], marker=style["marker"], alpha=0.7)

        plt.xlabel("k (Nombre de prédictions)")
        plt.ylabel("Precision @ k")
        plt.title(f"{net} - Fraction removed: {frac} - Precision Comparison")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        filename_p = f"{net}_f{frac}_comparison_precision.png"
        plt.savefig(os.path.join(out_dir, filename_p), dpi=200)
        plt.close()

        plt.figure(figsize=(10, 6))
        for metric, points in metrics_data.items():
            points.sort(key=lambda x: x["k"])
            ks = [x["k"] for x in points]
            recs = [x["recall_at_k"] for x in points]
            
            style = styles.get(metric, {"color": "black", "label": metric, "marker": "x"})
            plt.plot(ks, recs, label=style["label"], color=style["color"], marker=style["marker"], alpha=0.7)

        plt.xlabel("k (Nombre de prédictions)")
        plt.ylabel("Recall @ k")
        plt.title(f"{net} - Fraction removed: {frac} - Recall Comparison")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        filename_r = f"{net}_f{frac}_comparison_recall.png"
        plt.savefig(os.path.join(out_dir, filename_r), dpi=200)
        plt.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--graphs", nargs="*", default=None, help='Examples: Caltech36 MIT8 "Johns Hopkins55"')
    p.add_argument("--fractions", nargs="*", type=float, default=[0.05, 0.1, 0.15, 0.2])
    p.add_argument("--ks", nargs="*", type=int, default=[50, 100, 200, 300, 400])
    p.add_argument("--metrics", nargs="*", default=["cn", "jaccard", "aa"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = "results/q4"
    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(out_dir)
    ensure_dir(fig_dir)

    gml_files = [f for f in sorted(os.listdir(args.data_dir)) if f.endswith(".gml")]

    if args.graphs is None or len(args.graphs) == 0:
        selected = gml_files
    else:
        selected = []
        for g in args.graphs:
            fname = g if g.endswith(".gml") else g + ".gml"
            if fname not in gml_files:
                raise FileNotFoundError(f"{fname} not found in {args.data_dir}/")
            selected.append(fname)

    graph_paths = [os.path.join(args.data_dir, f) for f in selected]

    all_rows = []

    if args.workers <= 1 or len(graph_paths) == 1:
        for gp in tqdm(graph_paths, desc="Graphs", unit="graph"):
            _, rows = run_for_graph(gp, args.fractions, args.metrics, args.ks, args.seed)
            all_rows.extend(rows)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_for_graph, gp, args.fractions, args.metrics, args.ks, args.seed) for gp in graph_paths]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Graphs", unit="graph"):
                _, rows = fut.result()
                all_rows.extend(rows)

    csv_path = os.path.join(out_dir, "link_prediction_eval.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved: {csv_path}")

    if not args.no_plots:
        plot_curves(all_rows, fig_dir)
        print(f"Saved figures in: {fig_dir}")


if __name__ == "__main__":
    main()
