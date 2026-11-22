"""
Calibrate the out-of-syllabus threshold (Feature 6).

    python calibrate_threshold.py eval_queries.json

eval_queries.json = {"in": [...in-syllabus questions...],
                     "out": [...out-of-syllabus questions...],
                     "hard_out": [...optional: near-topic but not covered...]}

Positive class = "refuse". A query is refused when its top dense cosine score < t.
Writes threshold_sweep.csv and threshold_sweep.png (for the report) and prints
the threshold with the best F1. Needs no Groq key (retrieval only).
"""
import csv
import json
import sys

from rag_engine import RAGEngine


def sweep(in_scores, out_scores, lo=0.10, hi=0.70, step=0.01):
    rows, t = [], lo
    while t <= hi + 1e-9:
        tp = sum(s < t for s in out_scores)   # out-of-syllabus correctly refused
        fn = len(out_scores) - tp             # out-of-syllabus wrongly answered
        fp = sum(s < t for s in in_scores)    # in-syllabus wrongly refused
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append(dict(threshold=round(t, 2), precision=round(prec, 3), recall=round(rec, 3),
                         f1=round(f1, 3), in_refused=fp, out_answered=fn))
        t += step
    return rows


def main(path):
    data = json.load(open(path, encoding="utf-8"))
    engine = RAGEngine()
    top = lambda q: engine.retrieve(q, top_k=1)[0]["score"]

    in_s = [top(q) for q in data["in"]]
    out_s = [top(q) for q in data["out"]]

    print("\nPer-query top scores (check these manually):")
    for tag, qs, ss in (("IN ", data["in"], in_s), ("OUT", data["out"], out_s)):
        for q, s in sorted(zip(qs, ss), key=lambda x: x[1]):
            print(f"  [{tag}] {s:.3f}  {q}")

    rows = sweep(in_s, out_s)
    best = max(rows, key=lambda r: (r["f1"], -r["in_refused"]))
    with open("threshold_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ts = [r["threshold"] for r in rows]
        plt.figure(figsize=(6, 4))
        for k, lab in (("precision", "Refusal precision"), ("recall", "Refusal recall"), ("f1", "F1")):
            plt.plot(ts, [r[k] for r in rows], label=lab)
        plt.axvline(best["threshold"], ls="--", c="gray")
        plt.xlabel("Cosine threshold"); plt.ylabel("Score")
        plt.title("Out-of-syllabus refusal vs threshold")
        plt.legend(); plt.tight_layout(); plt.savefig("threshold_sweep.png", dpi=150)
    except ImportError:
        print("(matplotlib not installed; skipped plot)")

    print(f"\nIn-syllabus  mean top score: {sum(in_s)/len(in_s):.3f} (min {min(in_s):.3f})")
    print(f"Out-of-syll. mean top score: {sum(out_s)/len(out_s):.3f} (max {max(out_s):.3f})")
    print(f"Best threshold by F1: {best}")

    # Optional: security topics that are NOT in the slides but sit close to ones that are.
    # A similarity threshold usually cannot catch these; report it honestly in the write-up.
    if data.get("hard_out"):
        h = [(q, top(q)) for q in data["hard_out"]]
        refused = sum(s < best["threshold"] for _, s in h)
        print(f"\nHard negatives refused at t={best['threshold']}: {refused}/{len(h)}")
        for q, s in sorted(h, key=lambda x: x[1]):
            print(f"  {s:.3f}  {'REFUSED' if s < best['threshold'] else 'answered'}  {q}")
    print("Set SIMILARITY_THRESHOLD=<that value> in .env")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval_queries.json")