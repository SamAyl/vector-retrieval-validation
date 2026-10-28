"""Figure 4 for Section 5.5, drawn from results.json produced by vdb_experiments.py."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("results.json"))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
DARK, MID, ACC, GREY = "#1f3b57", "#5b7a99", "#b5532e", "#9aa9b8"
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=300)

# (a) candidates needed for 20 eligible results, model vs measured median, by predicate type
e2 = R["E2"]; kinds = ["independent", "positive", "negative"]; cols = {"independent": MID, "positive": ACC, "negative": DARK}
x = np.arange(2); w = 0.22
for j, k in enumerate(kinds):
    vals = [next(r for r in e2 if r["predicate"] == k and r["selectivity"] == s)["measured_median_candidates"] for s in (0.1, 0.01)]
    ax[0].bar(x + (j - 1) * w, vals, w, color=cols[k], label=f"{k} predicate (measured median)")
    for xi, v in zip(x + (j - 1) * w, vals):
        ax[0].text(xi, v * 1.12, f"{v:,}", ha="center", fontsize=6.6)
model = [200, 2000]
for xi, m in zip(x, model):
    ax[0].hlines(m, xi - 1.6 * w, xi + 1.6 * w, colors="black", lw=1.2, ls="--")
ax[0].plot([], [], "k--", lw=1.2, label="independence model k/s")
ax[0].set_yscale("log"); ax[0].set_ylim(10, 3e7)
ax[0].set_xticks(x, ["s = 0.1", "s = 0.01"]); ax[0].set_ylabel("Candidates needed for 20 eligible")
ax[0].legend(frameon=False, fontsize=6.6, loc="upper left"); ax[0].set_title("(a) Predicate correlation and candidate demand", fontsize=8.6, loc="left")

# (b) all-shard completion: model, measured latencies, common cause
e3 = R["E3"]; S = [r["shards"] for r in e3]
ax[1].plot(S, [r["model"] for r in e3], "k--", lw=1.2, label="model (1 − p)^S")
ax[1].plot(S, [r["measured_latency_independent"] for r in e3], "o", color=MID, ms=5, label="measured shard latencies, independent")
ax[1].plot(S, [r["common_cause"] for r in e3], "s-", color=ACC, ms=4, lw=1.2, label="same marginal p, 0.5% common cause")
ax[1].set_xscale("log", base=2); ax[1].set_xticks(S, [str(s) for s in S])
ax[1].set_xlabel("Contacted shards S"); ax[1].set_ylabel("All-shard completion probability")
ax[1].set_ylim(0.6, 1.0); ax[1].legend(frameon=False, fontsize=6.8, loc="lower left")
ax[1].set_title("(b) Fan-out completion at p = 0.01", fontsize=8.6, loc="left")
fig.tight_layout(); fig.savefig("fig4.png"); print("fig4.png written")
