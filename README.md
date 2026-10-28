# Vector Databases for Semantic Retrieval — validation experiments

Code and outputs for the empirical validation (Section 5.5) of:

> S. Ayloo, *Vector Databases for Semantic Information Retrieval and Scalable Enterprise Artificial Intelligence* (manuscript under submission, 2026).

The experiments test the paper's three analytical relationships against real Faiss indexes on
100,000 synthetic, clustered, unit-normalised 768-dimensional embeddings (no proprietary data):

| Experiment | Paper | Question |
|---|---|---|
| E1 | Table 5 | Does measured index storage match the payload equation `N·d·b·r`, and what do HNSW and IVF-PQ structures add? |
| E2 | Table 6, Fig. 4a | Does the binomial candidate/fill model hold for independent predicates, and how far do correlated predicates move it? Post-filter vs in-search filtering on HNSW. |
| E3 | Fig. 4b | Do measured shard latencies reproduce all-shard completion `(1-p)^S`, and how does a common-cause event change it? |

## Run

```bash
pip install -r requirements.txt
python vdb_experiments.py   # ~10 min on 2 cores -> results.json, e1.csv, e2.csv, e3.csv
python vdb_figure.py        # figures/fig4.png
```

Seeds are fixed. The committed CSV/JSON files are the outputs reported in the paper (Faiss 1.15.1).
Timings come from a two-core virtual machine and are not performance claims.

## License

MIT
