"""Controlled validation experiments for the vector-database manuscript (Section 5.5).

E1  Measured index storage per vector vs. the payload equations of Section 3.2.
E2  Candidate demand and fill probability under independent and correlated predicates
    (exact search), compared with the binomial model; post-filter vs. in-search filtering on HNSW.
E3  All-shard deadline completion from measured shard latencies: independent vs. common-cause.

Synthetic data: clustered, anisotropic, unit-normalised 768-d vectors (no proprietary data).
Run: python vdb_experiments.py  (about 10-15 min on 2 cores) -> results.json, e1.csv, e2.csv, e3.csv
"""
import json, time, csv, math
import numpy as np
import faiss
from scipy.stats import binom

SEED = 7
N, D, NC = 100_000, 768, 200          # vectors, dimension, clusters
NQ = 400                              # queries
K = 20
rng = np.random.default_rng(SEED)
faiss.omp_set_num_threads(2)

# ------------------------------------------------------------------ data
def make_data():
    latent = 64
    proj = rng.normal(size=(latent, D)).astype("float32") / math.sqrt(latent)
    centers = rng.normal(size=(NC, latent)).astype("float32")
    sizes = rng.dirichlet(np.full(NC, 2.0)) * N
    sizes = np.maximum(1, np.round(sizes)).astype(int); sizes[-1] += N - sizes.sum()
    labels = np.repeat(np.arange(NC), sizes)[:N]
    spread = rng.uniform(0.35, 0.8, NC).astype("float32")
    z = centers[labels] + rng.normal(size=(N, latent)).astype("float32") * spread[labels, None]
    x = z @ proj + 0.15 * rng.normal(size=(N, D)).astype("float32")
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    qlab = rng.choice(labels, NQ)
    zq = centers[qlab] + rng.normal(size=(NQ, latent)).astype("float32") * spread[qlab, None]
    q = zq @ proj + 0.15 * rng.normal(size=(NQ, D)).astype("float32")
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return x.astype("float32"), labels, q.astype("float32"), qlab, centers

x, labels, q, qlab, centers = make_data()
results = {"N": N, "D": D, "clusters": NC, "queries": NQ, "k": K}

# ------------------------------------------------------------------ E1 storage
def size_of(index):
    return faiss.serialize_index(index).nbytes
e1 = []
def record(name, index, payload_bytes_per_vec, t_build):
    b = size_of(index) / N
    e1.append(dict(index=name, bytes_per_vector=round(b, 1), payload_model=payload_bytes_per_vec,
                   overhead_pct=round(100 * (b / payload_bytes_per_vec - 1), 1) if payload_bytes_per_vec else None,
                   gb_10M_x3=round(b * 10_000_000 * 3 / 1e9, 2), build_s=round(t_build, 1)))
t = time.time(); flat = faiss.IndexFlatIP(D); flat.add(x); record("Flat float32", flat, D * 4, time.time() - t)
t = time.time(); f16 = faiss.IndexScalarQuantizer(D, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT); f16.train(x); f16.add(x); record("Scalar fp16", f16, D * 2, time.time() - t)
t = time.time(); sq8 = faiss.IndexScalarQuantizer(D, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT); sq8.train(x); sq8.add(x); record("Scalar int8", sq8, D * 1, time.time() - t)
t = time.time()
ivfpq = faiss.IndexIVFPQ(faiss.IndexFlatIP(D), D, 1024, 96, 8, faiss.METRIC_INNER_PRODUCT)
ivfpq.train(x[:60000]); ivfpq.add(x); record("IVF1024,PQ96", ivfpq, 96, time.time() - t)
hnsw = {}
for M in (16, 32):
    t = time.time(); h = faiss.IndexHNSWFlat(D, M, faiss.METRIC_INNER_PRODUCT); h.hnsw.efConstruction = 80; h.add(x)
    record(f"HNSW M={M} + float32", h, D * 4, time.time() - t); hnsw[M] = h
results["E1"] = e1
print("E1", json.dumps(e1, indent=1))

# ------------------------------------------------------------------ exact ranking (reference)
t = time.time()
_, I_all = flat.search(q, N)            # full exact ranking per query (400 x 100k)
results["exact_rank_s"] = round(time.time() - t, 1)

# cluster distances for correlated predicates
cn = centers / np.linalg.norm(centers, axis=1, keepdims=True)
def eligibility(kind, s, qi, rs):
    if kind == "independent":
        return rs.random(N) < s
    ncl = max(1, int(round(s * NC)))
    sims = cn @ cn[qlab[qi]]
    order = np.argsort(-sims)
    if kind == "positive":       # predicate aligned with the query's neighbourhood
        chosen = order[:ncl]
    else:                         # predicate selecting regions far from the query
        chosen = order[-ncl:]
    m = np.isin(labels, chosen)
    # rescale to target selectivity by random thinning/padding (keeps correlation, fixes s)
    cur = m.mean()
    if cur > s:
        m &= rs.random(N) < s / cur
    return m

# ------------------------------------------------------------------ E2 filtering
e2 = []
for s in (0.1, 0.01):
    for kind in ("independent", "positive", "negative"):
        rs = np.random.default_rng(100 + int(1 / s))
        need, fill_mean, fill_15 = [], [], []
        hn_post, hn_post_recall, hn_in, hn_in_recall = [], [], [], []
        actual_s = []
        for qi in range(NQ):
            m = eligibility(kind, s, qi, rs)
            actual_s.append(m.mean())
            elig_rank = np.nonzero(m[I_all[qi]])[0]
            c_need = elig_rank[K - 1] + 1 if len(elig_rank) >= K else N + 1
            need.append(c_need)
            cm = int(math.ceil(K / s))
            fill_mean.append(c_need <= cm); fill_15.append(c_need <= int(1.5 * cm))
            truth = set(I_all[qi][elig_rank[:K]])
            if qi < 150:   # HNSW: post-filter with candidate budget C = k/s vs in-search filter
                h = hnsw[32]
                sp = faiss.SearchParametersHNSW(); sp.efSearch = max(cm, 64)
                _, Ih = h.search(q[qi:qi + 1], cm, params=sp)
                got = [i for i in Ih[0] if i >= 0 and m[i]][:K]
                hn_post.append(len(got) >= K); hn_post_recall.append(len(truth & set(got)) / K)
                sel = faiss.IDSelectorBatch(np.nonzero(m)[0].astype("int64"))
                sp2 = faiss.SearchParametersHNSW(); sp2.efSearch = 200; sp2.sel = sel
                _, Ih2 = h.search(q[qi:qi + 1], K, params=sp2)
                got2 = [i for i in Ih2[0] if i >= 0]
                hn_in.append(len(got2) >= K); hn_in_recall.append(len(truth & set(got2)) / K)
        need = np.array(need)
        cm = int(math.ceil(K / s))
        e2.append(dict(selectivity=s, predicate=kind, actual_selectivity=round(float(np.mean(actual_s)), 4),
                       model_mean_candidates=cm, measured_median_candidates=int(np.median(need)),
                       measured_p90_candidates=int(np.percentile(need, 90)),
                       model_fill_at_k_over_s=round(1 - binom.cdf(K - 1, cm, s), 3),
                       measured_fill_at_k_over_s=round(float(np.mean(fill_mean)), 3),
                       model_fill_at_1_5x=round(1 - binom.cdf(K - 1, int(1.5 * cm), s), 3),
                       measured_fill_at_1_5x=round(float(np.mean(fill_15)), 3),
                       hnsw_postfilter_fill=round(float(np.mean(hn_post)), 3),
                       hnsw_postfilter_recall=round(float(np.mean(hn_post_recall)), 3),
                       hnsw_infilter_fill=round(float(np.mean(hn_in)), 3),
                       hnsw_infilter_recall=round(float(np.mean(hn_in_recall)), 3)))
        print("E2", e2[-1])
results["E2"] = e2

# ------------------------------------------------------------------ E3 fan-out
faiss.omp_set_num_threads(1)
S_SHARDS = 8
shard_lat = []
idx = np.array_split(rng.permutation(N), S_SHARDS)
shards = []
for part in idx:
    hs = faiss.IndexHNSWFlat(D, 16, faiss.METRIC_INNER_PRODUCT); hs.hnsw.efConstruction = 40; hs.add(x[part]); shards.append(hs)
sp = faiss.SearchParametersHNSW(); sp.efSearch = 64
for rep in range(5):
    for qi in range(NQ):
        for hs in shards:
            t0 = time.perf_counter(); hs.search(q[qi:qi + 1], K, params=sp); shard_lat.append(time.perf_counter() - t0)
lat = np.array(shard_lat)
deadline = float(np.quantile(lat, 0.99))           # deadline set so the per-shard miss probability is 1%
p_hat = float(np.mean(lat > deadline))
mc = np.random.default_rng(11)
TR = 200_000
e3 = []
for S in (4, 8, 16, 32):
    ind = mc.choice(lat, size=(TR, S))
    p_ind = float(np.mean((ind <= deadline).all(axis=1)))
    # common cause: with prob 0.5% a shared event slows every shard past the deadline,
    # residual independent misses chosen so the marginal per-shard miss stays at p_hat
    q_cc = 0.005
    p_res = (p_hat - q_cc) / (1 - q_cc)
    shared = mc.random(TR) < q_cc
    resid = mc.random((TR, S)) < p_res
    miss = resid | shared[:, None]
    p_cc = float(np.mean(~miss.any(axis=1)))
    e3.append(dict(shards=S, model=round((1 - p_hat) ** S, 4), measured_latency_independent=round(p_ind, 4),
                   common_cause=round(p_cc, 4), model_common_cause=round((1 - q_cc) * (1 - p_res) ** S, 4),
                   marginal_miss_common_cause=round(float(miss.mean()), 4)))
    print("E3", e3[-1])
results["E3"] = e3
results["E3_shard_latency_ms"] = dict(p50=round(1e3 * float(np.median(lat)), 3), p99=round(1e3 * deadline, 3), p_hat=p_hat, samples=len(lat))
json.dump(results, open("results.json", "w"), indent=1)
for name in ("E1", "E2", "E3"):
    with open(f"{name.lower()}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[name][0].keys())); w.writeheader(); w.writerows(results[name])
print("done")
