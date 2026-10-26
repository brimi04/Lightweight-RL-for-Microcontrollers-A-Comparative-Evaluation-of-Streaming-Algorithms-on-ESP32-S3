"""
consolidated_final.py -- Final consolidation of the project results,
with the ADOPTED dataset for each agent x environment and the new
episodes_to_threshold criterion (window of 20 COMPLETE episodes,
conditional median over the seeds that cross with k>=1, k/n always
reported -- without requiring k>=n/2, as analysis/paper_numbers.py did).

It only READS existing data (raw_final/ unmodified, and the derived data already
generated in derived/). It writes only inside
results/derived/consolidated_final/.

ADOPTED SOURCES (see README.md, section 5):
  - GridWorld, the 3 lightweight agents + Standard DQN: results/raw_final/
    (10 seeds for the lightweight agents, 5 for Standard DQN). UNCHANGED with respect to the
    earlier manuscript version; only R20/AUC/threshold are recomputed with the new criterion.
  - CartPole, the 3 lightweight agents + Standard DQN: results/raw_final/
    (same). The Stream-Q reference configuration ([6,6,12,12]) is the
    same one validated by control C1.
  - MountainCar, Stream-Q and TinyDQN: results/raw_final/ with the 10 seeds
    (0-9) already present in the repository (an earlier manuscript version used only 5 of those
    10 for Stream-Q in the main table; here all 10 are adopted, already
    available and without re-running anything).
  - MountainCar, Stream-AC: ADOPTED results/derived/ablations_mountaincar_2000/
    stream_ac_documented/ (X3, Windows, 2026-10-19, 10 seeds), instead of
    results/raw_final/stream_ac_mountaincar_*.csv (which is KEPT untouched,
    but is no longer used for the final analysis because it does not correspond to the
    documented reproducible configuration -- see README.md, section 5).
  - MountainCar, Standard DQN: results/raw_final/baseline_dqn_mountaincar_*_full.csv
    (5 seeds). UNCHANGED.
  - CartPole ablations (X1, X2): results/derived/x1_x2_report/
    (already generated; only referenced here, not recomputed).
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

RESEARCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(RESEARCH, "results")
PY = os.path.join(RESULTS, "raw_final")
DERIVED = os.path.join(RESULTS, "derived")
X3_ADOPTED = os.path.join(DERIVED, "ablations_mountaincar_2000", "stream_ac_documented")
OUT = os.path.join(DERIVED, "consolidated_final")
os.makedirs(OUT, exist_ok=True)

TH = {"gridworld": 1.5, "cartpole": 50.0, "mountaincar": -175.0}   # config.yaml convergence_reward
LIGHT = ["stream_q", "stream_ac", "tiny_dqn"]
ENVS = ["gridworld", "cartpole", "mountaincar"]
DQN_FILE = {"gridworld": "baseline_dqn_gridworld_{s}.csv",
            "cartpole": "baseline_dqn_cartpole_{s}_full.csv",
            "mountaincar": "baseline_dqn_mountaincar_{s}_full.csv"}


def path_for(agent, env, seed):
    """ADOPTED source for (agent, environment, seed)."""
    if env == "mountaincar" and agent == "stream_ac":
        return os.path.join(X3_ADOPTED, f"stream_ac_mountaincar_{seed}.csv")
    return os.path.join(PY, f"{agent}_{env}_{seed}.csv")


def source_label(agent, env):
    if env == "mountaincar" and agent == "stream_ac":
        return "ADOPTED: derived/ablations_mountaincar_2000/stream_ac_documented/ (X3, Windows, 2026-10-19)"
    return "ORIGINAL: raw_final/ (unchanged from the earlier manuscript version)"


def per_seed_metrics(df, env):
    r = df["reward"]
    r20 = float(r.values[-20:].mean())
    auc = float(r.mean())
    roll = r.rolling(20, min_periods=20).mean().values
    idx = np.where(roll >= TH[env])[0]
    ep_thresh = int(idx[0]) if len(idx) else None
    steps_thresh = int(df["steps"].values[: idx[0] + 1].sum()) if len(idx) else None
    return dict(r20=r20, auc=auc, ep_threshold=ep_thresh, steps_threshold=steps_thresh,
               memory_bytes=int(df["memory_bytes"].iloc[0]), flops_per_update=int(df["flops_per_update"].iloc[0]))


rows = []
for env in ENVS:
    for agent in LIGHT:
        n_seeds = 10
        for s in range(n_seeds):
            df = pd.read_csv(path_for(agent, env, s)).sort_values("episode").reset_index(drop=True)
            m = per_seed_metrics(df, env)
            rows.append(dict(env=env, agent=agent, seed=s, n_seeds=n_seeds,
                             source=source_label(agent, env), **m))
    for s in range(5):
        df = pd.read_csv(os.path.join(PY, DQN_FILE[env].format(s=s))).sort_values("episode").reset_index(drop=True)
        m = per_seed_metrics(df, env)
        rows.append(dict(env=env, agent="standard_dqn", seed=s, n_seeds=5,
                         source="ORIGINAL: raw_final/ (unchanged)", **m))

per_seed = pd.DataFrame(rows)
per_seed.to_csv(os.path.join(OUT, "per_seed_all.csv"), index=False)


def summarize(env, agent):
    g = per_seed[(per_seed.env == env) & (per_seed.agent == agent)]
    crossed = g["ep_threshold"].dropna()
    k, n = len(crossed), len(g)
    ep_med = float(crossed.median()) if k >= 1 else None
    st_med = float(g["steps_threshold"].dropna().median()) if k >= 1 else None
    return dict(env=env, agent=agent, n=n, source=g.source.iloc[0],
               r20_mean=g.r20.mean(), r20_sd=g.r20.std(ddof=1),
               auc_mean=g.auc.mean(), auc_sd=g.auc.std(ddof=1),
               ep_threshold_median=ep_med, k_over_n=f"{k}/{n}", steps_threshold_median=st_med,
               memory_bytes=int(g.memory_bytes.iloc[0]), flops_per_update=int(g.flops_per_update.iloc[0]))


summary = pd.DataFrame([summarize(env, agent) for env in ENVS for agent in LIGHT + ["standard_dqn"]])
summary.to_csv(os.path.join(OUT, "summary_table_all_envs.csv"), index=False)
print(summary.to_string(index=False))

# ---- Kruskal-Wallis + Mann-Whitney-Holm: ONLY GridWorld and CartPole (no change
# of scope with respect to the earlier manuscript version; MountainCar is treated as an
# exploration diagnostic -- no between-agent test for
# MountainCar is computed here).
def holm(pvals):
    idx = np.argsort(pvals); n = len(pvals); out = np.empty(n); running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, pvals[i] * (n - rank))); out[i] = running
    return out


kw_results = {}
for env in ("gridworld", "cartpole"):
    groups = {a: per_seed[(per_seed.env == env) & (per_seed.agent == a)].r20.values for a in LIGHT}
    H, p = stats.kruskal(*groups.values())
    pairs = [(a, b, stats.mannwhitneyu(groups[a], groups[b], alternative="two-sided").pvalue) for a, b in combinations(LIGHT, 2)]
    pvals = np.array([x[2] for x in pairs])
    padj = holm(pvals)
    kw_results[env] = dict(H=float(H), p=float(p),
                           pairs=[dict(a=a, b=b, p=float(p0), p_holm=float(pa)) for (a, b, p0), pa in zip(pairs, padj)])
with open(os.path.join(OUT, "kruskal_wallis_gw_cp.json"), "w", encoding="utf-8") as f:
    json.dump(kw_results, f, indent=2)
print()
print(json.dumps(kw_results, indent=2))
print("\nFiles in", OUT)
