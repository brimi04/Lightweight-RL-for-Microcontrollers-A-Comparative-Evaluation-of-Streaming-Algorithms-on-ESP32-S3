"""
ablation_report_x1_x2.py -- Report of the final ablations X1
(Stream-Q/CartPole, discretization) and X2 (Stream-AC/CartPole, beta_e),
authorized and run in results/derived/ablations_cartpole_1500/.

It only READS already generated data (it never writes outside results/derived/):
  - X1: ablations_cartpole_1500/stream_q_bins/{ref,u6,u9,u12,u18,u24}/*.csv
        (ref = control C1, [6,6,12,12]; the other 5 = X1 grid)
  - X2: ablations_cartpole_1500/stream_ac_beta/{ref_b005,b000,b001,b010,b030}/*.csv
        (ref_b005 = control C2, beta_e=0.05; the other 4 = X2 grid)

Criteria (the same ones already used in the rest of the project, see
analysis/paper_numbers.py and analysis/statistical_tests.py, the latter not
included in this repository):
  * R20: mean of the last 20 episodes, per seed.
  * Normalized AUC: mean reward over ALL episodes, per seed
    (same definition as analysis/statistical_tests.py:auc_per_seed).
  * episodes_to_threshold: first episode at which the moving average of 20
    COMPLETE episodes (rolling(20, min_periods=20)) reaches the CartPole
    convergence threshold (config.yaml: convergence_reward = 50.0). The
    conditional median over the seeds that cross is reported (k>=1,
    without requiring k>=n/2) together with k/n (seeds that cross / total). .5
    values are not truncated. If k=0 there is no median to report ("---").
  * memory_bytes / flops_per_update: read directly from the CSV column
    (they are deterministic per configuration and do not vary across seeds or
    episodes within a configuration).

Paired design (same 10 seeds in all configurations of a family):
  * Friedman (chi2) across all configurations of the family, for R20 and AUC.
  * If Friedman is significant (or in any case, to report effect
    sizes): PAIRED Wilcoxon signed-rank test of each configuration vs the reference
    configuration of that family (ref for X1, ref_b005 for X2), with
    Holm-Bonferroni correction over the comparisons of that family.
  * Paired bootstrap (10000 resamples, resampling the seed index)
    for the 95% CI of the difference of medians (config - reference).

Outputs (all new, inside results/derived/x1_x2_report/):
  report.md, r20_auc_threshold_table.csv, memory_flops_table.csv,
  stats_R20.json, stats_AUC.json, per_seed_raw.csv
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

RESEARCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DERIVED = os.path.join(RESEARCH, "results", "derived")
ABL = os.path.join(DERIVED, "ablations_cartpole_1500")
OUT = os.path.join(DERIVED, "x1_x2_report")
os.makedirs(OUT, exist_ok=True)

TH_CARTPOLE = 50.0          # config.yaml: envs.cartpole.convergence_reward
SEEDS = list(range(10))
N_BOOT = 10000
RNG = np.random.default_rng(20261019)

FAMILIES = {
    "X1_stream_q_bins": {
        "dir": os.path.join(ABL, "stream_q_bins"),
        "agent": "stream_q", "env": "cartpole",
        "ref": "ref", "configs": ["ref", "u6", "u9", "u12", "u18", "u24"],
        "label": {"ref": "ref [6,6,12,12]", "u6": "u6 [6]^4", "u9": "u9 [9]^4",
                  "u12": "u12 [12]^4", "u18": "u18 [18]^4", "u24": "u24 [24]^4"},
    },
    "X2_stream_ac_beta": {
        "dir": os.path.join(ABL, "stream_ac_beta"),
        "agent": "stream_ac", "env": "cartpole",
        "ref": "ref_b005", "configs": ["ref_b005", "b000", "b001", "b010", "b030"],
        "label": {"ref_b005": "beta=0.05 (ref)", "b000": "beta=0.00", "b001": "beta=0.01",
                  "b010": "beta=0.10", "b030": "beta=0.30"},
    },
}


def load_csv(fam, config, seed):
    agent = fam["agent"]
    return pd.read_csv(os.path.join(fam["dir"], config, f"{agent}_cartpole_{seed}.csv")).sort_values("episode").reset_index(drop=True)


def per_seed_metrics(df):
    r = df["reward"]
    r20 = float(r.values[-20:].mean())
    auc = float(r.mean())                      # Normalized AUC = mean over ALL episodes
    roll = r.rolling(20, min_periods=20).mean().values
    idx = np.where(roll >= TH_CARTPOLE)[0]
    ep_thresh = int(idx[0]) if len(idx) else None
    steps_thresh = int(df["steps"].values[: idx[0] + 1].sum()) if len(idx) else None
    return r20, auc, ep_thresh, steps_thresh, int(df["memory_bytes"].iloc[0]), int(df["flops_per_update"].iloc[0])


rows = []
for fam_name, fam in FAMILIES.items():
    for config in fam["configs"]:
        for seed in SEEDS:
            df = load_csv(fam, config, seed)
            r20, auc, ep_t, st_t, mem, fl = per_seed_metrics(df)
            rows.append(dict(family=fam_name, config=config, label=fam["label"][config], seed=seed,
                             n_episodes=len(df), r20=r20, auc=auc, ep_threshold=ep_t, steps_threshold=st_t,
                             memory_bytes=mem, flops_per_update=fl, is_ref=(config == fam["ref"])))
raw = pd.DataFrame(rows)
raw.to_csv(os.path.join(OUT, "per_seed_raw.csv"), index=False)


def summarize(fam_name, fam):
    sub = raw[raw.family == fam_name]
    out_rows = []
    for config in fam["configs"]:
        g = sub[sub.config == config]
        crossed = g["ep_threshold"].dropna()
        n = len(g)
        k = len(crossed)
        # Conditional median over the seeds that cross (without requiring k>=n/2;
        # reported whenever k>=1, without truncating .5), with explicit k/n.
        ep_med = float(crossed.median()) if k >= 1 else None
        st_med = float(g["steps_threshold"].dropna().median()) if k >= 1 else None
        out_rows.append(dict(
            family=fam_name, config=config, label=fam["label"][config], n=n,
            r20_mean=g.r20.mean(), r20_sd=g.r20.std(ddof=1),
            auc_mean=g.auc.mean(), auc_sd=g.auc.std(ddof=1),
            ep_threshold_median=ep_med, k_over_n=f"{k}/{n}",
            steps_threshold_median=st_med,
            memory_bytes=int(g.memory_bytes.iloc[0]), flops_per_update=int(g.flops_per_update.iloc[0]),
        ))
    return pd.DataFrame(out_rows)


summary_tbl = pd.concat([summarize(f, FAMILIES[f]) for f in FAMILIES], ignore_index=True)
summary_tbl.to_csv(os.path.join(OUT, "r20_auc_threshold_table.csv"), index=False)
summary_tbl[["family", "config", "label", "memory_bytes", "flops_per_update"]].drop_duplicates().to_csv(
    os.path.join(OUT, "memory_flops_table.csv"), index=False)


def holm_bonferroni(p_values, alpha=0.05):
    idx = np.argsort(p_values)
    n = len(p_values)
    rejected = np.zeros(n, dtype=bool)
    p_adj = np.empty(n)
    running_max = 0.0
    for rank, i in enumerate(idx):
        adj = min(1.0, p_values[i] * (n - rank))
        running_max = max(running_max, adj)
        p_adj[i] = running_max
        if running_max < alpha:
            rejected[i] = True
    return p_adj, rejected


def paired_bootstrap_ci(x_ref, x_cfg, n_boot=N_BOOT, rng=RNG):
    """95% CI of the difference of medians (cfg - ref), resampling the
    seed index (paired: same seed in both arms in each
    resample)."""
    n = len(x_ref)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[b] = np.median(x_cfg[idx]) - np.median(x_ref[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi), float(np.median(x_cfg) - np.median(x_ref))


def analyze_metric(fam_name, fam, metric):
    sub = raw[raw.family == fam_name]
    configs = fam["configs"]
    ref = fam["ref"]
    # paired matrix (seed x config), with the same seed order for all 10
    wide = sub.pivot(index="seed", columns="config", values=metric)[configs]
    friedman_stat, friedman_p = stats.friedmanchisquare(*[wide[c].values for c in configs])

    others = [c for c in configs if c != ref]
    x_ref = wide[ref].values
    wres = []
    for c in others:
        x_c = wide[c].values
        try:
            w_stat, w_p = stats.wilcoxon(x_c, x_ref)
        except ValueError as e:
            w_stat, w_p = float("nan"), 1.0  # all differences are 0 (degenerate case)
        ci_lo, ci_hi, med_diff = paired_bootstrap_ci(x_ref, x_c)
        wres.append(dict(config=c, label=fam["label"][c], w_stat=w_stat, w_p=w_p,
                         median_diff=med_diff, boot_ci95=[ci_lo, ci_hi]))
    p_raw = np.array([r["w_p"] for r in wres])
    p_adj, rejected = holm_bonferroni(p_raw)
    for r, pa, rej in zip(wres, p_adj, rejected):
        r["w_p_holm"] = float(pa)
        r["rejected_holm_0.05"] = bool(rej)

    return dict(family=fam_name, metric=metric, ref=ref, n_seeds=len(SEEDS),
                friedman_chi2=float(friedman_stat), friedman_p=float(friedman_p),
                pairwise_vs_ref=wres)


for metric in ("r20", "auc"):
    results = {fam_name: analyze_metric(fam_name, fam, metric) for fam_name, fam in FAMILIES.items()}
    with open(os.path.join(OUT, f"stats_{metric.upper()}.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

print("Done. Files in", OUT)
print(summary_tbl.to_string(index=False))
