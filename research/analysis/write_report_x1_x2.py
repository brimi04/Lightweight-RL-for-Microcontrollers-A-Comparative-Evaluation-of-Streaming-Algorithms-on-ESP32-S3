import json, os
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "derived", "x1_x2_report")
tbl = pd.read_csv(os.path.join(OUT, "r20_auc_threshold_table.csv"))
R20 = json.load(open(os.path.join(OUT, "stats_R20.json")))
AUC = json.load(open(os.path.join(OUT, "stats_AUC.json")))

L = []
def o(s=""): L.append(s)

o("# X1 + X2 report -- final CartPole ablations\n")
o("Source: `results/derived/ablations_cartpole_1500/` (controls C1/C2 reused without re-running + X1/X2 grids run on 2026-10-19).")
o("Actual execution environment: Python 3.10.0, original `.venv`, NumPy 2.2.6, Gymnasium 0.29.1 (checked by `run_x1_x2.ps1` before each run).\n")

for fam_name, title, ref_label in [
    ("X1_stream_q_bins", "## X1 -- Stream-Q / CartPole: uniform discretization vs. reference [6,6,12,12]", "ref"),
    ("X2_stream_ac_beta", "## X2 -- Stream-AC / CartPole: entropy coefficient beta_e vs. reference 0.05", "ref_b005"),
]:
    o(title)
    sub = tbl[tbl.family == fam_name]
    o("\n### R20 / AUC / episodes-to-threshold table (criterion: moving average of 20 COMPLETE episodes >= 50.0; conditional median over the seeds that cross, k>=1, without requiring k>=n/2; .5 not truncated; '---' = no seed crosses)\n")
    o("| config | n | R20 mean +- SD | AUC mean +- SD | ep_threshold (median) | k/n | steps_threshold (median) |")
    o("|---|---|---|---|---|---|---|")
    for _, r in sub.iterrows():
        ep = f"{r.ep_threshold_median:.1f}" if pd.notna(r.ep_threshold_median) else "---"
        st = f"{r.steps_threshold_median:.0f}" if pd.notna(r.steps_threshold_median) else "---"
        o(f"| {r.label} | {r.n} | {r.r20_mean:.2f} +- {r.r20_sd:.2f} | {r.auc_mean:.2f} +- {r.auc_sd:.2f} | {ep} | {r.k_over_n} | {st} |")

    o("\n### Memory and FLOPs (deterministic per configuration; invariant across seeds and episodes)\n")
    o("| config | memory_bytes | flops_per_update |")
    o("|---|---|---|")
    for _, r in sub.drop_duplicates("config").iterrows():
        o(f"| {r.label} | {r.memory_bytes} | {r.flops_per_update} |")

    o("\n### Paired statistical analysis (10 shared seeds; Friedman + paired Wilcoxon signed-rank vs. reference, Holm-Bonferroni; paired bootstrap with 10000 resamples for the 95% CI of the difference of medians)\n")
    for metric, res in [("R20", R20[fam_name]), ("AUC", AUC[fam_name])]:
        o(f"\n**{metric}** -- Friedman chi2={res['friedman_chi2']:.2f}, p={res['friedman_p']:.4g}")
        o("\n| config vs. reference | median diff. | Wilcoxon p (raw) | Holm p-adj | rejects H0 (alpha=0.05) | 95% bootstrap CI (median diff.) |")
        o("|---|---|---|---|---|---|")
        for p in res["pairwise_vs_ref"]:
            rej = "yes" if p["rejected_holm_0.05"] else "no"
            o(f"| {p['label']} | {p['median_diff']:+.3f} | {p['w_p']:.4f} | {p['w_p_holm']:.4f} | {rej} | [{p['boot_ci95'][0]:+.3f}, {p['boot_ci95'][1]:+.3f}] |")
    o("")

o("## Direct readings of the data (no causal interpretation for the paper yet)\n")
o("- **X1:** Friedman significant for R20 (p=2.5e-5) and AUC (p=7.5e-6). Wilcoxon-Holm rejects H0 (vs. ref [6,6,12,12]) for u18 and u24 in both metrics, and for u12 in AUC (not in R20, p_holm=0.082). u6 and u9 do not differ from ref after Holm. The number of seeds reaching the convergence threshold (50.0) decreases with the number of bins but NOT strictly monotonically: u6 10/10, u9 7/10, u12 2/10, u18 0/10, u24 1/10 (ref, with mixed bins [6,6,12,12], 8/10). The drop is clear from u6 to u18, but u24 (1/10) crosses one seed more than u18 (0/10); with n=10 seeds that 1-case difference does not allow claiming strict monotonicity between u18 and u24. Where there are crossings (ref, u6, u9, u12, u24), the conditional median of episodes-to-threshold also rises with more bins (494.5 to 784.5), consistent with slower tabular learning in larger state spaces.")
o("- **X2:** Friedman NOT significant for R20 (p=0.48): there is no evidence that beta_e in {0.00,0.01,0.05,0.10,0.30} changes the final reward of Stream-AC/CartPole. It is significant for AUC (p=7.7e-5), and only b030 survives Holm (median diff. +1.10 over ref, 95% CI does not cross 0). No X2 configuration crosses the convergence threshold (50.0) in any seed (0/10 in all); R20 stays between ~9.8 and ~10.1 in the 5 configurations, consistent with Stream-AC/CartPole with these hyperparameters not learning a policy that keeps the pole up for more than ~10 steps in the last 20 episodes.")
o("\nThe manuscript, figures and firmware were not modified by this report; it is raw material for deciding which conclusions of the paper change.")

with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print("report.md written:", os.path.join(OUT, "report.md"))
