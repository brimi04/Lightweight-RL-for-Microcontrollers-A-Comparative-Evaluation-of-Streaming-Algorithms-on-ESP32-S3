# X1 + X2 report -- final CartPole ablations

Source: `results/derived/ablations_cartpole_1500/` (controls C1/C2 reused without re-running + X1/X2 grids run on 2026-10-19).
Actual execution environment: Python 3.10.0, original `.venv`, NumPy 2.2.6, Gymnasium 0.29.1 (checked by `run_x1_x2.ps1` before each run).

## X1 -- Stream-Q / CartPole: uniform discretization vs. reference [6,6,12,12]

### R20 / AUC / episodes-to-threshold table (criterion: moving average of 20 COMPLETE episodes >= 50.0; conditional median over the seeds that cross, k>=1, without requiring k>=n/2; .5 not truncated; '---' = no seed crosses)

| config | n | R20 mean +- SD | AUC mean +- SD | ep_threshold (median) | k/n | steps_threshold (median) |
|---|---|---|---|---|---|---|
| ref [6,6,12,12] | 10 | 94.81 +- 56.81 | 52.02 +- 32.19 | 784.5 | 8/10 | 12485 |
| u6 [6]^4 | 10 | 90.09 +- 14.31 | 53.77 +- 11.45 | 494.5 | 10/10 | 9324 |
| u9 [9]^4 | 10 | 84.73 +- 102.13 | 33.87 +- 22.87 | 748.0 | 7/10 | 11933 |
| u12 [12]^4 | 10 | 39.07 +- 43.16 | 26.83 +- 27.78 | 564.5 | 2/10 | 10452 |
| u18 [18]^4 | 10 | 16.88 +- 6.42 | 13.06 +- 3.24 | --- | 0/10 | --- |
| u24 [24]^4 | 10 | 19.33 +- 16.26 | 14.22 +- 7.02 | 746.0 | 1/10 | 10933 |

### Memory and FLOPs (deterministic per configuration; invariant across seeds and episodes)

| config | memory_bytes | flops_per_update |
|---|---|---|
| ref [6,6,12,12] | 82944 | 41472 |
| u6 [6]^4 | 20736 | 10368 |
| u9 [9]^4 | 104976 | 52488 |
| u12 [12]^4 | 331776 | 165888 |
| u18 [18]^4 | 1679616 | 839808 |
| u24 [24]^4 | 5308416 | 2654208 |

### Paired statistical analysis (10 shared seeds; Friedman + paired Wilcoxon signed-rank vs. reference, Holm-Bonferroni; paired bootstrap with 10000 resamples for the 95% CI of the difference of medians)


**R20** -- Friedman chi2=28.86, p=2.473e-05

| config vs. reference | median diff. | Wilcoxon p (raw) | Holm p-adj | rejects H0 (alpha=0.05) | 95% bootstrap CI (median diff.) |
|---|---|---|---|---|---|
| u6 [6]^4 | -4.100 | 1.0000 | 1.0000 | no | [-34.625, +38.900] |
| u9 [9]^4 | -43.325 | 0.2754 | 0.5508 | no | [-85.100, +7.775] |
| u12 [12]^4 | -77.925 | 0.0273 | 0.0820 | no | [-106.800, -25.100] |
| u18 [18]^4 | -83.050 | 0.0039 | 0.0156 | yes | [-111.350, -38.700] |
| u24 [24]^4 | -86.825 | 0.0020 | 0.0098 | yes | [-116.100, -40.825] |

**AUC** -- Friedman chi2=31.49, p=7.51e-06

| config vs. reference | median diff. | Wilcoxon p (raw) | Holm p-adj | rejects H0 (alpha=0.05) | 95% bootstrap CI (median diff.) |
|---|---|---|---|---|---|
| u6 [6]^4 | +4.396 | 0.9219 | 0.9219 | no | [-21.967, +31.771] |
| u9 [9]^4 | -20.844 | 0.1934 | 0.3867 | no | [-58.438, +12.284] |
| u12 [12]^4 | -36.141 | 0.0039 | 0.0117 | yes | [-60.999, -7.927] |
| u18 [18]^4 | -38.323 | 0.0020 | 0.0098 | yes | [-64.815, -10.476] |
| u24 [24]^4 | -39.818 | 0.0020 | 0.0098 | yes | [-65.406, -11.149] |

## X2 -- Stream-AC / CartPole: entropy coefficient beta_e vs. reference 0.05

### R20 / AUC / episodes-to-threshold table (criterion: moving average of 20 COMPLETE episodes >= 50.0; conditional median over the seeds that cross, k>=1, without requiring k>=n/2; .5 not truncated; '---' = no seed crosses)

| config | n | R20 mean +- SD | AUC mean +- SD | ep_threshold (median) | k/n | steps_threshold (median) |
|---|---|---|---|---|---|---|
| beta=0.05 (ref) | 10 | 9.87 +- 0.43 | 11.68 +- 0.41 | --- | 0/10 | --- |
| beta=0.00 | 10 | 10.04 +- 0.53 | 11.39 +- 0.27 | --- | 0/10 | --- |
| beta=0.01 | 10 | 10.09 +- 0.71 | 11.40 +- 0.27 | --- | 0/10 | --- |
| beta=0.10 | 10 | 9.78 +- 0.40 | 11.78 +- 0.40 | --- | 0/10 | --- |
| beta=0.30 | 10 | 9.97 +- 0.31 | 12.78 +- 0.66 | --- | 0/10 | --- |

### Memory and FLOPs (deterministic per configuration; invariant across seeds and episodes)

| config | memory_bytes | flops_per_update |
|---|---|---|
| beta=0.05 (ref) | 92 | 80 |
| beta=0.00 | 92 | 80 |
| beta=0.01 | 92 | 80 |
| beta=0.10 | 92 | 80 |
| beta=0.30 | 92 | 80 |

### Paired statistical analysis (10 shared seeds; Friedman + paired Wilcoxon signed-rank vs. reference, Holm-Bonferroni; paired bootstrap with 10000 resamples for the 95% CI of the difference of medians)


**R20** -- Friedman chi2=3.47, p=0.4817

| config vs. reference | median diff. | Wilcoxon p (raw) | Holm p-adj | rejects H0 (alpha=0.05) | 95% bootstrap CI (median diff.) |
|---|---|---|---|---|---|
| beta=0.00 | +0.175 | 0.3223 | 1.0000 | no | [-0.150, +0.600] |
| beta=0.01 | -0.075 | 0.3535 | 1.0000 | no | [-0.475, +0.650] |
| beta=0.10 | -0.275 | 0.5117 | 1.0000 | no | [-0.725, +0.300] |
| beta=0.30 | +0.075 | 0.5801 | 1.0000 | no | [-0.450, +0.550] |

**AUC** -- Friedman chi2=24.08, p=7.698e-05

| config vs. reference | median diff. | Wilcoxon p (raw) | Holm p-adj | rejects H0 (alpha=0.05) | 95% bootstrap CI (median diff.) |
|---|---|---|---|---|---|
| beta=0.00 | -0.163 | 0.1055 | 0.2520 | no | [-0.770, +0.114] |
| beta=0.01 | -0.126 | 0.0840 | 0.2520 | no | [-0.743, +0.136] |
| beta=0.10 | +0.282 | 0.5566 | 0.5566 | no | [-0.344, +0.547] |
| beta=0.30 | +1.098 | 0.0020 | 0.0078 | yes | [+0.617, +1.259] |

## Direct readings of the data (no causal interpretation for the paper yet)

- **X1:** Friedman significant for R20 (p=2.5e-5) and AUC (p=7.5e-6). Wilcoxon-Holm rejects H0 (vs. ref [6,6,12,12]) for u18 and u24 in both metrics, and for u12 in AUC (not in R20, p_holm=0.082). u6 and u9 do not differ from ref after Holm. The number of seeds reaching the convergence threshold (50.0) decreases with the number of bins but NOT strictly monotonically: u6 10/10, u9 7/10, u12 2/10, u18 0/10, u24 1/10 (ref, with mixed bins [6,6,12,12], 8/10). The drop is clear from u6 to u18, but u24 (1/10) crosses one seed more than u18 (0/10); with n=10 seeds that 1-case difference does not allow claiming strict monotonicity between u18 and u24. Where there are crossings (ref, u6, u9, u12, u24), the conditional median of episodes-to-threshold also rises with more bins (494.5 to 784.5), consistent with slower tabular learning in larger state spaces.
- **X2:** Friedman NOT significant for R20 (p=0.48): there is no evidence that beta_e in {0.00,0.01,0.05,0.10,0.30} changes the final reward of Stream-AC/CartPole. It is significant for AUC (p=7.7e-5), and only b030 survives Holm (median diff. +1.10 over ref, 95% CI does not cross 0). No X2 configuration crosses the convergence threshold (50.0) in any seed (0/10 in all); R20 stays between ~9.8 and ~10.1 in the 5 configurations, consistent with Stream-AC/CartPole with these hyperparameters not learning a policy that keeps the pole up for more than ~10 steps in the last 20 episodes.

The manuscript, figures and firmware were not modified by this report; it is raw material for deciding which conclusions of the paper change.
