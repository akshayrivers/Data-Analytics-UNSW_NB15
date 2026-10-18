"""
UNSW-NB15 Statistical Analysis — Assignment
Samples: n=500, n=2000, n=10000 + Full (N=257673)
Seeded random sampling for reproducibility.
Outputs: outputs/tables/*.csv , outputs/figures/*.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

SEED = 42
SAMPLES = {"n500": (500, 11), "n2000": (2000, 22), "n10000": (10000, 33)}
FOCUS = ["dur", "sbytes", "dbytes", "rate", "smean", "dmean"]
TOP6 = ["Normal", "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance"]

OUT_TAB = "outputs/tables"
OUT_FIG = "outputs/figures"
os.makedirs(OUT_TAB, exist_ok=True)
os.makedirs(OUT_FIG, exist_ok=True)
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10})

# ---------- Load ----------
train = pd.read_parquet("archive/UNSW_NB15_training-set.parquet")
test = pd.read_parquet("archive/UNSW_NB15_testing-set.parquet")
df = pd.concat([train, test], ignore_index=True)
N_FULL = len(df)
print(f"Full N = {N_FULL}")
df["proto_grp"] = df["proto"].apply(lambda x: x if x in ["tcp", "udp"] else "other")

# ---------- Samples ----------
samples = {}
for name, (n, seed) in SAMPLES.items():
    samples[name] = df.sample(n=n, random_state=seed)
samples["full"] = df
order = ["n500", "n2000", "n10000", "full"]
nvals = {"n500": 500, "n2000": 2000, "n10000": 10000, "full": N_FULL}

# ---------- 1. Descriptive statistics ----------
rows = []
for sname in order:
    d = samples[sname]
    for v in FOCUS:
        x = d[v].dropna().to_numpy(dtype=float)
        n = len(x)
        mean = float(np.mean(x))
        median = float(np.median(x))
        var = float(np.var(x, ddof=1)) if n > 1 else np.nan
        std = float(np.std(x, ddof=1)) if n > 1 else np.nan
        q1, q3 = float(np.quantile(x, .25)), float(np.quantile(x, .75))
        iqr = q3 - q1
        rows.append({
            "sample": sname, "n": n, "variable": v,
            "count": n, "mean": mean, "median": median,
            "variance": var, "std": std,
            "min": float(np.min(x)), "max": float(np.max(x)),
            "range": float(np.max(x) - np.min(x)),
            "q1": q1, "q3": q3, "iqr": iqr,
            "skew": float(stats.skew(x)) if n > 2 else np.nan,
            "kurtosis": float(stats.kurtosis(x)) if n > 3 else np.nan,
            "cv": float(std / mean) if mean != 0 else np.nan,
            "se_mean": float(std / np.sqrt(n)) if n > 0 else np.nan,
        })
desc = pd.DataFrame(rows)
desc.to_csv(f"{OUT_TAB}/01_descriptive_all.csv", index=False)
# pivot of key stats for report
for stat in ["mean", "std", "variance", "median"]:
    desc.pivot(index="variable", columns="sample", values=stat).to_csv(f"{OUT_TAB}/01_descriptive_{stat}.csv")
print("descriptives saved")
print(desc[desc["variable"]=="dmean"][["sample","n","mean","median","variance","std","skew"]].to_string(index=False))

# ---------- 2. 95% CI for means (t-interval) ----------
ci_rows = []
for sname in order:
    d = samples[sname]
    for v in FOCUS:
        x = d[v].dropna().to_numpy(dtype=float)
        n = len(x); m = float(np.mean(x)); s = float(np.std(x, ddof=1))
        se = s / np.sqrt(n)
        h = stats.t.ppf(0.975, n - 1) * se
        ci_rows.append({"sample": sname, "n": n, "variable": v, "mean": m, "std": s,
                        "se": se, "ci95_low": m - h, "ci95_high": m + h, "ci_width": 2*h})
ci_means = pd.DataFrame(ci_rows)
ci_means.to_csv(f"{OUT_TAB}/02_ci_means.csv", index=False)

# ---------- 3. 95% CI for attack proportion (Normal approx + Wilson) ----------
prop_rows = []
for sname in order:
    d = samples[sname]
    n = len(d); k = int((d["label"] == 1).sum()); p = k / n
    se = np.sqrt(p*(1-p)/n)
    z = stats.norm.ppf(0.975)
    low_w, high_w = p - z*se, p + z*se
    # Wilson
    denom = 1 + z**2/n
    centre = (p + z**2/(2*n)) / denom
    half = z*np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    prop_rows.append({"sample": sname, "n": n, "attacks": k, "p_hat": p,
                      "wald_low": low_w, "wald_high": high_w,
                      "wilson_low": centre-half, "wilson_high": centre+half,
                      "wilson_width": 2*half})
ci_prop = pd.DataFrame(prop_rows)
ci_prop.to_csv(f"{OUT_TAB}/03_ci_proportion_attack.csv", index=False)
print(ci_prop.to_string(index=False))

# ---------- 4. Welch t-test: dmean Normal vs Attack ----------
ttest_rows = []
for sname in order:
    d = samples[sname]
    a = d.loc[d["label"]==0, "dmean"].to_numpy(dtype=float)
    b = d.loc[d["label"]==1, "dmean"].to_numpy(dtype=float)
    m0, m1 = float(np.mean(a)), float(np.mean(b))
    diff = m0 - m1
    res = stats.ttest_ind(a, b, equal_var=False)
    # 95% CI for difference (Welch)
    s0, s1 = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
    n0, n1 = len(a), len(b)
    se_d = np.sqrt(s0**2/n0 + s1**2/n1)
    df_w = (s0**2/n0 + s1**2/n1)**2 / ((s0**2/n0)**2/(n0-1) + (s1**2/n1)**2/(n1-1))
    h = stats.t.ppf(0.975, df_w) * se_d
    # Cohen's d (pooled)
    sp = np.sqrt(((n0-1)*s0**2 + (n1-1)*s1**2) / (n0+n1-2))
    cohen = diff / sp
    ttest_rows.append({"sample": sname, "n": n0+n1, "n_normal": n0, "n_attack": n1,
                       "mean_normal": m0, "mean_attack": m1, "diff": diff,
                       "t_stat": float(res.statistic), "df_welch": float(df_w),
                       "p_value": float(res.pvalue), "se_diff": se_d,
                       "ci95_low": diff-h, "ci95_high": diff+h, "cohens_d": cohen,
                       "reject_H0_05": bool(res.pvalue < 0.05)})
ttest = pd.DataFrame(ttest_rows)
ttest.to_csv(f"{OUT_TAB}/04_ttest_dmean_normal_vs_attack.csv", index=False)
print(ttest.to_string(index=False))

# ---------- 5. Chi-square: proto_grp x label ----------
chi_rows = []
for sname in order:
    d = samples[sname]
    ct = pd.crosstab(d["proto_grp"], d["label"])
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    cramer = np.sqrt(chi2 / (len(d) * (min(ct.shape)-1)))
    chi_rows.append({"sample": sname, "n": len(d), "chi2": chi2, "df": dof,
                     "p_value": p, "cramers_v": cramer,
                     "reject_H0_05": bool(p < 0.05)})
    if sname == "full":
        ct.to_csv(f"{OUT_TAB}/05_crosstab_proto_label_full.csv")
        pd.DataFrame(exp, index=ct.index, columns=ct.columns).to_csv(f"{OUT_TAB}/05_expected_proto_label_full.csv")
chi = pd.DataFrame(chi_rows)
chi.to_csv(f"{OUT_TAB}/05_chisquare_proto_vs_label.csv", index=False)
print(chi.to_string(index=False))

# ---------- 6. One-way ANOVA: smean across TOP6 attack_cat ----------
anova_rows = []
for sname in order:
    d = samples[sname]
    groups = [d.loc[d["attack_cat"]==c, "smean"].to_numpy(dtype=float) for c in TOP6]
    groups = [g for g in groups if len(g) > 1]
    F, p = stats.f_oneway(*groups)
    # eta-squared
    allx = np.concatenate(groups)
    grand = np.mean(allx)
    ssb = sum(len(g)*(np.mean(g)-grand)**2 for g in groups)
    sst = np.sum((allx-grand)**2)
    eta2 = ssb/sst if sst>0 else np.nan
    # group ns
    ns = [int((d["attack_cat"]==c).sum()) for c in TOP6]
    anova_rows.append({"sample": sname, "n_top6": int(sum(ns)), "F": float(F),
                       "df_between": len(groups)-1, "df_within": len(allx)-len(groups),
                       "p_value": float(p), "eta_squared": float(eta2),
                       "reject_H0_05": bool(p < 0.05),
                       **{f"n_{c}": ns[i] for i, c in enumerate(TOP6)}})
anova = pd.DataFrame(anova_rows)
anova.to_csv(f"{OUT_TAB}/06_anova_smean_by_attackcat.csv", index=False)
print(anova.to_string(index=False))
# group means full for report
grp_full = df[df["attack_cat"].isin(TOP6)].groupby("attack_cat")["smean"].agg(["count","mean","std","median"])
grp_full.to_csv(f"{OUT_TAB}/06_groupmeans_smean_full.csv")
print(grp_full.round(2).to_string())

# ---------- 7. Correlation: sbytes vs dbytes (Pearson + Spearman) + log version ----------
corr_rows = []
for sname in order:
    d = samples[sname]
    x = d["sbytes"].to_numpy(dtype=float); y = d["dbytes"].to_numpy(dtype=float)
    r, p_r = stats.pearsonr(x, y)
    rs, p_s = stats.spearmanr(x, y)
    # 95% CI for Pearson via Fisher z
    zr = np.arctanh(np.clip(r, -0.999999, 0.999999))
    sez = 1/np.sqrt(len(x)-3)
    lo_z, hi_z = zr - 1.96*sez, zr + 1.96*sez
    corr_rows.append({"sample": sname, "n": len(x), "pearson_r": r, "pearson_p": p_r,
                      "pearson_ci_low": float(np.tanh(lo_z)), "pearson_ci_high": float(np.tanh(hi_z)),
                      "spearman_rho": float(rs), "spearman_p": float(p_s)})
corr = pd.DataFrame(corr_rows)
corr.to_csv(f"{OUT_TAB}/07_correlation_sbytes_dbytes.csv", index=False)
print(corr.to_string(index=False))

print("ALL TABLES DONE")

# ================= FIGURES =================
# 1. histograms
for v, bins, title, xlabel in [
    ("smean", 60, "Distribution of smean (mean packet size, source)", "smean (bytes)"),
    ("dmean", 60, "Distribution of dmean (mean packet size, destination)", "dmean (bytes)"),
    ("dur", 60, "Distribution of dur (connection duration, s) — clipped at 10 s", "dur (s)"),
    ("rate", 60, "Distribution of rate (packets/sec) — clipped at 400k", "rate (packets/s)"),
]:
    plt.figure(figsize=(7, 4.2))
    x = df[v].to_numpy(dtype=float)
    if v == "dur": x = np.clip(x, 0, 10)
    if v == "rate": x = np.clip(x, 0, 400000)
    plt.hist(x, bins=bins, edgecolor="white", linewidth=0.3)
    plt.axvline(np.mean(df[v].to_numpy(dtype=float)), color="red", linestyle="--", label=f"Mean={np.mean(df[v].to_numpy(dtype=float)):.1f}")
    plt.axvline(np.median(df[v].to_numpy(dtype=float)), color="black", linestyle=":", label=f"Median={np.median(df[v].to_numpy(dtype=float)):.1f}")
    plt.title(title + f"  (Full N={N_FULL:,})")
    plt.xlabel(xlabel); plt.ylabel("Frequency")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUT_FIG}/hist_{v}_full.png"); plt.close()

# log histograms for skewed vars
for v in ["sbytes", "dbytes"]:
    plt.figure(figsize=(7, 4.2))
    x = np.log1p(df[v].to_numpy(dtype=float))
    plt.hist(x, bins=60, edgecolor="white", linewidth=0.3)
    plt.title(f"Distribution of log1p({v}) — full data (N={N_FULL:,})")
    plt.xlabel(f"log1p({v})"); plt.ylabel("Frequency")
    plt.tight_layout(); plt.savefig(f"{OUT_FIG}/hist_log_{v}_full.png"); plt.close()

# 2. box plots (use n10000 for readability + full annotation)
d10 = samples["n10000"]
plt.figure(figsize=(7, 4.5))
sns.boxplot(data=d10, x="label", y="dmean", showfliers=False)
plt.xticks([0, 1], ["Normal (0)", "Attack (1)"])
plt.title("Box plot: dmean by class (n=10,000 sample, outliers hidden)")
plt.xlabel("Class"); plt.ylabel("dmean (bytes)")
plt.tight_layout(); plt.savefig(f"{OUT_FIG}/box_dmean_by_label.png"); plt.close()

plt.figure(figsize=(9, 4.8))
sub = df[df["attack_cat"].isin(TOP6)]
order_cats = ["Normal","Generic","Exploits","Fuzzers","DoS","Reconnaissance"]
sns.boxplot(data=sub.sample(n=10000, random_state=7), x="attack_cat", y="smean", order=order_cats, showfliers=False)
plt.title("Box plot: smean by attack category (10k subsample, outliers hidden)")
plt.xlabel("Attack category"); plt.ylabel("smean (bytes)")
plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(f"{OUT_FIG}/box_smean_by_attackcat.png"); plt.close()

# 3. bar charts
plt.figure(figsize=(9, 4.8))
ct = df["attack_cat"].value_counts()
plt.bar(ct.index.astype(str), ct.values)
plt.title(f"Bar chart: record counts by attack category (Full N={N_FULL:,})")
plt.xlabel("Attack category"); plt.ylabel("Count")
plt.xticks(rotation=25, ha="right")
for i, v in enumerate(ct.values): plt.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT_FIG}/bar_attackcat_counts.png"); plt.close()

plt.figure(figsize=(7, 4.5))
ct2 = pd.crosstab(df["proto_grp"], df["label"], normalize="index") * 100
ct2 = ct2.reindex(["tcp", "udp", "other"])
x = np.arange(len(ct2)); w = 0.35
plt.bar(x - w/2, ct2[0], w, label="Normal %")
plt.bar(x + w/2, ct2[1], w, label="Attack %")
plt.xticks(x, ["tcp", "udp", "other"])
plt.title("Grouped bar: class composition within protocol group (full data)")
plt.xlabel("Protocol group"); plt.ylabel("Percent (%)"); plt.legend()
plt.tight_layout(); plt.savefig(f"{OUT_FIG}/bar_proto_by_label.png"); plt.close()

# 4. scatter plots (subsample for clarity)
s2 = samples["n2000"]
plt.figure(figsize=(7, 4.8))
plt.scatter(np.log1p(s2["sbytes"]), np.log1p(s2["dbytes"]), c=s2["label"], alpha=0.4, s=10)
plt.title("Scatter: log1p(sbytes) vs log1p(dbytes), coloured by class (n=2,000)")
plt.xlabel("log1p(sbytes)"); plt.ylabel("log1p(dbytes)")
plt.tight_layout(); plt.savefig(f"{OUT_FIG}/scatter_sbytes_dbytes.png"); plt.close()

plt.figure(figsize=(7, 4.8))
plt.scatter(s2["spkts"].clip(0, 100), np.log1p(s2["sbytes"]), c=s2["label"], alpha=0.4, s=10)
plt.title("Scatter: spkts (clipped 0–100) vs log1p(sbytes) (n=2,000)")
plt.xlabel("spkts"); plt.ylabel("log1p(sbytes)")
plt.tight_layout(); plt.savefig(f"{OUT_FIG}/scatter_spkts_sbytes.png"); plt.close()

# 5. line graphs: CI width vs n + mean stability
plt.figure(figsize=(7, 4.5))
for v in ["dmean", "smean", "dur"]:
    sub = ci_means[ci_means["variable"]==v].sort_values("n")
    plt.plot(sub["n"], sub["ci_width"], marker="o", label=v)
plt.xscale("log")
plt.title("Line graph: 95% CI width vs sample size (log x-axis)")
plt.xlabel("Sample size n (log scale)"); plt.ylabel("CI width (upper − lower)")
plt.legend(); plt.tight_layout(); plt.savefig(f"{OUT_FIG}/line_ciwidth_vs_n.png"); plt.close()

plt.figure(figsize=(7, 4.5))
sub = ci_means[ci_means["variable"]=="dmean"].sort_values("n")
plt.errorbar(sub["n"].astype(str), sub["mean"], yerr=(sub["ci_width"]/2), fmt="-o", capsize=5)
plt.title("Line graph: estimated mean of dmean ± 95% CI across sample sizes")
plt.xlabel("Sample"); plt.ylabel("dmean mean (bytes)")
plt.tight_layout(); plt.savefig(f"{OUT_FIG}/line_mean_dmean_vs_n.png"); plt.close()

# 6. grouped means across sample sizes (Normal vs Attack dmean)
plt.figure(figsize=(7, 4.5))
x = np.arange(len(order)); w = 0.35
plt.bar(x - w/2, ttest["mean_normal"], w, label="Normal mean dmean", yerr=ttest.apply(lambda r: (r["ci95_high"]-r["ci95_low"])/4, axis=1))
plt.bar(x + w/2, ttest["mean_attack"], w, label="Attack mean dmean")
plt.xticks(x, order)
plt.title("Bar: mean dmean (Normal vs Attack) across sample sizes")
plt.xlabel("Sample"); plt.ylabel("Mean dmean (bytes)"); plt.legend()
plt.tight_layout(); plt.savefig(f"{OUT_FIG}/bar_dmean_means_vs_n.png"); plt.close()

print("ALL FIGURES DONE")
