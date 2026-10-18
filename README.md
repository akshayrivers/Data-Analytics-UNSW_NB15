# Statistical Analysis of the UNSW-NB15 Network Traffic Dataset

Parametric & non-parametric inference, confidence intervals, and empirical scaling of sample size.

This repository contains a reproducible statistical analysis of the UNSW-NB15 network-flow dataset. The analysis is implemented in `analysis.py`, produces CSV tables in `outputs/tables/` and publication-quality figures in `outputs/figures/`, and uses deterministic sampling seeds to demonstrate how estimates, intervals, and inferential tests change with sample size.

Contents
- Dataset and selected variables
- Mathematical definitions (corrected and unambiguous)
- Inferential procedures implemented
- Summary of results and recommendations
- Reproducibility and execution
- Repository layout and license

Dataset and selected variables
------------------------------
The analysis concatenates the provided training and testing parquet partitions into a single population of N = 257,673 records. Key derived and used variables include:
- label: binary class (0 = Normal, 1 = Attack)
- attack_cat: multi-class attack label (Generic, Exploits, Fuzzers, DoS, Reconnaissance, ...)
- proto_grp: protocol aggregated to {tcp, udp, other}
- Focus continuous features: dur, sbytes, dbytes, rate, smean, dmean

Mathematical definitions (precise)
---------------------------------
All formulas below are sample-based unless stated otherwise. The code uses sample variance with Bessel correction (ddof=1) and SciPy test routines.

Notation: x_1,...,x_n are sample observations; \bar{x} denotes the sample mean; s^2 denotes the sample variance (ddof=1).

1) Point estimates and dispersion
- Sample mean:
  \[\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i.\]
- Sample variance (Bessel correction):
  \[s^2 = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})^2.\]
- Sample standard deviation: s = \sqrt{s^2}.
- Standard error of the mean (SEM):
  \[\mathrm{SE}(\bar{x}) = \frac{s}{\sqrt{n}}.\]
- Interquartile range: IQR = Q_{0.75} - Q_{0.25}.
- Sample skewness (third standardized moment, library-dependent bias corrections noted):
  \[g_1 = \frac{\frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^3}{\left(\frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2\right)^{3/2}}.\]
- Excess kurtosis (Fisher definition):
  \[g_2 = \frac{\frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^4}{\left(\frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2\right)^2} - 3.\]

Note: SciPy's `stats.skew` and `stats.kurtosis` return sample estimates (the latter returns excess kurtosis by default). The README and code are aligned: code uses SciPy.

2) Confidence intervals
- Student's t two-sided (1-\alpha) CI for a mean using the sample sd:
  \[\mathrm{CI}_{1-\alpha}(\mu) = \left[\bar{x} - t_{1-\alpha/2,\,n-1}\frac{s}{\sqrt{n}},\;\bar{x} + t_{1-\alpha/2,\,n-1}\frac{s}{\sqrt{n}}\right].\]
- Proportion CI (\hat{p}=k/n):
  - Wald (approximate):
    \[\hat{p} \pm z_{1-\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}.\]
  - Wilson (recommended when n is moderate or \hat{p} near 0/1):
    \[\mathrm{CI}_{\text{Wilson}} = \frac{\hat{p} + \frac{z^2}{2n}}{1+\frac{z^2}{n}} \pm \frac{z}{1+\frac{z^2}{n}} \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}.\]
  where z = \Phi^{-1}(1-\alpha/2).

3) Hypothesis testing (concise, as implemented)
- Welch two-sample t (unequal variances):
  test statistic:
  \[t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}},\]
  degrees of freedom (Welch–Satterthwaite):
  \[\nu \approx \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_2^2/n_2)^2}{n_2-1}}.\]
  Cohen's d (descriptive effect size using pooled SD):
  \[d = \frac{\bar{x}_1 - \bar{x}_2}{s_p},\quad s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}.\]
  (Important: Cohen's d is a descriptive, standardized magnitude; Welch t does not assume pooled variance.)

- Pearson's chi-square for contingency table (observed O_{ij}, expected E_{ij}=R_i C_j / N):
  \[\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}},\qquad df=(r-1)(c-1).\]
  Cramér's V effect size:
  \[V = \sqrt{\frac{\chi^2}{N \cdot \min(r-1,c-1)}}.\]

- One-way ANOVA for k groups:
  SSB, SSW, SST as usual; F = MSB/MSW, and eta-squared = SSB/SST as implemented.

4) Correlation (Pearson and Spearman) — corrected and explicit
- Pearson sample correlation (implemented with SciPy `pearsonr`):
  \[r = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2}\sqrt{\sum_{i=1}^n (y_i - \bar{y})^2}}.\]
- Fisher z transform (for approximate CI on r):
  \[z_r = \operatorname{arctanh}(r) = \tfrac{1}{2}\ln\left(\frac{1+r}{1-r}\right),\qquad \mathrm{SE}_z = \frac{1}{\sqrt{n-3}}.\]
  Transform back to r using tanh. Implementation note: values of r are clipped inside the code to avoid numerical overflow before arctanh.
- Spearman rank correlation:
  Computed by first ranking X and Y (average ranks for ties) and computing Pearson correlation on ranks; if no ties the closed form is \(\rho_s = 1 - \frac{6\sum d_i^2}{n(n^2-1)}\).

Implementation note: the code uses SciPy (`stats.pearsonr` and `stats.spearmanr`) and applies Fisher-z CI for Pearson.

Summary of inferential workflow
-------------------------------
The analysis implements the following tests and summaries (all at nominal \(\alpha = 0.05\)):
- T1: Welch two-sample t-test on `dmean` (Normal vs Attack), report t, df (Welch), p-value, 95% CI for mean difference, Cohen's d.
- T2: Pearson chi-square on `proto_grp` × `label`, report chi2, df, p-value, Cramér's V.
- T3: One-way ANOVA on `smean` across top attack categories, report F, p-value, eta-squared.
- T4: Pearson and Spearman correlation between `sbytes` and `dbytes`, report r, p-value, Fisher-z CI; report Spearman rho and p-value.

Practical interpretation guidance
--------------------------------
- In large samples, p-values can be extremely small for modest effects; always inspect effect sizes (Cohen's d, Cramér's V, eta-squared, Spearman rho) and confidence intervals.
- For heavy-tailed network-flow variables, medians and rank-based summaries are more robust and often more informative than raw means.
- Use Wilson intervals for proportions when sample sizes are not very large or proportions are near 0/1.

Reproducibility and execution
-----------------------------
Run the single analysis script which reads the parquet files in `archive/`, performs deterministic sampling, computes all tables, and writes outputs:

```bash
python analysis.py
```

Environment (Python 3.10+):
```bash
pip install numpy pandas scipy matplotlib seaborn pyarrow
```

Repository layout
-----------------
- analysis.py — analysis engine
- archive/ — input parquet datasets (training + testing)
- outputs/tables/ — CSV tables produced by the analysis
- outputs/figures/ — PNG figures produced by the analysis
- README.md — this file
- LICENSE — MIT license

License
-------
MIT License. The raw UNSW-NB15 dataset is the property of the Australian Centre for Cyber Security (UNSW Canberra); this repository documents an analysis of that dataset.
