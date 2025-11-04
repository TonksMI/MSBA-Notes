# Technical Insights – CPSC 540 Homework 1

## Modeling Choices

- **Weighting:** Inverse-variance weights were computed within each `(campaign_variant, email_signup)` cell to stabilize heteroscedastic campaign spend, then normalized to mean 1 so model estimation preserves an effective sample size. A `.Machine$double.eps` offset prevents division-by-zero in low-variance strata.
- **Count Models:** Overdispersion diagnostics (variance/mean ratio > 6; `check_overdispersion`) ruled out Poisson regression. Weighted Negative Binomial models were compared against Poisson and GAM alternatives; the simplified NB model delivered the lowest AIC.
- **Continuous Outcome:** The ANCOVA includes age, gender, past purchases, email status, and ad source, mirroring lecture guidance. Type III sums of squares (`car::Anova`) accommodate the unbalanced design.
- **Effect Sizes:** `effectsize::cohens_d` supplies weighted Cohen’s *d* for each campaign pair, with Sawilowsky thresholds for interpretation. Pairwise Tukey HSD uses the weighted `aov` object to satisfy the assignment brief.

## Diagnostics & Validation

- **Negative Binomial:** Deviance residuals vs. fitted plot shows no pattern, supporting fit adequacy after weighting.  
- **ANCOVA:** Standardized residual plot indicates mild funneling at extreme fitted values but within acceptable limits; no multicollinearity flags were triggered.
- **Overdispersion:** Chi-square test strongly rejects equidispersion (p ≈ 0), reinforcing the NB choice.

## Assumptions & Limitations

- The dataset is synthetic and cross-sectional, so causal claims are avoided; recommendations emphasize association and practical deployment.  
- Email signup is self-selected, introducing potential omitted-variable bias—highlighted in the discussion as a future quasi-experimental priority.  
- Weighted Tukey comparisons inherit the linear model’s covariate adjustment; however, they do not fully replicate a generalized linear hypothesis for NB outcomes.

## Potential Extensions

- **Robustness:** Fit a zero-inflated NB for sensitivity; early tests suggested marginal gains but at the cost of interpretability.  
- **Channel Strategy:** Explore interaction between `ad_source` and campaign variant in the ANCOVA to tailor media buying guidance.  
- **Profitability:** Layer average margin estimates onto adjusted spend to translate findings directly into projected profit.
