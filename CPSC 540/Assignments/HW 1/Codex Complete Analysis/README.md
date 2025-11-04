# CPSC 540 Homework 1 – Weighted Campaign Analysis

## Overview

This repository contains the full workflow for CPSC 540 Homework 1. The assignment analyzes Company X’s “Be Well” marketing experiment using inverse-variance weighting and model comparisons tailored to the campaign’s two business questions:

- **Question 1:** Which customer characteristics explain higher purchase counts?  
  → Answered with weighted Negative Binomial models, incident rate ratios, and diagnostics for overdispersion.
- **Question 2:** Which campaign variant maximizes spend?  
  → Addressed via weighted ANCOVA, Type III inference, Tukey HSD, Cohen’s *d*, and marginal means.

## Files

- `final_submission_v2.Rmd` – End-to-end R Markdown analysis with documentation and visuals.  
- `final_submission_v2.html` – Rendered stakeholder report (open in a browser).  
- `insights.md` – Technical notes, open issues, and follow-up ideas.  
- `marketingcampaign.csv` – Provided data source (5,000 customers).

## Reproducibility

1. Ensure R ≥ 4.3 with the packages below installed:
   - `tidyverse`, `broom`, `MASS`, `mgcv`, `performance`, `car`, `knitr`, `emmeans`, `effectsize`, `glue`, `scales`.
2. From the `Complete Analysis` directory run:
   ```bash
   Rscript -e "rmarkdown::render('final_submission_v2.Rmd', output_file = 'final_submission_v2.html')"
   ```
3. The command sets `set.seed(1818)` and recreates all tables, visuals, and narrative text.

## Key Findings

- Purchase frequency dips about **21% per decade of age** (IRR = 0.78 per 10 years) while email enrollment remains statistically neutral; incremental campaign spend produces a small but significant lift.  
- Campaign **Variant B** delivers the strongest weighted spend at **$217 per customer**, beating Variant A by about **$69** (Tukey-adjusted p < 0.0001) and translating to roughly **$346K** in incremental revenue across 5,000 customers.  
- Weighted diagnostics confirm the modeling assumptions: Negative Binomial handles overdispersion in counts, and residual checks show no major violations for the ANCOVA.

## Notes

- All models apply the required inverse-variance weights grouped by `campaign_variant` and `email_signup`.  
- Tables rely on `knitr::kable()` and the HTML report disables console `cat()` output, satisfying assignment formatting rules.  
- The analysis is designed for non-technical stakeholders while preserving methodological transparency for grading.
