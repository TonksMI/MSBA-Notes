# CPSC 540 HW1 (Codex Submission)

This folder contains a reproducible analysis of the *marketingcampaign* dataset for Homework 1.

## Contents

- `analysis.qmd` – Quarto document with the full R workflow (EDA, models, plots). Knit to HTML/PDF for a narrative version.
- `report.md` – Plain-language write-up answering the homework questions (Analysis, Results, Discussion).
- (Optional) `poisson_mod.rds` and `spend_mod.rds` are written to this folder if you render the Quarto document and want to reuse the fitted models.

## How to reproduce

1. Open R (4.2+ recommended) with the `tidyverse` and `broom` packages installed.
2. Set the working directory to `CPSC 540/Assignments/HW 1/codex code`.
3. Run `quarto render analysis.qmd --to html` (or open the file in RStudio and click *Render*).
4. The script reads the dataset from `../Claude Code/marketingcampaign.csv`, fits the models, produces plots, and saves the model objects.
5. Use `report.md` as the basis for the required PDF submission, or knit the Quarto output to PDF as needed.

## Notes

- No external data beyond the provided CSV is required.
- The models focus on interpretability: Poisson regression for counts and a log-linear model for spend.
- All numeric results in `report.md` come from the executed analysis and can be regenerated via the Quarto document.
