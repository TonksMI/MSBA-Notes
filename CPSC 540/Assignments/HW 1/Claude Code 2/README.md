# CPSC 540 Homework 1: Marketing Campaign Analysis

**Student:** Matthew Tonks
**Due Date:** September 29th, 2025
**Course:** CPSC 540 - Statistical Machine Learning

---

## Project Overview

This project analyzes synthetic marketing data from Company X (a wellness brand) to answer two critical business questions:

1. **What kinds of people make MORE purchases?**
   Understanding customer characteristics associated with higher purchase frequency to inform targeting and retention strategies.

2. **Which campaign variant should Company X use in the future?**
   Evaluating three variants of the "Be Well" campaign (A: control, B: physical health focus, C: mental health focus) to determine optimal marketing strategy.

---

## Repository Structure

```
Claude Code 2/
├── README.md                    # This file - project overview and instructions
├── analysis.Rmd                 # Complete R Markdown analysis with all models
├── insights.md                  # Technical notes, modeling decisions, and refinements
├── final_submission.md          # Polished report for submission (PDF ready)
└── marketingcampaign.csv       # Dataset (5,000 customers)
```

---

## Files Description

### 1. `analysis.Rmd` - Modeling File

**Purpose:** Complete statistical analysis with all code, models, and outputs

**Contents:**
- Data loading and preparation
- Comprehensive exploratory data analysis (EDA)
- Question 1 models: Negative Binomial regression for count data (past purchases)
- Question 2 models: ANCOVA/linear regression for continuous data (campaign spend)
- Model diagnostics and validation
- Results visualization
- Model comparison and selection

**How to Use:**
```r
# Open in RStudio
# Install required packages (see Dependencies section below)
# Knit to HTML or run chunks interactively
rmarkdown::render("analysis.Rmd")
```

**Key Features:**
- Reproducible (seed = 1818)
- Well-commented code
- Publication-quality visualizations
- Comprehensive model diagnostics

---

### 2. `insights.md` - Analysis Notes

**Purpose:** Document the analytical process, decisions, and iterative refinements

**Contents:**
- Data exploration findings
- Model selection rationale (why Negative Binomial? why ANCOVA?)
- Key insights and patterns discovered
- Errors encountered and solutions
- Refinements made during analysis
- Technical notes for reproducibility
- Communication strategy for non-technical audiences

**How to Use:**
- Read to understand the "why" behind modeling choices
- Reference when writing similar analyses
- Use as checklist for comprehensive analysis

**Key Features:**
- Transparent documentation of analytical decisions
- Bridges technical analysis and business communication
- Captures lessons learned

---

### 3. `final_submission.md` - Report for Submission

**Purpose:** Polished, submission-ready report answering the homework questions

**Contents:**

#### Analysis Section
- Methodological approach for each question
- Justification for model choices
- Why these methods are appropriate

#### Results Section
- Clear answers to Question 1: Characteristics of high-frequency purchasers
- Clear answer to Question 2: Recommended campaign variant with evidence
- Key findings with statistical and practical significance
- Tables and figures supporting conclusions

#### Discussion Section
- Potential impacts and business applications
- Strategic implications for Company X
- What I would do differently next time (methodological improvements)
- Limitations and caveats

**How to Use:**
- Convert to PDF for submission: `pandoc final_submission.md -o final_submission.pdf`
- Or copy/paste into Word and format for submission
- Can be read as standalone document

**Key Features:**
- Non-technical language for business stakeholders
- Clear, actionable recommendations
- Comprehensive but accessible
- Addresses all homework requirements

---

### 4. `marketingcampaign.csv` - Dataset

**Source:** https://raw.githubusercontent.com/chelseaparlett/CPSC540ParlettPelleriti/refs/heads/main/Data/marketingcampaign.csv

**Description:** Synthetic data for 5,000 customers of Company X wellness brand

**Variables:**
- `age`: Customer age (years)
- `gender`: Self-identified gender (Man, Woman, Non-Binary)
- `past_purchases`: Number of purchases over past 2 years (count)
- `campaign_variant`: Campaign version assigned (A = control, B = Be Well physical, C = Be Well mental)
- `ad_source`: Platform where customer saw campaign (TikTok, Instagram, Facebook, Google Search)
- `email_signup`: Enrolled in promotional emails (TRUE/FALSE)
- `campaign_spend`: Amount spent on Be Well campaign items (dollars)

**Sample Size:** n = 5,000 customers (~1,667 per campaign variant)

---

## Dependencies

### R Version
- R version 4.0 or higher recommended

### Required R Packages

```r
# Install all required packages
install.packages(c(
  "tidyverse",      # Data manipulation and visualization
  "broom",          # Tidy model outputs
  "MASS",           # Negative binomial regression (glm.nb)
  "mgcv",           # Generalized additive models (GAM)
  "performance",    # Model diagnostics
  "see",            # Enhanced visualizations for performance
  "gridExtra",      # Multiple plot arrangement
  "knitr",          # Table formatting
  "car",            # ANOVA, VIF diagnostics
  "effectsize",     # Cohen's d and effect sizes
  "emmeans"         # Estimated marginal means and contrasts
))
```

### Loading Packages

All required packages are loaded in the setup chunk of `analysis.Rmd`:

```r
library(tidyverse)
library(broom)
library(MASS)
library(mgcv)
library(performance)
library(see)
library(gridExtra)
library(knitr)
library(car)
library(effectsize)  # Install if not available
library(emmeans)     # Install if not available
```

---

## How to Run the Analysis

### Option 1: RStudio (Recommended)

1. **Open RStudio**
2. **Set working directory:**
   ```r
   setwd("/Users/matthewtonks/Repositories/CPSC 540/Assignments/HW 1/Claude Code 2")
   ```
3. **Open `analysis.Rmd`**
4. **Install packages** (if not already installed - see Dependencies)
5. **Click "Knit"** to render full HTML report
   - Or run chunks interactively for exploration

### Option 2: Command Line

```bash
# Navigate to directory
cd "/Users/matthewtonks/Repositories/CPSC 540/Assignments/HW 1/Claude Code 2"

# Render R Markdown to HTML
Rscript -e "rmarkdown::render('analysis.Rmd')"

# Output will be analysis.html
```

### Option 3: Interactive R Session

```r
# Set working directory
setwd("/Users/matthewtonks/Repositories/CPSC 540/Assignments/HW 1/Claude Code 2")

# Load required libraries
source("analysis.Rmd")  # Or copy chunks into console

# Explore interactively
```

---

## Analysis Workflow

The analysis follows this systematic workflow:

### 1. Data Preparation
- Load CSV data
- Convert categorical variables to factors
- Check for missing values and data quality issues
- Summary statistics

### 2. Exploratory Data Analysis (EDA)
- Distribution plots for all variables
- Relationships between predictors and outcomes
- Identify potential issues (outliers, skewness, etc.)

### 3. Question 1: Purchase Frequency Modeling
- **Check overdispersion** (variance vs. mean)
- **Fit Negative Binomial regression** (handles overdispersion)
- **Compare models:** Full, simplified, Poisson, GAM
- **Diagnostics:** Residual plots, model fit checks
- **Interpret:** Incident rate ratios (IRR) for effect sizes
- **Profile:** High vs. low frequency purchaser characteristics

### 4. Question 2: Campaign Variant Comparison
- **Fit ANCOVA** (linear regression with categorical + continuous predictors)
- **Control for covariates** (age, gender, past purchases, email, ad source)
- **Post-hoc tests:** Tukey HSD for pairwise comparisons
- **Effect sizes:** Cohen's d for practical significance
- **Subgroup analysis:** Test interactions and differential effects
- **Visualize:** Adjusted means with confidence intervals

### 5. Model Comparison
- AIC/BIC for statistical comparison
- Practical interpretation for business decisions
- Select best model for each question

### 6. Final Recommendations
- Clear answers to both questions
- Quantify business impact
- Provide actionable insights

---

## Key Analytical Decisions

### Question 1: Why Negative Binomial?

**Data Characteristics:**
- **Count variable:** Past purchases (0, 1, 2, 3, ...)
- **Overdispersed:** Variance >> Mean (some customers make many more purchases than average)

**Model Choice:**
- ❌ **Linear regression:** Inappropriate for count data (can predict negative values)
- ❌ **Poisson regression:** Assumes variance = mean (violated here)
- ✅ **Negative Binomial:** Explicitly models overdispersion, standard for purchase frequency

**Interpretation:**
- Coefficients give **Incident Rate Ratios (IRR)**
- IRR > 1: Factor increases expected purchases
- IRR < 1: Factor decreases expected purchases

---

### Question 2: Why ANCOVA/Linear Regression?

**Data Characteristics:**
- **Continuous outcome:** Campaign spend (dollars)
- **Multiple groups:** Three campaign variants (A, B, C)
- **Confounders present:** Customer characteristics may differ across groups

**Model Choice:**
- ❌ **Simple ANOVA:** Ignores confounding variables
- ✅ **ANCOVA:** Compares group means while controlling for covariates
- ✅ **Gamma GLM:** Robustness check for skewed positive data

**Interpretation:**
- **Adjusted means:** Average spend by variant after controlling for customer differences
- **Pairwise comparisons:** Statistical tests of differences between variants
- **Effect sizes:** Practical magnitude of differences (Cohen's d)

---

## Expected Outputs

### From `analysis.Rmd` (when knitted)

1. **HTML Report** (`analysis.html`) containing:
   - All code and outputs
   - Visualizations (20+ plots)
   - Statistical tables
   - Model summaries
   - Diagnostic plots

2. **Key Results:**
   - **Q1:** Profile of high-frequency purchasers with IRRs
   - **Q2:** Recommended campaign variant with evidence
   - **Both:** Model comparisons, effect sizes, confidence intervals

### From `final_submission.md`

- **Submission-ready report** addressing:
  - ✅ Analysis section: Methods and justification
  - ✅ Results section: Clear answers to both questions
  - ✅ Discussion section: Implications and improvements

---

## Deliverables for Homework Submission

Per homework requirements, submit:

1. **PDF of Report**
   - Convert `final_submission.md` to PDF
   - Or export `analysis.html` to PDF from browser
   - Ensure all figures and tables are included

2. **Code File**
   - Submit `analysis.Rmd`
   - Fully commented and reproducible
   - Can be run to regenerate all results

3. **README**
   - This file (`README.md` or convert to `README.txt`)
   - Documents project structure and how to run analysis

---

## Reproducibility

### Ensuring Reproducible Results

1. **Set seed:** `set.seed(1818)` in setup chunk
2. **Package versions:** Documented in Session Info at end of `analysis.Rmd`
3. **Data source:** URL documented, data included in directory
4. **Complete code:** All analysis steps included in `analysis.Rmd`

### Verifying Reproducibility

```r
# Run this to verify analysis reproduces
rmarkdown::render("analysis.Rmd", output_file = "verification.html")

# Compare to original analysis.html
# Results should be identical (given same package versions)
```

---

## Contact and Questions

**Student:** Matthew Tonks
**Course:** CPSC 540 - Statistical Machine Learning
**Instructor:** [Instructor name]

For questions about this analysis:
- Review `insights.md` for technical details
- Check `final_submission.md` Discussion section for methodology notes
- Examine `analysis.Rmd` code comments for implementation details

---

## Acknowledgments

- **Data Source:** CPSC 540 course materials (synthetic data)
- **Statistical Methods:** Based on course lectures and readings
- **R Packages:** Grateful to all package authors (see Session Info)

---

## License and Usage

This is an academic assignment for CPSC 540. Code and analysis methods may be adapted for educational purposes with proper attribution.

---

## Appendix: Quick Reference

### Model Summary

| Question | Outcome Variable | Model Type | Key Predictors | Recommended Action |
|----------|------------------|------------|----------------|-------------------|
| Q1 | `past_purchases` (count) | Negative Binomial | Age, email signup, campaign spend | Focus on email list growth |
| Q2 | `campaign_spend` (dollars) | ANCOVA/Linear | Campaign variant (controlling for covariates) | Use variant [B/C] - see results |

### File Sizes (Approximate)

- `marketingcampaign.csv`: ~180 KB
- `analysis.Rmd`: ~30 KB
- `analysis.html`: ~3-5 MB (with plots)
- `insights.md`: ~25 KB
- `final_submission.md`: ~35 KB

### Time Estimates

- **Running full analysis:** 2-5 minutes (depending on system)
- **Reading insights.md:** 15-20 minutes
- **Reading final_submission.md:** 20-30 minutes
- **Reviewing analysis.Rmd interactively:** 45-60 minutes

---

**Last Updated:** October 13, 2025
**Version:** 1.0
