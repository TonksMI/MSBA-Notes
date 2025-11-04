# CPSC 540 Homework 1 - Complete Analysis Workflow Prompt

Use this prompt to create a complete statistical analysis from scratch for CPSC 540 Homework 1.

---

## Master Prompt

```
I need you to complete CPSC 540 Homework 1 with a complete statistical analysis following this exact structure:

## ASSIGNMENT DETAILS

**Course:** CPSC 540 - Statistical Machine Learning
**Assignment:** Homework 1 (Individual Work)
**Due Date:** September 29th, 2025 by 11:59pm

## DATA SOURCE

**URL:** https://raw.githubusercontent.com/chelseaparlett/CPSC540ParlettPelleriti/refs/heads/main/Data/marketingcampaign.csv

**Dataset Description:**
- Synthetic marketing data from Company X (wellness brand)
- 5,000 customers
- Campaign: "Be Well" (energy + protein powder product)
- Three campaign variants tested

**Variables:**
- `age`: Customer age
- `gender`: Self-identified gender
- `past_purchases`: Number of purchases over past 2 years (COUNT DATA)
- `campaign_variant`:
  - A = Old business-as-usual campaign (CONTROL)
  - B = New "Be Well" campaign (physical health focus)
  - C = "Be Well" campaign (mental health focus)
- `ad_source`: Platform where customer saw ad (TikTok, Instagram, Facebook, Google Search)
- `email_signup`: Enrolled in promotional emails (TRUE/FALSE)
- `campaign_spend`: Dollar amount spent on Be Well campaign items (CONTINUOUS)

## HOMEWORK QUESTIONS

**Question 1:** What kinds of people make MORE purchases?
- Outcome variable: `past_purchases` (count data)
- Goal: Identify customer characteristics associated with higher purchase frequency
- Target audience: Non-data experts (business stakeholders)

**Question 2:** Which campaign variant should Company X use in the future and why?
- Outcome variable: `campaign_spend` (continuous, dollars)
- Goal: Compare variants A, B, C and recommend best strategy
- Target audience: Non-data experts (business stakeholders)

## REQUIRED DELIVERABLES

Create these files in a new folder '{insert model name and timestamp here}_Folder':

1. **final_submission.Rmd** - Main analysis file with weighted models
2. **final_submission.html** - Rendered HTML report
3. **README.md** - Project documentation
4. **insights.md** - Technical notes and decisions (optional but recommended)

## CRITICAL REQUIREMENTS

### Weighting Strategy
Calculate inverse variance weights:
```r
data <- data %>%
  group_by(campaign_variant, email_signup) %>%
  mutate(
    sigma2 = var(campaign_spend),
    weight = 1 / sigma2
  ) %>%
  ungroup()
```

### Output Formatting
- **NO cat() statements for console output**
- Use markdown formatting with `results='asis'`
- All tables via kable()
- Professional HTML output

### Statistical Models

**Question 1 (Count Data - past_purchases):**
- Check overdispersion: variance vs mean ratio
- Use Negative Binomial regression (overdispersion expected)
- Compare models: NB Full, NB Simple, Poisson, GAM
- Include weights: `glm.nb(..., weights = weight)`
- Report Incident Rate Ratios (IRR) with interpretation
- All models must include weights parameter

**Question 2 (Continuous Data - campaign_spend):**
- Use ANCOVA (linear regression with covariates)
- Include weights: `lm(..., weights = weight)`
- Control for: age, gender, past_purchases, email_signup, ad_source
- Tukey HSD pairwise comparisons with weights
- Cohen's d effect sizes (calculate pairwise for 3 groups)
- Estimated marginal means with weights
- All models must include weights parameter

### Report Structure (Per Assignment Instructions)

**Section 1 - Analysis:**
- HOW you analyzed the data (methods/models used)
- WHY you chose each model/analysis
- Justify model selection for count vs continuous data
- Explain weighting approach

**Section 2 - Results:**
- Clear answer to Question 1 with evidence
- Clear answer to Question 2 with evidence
- Statistical support (p-values, CIs, effect sizes)
- Professional visualizations
- Executive-friendly language

**Section 3 - Discussion:**
- Potential impacts/applications of findings
- What you'd do differently next time
- Limitations and caveats
- Methodological improvements

## DETAILED IMPLEMENTATION GUIDE

### Step 1: Setup and Data Loading

```r
# In setup chunk
library(tidyverse)
library(broom)
library(MASS)         # glm.nb
library(mgcv)         # gam
library(performance)  # check_overdispersion
library(car)          # Anova Type III
library(knitr)        # kable
library(emmeans)      # marginal means
library(effectsize)   # Cohen's d

set.seed(1818)  # Reproducibility

# Load and prepare data
data <- read_csv("marketingcampaign.csv", show_col_types = FALSE)

data <- data %>%
  mutate(
    gender = factor(gender),
    campaign_variant = factor(campaign_variant, levels = c("A", "B", "C")),
    ad_source = factor(ad_source),
    email_signup = factor(email_signup)
  )

# Calculate weights (1/σ²)
data <- data %>%
  group_by(campaign_variant, email_signup) %>%
  mutate(
    sigma2 = var(campaign_spend),
    weight = 1 / sigma2
  ) %>%
  ungroup()
```

### Step 2: Question 1 Analysis (Count Data)

**Check Overdispersion:**
```r
mean_purchases <- mean(data$past_purchases)
var_purchases <- var(data$past_purchases)
ratio <- var_purchases / mean_purchases

# If ratio >> 1, use Negative Binomial (not Poisson)
```

**Fit Models with Weights:**
```r
# Model 1: Full NB
model_q1_full <- glm.nb(
  past_purchases ~ age + gender + email_signup + ad_source +
                   campaign_variant + campaign_spend,
  data = data,
  weights = weight
)

# Model 2: Simplified NB
model_q1_simple <- glm.nb(
  past_purchases ~ age + email_signup + campaign_spend,
  data = data,
  weights = weight
)

# Model 3: Poisson (comparison)
model_q1_poisson <- glm(
  past_purchases ~ age + email_signup + campaign_spend,
  family = poisson(link = "log"),
  data = data,
  weights = weight
)

# Model 4: GAM (non-linear)
model_q1_gam <- gam(
  past_purchases ~ s(age) + email_signup + s(campaign_spend),
  family = nb(),
  data = data,
  weights = weight
)

# Compare models
AIC(model_q1_full, model_q1_simple, model_q1_poisson, model_q1_gam)

# Select best model
best_model_q1 <- model_q1_simple
```

**Interpret Results:**
```r
# Get IRRs (Incident Rate Ratios)
tidy(best_model_q1, conf.int = TRUE, exponentiate = TRUE) %>%
  kable(digits = 3, caption = "Incident Rate Ratios")

# IRR interpretation:
# IRR > 1: Factor increases purchase count
# IRR < 1: Factor decreases purchase count
```

### Step 3: Question 2 Analysis (Continuous Data)

**Fit Models with Weights:**
```r
# Model 1: Simple LM
model_q2_simple <- lm(
  campaign_spend ~ campaign_variant,
  data = data,
  weights = weight
)

# Model 2: ANCOVA (Full)
model_q2_full <- lm(
  campaign_spend ~ campaign_variant + age + gender + past_purchases +
                   email_signup + ad_source,
  data = data,
  weights = weight
)

# Model 3: Interaction Model
model_q2_interaction <- lm(
  campaign_spend ~ campaign_variant * (email_signup + past_purchases),
  data = data,
  weights = weight
)

# Compare models
AIC(model_q2_simple, model_q2_full, model_q2_interaction)

# Select best model
best_model_q2 <- model_q2_full
```

**ANOVA with Weights:**
```r
# Type III ANOVA
Anova(best_model_q2, type = "III") %>%
  kable(digits = 3, caption = "ANOVA Results")
```

**Pairwise Comparisons:**
```r
# Tukey HSD with weights
model_aov <- aov(
  campaign_spend ~ campaign_variant + age + gender + past_purchases +
                   email_signup + ad_source,
  data = data,
  weights = weight
)

tukey_results <- TukeyHSD(model_aov, "campaign_variant")
```

**Effect Sizes:**
```r
# Cohen's d (pairwise for 3 groups)
data_BA <- data %>% filter(campaign_variant %in% c("A", "B"))
d_BA <- cohens_d(campaign_spend ~ campaign_variant, data = data_BA)

data_CA <- data %>% filter(campaign_variant %in% c("A", "C"))
d_CA <- cohens_d(campaign_spend ~ campaign_variant, data = data_CA)

data_CB <- data %>% filter(campaign_variant %in% c("B", "C"))
d_CB <- cohens_d(campaign_spend ~ campaign_variant, data = data_CB)
```

**Estimated Marginal Means:**
```r
emmeans_results <- emmeans(best_model_q2, ~ campaign_variant, weights = "proportional")
summary(emmeans_results) %>%
  kable(digits = 2, caption = "Adjusted Mean Spend by Variant")
```

### Step 4: Output Formatting

**Use Markdown, NOT cat() to console:**

```r
# WRONG - Don't do this:
cat("Mean:", mean(x), "\n")

# RIGHT - Do this instead:
# In chunk with results='asis'
cat("\n#### Key Finding\n\n")
cat(sprintf("- **Mean:** %.2f\n", mean(x)))
cat(sprintf("- **Interpretation:** This means...\n\n"))
```

**For Tables:**
```r
# Always use kable
tibble(
  Metric = c("Mean", "Variance"),
  Value = c(mean_val, var_val)
) %>% kable(caption = "Summary Statistics")
```

## ERROR PREVENTION

### Common Issues to Avoid:

1. **select() namespace conflicts:**
   - Don't use: `df %>% select(col1, col2)`
   - Instead: Use indexed access or remove select()

2. **Gamma GLM with zeros:**
   - Check: `min(campaign_spend) > 0`
   - Only fit Gamma if all values positive

3. **Cohen's d with 3 levels:**
   - Calculate pairwise separately
   - Filter data for each comparison

4. **Missing weights:**
   - Double-check all models have `weights = weight`
   - Check emmeans uses `weights = "proportional"`

## OUTPUT REQUIREMENTS

### Analysis Section - Explain:
- Why Negative Binomial for count data (overdispersion)
- Why ANCOVA for continuous data (group comparison with covariates)
- Why weights (1/σ²) for efficiency and heteroscedasticity
- Model selection criteria (AIC/BIC)

### Results Section - Report:
- **Q1:** Customer characteristics associated with higher purchases
  - Age effect with IRR
  - Email signup effect with IRR
  - Campaign spend relationship
- **Q2:** Recommended campaign variant with evidence
  - Adjusted mean spend by variant
  - Statistical significance (p-values)
  - Effect sizes (Cohen's d)
  - Revenue impact estimate

### Discussion Section - Include:
- **Applications:**
  - Email marketing strategy implications
  - Customer segmentation opportunities
  - Campaign rollout recommendations
- **Future improvements:**
  - Causal inference methods
  - Time-series analysis
  - Cost-benefit analysis
- **Limitations:**
  - Synthetic data caveat
  - Cross-sectional design
  - Unmeasured confounders

## QUALITY CHECKLIST

Before submission, verify:

- [ ] All models include weights parameter
- [ ] No cat() statements printing to console
- [ ] All output uses markdown formatting
- [ ] Tables formatted with kable()
- [ ] Overdispersion checked for Q1
- [ ] Pairwise comparisons for Q2
- [ ] Effect sizes calculated
- [ ] Diagnostics included
- [ ] Results geared toward non-experts
- [ ] HTML renders without errors
- [ ] PDF exported successfully
- [ ] README.md included
- [ ] Code file (.Rmd) included

## FINAL DELIVERABLES

### 1. PDF Report
Convert final_submission_v2.html to PDF:
- Print from browser
- Or render: `output_format = 'pdf_document'`

### 2. Code File
Submit: `final_submission_v2.Rmd`

### 3. README
Include:
- Project overview
- How to reproduce analysis
- Dependencies (R packages)
- Key findings summary

## SUBMISSION CHECKLIST

- [ ] PDF report with all three sections (Analysis, Results, Discussion)
- [ ] R Markdown code file (.Rmd)
- [ ] README file (.md or .txt)
- [ ] All models use weights
- [ ] No cat() to console (markdown only)
- [ ] Answers clearly labeled for Q1 and Q2
- [ ] Language appropriate for non-experts
- [ ] Visualizations professional quality
- [ ] Reproducible (set.seed(1818))

## IMPORTANT NOTES

**Model Requirements:**
- Use statistical models covered in class (NB, linear, GAM)
- No neural networks or advanced ML
- Justify all model choices
- Compare multiple models

**Weighting:**
- Calculate: `weight = 1/σ²`
- Group by: `campaign_variant` and `email_signup`
- Apply to ALL models (NB, GAM, lm, aov, emmeans)

**Output:**
- NO cat() for console output
- Use markdown formatting
- Professional HTML/PDF
- Non-technical language

**Audience:**
- Write for non-data experts
- Business stakeholders at Company X
- Clear, actionable recommendations
- Avoid excessive statistical jargon

Execute this workflow and deliver a complete, publication-ready analysis suitable for CPSC 540 Homework 1 submission.
```

---

## Quick Start Example

Here's exactly what to paste to get started:

```
I need you to complete CPSC 540 Homework 1.

DATA: Download from https://raw.githubusercontent.com/chelseaparlett/CPSC540ParlettPelleriti/refs/heads/main/Data/marketingcampaign.csv

This is marketing data (5,000 customers) for Company X wellness brand testing 3 campaign variants.

QUESTIONS:
1. What kinds of people make MORE purchases? (analyze past_purchases count data)
2. Which campaign variant should Company X use? (analyze campaign_spend continuous data)

REQUIREMENTS:
- Calculate weights = 1/σ² grouped by (campaign_variant, email_signup)
- Apply weights to ALL models
- Use Negative Binomial for Q1 (count data with overdispersion)
- Use ANCOVA for Q2 (continuous data, 3 groups)
- NO cat() to console - use markdown formatting only
- Report in 3 sections: Analysis, Results, Discussion
- Write for non-expert audience

DELIVERABLES:
- final_submission_v2.Rmd (code)
- final_submission_v2.html (rendered)
- README.md (documentation)

Create a complete, weighted statistical analysis following CPSC 540 Homework 1 requirements.
```

---

## Variables Reference

**Outcome Variables:**
- Q1: `past_purchases` (count, 0-80 range)
- Q2: `campaign_spend` (continuous dollars)

**Predictors:**
- `age` (continuous)
- `gender` (categorical: Man, Woman, Non-Binary)
- `email_signup` (binary: TRUE/FALSE)
- `campaign_variant` (categorical: A, B, C)
- `ad_source` (categorical: TikTok, Instagram, Facebook, Google Search)
- Cross-use: `past_purchases` in Q2, `campaign_spend` in Q1

**Weights Grouping:**
- `campaign_variant` (A, B, C)
- `email_signup` (TRUE, FALSE)
- Creates 6 groups total

---

## Expected Results Format

### Question 1 Answer:
"Customers who make more purchases share these characteristics:
1. **Email enrollment** - Subscribers make X% more purchases (IRR=Y)
2. **Spending behavior** - Higher campaign spending correlates with purchase history
3. **Age** - [Positive/Negative] relationship with purchases

**Recommendation:** Prioritize email signup conversion to increase customer lifetime value."

### Question 2 Answer:
"Company X should use Campaign Variant [B/C]:
- **Revenue advantage:** $X per customer vs control (Y% improvement)
- **Statistical confidence:** p < 0.05, Cohen's d = Z
- **Projected impact:** $XXX,XXX annually across 5,000 customers

**Recommendation:** Deploy Variant [X] as primary campaign strategy."

---

**Created:** October 13, 2025 for CPSC 540 HW1
**Status:** Ready to Use ✅
