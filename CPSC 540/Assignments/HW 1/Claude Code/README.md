# CPSC 540 Homework 1 - Marketing Campaign Analysis

## Assignment Overview

This assignment analyzes synthetic marketing data for Company X (wellness brand) to answer two key business questions:

1. **What kinds of people make MORE purchases?**
2. **Which campaign variant should Company X use in the future and why?**

---

## Quick Start

### View Results Immediately
- **Key Findings**: Open `KEY_FINDINGS.md` for a complete summary of results and recommendations
- **Visualizations**: Browse the `plots/` directory (14 plots generated)
- **Step-by-Step Guide**: See `Step_by_Step_Guide.md` for detailed methodology

### Run the Analysis

**Option 1: Quick Script (Recommended)**
```bash
Rscript generate_plots.R
```
This generates all plots and prints statistical results to console (~30 seconds).

**Option 2: Full R Markdown**
```bash
Rscript -e "rmarkdown::render('HW1_Analysis.Rmd')"
```
This creates an HTML report with all analysis, code, and visualizations (~2 minutes).

**Option 3: Interactive Analysis**
Open `HW1_Analysis.Rmd` in RStudio and run interactively.

---

## File Structure

```
HW 1/
├── README.md                          # This file
├── hw.md                              # Original assignment description
├── marketingcampaign.csv             # Dataset (5,000 customers, 7 variables)
│
├── HW1_Analysis.Rmd                  # Complete R Markdown analysis
├── generate_plots.R                  # Standalone R script for plots
├── Step_by_Step_Guide.md             # Detailed methodology guide
├── KEY_FINDINGS.md                   # Executive summary of results
│
└── plots/                            # Generated visualizations (14 plots)
    ├── 01_past_purchases_distribution.png
    ├── 02_campaign_spend_distribution.png
    ├── 03_age_distribution.png
    ├── 04_purchases_by_gender.png
    ├── 05_purchases_by_age.png
    ├── 06_purchases_by_email.png
    ├── 07_purchases_by_ad_source.png
    ├── 08_spend_by_variant.png
    ├── 09_mean_spend_by_variant.png
    ├── 10_spend_by_variant_and_gender.png
    ├── 11_spend_by_variant_and_email.png
    ├── 12_q1_coefficient_plot.png
    ├── 13_q2_coefficient_plot.png
    └── 14_predicted_spend_by_variant.png
```

---

## Dataset Description

**Source**: Synthetic data for Company X wellness brand marketing campaign
**Size**: 5,000 customers
**Campaign**: "Be Well" campaign for energy + protein powder

### Variables

| Variable | Type | Description |
|----------|------|-------------|
| `age` | Numeric | Customer age (18-80) |
| `gender` | Categorical | Self-identified gender (Man, Woman, Non-Binary, Other) |
| `past_purchases` | Count | Number of purchases in past 2 years (0-100) |
| `campaign_variant` | Categorical | Campaign version assigned (A, B, or C) |
| `ad_source` | Categorical | Where customer saw ad (Facebook, Google Search, Instagram, TikTok) |
| `email_signup` | Binary | Signed up for promotional emails (TRUE/FALSE) |
| `campaign_spend` | Numeric | Amount spent on campaign items ($0-$739) |

### Campaign Variants
- **A**: Original "business as usual" campaign (baseline)
- **B**: New "Be Well" campaign focusing on physical health
- **C**: "Be Well" campaign variant focusing on mental health

---

## Key Results

### Question 1: Customer Purchase Patterns

**Model**: Negative Binomial Regression (handles overdispersed count data)

**High-purchasing customers are:**
- ✓ **Women** (31% more purchases than men)
- ✓ **Non-Binary** (41% more purchases than men)
- ✓ **Younger** (2% fewer purchases per year of age)

**Business Impact**: Target women and non-binary customers aged 18-35 for highest engagement.

### Question 2: Optimal Campaign Variant

**Model**: Linear Regression (R² = 0.79)

**Recommendation**: **Use Variant B (Physical Health Focus)**

**Performance**:
- Variant B: $217/customer
- Variant C: $202/customer
- Variant A: $148/customer

**Impact**: Variant B generates **$71 more per customer** than the original campaign (48% increase, p < 0.001).

**Annual Revenue Potential**: +$3.45M more than continuing with Variant A (based on 50K customers/year).

---

## Statistical Methods

### Question 1: Negative Binomial Regression
- **Why?** Past purchases is count data with overdispersion (variance >> mean)
- **Link function**: Log link
- **Interpretation**: Exponentiated coefficients = Incident Rate Ratios (IRR)
- **Model fit**: AIC = 42,749; Theta = 5.67

### Question 2: Linear Regression
- **Why?** Campaign spend is continuous, approximately normal
- **Controls**: Age, gender, past purchases, email signup, ad source
- **Model fit**: R² = 0.787, F(11, 4988) = 1,672, p < 0.001
- **Diagnostics**: All assumptions satisfied (normality, homoscedasticity, no multicollinearity)

---

## Requirements

### R Packages
```r
install.packages(c(
  "tidyverse",    # Data manipulation and ggplot2
  "broom",        # Tidy model outputs
  "MASS",         # Negative binomial regression (glm.nb)
  "car",          # VIF and diagnostic tests
  "GGally",       # Pairwise correlation plots
  "gridExtra",    # Arrange multiple plots
  "scales",       # Formatting functions
  "viridis"       # Color palettes
))
```

### R Version
Tested on R version 4.0+

---

## Assignment Deliverables

### For Submission:
1. **PDF Report** - Knit from `HW1_Analysis.Rmd` or write separately
2. **Code File** - `HW1_Analysis.Rmd` or `generate_plots.R`
3. **README** - This file (explains project structure)

### Report Sections Required:
- **Analysis**: Model selection and methodology
- **Results**: Clear answers to Questions 1 and 2 with supporting evidence
- **Discussion**: Business implications and recommendations

---

## Reproducibility

All analyses are fully reproducible:
- **Seed**: `set.seed(1818)` for consistency
- **Data**: Included in repository (`marketingcampaign.csv`)
- **Code**: Fully commented and organized
- **Environment**: Session info saved in R Markdown output

To reproduce exactly:
1. Ensure all required packages are installed
2. Run from the `HW 1/` directory
3. Use R 4.0 or higher

---

## Visualizations Guide

### Exploratory Plots (1-3)
Basic distributions to understand the data structure and identify potential issues.

### Question 1 Plots (4-7, 12)
- **Boxplots**: Show distributions of purchases across categorical variables
- **Scatterplot**: Relationship between age and purchases
- **Coefficient plot**: Statistical effects with confidence intervals

### Question 2 Plots (8-11, 13-14)
- **Boxplots**: Campaign spending across variants
- **Bar charts**: Mean comparisons with error bars
- **Interaction plots**: How effects vary by customer characteristics
- **Coefficient plots**: Statistical effects from regression

---

## Interpretation Tips

### For Question 1 (Negative Binomial)
- **IRR > 1**: Variable increases purchase count
- **IRR < 1**: Variable decreases purchase count
- **IRR = 1.31**: 31% increase in expected purchases
- **p < 0.05**: Statistically significant effect

### For Question 2 (Linear Regression)
- **Positive coefficient**: Increases campaign spend
- **Negative coefficient**: Decreases campaign spend
- **campaign_variantB = $70.89**: B generates $70.89 MORE than baseline (A)
- **95% CI not crossing 0**: Significant effect

---

## Additional Resources

### Files for Learning:
- `Step_by_Step_Guide.md` - Complete walkthrough of methodology
- `KEY_FINDINGS.md` - Executive summary for non-technical audiences
- `HW1_Analysis.Rmd` - Annotated code with explanations

### Concepts Covered:
- Generalized Linear Models (GLMs)
- Negative Binomial Regression
- Model selection (AIC, diagnostics)
- Effect size interpretation
- Statistical vs. practical significance
- Business impact analysis

---

## Contact

For questions about the analysis methodology, consult:
1. `Step_by_Step_Guide.md` (comprehensive methodology)
2. Course lecture notes on GLMs
3. Office hours or course forums

---

## License

This is academic coursework for CPSC 540. The data is synthetic and provided for educational purposes.

---

**Last Updated**: September 29, 2025
**Course**: CPSC 540 - Statistical Machine Learning I
**Assignment**: Homework 1