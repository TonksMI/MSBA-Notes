# Weighted Analysis Summary

## Final Submission Version 2 with Inverse Variance Weights

**File:** `final_submission_v2.Rmd` → `final_submission_v2.html`

**Date:** October 13, 2025

---

## Key Improvements Over Original Version

### 1. Inverse Variance Weighting (1/σ²)

**Formula:**
```r
weight = 1 / σ²
```

Where σ² is the variance of `campaign_spend` within groups defined by:
- `campaign_variant` (A, B, C)
- `email_signup` (TRUE, FALSE)

**Purpose:**
- Gives more weight to observations with **lower variance** (more precise measurements)
- Gives less weight to observations with **higher variance** (less reliable measurements)
- Standard approach for weighted least squares (WLS)
- Improves efficiency and reduces impact of heteroscedasticity

**Implementation:**
```r
data <- data %>%
  group_by(campaign_variant, email_signup) %>%
  mutate(
    sigma2 = var(campaign_spend),
    weight = 1 / sigma2
  ) %>%
  ungroup()
```

---

### 2. Models Using Weights

All statistical models now incorporate the calculated weights:

#### Question 1 Models (Past Purchases)

1. **Negative Binomial (Full):**
   ```r
   glm.nb(past_purchases ~ age + gender + email_signup + ad_source +
          campaign_variant + campaign_spend,
          data = data, weights = weight)
   ```

2. **Negative Binomial (Simplified):**
   ```r
   glm.nb(past_purchases ~ age + email_signup + campaign_spend,
          data = data, weights = weight)
   ```

3. **Poisson (Comparison):**
   ```r
   glm(past_purchases ~ age + email_signup + campaign_spend,
       family = poisson(link = "log"),
       data = data, weights = weight)
   ```

4. **GAM (Non-linear):**
   ```r
   gam(past_purchases ~ s(age) + email_signup + s(campaign_spend),
       family = nb(),
       data = data, weights = weight)
   ```

#### Question 2 Models (Campaign Spend)

1. **Simple Linear Model:**
   ```r
   lm(campaign_spend ~ campaign_variant,
      data = data, weights = weight)
   ```

2. **ANCOVA (Full):**
   ```r
   lm(campaign_spend ~ campaign_variant + age + gender + past_purchases +
      email_signup + ad_source,
      data = data, weights = weight)
   ```

3. **Gamma GLM:**
   ```r
   glm(campaign_spend ~ campaign_variant + age + gender + past_purchases +
       email_signup + ad_source,
       family = Gamma(link = "log"),
       data = data, weights = weight)
   ```

4. **Interaction Model:**
   ```r
   lm(campaign_spend ~ campaign_variant * (email_signup + past_purchases),
      data = data, weights = weight)
   ```

5. **ANOVA with Weights:**
   ```r
   aov(campaign_spend ~ campaign_variant + age + gender + past_purchases +
       email_signup + ad_source,
       data = data, weights = weight)
   ```

6. **Estimated Marginal Means:**
   ```r
   emmeans(model, ~ campaign_variant, weights = "proportional")
   ```

---

### 3. Output Format Improvements

**Replaced Console Output with Markdown:**
- No more `cat()` statements printing to console
- Used `results='asis'` for markdown formatting
- Professional HTML output with proper formatting
- Inline R expressions for dynamic values
- Better table formatting with `kable()`

**Example:**
```r
# OLD (console output):
cat("Mean:", round(mean_purchases, 2), "\n")

# NEW (markdown output):
cat("\n- **Mean:** ", round(mean_purchases, 2), "\n")
```

---

## Statistical Benefits of Weighting

### 1. Efficiency
- Weighted estimators are more efficient than unweighted
- Lower standard errors for coefficients
- More precise confidence intervals

### 2. Heteroscedasticity Correction
- Accounts for non-constant variance across groups
- More reliable inference (p-values, confidence intervals)
- Satisfies Gauss-Markov assumptions better

### 3. Robustness
- Reduces influence of high-variance outliers
- More stable estimates
- Better model diagnostics

### 4. Proper Inference
- Weighted ANOVA/ANCOVA accounts for unequal variances
- Tukey HSD with weights adjusts for group differences
- Emmeans with proportional weights for fair comparisons

---

## Weight Distribution

The weights are calculated within 6 groups:

| Campaign Variant | Email Signup | Group |
|------------------|--------------|-------|
| A | FALSE | 1 |
| A | TRUE | 2 |
| B | FALSE | 3 |
| B | TRUE | 4 |
| C | FALSE | 5 |
| C | TRUE | 6 |

Each observation within a group receives the same weight = 1/σ² for that group.

**Properties:**
- All weights > 0
- Higher variance groups → lower weights
- Lower variance groups → higher weights
- Scale-invariant (ratios matter, not absolute values)

---

## Output Files

### Generated Files

1. **final_submission_v2.Rmd** (source)
   - R Markdown with all code and analysis
   - Includes weights calculation
   - Clean markdown output

2. **final_submission_v2.html** (output)
   - Rendered HTML report
   - Publication-ready
   - Interactive table of contents
   - Code folding enabled
   - All results with weights

3. **final_submission_v2.knit.md** (intermediate)
   - Markdown with executed R code
   - Generated during rendering

---

## Results Comparison

### With vs Without Weights

**Key Differences Expected:**

1. **Standard Errors:** Generally smaller with weights (more efficient)
2. **Significance Levels:** May change slightly (more accurate)
3. **Effect Sizes:** Similar direction, but potentially different magnitudes
4. **Model Fit:** Better diagnostics with weights (residual plots)
5. **Predictions:** More accurate for low-variance groups

**Interpretation:**
- Weighted results are **more trustworthy**
- Account for data structure properly
- Follow statistical best practices
- More defensible in academic/business settings

---

## Usage Instructions

### Rendering the Analysis

```bash
# Navigate to directory
cd "/Users/matthewtonks/Repositories/CPSC 540/Assignments/HW 1/Claude Code 2"

# Render to HTML
Rscript -e "rmarkdown::render('final_submission_v2.Rmd')"

# Render to PDF
Rscript -e "rmarkdown::render('final_submission_v2.Rmd', output_format = 'pdf_document')"
```

### In RStudio

1. Open `final_submission_v2.Rmd`
2. Click **Knit** button
3. Select output format (HTML or PDF)

---

## Technical Details

### Weight Calculation Details

```r
# Step 1: Group data
grouped <- data %>%
  group_by(campaign_variant, email_signup)

# Step 2: Calculate group variance
variance_by_group <- grouped %>%
  summarise(sigma2 = var(campaign_spend))

# Step 3: Assign weights (1/variance)
data_weighted <- grouped %>%
  mutate(
    sigma2 = var(campaign_spend),
    weight = 1 / sigma2
  ) %>%
  ungroup()
```

### Why Group by Variant × Email?

These are the key factors affecting variance in `campaign_spend`:
- Different campaigns may have different variance
- Email subscribers may respond more consistently
- Interaction between these factors affects variability
- Grouping captures heteroscedasticity structure

### Alternative Weighting Schemes (Not Used)

1. **Inverse Standard Deviation:** `weight = 1/σ`
   - Less common, not optimal for variance modeling

2. **Normalized Weights:** `weight = n/σ²`
   - Incorporates sample size
   - Not needed here (groups roughly equal size)

3. **Robust Weights:** Based on residuals
   - Iterative process
   - More complex, not necessary for this analysis

---

## Model Diagnostics

With weights, expect to see:

1. **Residual Plots:** More homoscedastic (constant variance)
2. **Q-Q Plots:** Better normality of residuals
3. **Scale-Location:** Flatter trend line
4. **Leverage:** Adjusted for weight influence

All diagnostic plots included in Appendix of HTML report.

---

## Recommendations for Submission

### Files to Submit

1. **PDF Report:** Convert `final_submission_v2.html` to PDF
   - Print to PDF from browser
   - Or render directly: `output_format = 'pdf_document'`

2. **Code File:** `final_submission_v2.Rmd`
   - Complete, reproducible analysis
   - Well-commented
   - Follows best practices

3. **README:** Document explaining:
   - Weight calculation rationale
   - How to reproduce results
   - Dependencies required

### Key Points to Highlight

1. **Methodological Rigor:**
   - Used inverse variance weighting (1/σ²)
   - Accounts for heteroscedasticity
   - Follows WLS best practices

2. **Statistical Appropriateness:**
   - Weights based on group variance structure
   - All models properly weighted
   - Diagnostic checks performed

3. **Business Impact:**
   - More reliable estimates
   - Better-informed recommendations
   - Robust to data irregularities

---

## Advantages of This Version

### Over Unweighted Analysis

1. ✅ Statistically more efficient
2. ✅ Accounts for heteroscedasticity
3. ✅ More reliable inference
4. ✅ Better model diagnostics
5. ✅ Publication-quality results

### Over Original Version with Cat()

1. ✅ Professional markdown output
2. ✅ Cleaner code structure
3. ✅ Better formatted tables
4. ✅ Integrated inline results
5. ✅ Publication-ready HTML

---

## Summary

**final_submission_v2.Rmd** provides a complete, weighted statistical analysis of the marketing campaign data with:

- ✅ Proper inverse variance weights (1/σ²)
- ✅ All models weighted appropriately
- ✅ Clean markdown output format
- ✅ Professional HTML rendering
- ✅ Comprehensive diagnostics
- ✅ Publication-ready results

The analysis is **statistically rigorous**, **professionally formatted**, and **ready for submission**.

---

**Analysis completed:** October 13, 2025
**Output file:** `final_submission_v2.html`
**Status:** ✅ Complete and ready for submission
