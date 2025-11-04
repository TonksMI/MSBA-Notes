# Complete Homework Analysis Workflow Prompt

Use this prompt to replicate the entire homework analysis process from scratch.

---

## Master Prompt

```
I need you to complete a statistical analysis homework assignment with the following structure and requirements:

## TASK OVERVIEW
Create a complete homework analysis in a new folder with:
1. **analysis.Rmd** - Full R Markdown analysis with all models and code
2. **insights.md** - Technical notes, decisions, and refinements
3. **final_submission_v2.Rmd** - Weighted analysis with clean markdown output (no cat())
4. **README.md** - Complete project documentation

## DATA SOURCE
Download data from: [INSERT DATA URL]
Dataset contains: [DESCRIBE DATASET - e.g., "5,000 customers with demographics, purchase history, campaign variant assignments"]

## HOMEWORK QUESTIONS
[INSERT YOUR SPECIFIC QUESTIONS - e.g.:]
1. Question 1: What kinds of people make MORE purchases?
2. Question 2: Which campaign variant should Company X use in the future?

## REQUIRED ANALYSIS STRUCTURE

### Part 1: Exploratory Analysis (analysis.Rmd)
- Load and prepare data
- Comprehensive EDA with visualizations
- Check distributions and relationships
- Test multiple model types for each question
- Compare models using AIC/BIC
- Full diagnostics and validation

### Part 2: Technical Documentation (insights.md)
- Document all modeling decisions
- Explain model selection rationale
- Note issues encountered and solutions
- Track refinements and iterations
- Include communication strategy for non-technical audience

### Part 3: Final Submission (final_submission_v2.Rmd)
**CRITICAL REQUIREMENTS:**
- Calculate weights: `weight = 1/σ²` based on group variance
- Apply weights to ALL models (NB, GAM, linear, ANOVA, emmeans)
- Replace ALL cat() console output with markdown formatting
- Use `results='asis'` for markdown output chunks
- Create publication-ready HTML report

### Part 4: Project Documentation (README.md)
- Project overview and structure
- Dependencies and installation instructions
- How to run the analysis
- Model selection justification
- Key findings summary

## STATISTICAL REQUIREMENTS

### For Count Data (e.g., purchase frequency):
- Check for overdispersion (variance vs mean)
- Use Negative Binomial if overdispersed
- Include weights parameter: `glm.nb(..., weights = weight)`
- Fit comparison models: NB full, NB simple, Poisson, GAM
- Report Incident Rate Ratios (IRR) with interpretation

### For Continuous Data (e.g., spending):
- Use ANCOVA with covariates
- Include weights parameter: `lm(..., weights = weight)`
- Conduct Tukey HSD pairwise comparisons
- Calculate Cohen's d effect sizes (pairwise for 3+ groups)
- Report adjusted marginal means with emmeans

### Weight Calculation:
```r
data <- data %>%
  group_by([KEY_GROUPING_VARIABLES]) %>%
  mutate(
    sigma2 = var([OUTCOME_VARIABLE]),
    weight = 1 / sigma2
  ) %>%
  ungroup()
```

### Models Must Include Weights:
- `glm.nb(..., weights = weight)`
- `gam(..., weights = weight)`
- `lm(..., weights = weight)`
- `aov(..., weights = weight)`
- `emmeans(..., weights = "proportional")`

## OUTPUT FORMATTING REQUIREMENTS

### DO NOT Use cat() for Console Output
**WRONG:**
```r
cat("Mean:", mean(x), "\n")
cat("The best model is:", model_name, "\n")
```

**CORRECT - Use Markdown:**
```r
# In chunk with results='asis':
cat("\n#### Key Finding\n\n")
cat(sprintf("- **Mean:** %.2f\n", mean(x)))
cat(sprintf("- **Best model:** %s\n\n", model_name))
```

### Use These Patterns:

**For Statistics:**
```r
tibble(
  Statistic = c("Mean", "Variance", "Ratio"),
  Value = c(mean_val, var_val, ratio)
) %>% kable(caption = "Summary Statistics")
```

**For Model Results:**
```r
tidy(model, conf.int = TRUE) %>%
  kable(digits = 3, caption = "Model Coefficients")
```

**For Text Output:**
```r
# chunk header: results='asis'
cat("\n#### Section Title\n\n")
cat(sprintf("- **Finding:** %.1f%% increase\n", percent))
cat("- **Interpretation:** This means...\n\n")
```

## ERROR HANDLING

### Common Issues to Prevent:

1. **select() namespace conflicts:**
   - Remove problematic select() calls
   - Use indexed access: `variable[1]` instead of `variable`

2. **Gamma GLM requires positive values:**
   - Check: `min(data$outcome) > 0`
   - Conditionally fit Gamma only if all positive

3. **Cohen's d requires 2 levels:**
   - Calculate pairwise for 3+ groups
   - Filter data for each comparison

4. **Tukey HSD with weights:**
   - Use aov() with weights parameter
   - Apply TukeyHSD to weighted model

## DELIVERABLE CHECKLIST

Create these files in order:

- [ ] **Folder structure** created (e.g., "Claude Code 2")
- [ ] **Data downloaded** and verified
- [ ] **analysis.Rmd** - Complete exploratory analysis
  - [ ] Multiple models tested
  - [ ] Diagnostics included
  - [ ] Comparisons made (AIC/BIC)
- [ ] **insights.md** - Technical documentation
  - [ ] Model selection rationale
  - [ ] Issues and solutions
  - [ ] Refinements documented
- [ ] **final_submission_v2.Rmd** - Publication version
  - [ ] Weights calculated (1/σ²)
  - [ ] All models weighted
  - [ ] No cat() to console
  - [ ] Markdown output only
  - [ ] Professional formatting
- [ ] **README.md** - Project overview
  - [ ] Installation instructions
  - [ ] How to run
  - [ ] Model justifications
- [ ] **Rendered HTML** - Run the analysis
  - [ ] No errors during rendering
  - [ ] All results displayed correctly
  - [ ] Weights applied properly

## REPORT SECTIONS (for final_submission_v2.Rmd)

### Required Sections:

1. **Analysis Section**
   - How you analyzed the data (methods)
   - Why you chose each model
   - Justification for approaches

2. **Results Section**
   - Clear answer to each question
   - What the analyses revealed
   - Statistical evidence (p-values, CIs, effect sizes)

3. **Discussion Section**
   - Potential impacts/applications
   - What you'd do differently next time
   - Limitations and caveats

## QUALITY STANDARDS

### Code Quality:
- All chunks have descriptive names
- Code is well-commented
- Reproducible (set seed)
- Uses tidyverse style
- No hardcoded values

### Statistical Quality:
- Appropriate models for data types
- Assumptions checked
- Diagnostics included
- Multiple models compared
- Weights properly applied

### Presentation Quality:
- Professional HTML output
- Clear visualizations
- Formatted tables (kable)
- Logical flow
- Executive-friendly language

### Documentation Quality:
- Complete README
- Technical insights documented
- Decisions explained
- Replication instructions clear

## EXAMPLE WORKFLOW

```r
# 1. Create folder and download data
mkdir "Project Folder"
curl -o data.csv [URL]

# 2. Create analysis.Rmd
# - Load data
# - EDA
# - Multiple models
# - Diagnostics

# 3. Create insights.md
# - Document decisions
# - Note issues
# - Record solutions

# 4. Create final_submission_v2.Rmd
# - Calculate weights
# - Apply to all models
# - Clean markdown output
# - Professional formatting

# 5. Render
rmarkdown::render("final_submission_v2.Rmd")

# 6. Create README.md
# - Document everything
```

## FINAL OUTPUT

When complete, I should have:

1. **Clean folder** with all files organized
2. **Reproducible analysis** that runs without errors
3. **Professional HTML report** ready for submission
4. **Complete documentation** for replication
5. **Weighted models** using 1/σ² throughout
6. **Markdown output** with no console cat() statements

## IMPORTANT NOTES

- **DO use:** weights = 1/σ² in all models
- **DO use:** markdown formatting for output
- **DO use:** results='asis' for text chunks
- **DO use:** kable() for all tables
- **DO NOT use:** cat() for console output
- **DO NOT use:** select() if it causes namespace conflicts
- **DO NOT use:** unweighted models
- **DO check:** All models have weights parameter
- **DO verify:** HTML renders without errors
- **DO ensure:** All cat() statements produce markdown syntax

Execute this entire workflow and deliver a complete, publication-ready analysis.
```

---

## Usage Instructions

### To Use This Prompt:

1. **Copy the entire prompt** from the code block above
2. **Fill in the placeholders** with your specific information:
   - Data URL
   - Dataset description
   - Specific homework questions
   - Key grouping variables for weights
   - Outcome variables

3. **Provide to Claude** with any additional context:
   - Course requirements
   - Specific rubric items
   - Additional constraints
   - Style preferences

### Example Filled Prompt:

```
I need you to complete a statistical analysis homework assignment with the following structure and requirements:

## TASK OVERVIEW
Create a complete homework analysis in a new folder called "Homework Analysis" with:
1. analysis.Rmd - Full R Markdown analysis with all models and code
2. insights.md - Technical notes, decisions, and refinements
3. final_submission_v2.Rmd - Weighted analysis with clean markdown output (no cat())
4. README.md - Complete project documentation

## DATA SOURCE
Download data from: https://raw.githubusercontent.com/example/data.csv
Dataset contains: 5,000 customers with demographics (age, gender), purchase history,
campaign variant assignments (A, B, C), ad source, email signup status, and campaign spending.

## HOMEWORK QUESTIONS
1. Question 1: What kinds of people make MORE purchases?
   - Analyze past_purchases (count variable)
   - Identify customer characteristics associated with higher frequency

2. Question 2: Which campaign variant should Company X use in the future?
   - Compare variants A, B, C
   - Outcome: campaign_spend (continuous, dollars)
   - Recommend optimal strategy

[... continue with rest of prompt ...]

Weight calculation should group by: campaign_variant and email_signup
Outcome variables: past_purchases (Q1), campaign_spend (Q2)
```

---

## Quick Reference Checklist

When using this prompt, ensure:

- [ ] Data URL provided
- [ ] Questions clearly stated
- [ ] Grouping variables for weights specified
- [ ] Outcome variables identified
- [ ] Any special requirements noted
- [ ] Expected deliverables listed
- [ ] Timeline/deadline specified (if applicable)

---

## Customization Options

### For Different Data Types:

**Binary Outcome:**
```
Use logistic regression with weights:
glm(outcome ~ predictors, family = binomial, weights = weight)
```

**Time-to-Event:**
```
Use Cox proportional hazards with weights:
coxph(Surv(time, event) ~ predictors, weights = weight)
```

**Ordinal Outcome:**
```
Use ordinal regression with weights:
polr(outcome ~ predictors, weights = weight)
```

### For Different Assignment Types:

**For Kaggle Competition:**
- Add train/test split requirements
- Include cross-validation
- Emphasize prediction accuracy
- Add model ensembling

**For Research Project:**
- Add literature review section
- Include hypothesis testing framework
- Emphasize causal inference
- Add sensitivity analyses

**For Business Case:**
- Add ROI calculations
- Include executive summary
- Emphasize actionable recommendations
- Add cost-benefit analysis

---

## Troubleshooting Guide

If you encounter issues using this prompt:

### Issue: Models won't fit with weights
**Solution:** Check for zero or infinite variances. Add small constant:
```r
weight = 1 / (sigma2 + 0.001)
```

### Issue: Rendering fails
**Solution:** Check for:
- Missing libraries
- Syntax errors in markdown chunks
- Undefined variables in inline R

### Issue: Output looks wrong
**Solution:** Verify:
- `results='asis'` in chunk header
- Proper markdown syntax in cat()
- Table formatting with kable()

---

## Version History

- **v1.0** - Initial prompt creation
- **v2.0** - Added weights requirement (1/σ²)
- **v2.1** - Replaced cat() with markdown output
- **v2.2** - Added error handling section

---

**Created:** October 13, 2025
**Last Updated:** October 13, 2025
**Status:** Production Ready ✅
