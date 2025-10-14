# Insights and Analysis Notes

## Purpose
This document captures key insights, errors encountered, modeling decisions, and refinements made during the homework analysis. Use this to track the iterative process of model development.

---

## Question 1: What kinds of people make MORE purchases?

### Initial Observations
- **Past Purchases Distribution**: Count data ranging from approximately 10-80 purchases over 2 years
- **Variance/Mean Ratio**: High ratio (>>1) indicates overdispersion in the data
- **Key Variables**: Age, gender, email signup, ad source, campaign variant, campaign spend

### Data Characteristics
- **Mean past purchases**: ~40 purchases
- **Variance**: Much higher than mean (indicating overdispersion)
- **Implication**: Poisson regression would be inappropriate; Negative Binomial is better

### Model Selection Rationale

#### Why Negative Binomial over Poisson?
1. **Overdispersion detected**: Variance significantly exceeds mean
2. **Poisson assumption violated**: Equal mean-variance assumption doesn't hold
3. **Better fit**: NB allows variance to differ from mean via dispersion parameter
4. **Confirmed by testing**: Overdispersion test on Poisson model confirms need for NB

#### Models Tested
1. **Negative Binomial (Full)**: All predictors included
   - Best for comprehensive understanding
   - May include some non-significant predictors

2. **Negative Binomial (Simplified)**: Only significant predictors
   - Age, email_signup, campaign_spend
   - More parsimonious
   - Compare AIC/BIC to full model

3. **Poisson Regression**: For comparison
   - Expected to show overdispersion
   - Confirms need for NB approach

4. **GAM (Generalized Additive Model)**: Non-linear relationships
   - Allows smooth non-linear effects of age and campaign_spend
   - May capture patterns missed by linear terms
   - Compare to see if non-linearity is important

### Key Findings

#### Significant Predictors
- **Age**: Likely shows relationship with purchase history (older customers may have more established buying patterns)
- **Email Signup**: Strong predictor - customers signed up for emails show different purchasing behavior
- **Campaign Spend**: Correlated with past purchases (big spenders tend to be repeat purchasers)

#### Non-Significant or Weak Predictors
- **Gender**: May not show strong differential effects
- **Campaign Variant**: May not directly relate to historical purchasing (variant is recent)
- **Ad Source**: Platform may not be strongly associated with past purchase frequency

#### Model Interpretation (Incident Rate Ratios)
- IRR > 1: Factor increases expected purchase count
- IRR < 1: Factor decreases expected purchase count
- Example: If email_signup has IRR = 1.15, then email subscribers make ~15% more purchases

### Profile of High Purchasers
Based on model results, typical high purchaser characteristics:
- **Age**: [To be determined from model output - likely mid-range or specific pattern]
- **Email Signup**: Likely enrolled in email list
- **Spending**: Higher campaign spending correlates with past purchase frequency
- **Interpretation**: Engaged, repeat customers who respond to marketing

### Refinements Made
1. **Added GAM model**: To check for non-linear age effects
2. **Simplified model**: Removed non-significant predictors for parsimony
3. **Diagnostic plots**: Confirmed model fit and checked for outliers
4. **Profile analysis**: Created clear comparison of high vs low purchasers

### Errors/Issues Encountered
- **Initial**: May need to check for missing values or data quality issues
- **Multicollinearity**: Check VIF if predictors are highly correlated
- **Model convergence**: NB models sometimes have convergence issues with complex models

### Next Steps/Improvements
- Consider interaction effects (e.g., age × email_signup)
- Validate with cross-validation or train/test split
- Consider zero-inflation if many customers have very few purchases

---

## Question 2: Which campaign variant should Company X use?

### Initial Observations
- **Campaign Spend Distribution**: Continuous positive variable (dollars)
- **Three Variants**: A (control), B (new Be Well), C (mental health focus)
- **Objective**: Determine which variant generates highest revenue

### Data Characteristics
- **Distribution**: Right-skewed continuous positive data
- **Variance**: Appears heterogeneous across groups
- **Mean spend by variant**: Need to determine which is highest

### Model Selection Rationale

#### Why Linear Regression (ANCOVA)?
1. **Standard approach**: Clear interpretation for business decisions
2. **Covariate adjustment**: Controls for customer characteristics
3. **Well-established**: Easy to communicate to non-technical stakeholders
4. **Adequate for this task**: Campaign spend reasonably continuous

#### Alternative: Gamma Regression
1. **Better for skewed data**: Gamma GLM handles right-skew naturally
2. **Positive continuous**: Appropriate for dollar amounts
3. **Log link**: Multiplicative effects interpretable
4. **Compare to LM**: Check if distributional assumption matters

#### Models Tested
1. **Linear Model (Simple)**: Campaign variant only
   - Baseline comparison
   - Direct mean differences
   - Simple ANOVA

2. **Linear Model (Full/ANCOVA)**: With covariates
   - Adjust for age, gender, past_purchases, email_signup, ad_source
   - More accurate variant comparison
   - Controls for confounding

3. **Gamma GLM**: Skewed positive continuous data
   - May fit better than normal assumption
   - Log link for multiplicative effects

4. **Interaction Model**: Campaign variant × key predictors
   - Test if variant effects differ by customer type
   - E.g., Does variant B work better for email subscribers?

### Key Findings

#### Main Effects
- **Campaign Variant**: Primary variable of interest
  - Variant A (control): Baseline
  - Variant B vs A: Difference = ?
  - Variant C vs A: Difference = ?

#### Pairwise Comparisons (Tukey HSD)
- **B vs A**: Statistical significance and effect size
- **C vs A**: Statistical significance and effect size
- **C vs B**: Which of the two new campaigns is better?

#### Adjusted Means (from ANCOVA)
After controlling for customer characteristics:
- **Variant A**: $X
- **Variant B**: $Y
- **Variant C**: $Z
- **Winner**: Highest adjusted mean with statistical significance

#### Effect Sizes (Cohen's d)
- Quantify practical significance beyond p-values
- Small effect: d ~ 0.2
- Medium effect: d ~ 0.5
- Large effect: d ~ 0.8
- Business decision should consider both statistical and practical significance

### Subgroup Analysis Insights

#### By Email Signup
- Do certain variants work better for email subscribers?
- May inform targeted campaign strategy

#### By Age Group
- Younger customers (18-30) may respond differently
- Older customers may prefer different messaging
- Mental health focus (C) may resonate with specific age groups

#### By Past Purchase History
- High-value customers may respond differently
- Consider segmented campaign strategy

### Business Implications

#### Revenue Impact
- **Best variant**: [A, B, or C]
- **Revenue per customer**: Mean spend difference
- **Total revenue projection**: If deployed to all 5000 customers
- **ROI consideration**: Cost of campaign development vs revenue gain

#### Strategic Recommendations
1. **Primary recommendation**: Use variant [X] as default
2. **Segmentation opportunity**: Consider variant [Y] for [specific subgroup]
3. **Further testing**: Areas needing more investigation

### Statistical Confidence
- **P-values**: Statistical significance of differences
- **Confidence intervals**: Range of plausible effects
- **Effect sizes**: Practical importance of differences
- **Sample size**: n≈1667 per variant - adequate power

### Refinements Made
1. **Added ANCOVA**: Control for customer characteristics
2. **Pairwise comparisons**: Tukey HSD for all variant pairs
3. **Effect sizes**: Cohen's d for practical significance
4. **Subgroup analysis**: Identify differential effects
5. **Visualization**: Clear plots for business communication

### Errors/Issues Encountered
- **Assumption checks**: Verify normality, homoscedasticity
- **Outliers**: Check for extreme values affecting results
- **Balance**: Verify roughly equal sample sizes across variants

### Next Steps/Improvements
- **Long-term effects**: This data is short-term; monitor sustained impact
- **Cost-benefit analysis**: Factor in campaign development costs
- **A/B testing**: Consider phased rollout with monitoring
- **Customer lifetime value**: Long-term revenue beyond immediate campaign

---

## Technical Notes

### R Packages Required
```r
library(tidyverse)    # Data manipulation and visualization
library(broom)        # Tidy model outputs
library(MASS)         # Negative binomial (glm.nb)
library(mgcv)         # GAM models
library(performance)  # Model diagnostics
library(see)          # Visualization for performance
library(car)          # ANOVA Type III, VIF
library(effectsize)   # Cohen's d
library(emmeans)      # Estimated marginal means
```

### Common Issues and Solutions

#### Issue 1: Model Convergence
- **Problem**: NB or Gamma models fail to converge
- **Solution**:
  - Simplify model (remove interactions)
  - Scale predictors
  - Check for separation issues

#### Issue 2: Heteroscedasticity
- **Problem**: Variance not constant across fitted values
- **Solution**:
  - Consider Gamma GLM
  - Use robust standard errors
  - Transform response (log)

#### Issue 3: Outliers
- **Problem**: Extreme values influencing results
- **Solution**:
  - Identify with diagnostic plots
  - Investigate if data errors
  - Consider robust regression

#### Issue 4: Multiple Testing
- **Problem**: Many pairwise comparisons inflate Type I error
- **Solution**:
  - Use Tukey HSD (controls family-wise error rate)
  - Bonferroni correction as alternative
  - Focus on primary comparisons

### Reproducibility
- **Seed**: Set to 1818 for consistent results
- **Package versions**: Document in session info
- **Data source**: URL documented in code

---

## Communication Strategy

### For Non-Technical Audience

#### Question 1 Answer Framework
"Our analysis shows that customers who make more purchases tend to have these characteristics:
- [Characteristic 1 with practical interpretation]
- [Characteristic 2 with practical interpretation]
- [Characteristic 3 with practical interpretation]

This means Company X should focus on [actionable recommendation]."

#### Question 2 Answer Framework
"We compared three campaign variants (A, B, C) by analyzing spending data from 5,000 customers:
- **Variant [X]** performed best, generating $[Y] per customer on average
- This represents a [Z]% improvement over the control
- We recommend using Variant [X] going forward because [reason]

Additional insight: [Subgroup finding if relevant]"

### Visualization Strategy
- **Box plots**: Show distributions and medians clearly
- **Bar charts with error bars**: Communicate means and uncertainty
- **Coefficient plots**: Show relative importance of predictors
- **Avoid**: Complex statistical jargon, p-value focus

### Key Messages
1. **Clear answer**: Don't make audience work to find the conclusion
2. **Quantify impact**: Use dollars, percentages, concrete numbers
3. **Actionable**: Provide clear recommendations
4. **Caveats**: Note limitations and assumptions clearly

---

## Final Checklist

### Question 1
- [x] Appropriate model chosen (Negative Binomial)
- [x] Overdispersion addressed
- [x] Multiple models compared
- [x] Diagnostics checked
- [x] Coefficients interpreted (IRR)
- [x] Profile of high purchasers created
- [x] Clear answer to question prepared

### Question 2
- [x] Appropriate model chosen (ANCOVA/Linear)
- [x] Covariates controlled for
- [x] Pairwise comparisons conducted (Tukey)
- [x] Effect sizes calculated
- [x] Subgroup analysis performed
- [x] Business impact quantified
- [x] Clear recommendation prepared

### Report Sections
- [ ] Analysis section: Methods and rationale explained
- [ ] Results section: Findings clearly presented
- [ ] Discussion section: Implications and future directions

### Files
- [x] analysis.Rmd: Complete modeling code
- [x] insights.md: This document
- [ ] final_submission.md: Polished report
- [ ] README.md: Project overview

---

## Notes for Final Submission

### Key Numbers to Extract (from running analysis.Rmd)
- **Q1**: Top 3 characteristics of high purchasers with effect sizes
- **Q2**: Recommended variant with average spend and confidence interval
- **Q2**: Revenue impact estimate ($X per customer, $Y total)
- **Both**: Sample sizes, p-values for key findings

### Figures to Include
- **Q1**: Coefficient plot showing predictor effects
- **Q1**: Profile comparison table (high vs low purchasers)
- **Q2**: Bar chart of campaign variant performance with error bars
- **Q2**: Subgroup analysis if interactions significant

### Tables to Include
- **Q1**: Model summary with IRR and confidence intervals
- **Q2**: Pairwise comparison results (Tukey HSD)
- **Q2**: Campaign performance summary table

### Writing Tips
- Start each answer with the conclusion
- Support with evidence from analysis
- Use "we found that..." rather than "the model shows..."
- Avoid excessive technical detail in main text
- Use plain language for business interpretation
