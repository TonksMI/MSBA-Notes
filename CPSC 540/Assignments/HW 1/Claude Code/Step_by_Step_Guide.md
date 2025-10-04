# CPSC 540 HW1: Step-by-Step Analysis Guide

## Overview
This guide walks you through the complete analysis for Homework 1, answering:
- **Question 1**: What kinds of people make MORE purchases?
- **Question 2**: Which campaign variant should Company X use in the future and why?

---

## Part 1: Data Preparation and Exploration

### Step 1.1: Load Required Packages
```r
library(tidyverse)      # Data manipulation and visualization
library(broom)          # Tidy model outputs
library(MASS)           # For negative binomial regression
library(car)            # For VIF and diagnostic tests
library(GGally)         # For correlation plots
```

**Why?** These packages provide tools for data manipulation, visualization, and statistical modeling using GLMs.

### Step 1.2: Load and Inspect the Data
```r
# Load data
data <- read.csv("marketingcampaign.csv")

# Quick inspection
head(data)        # First 6 rows
str(data)         # Structure of data
summary(data)     # Summary statistics
dim(data)         # Dimensions (rows, columns)
```

**What to look for:**
- Number of observations (should be 5000)
- Variable types (numeric vs categorical)
- Any missing values
- Range of values for each variable

### Step 1.3: Check for Missing Values
```r
# Check missing values
colSums(is.na(data))

# Proportion of missing values
colMeans(is.na(data)) * 100
```

**Interpretation:** If missing values exist, decide on handling strategy (removal, imputation, etc.)

### Step 1.4: Convert Variables to Appropriate Types
```r
# Convert categorical variables to factors
data <- data %>%
  mutate(
    gender = factor(gender),
    campaign_variant = factor(campaign_variant, levels = c("A", "B", "C")),
    ad_source = factor(ad_source),
    email_signup = factor(email_signup)
  )

# Verify conversions
str(data)
```

**Why?** R needs to know which variables are categorical (factors) vs numeric for proper modeling.

---

## Part 2: Exploratory Data Analysis (EDA)

### Step 2.1: Distribution of Response Variables

**For Question 1 (Past Purchases):**
```r
# Histogram of past purchases
ggplot(data, aes(x = past_purchases)) +
  geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7) +
  geom_vline(aes(xintercept = mean(past_purchases)),
             color = "red", linetype = "dashed", size = 1) +
  labs(title = "Distribution of Past Purchases",
       x = "Number of Past Purchases",
       y = "Count") +
  theme_minimal()

# Summary statistics
summary(data$past_purchases)
```

**What to observe:**
- Is the distribution symmetric or skewed?
- Are there outliers?
- What's the typical range of purchases?
- Since this is count data, consider Poisson or Negative Binomial models

**For Question 2 (Campaign Spend):**
```r
# Histogram of campaign spend
ggplot(data, aes(x = campaign_spend)) +
  geom_histogram(bins = 30, fill = "coral", alpha = 0.7) +
  geom_vline(aes(xintercept = mean(campaign_spend)),
             color = "red", linetype = "dashed", size = 1) +
  labs(title = "Distribution of Campaign Spend",
       x = "Campaign Spend ($)",
       y = "Count") +
  theme_minimal()

# Summary statistics
summary(data$campaign_spend)
```

**What to observe:**
- Is the distribution approximately normal?
- Are there any extreme values?
- Since this is continuous positive data, consider Linear or Gamma regression

### Step 2.2: Explore Categorical Variables
```r
# Create bar plots for categorical variables
table(data$gender)
table(data$campaign_variant)
table(data$ad_source)
table(data$email_signup)

# Visualize
ggplot(data, aes(x = gender, fill = gender)) +
  geom_bar(alpha = 0.7) +
  geom_text(stat = 'count', aes(label = after_stat(count)), vjust = -0.5) +
  labs(title = "Distribution of Gender") +
  theme_minimal()
```

**What to look for:**
- Are the groups balanced?
- Any categories with very few observations?

### Step 2.3: Examine Relationships Between Variables
```r
# Correlation matrix for numeric variables
numeric_data <- data %>% select(age, past_purchases, campaign_spend)
cor(numeric_data)

# Pairwise scatter plots
ggpairs(numeric_data)
```

**What to observe:**
- Strong correlations between predictors (multicollinearity concern)
- Relationships between response and predictors

---

## Part 3: Question 1 Analysis - What kinds of people make MORE purchases?

### Step 3.1: Exploratory Visualizations for Question 1

**Past Purchases by Gender:**
```r
ggplot(data, aes(x = gender, y = past_purchases, fill = gender)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Past Purchases by Gender",
       x = "Gender",
       y = "Number of Past Purchases") +
  theme_minimal()

# Calculate means
data %>%
  group_by(gender) %>%
  summarise(mean_purchases = mean(past_purchases),
            median_purchases = median(past_purchases),
            n = n())
```

**Interpretation:** Do certain genders have higher purchase rates?

**Past Purchases by Age:**
```r
ggplot(data, aes(x = age, y = past_purchases)) +
  geom_point(alpha = 0.3, color = "steelblue") +
  geom_smooth(method = "loess", color = "red", size = 1.5) +
  labs(title = "Past Purchases vs Age",
       x = "Age",
       y = "Number of Past Purchases") +
  theme_minimal()
```

**Interpretation:** Is there a linear or non-linear relationship with age?

**Past Purchases by Email Signup:**
```r
ggplot(data, aes(x = email_signup, y = past_purchases, fill = email_signup)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Past Purchases by Email Signup Status") +
  theme_minimal()

# Calculate means
data %>%
  group_by(email_signup) %>%
  summarise(mean_purchases = mean(past_purchases),
            median_purchases = median(past_purchases),
            n = n())
```

**Interpretation:** Do email subscribers purchase more?

**Past Purchases by Ad Source:**
```r
ggplot(data, aes(x = ad_source, y = past_purchases, fill = ad_source)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Past Purchases by Ad Source") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Calculate means by ad source
data %>%
  group_by(ad_source) %>%
  summarise(mean_purchases = mean(past_purchases),
            n = n()) %>%
  arrange(desc(mean_purchases))
```

**Interpretation:** Which ad sources are associated with higher purchase customers?

### Step 3.2: Check for Overdispersion

Since `past_purchases` is count data, we need to check if Poisson or Negative Binomial is appropriate.

```r
# Check mean vs variance
mean_purchases <- mean(data$past_purchases)
var_purchases <- var(data$past_purchases)

cat("Mean:", mean_purchases, "\n")
cat("Variance:", var_purchases, "\n")
cat("Variance/Mean Ratio:", var_purchases / mean_purchases, "\n")
```

**Decision Rule:**
- If variance ≈ mean: Use **Poisson Regression**
- If variance >> mean (ratio > 2): Use **Negative Binomial Regression** (overdispersion)

### Step 3.3: Fit Statistical Models

**Option A: Poisson Regression**
```r
# Fit Poisson model
poisson_model <- glm(past_purchases ~ age + gender + email_signup + ad_source,
                     data = data,
                     family = poisson(link = "log"))

summary(poisson_model)
```

**Option B: Negative Binomial Regression (recommended if overdispersed)**
```r
library(MASS)

# Fit Negative Binomial model
negbin_model <- glm.nb(past_purchases ~ age + gender + email_signup + ad_source,
                       data = data)

summary(negbin_model)
```

**Model Comparison:**
```r
# Compare AIC (lower is better)
AIC(poisson_model)
AIC(negbin_model)

# Choose the model with lower AIC
```

### Step 3.4: Interpret Model Coefficients

**Extract and interpret coefficients:**
```r
# Get tidy output with confidence intervals
# Exponentiate to get Incident Rate Ratios (IRR)
coef_table <- tidy(negbin_model, conf.int = TRUE, exponentiate = TRUE)
print(coef_table)
```

**Interpretation Guide:**
- **IRR > 1**: Variable is associated with MORE purchases
  - Example: IRR = 1.15 means 15% increase in purchases
- **IRR < 1**: Variable is associated with FEWER purchases
  - Example: IRR = 0.85 means 15% decrease in purchases
- **IRR = 1**: No effect
- **p-value < 0.05**: Statistically significant effect

**Example Interpretation:**
```r
# If age coefficient IRR = 1.01, p < 0.001:
"For each additional year of age, the expected number of past purchases
increases by 1% (IRR = 1.01), holding all other variables constant.
This effect is statistically significant (p < 0.001)."
```

### Step 3.5: Model Diagnostics

```r
# Check residuals
plot(negbin_model)

# Check for multicollinearity
vif(negbin_model)
```

**What to check:**
- Residual plots should show random scatter
- VIF < 5 (ideally < 3) indicates no severe multicollinearity

### Step 3.6: Visualize Effects

**Coefficient Plot:**
```r
# Plot Incident Rate Ratios
coef_plot_data <- coef_table %>%
  filter(term != "(Intercept)")

ggplot(coef_plot_data, aes(x = estimate, y = term)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "red") +
  geom_point(size = 4, color = "steelblue") +
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.2) +
  labs(title = "Incident Rate Ratios for Past Purchases",
       x = "IRR (95% CI)",
       y = "Predictor") +
  theme_minimal()
```

**Predicted Values by Age:**
```r
# Create prediction data
age_pred <- data.frame(
  age = seq(min(data$age), max(data$age), length.out = 100),
  gender = "Man",
  email_signup = "TRUE",
  ad_source = "Facebook"
)

# Generate predictions
age_pred$predicted <- predict(negbin_model, newdata = age_pred, type = "response")

# Plot
ggplot() +
  geom_point(data = data, aes(x = age, y = past_purchases), alpha = 0.2) +
  geom_line(data = age_pred, aes(x = age, y = predicted),
            color = "red", size = 1.5) +
  labs(title = "Predicted Past Purchases by Age",
       x = "Age",
       y = "Predicted Number of Purchases") +
  theme_minimal()
```

### Step 3.7: Answer Question 1

**Template for Answer:**

Based on the Negative Binomial regression analysis, the following customer characteristics are associated with MORE past purchases:

1. **[Significant Variable 1]**: [Describe effect size and direction]
   - IRR = [value], 95% CI: [lower, upper], p < 0.05
   - Interpretation: [Practical meaning]

2. **[Significant Variable 2]**: [Describe effect size and direction]
   - IRR = [value], 95% CI: [lower, upper], p < 0.05
   - Interpretation: [Practical meaning]

**Example:**
"Customers who are signed up for promotional emails make approximately 25% more purchases (IRR = 1.25, p < 0.001) compared to non-subscribers, holding all other factors constant. This suggests that email marketing effectively drives repeat purchases."

---

## Part 4: Question 2 Analysis - Which campaign variant should Company X use?

### Step 4.1: Exploratory Visualizations for Question 2

**Campaign Spend by Variant:**
```r
# Boxplot
ggplot(data, aes(x = campaign_variant, y = campaign_spend, fill = campaign_variant)) +
  geom_boxplot(alpha = 0.7) +
  scale_fill_manual(values = c("A" = "#E69F00", "B" = "#56B4E9", "C" = "#009E73")) +
  labs(title = "Campaign Spend by Variant",
       x = "Campaign Variant",
       y = "Campaign Spend ($)") +
  theme_minimal()

# Calculate summary statistics
data %>%
  group_by(campaign_variant) %>%
  summarise(
    n = n(),
    mean_spend = mean(campaign_spend),
    median_spend = median(campaign_spend),
    sd_spend = sd(campaign_spend),
    total_spend = sum(campaign_spend)
  ) %>%
  arrange(desc(mean_spend))
```

**Key Questions:**
- Which variant has the highest mean spend?
- Are the differences substantial?
- Is there high variability within variants?

**Mean Spend with Error Bars:**
```r
# Calculate means and standard errors
spend_summary <- data %>%
  group_by(campaign_variant) %>%
  summarise(
    mean_spend = mean(campaign_spend),
    se = sd(campaign_spend) / sqrt(n())
  )

# Plot with 95% confidence intervals
ggplot(spend_summary, aes(x = campaign_variant, y = mean_spend, fill = campaign_variant)) +
  geom_col(alpha = 0.7, color = "black") +
  geom_errorbar(aes(ymin = mean_spend - 1.96*se,
                    ymax = mean_spend + 1.96*se),
                width = 0.2, size = 1) +
  geom_text(aes(label = paste0("$", round(mean_spend, 2))), vjust = -2.5) +
  scale_fill_manual(values = c("A" = "#E69F00", "B" = "#56B4E9", "C" = "#009E73")) +
  labs(title = "Mean Campaign Spend by Variant (95% CI)",
       x = "Campaign Variant",
       y = "Mean Campaign Spend ($)") +
  theme_minimal()
```

**Interpretation:** Do the confidence intervals overlap? Non-overlapping CIs suggest significant differences.

### Step 4.2: Initial Statistical Test (ANOVA)

```r
# One-way ANOVA
anova_result <- aov(campaign_spend ~ campaign_variant, data = data)
summary(anova_result)
```

**Interpretation:**
- p-value < 0.05: At least one variant is significantly different from others
- p-value ≥ 0.05: No significant differences detected

### Step 4.3: Fit Regression Models

Since we want to control for other variables (age, gender, past purchases, etc.), we'll use regression.

**Option A: Linear Regression**
```r
# Fit linear model
linear_model <- lm(campaign_spend ~ campaign_variant + age + gender +
                   past_purchases + email_signup + ad_source,
                   data = data)

summary(linear_model)
```

**Option B: Gamma GLM (for positive continuous data)**
```r
# Fit Gamma model
gamma_model <- glm(campaign_spend ~ campaign_variant + age + gender +
                   past_purchases + email_signup + ad_source,
                   data = data,
                   family = Gamma(link = "log"))

summary(gamma_model)
```

**Model Comparison:**
```r
# Compare AIC
AIC(linear_model)
AIC(gamma_model)

# Choose model with lower AIC
```

### Step 4.4: Interpret Campaign Variant Effects

**Extract coefficients:**
```r
# Tidy output
coef_table_q2 <- tidy(linear_model, conf.int = TRUE)

# Filter to campaign variants only
campaign_effects <- coef_table_q2 %>%
  filter(str_detect(term, "campaign_variant"))

print(campaign_effects)
```

**Key Points:**
- Variant A is the **reference category** (baseline)
- Coefficients for B and C show difference from A
- Positive coefficient: Variant spends MORE than A
- Negative coefficient: Variant spends LESS than A

**Example Interpretation:**
```r
# If campaign_variantB coefficient = 45.50, p = 0.001:
"Variant B generates $45.50 more in campaign spending per customer
compared to Variant A (the original campaign), after controlling for
customer demographics and behavior (p = 0.001)."
```

### Step 4.5: Post-hoc Pairwise Comparisons

```r
# Tukey's HSD for all pairwise comparisons
tukey_result <- TukeyHSD(anova_result)
print(tukey_result)

# Visualize
plot(tukey_result, las = 1, col = "steelblue")
```

**What to look for:**
- B-A comparison: Is B different from A?
- C-A comparison: Is C different from A?
- C-B comparison: Is C different from B?
- Adjusted p-values < 0.05 indicate significant differences

### Step 4.6: Model Diagnostics

```r
# Check assumptions
par(mfrow = c(2, 2))
plot(linear_model)
par(mfrow = c(1, 1))

# Check multicollinearity
vif(linear_model)
```

**What to check:**
- Residuals vs Fitted: Should show random scatter (linearity)
- Q-Q Plot: Points should follow diagonal line (normality)
- Scale-Location: Should show random scatter (homoscedasticity)
- Residuals vs Leverage: Check for influential outliers

### Step 4.7: Calculate ROI Metrics

```r
# Calculate total revenue by variant
roi_summary <- data %>%
  group_by(campaign_variant) %>%
  summarise(
    n_customers = n(),
    mean_spend_per_customer = mean(campaign_spend),
    total_revenue = sum(campaign_spend),
    median_spend = median(campaign_spend)
  ) %>%
  arrange(desc(mean_spend_per_customer))

print(roi_summary)

# Calculate percentage improvement over baseline (A)
baseline_mean <- roi_summary %>%
  filter(campaign_variant == "A") %>%
  pull(mean_spend_per_customer)

roi_summary <- roi_summary %>%
  mutate(
    improvement_over_A = ((mean_spend_per_customer - baseline_mean) / baseline_mean) * 100
  )

print(roi_summary)
```

### Step 4.8: Visualize Model Predictions

```r
# Get predicted values adjusted for covariates
predicted_by_variant <- data %>%
  mutate(predicted = predict(linear_model, type = "response")) %>%
  group_by(campaign_variant) %>%
  summarise(
    mean_predicted = mean(predicted),
    se = sd(predicted) / sqrt(n())
  )

# Plot
ggplot(predicted_by_variant, aes(x = campaign_variant, y = mean_predicted,
                                  fill = campaign_variant)) +
  geom_col(alpha = 0.7, color = "black") +
  geom_errorbar(aes(ymin = mean_predicted - 1.96*se,
                    ymax = mean_predicted + 1.96*se),
                width = 0.2) +
  geom_text(aes(label = paste0("$", round(mean_predicted, 2))), vjust = -2) +
  scale_fill_manual(values = c("A" = "#E69F00", "B" = "#56B4E9", "C" = "#009E73")) +
  labs(title = "Model-Predicted Campaign Spend by Variant",
       subtitle = "Adjusted for customer characteristics",
       x = "Campaign Variant",
       y = "Predicted Spend ($)") +
  theme_minimal()
```

### Step 4.9: Check for Interaction Effects (Optional but Recommended)

Do certain customer groups respond differently to different variants?

```r
# Test interaction with email signup
interaction_model <- lm(campaign_spend ~ campaign_variant * email_signup +
                        age + gender + past_purchases + ad_source,
                        data = data)

summary(interaction_model)

# Visualize interaction
interaction_data <- data %>%
  group_by(campaign_variant, email_signup) %>%
  summarise(mean_spend = mean(campaign_spend),
            se = sd(campaign_spend) / sqrt(n()))

ggplot(interaction_data, aes(x = campaign_variant, y = mean_spend,
                             color = email_signup, group = email_signup)) +
  geom_line(size = 1.5) +
  geom_point(size = 4) +
  geom_errorbar(aes(ymin = mean_spend - 1.96*se,
                    ymax = mean_spend + 1.96*se),
                width = 0.1) +
  labs(title = "Campaign Variant × Email Signup Interaction",
       x = "Campaign Variant",
       y = "Mean Spend ($)") +
  theme_minimal()
```

**Interpretation:** If lines are parallel, no interaction. If lines cross or diverge, there's an interaction effect.

### Step 4.10: Answer Question 2

**Template for Answer:**

Based on the regression analysis controlling for customer demographics and behavior, I recommend **Campaign Variant [B/C]** for the following reasons:

1. **Revenue Performance**:
   - Variant [X] generates $[amount] more per customer than the baseline (Variant A)
   - This represents a [percentage]% increase in campaign spending
   - The difference is statistically significant (p < 0.05)

2. **Statistical Evidence**:
   - Coefficient = [value], 95% CI: [lower, upper]
   - Tukey's HSD confirms significant differences between variants

3. **Business Impact**:
   - If applied to all 5000 customers, Variant [X] would generate $[total] more revenue
   - [Additional business considerations]

**Caveats/Recommendations**:
- [Mention any important findings about customer segments]
- [Note any interaction effects]
- [Suggest follow-up analyses or A/B testing]

---

## Part 5: Model Selection Checklist

### For Question 1 (Count Data - Past Purchases):

| Model | When to Use | R Function |
|-------|-------------|------------|
| **Poisson Regression** | Variance ≈ Mean | `glm(..., family = poisson)` |
| **Negative Binomial** | Variance > Mean (overdispersion) | `glm.nb(...)` from MASS package |
| **Zero-Inflated Models** | Many zeros in data | `zeroinfl()` from pscl package |

**Decision:** Check variance/mean ratio. If > 2, use Negative Binomial.

### For Question 2 (Continuous Positive Data - Campaign Spend):

| Model | When to Use | R Function |
|-------|-------------|------------|
| **Linear Regression** | Approximately normal residuals | `lm(...)` |
| **Gamma GLM** | Right-skewed positive data | `glm(..., family = Gamma)` |
| **Log-transformed Linear** | Log-transformed response is normal | `lm(log(y) ~ ...)` |

**Decision:** Compare AIC values. Check residual plots for assumptions.

---

## Part 6: Reporting Checklist

### For Your Written Report:

**Analysis Section:**
- [ ] Explain why you chose each model (e.g., "count data → negative binomial")
- [ ] Mention any data transformations or preprocessing
- [ ] Justify inclusion of covariates (control variables)

**Results Section:**
- [ ] Report key coefficients with confidence intervals and p-values
- [ ] Interpret effect sizes in practical terms (avoid jargon)
- [ ] Include visualizations (coefficient plots, predicted values)
- [ ] State clear answers to Questions 1 and 2

**Discussion Section:**
- [ ] Business implications of findings
- [ ] Recommendations for Company X
- [ ] Limitations of the analysis
- [ ] What you would do differently next time
- [ ] Suggestions for future research/analysis

### Files to Submit:
1. **PDF Report** (from R Markdown knitted to HTML/PDF)
2. **Code File** (.Rmd or .R with all code)
3. **README** (.txt or .md explaining file structure)

---

## Part 7: Common Pitfalls to Avoid

1. **Not checking model assumptions**: Always plot residuals!
2. **Ignoring overdispersion**: Check variance/mean ratio for count data
3. **Confusing statistical vs practical significance**: A p-value < 0.05 doesn't always mean the effect is large enough to matter
4. **Forgetting to interpret exponentiated coefficients**: For Poisson/Negative Binomial with log link, exponentiate to get IRR
5. **Not controlling for confounders**: Include relevant covariates in models
6. **Over-interpreting non-significant results**: Focus on significant predictors
7. **Ignoring the audience**: Write for non-experts (avoid statistical jargon)

---

## Quick Reference: Key R Functions

```r
# Data manipulation
library(tidyverse)
data %>% filter(...) %>% select(...) %>% mutate(...) %>% group_by(...) %>% summarise(...)

# Visualization
ggplot(data, aes(x = , y = )) + geom_point() + theme_minimal()

# Models
lm(y ~ x1 + x2, data = data)                    # Linear regression
glm(y ~ x, family = poisson, data = data)       # Poisson regression
glm.nb(y ~ x, data = data)                      # Negative binomial (from MASS)
glm(y ~ x, family = Gamma(link="log"), data=data) # Gamma GLM

# Model summaries
summary(model)                                   # Full summary
tidy(model, conf.int = TRUE)                    # Tidy coefficients (from broom)
glance(model)                                    # Model fit statistics
AIC(model)                                       # Akaike Information Criterion

# Diagnostics
plot(model)                                      # Diagnostic plots
vif(model)                                       # Variance Inflation Factors (from car)

# Predictions
predict(model, newdata = ..., type = "response")
```

---

## Additional Resources

- **GLM Reference**: Chapter on Generalized Linear Models in "An Introduction to Statistical Learning"
- **Count Models**: "Regression Models for Count Data in R" (Zeileis et al.)
- **Visualization**: "ggplot2: Elegant Graphics for Data Analysis" by Hadley Wickham
- **Interpretation**: "Statistical Rethinking" by Richard McElreath (Bayesian perspective)

---

Good luck with your analysis!