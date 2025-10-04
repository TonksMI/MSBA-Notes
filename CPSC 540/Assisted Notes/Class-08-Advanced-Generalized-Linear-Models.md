# Class 8: Advanced Generalized Linear Models
**Date:** September 22, 2025 (Monday)
**Topics:** Advanced GLMs, Overdispersion, Mixed Effects, and Extensions
**Reading:** Regression and Other Stories Chapters 15-16, Additional GLM References

## Overview
Advanced topics in Generalized Linear Models, including handling overdispersion, mixed effects GLMs, specialized distributions, and extensions to more complex data structures. This builds on the fundamental GLM framework to address real-world modeling challenges.

## Advanced GLM Distributions

### Negative Binomial Regression
**Purpose:** Handle overdispersion in count data when Poisson assumptions fail.

#### Mathematical Framework
The negative binomial distribution can be parameterized as:
$$Y \sim NB(\mu, \alpha)$$

**PDF:**
$$P(Y = y) = \frac{\Gamma(y + 1/\alpha)}{\Gamma(1/\alpha)\Gamma(y+1)} \left(\frac{1/\alpha}{1/\alpha + \mu}\right)^{1/\alpha} \left(\frac{\mu}{1/\alpha + \mu}\right)^y$$

**Key Properties:**
```
E[Y] = μ
Var(Y) = μ + α·μ² (overdispersion when α > 0)
```

**Link Function:** Log link (canonical)
```
g(μ) = log(μ)
μ = e^(X'β)
```

**When to use:**
- Count data with **variance > mean**
- Poisson model shows **overdispersion**
- **α → 0**: approaches Poisson (equidispersion)
- **α > 0**: allows for overdispersion

#### Implementation in R
```r
# Fit negative binomial model
library(MASS)
nb_model <- glm.nb(y ~ x1 + x2 + x3)

# Alternative using glm with negative binomial family
library(mgcv)
nb_model2 <- glm(y ~ x1 + x2 + x3, family = nb())

# Compare with Poisson
poisson_model <- glm(y ~ x1 + x2 + x3, family = poisson)

# Test for overdispersion
anova(poisson_model, nb_model, test = "Chisq")
```

**Business Application:** Customer complaint modeling
```r
# Example: Daily customer complaints
complaints_data <- data.frame(
  complaints = c(5, 12, 8, 15, 3, 20, 7, 11, 16, 4),
  temperature = c(75, 85, 78, 88, 70, 92, 73, 82, 87, 68),
  weekend = c(0, 0, 0, 0, 0, 1, 1, 0, 0, 0)
)

# Fit models
poisson_fit <- glm(complaints ~ temperature + weekend,
                   family = poisson, data = complaints_data)
nb_fit <- glm.nb(complaints ~ temperature + weekend, data = complaints_data)

# Interpret coefficients (rate ratios)
exp(coef(nb_fit))
```

### Beta Regression
**Purpose:** Model proportions, rates, and percentages bounded between 0 and 1.

#### Mathematical Framework
$$Y \sim Beta(\mu, \phi)$$

**Parameterization:**
- **μ**: Mean parameter (0 < μ < 1)
- **φ**: Precision parameter (φ > 0)

**Variance:**
$$Var(Y) = \frac{\mu(1-\mu)}{1+\phi}$$

**Link Function:** Logit link (most common)
```
g(μ) = log(μ/(1-μ)) = logit(μ)
μ = e^η/(1 + e^η) where η = X'β
```

#### Implementation in R
```r
library(betareg)

# Fit beta regression
beta_model <- betareg(y ~ x1 + x2, data = data)

# Alternative link functions
beta_model_probit <- betareg(y ~ x1 + x2, link = "probit")
beta_model_cloglog <- betareg(y ~ x1 + x2, link = "cloglog")
```

**Business Application:** Market share analysis
```r
# Example: Product market share prediction
market_data <- data.frame(
  market_share = c(0.15, 0.23, 0.31, 0.28, 0.42, 0.18, 0.35),
  advertising_spend = c(10, 15, 25, 20, 35, 12, 28),
  competitor_count = c(5, 4, 3, 4, 2, 6, 3),
  brand_age = c(2, 5, 8, 6, 12, 1, 9)
)

beta_fit <- betareg(market_share ~ advertising_spend + competitor_count + brand_age,
                    data = market_data)

# Marginal effects
summary(beta_fit)
```

### Robust GLMs with Student-t Distribution
**Purpose:** Robust regression for data with outliers or heavy tails.

#### Mathematical Framework
**Student-t GLM:**
```
Y ~ t(μ, σ², ν)
```

Where:
- **ν**: Degrees of freedom (controls tail heaviness)
- **ν → ∞**: approaches normal distribution
- **Small ν**: heavy tails, robust to outliers

**Link Function:** Usually identity link
```
g(μ) = μ = X'β
```

#### Implementation
```r
library(heavy)

# Fit robust t-GLM
t_model <- glm(y ~ x1 + x2, family = Student(df = 5))

# Compare with normal GLM
normal_model <- glm(y ~ x1 + x2, family = gaussian)

# Model comparison
AIC(t_model, normal_model)
```

## Handling Overdispersion

### Detection Methods

#### 1. Pearson Chi-Square Test
```r
# Calculate dispersion parameter
pearson_chisq <- sum(residuals(model, type = "pearson")^2)
dispersion_est <- pearson_chisq / df.residual(model)

cat("Dispersion estimate:", dispersion_est)
if(dispersion_est > 1.5) {
  cat("\nEvidence of overdispersion")
}
```

#### 2. Deviance-based Test
```r
# Deviance dispersion
dev_dispersion <- deviance(model) / df.residual(model)
cat("Deviance dispersion:", dev_dispersion)
```

#### 3. Formal Statistical Tests
```r
library(AER)
# Overdispersion test for Poisson models
dispersiontest(poisson_model, trafo = 1)  # Tests H0: dispersion = 1
```

### Solutions for Overdispersion

#### 1. Quasi-Poisson Models
```r
# Allow dispersion parameter ≠ 1
quasi_model <- glm(y ~ x1 + x2, family = quasipoisson)
summary(quasi_model)  # Note: no AIC available
```

#### 2. Negative Binomial Models
```r
# Explicitly model overdispersion
nb_model <- glm.nb(y ~ x1 + x2)
```

#### 3. Zero-Inflated Models
```r
library(pscl)
# When excess zeros cause overdispersion
zi_model <- zeroinfl(y ~ x1 + x2 | z1 + z2, dist = "poisson")
```

## Mixed Effects GLMs

### Motivation
**Standard GLMs assume independence**, but real data often has:
- **Repeated measures** on same subjects
- **Clustered data** (students within schools)
- **Hierarchical structure** (employees within departments)

### Framework
**Mixed effects GLM:**
$$g(E[Y_{ij}|u_i]) = X_{ij}'\beta + Z_{ij}'u_i$$

Where:
- **β**: Fixed effects (population-level)
- **u_i**: Random effects (group-specific deviations)
- **Z_{ij}**: Design matrix for random effects

#### Implementation in R
```r
library(lme4)

# Random intercept logistic regression
mixed_logit <- glmer(outcome ~ treatment + time + (1|subject),
                     family = binomial, data = longitudinal_data)

# Random slope and intercept
mixed_logit2 <- glmer(outcome ~ treatment + time + (time|subject),
                      family = binomial, data = longitudinal_data)

# Poisson mixed model
mixed_poisson <- glmer(count ~ treatment + (1|cluster),
                       family = poisson, data = cluster_data)
```

**Business Application:** Customer retention modeling
```r
# Example: Monthly customer activity
customer_data <- data.frame(
  customer_id = rep(1:100, each = 12),
  month = rep(1:12, 100),
  active = rbinom(1200, 1, 0.7),
  marketing_spend = rnorm(1200, 50, 15),
  satisfaction = rnorm(1200, 7, 2)
)

# Account for customer-specific effects
retention_model <- glmer(active ~ marketing_spend + satisfaction + month +
                         (1|customer_id),
                         family = binomial, data = customer_data)

# Interpret results
summary(retention_model)
ranef(retention_model)  # Customer-specific random effects
```

## Ordinal Response Models

### Cumulative Logit Models
**Purpose:** Model ordered categorical responses (ratings, severity levels).

#### Mathematical Framework
For response categories 1, 2, ..., K:
$$P(Y \leq k) = \frac{e^{\alpha_k + X'\beta}}{1 + e^{\alpha_k + X'\beta}}$$

**Key assumption:** **Proportional odds** - effect of predictors same across all cutpoints.

#### Implementation
```r
library(MASS)
# Ordered logistic regression
ordered_model <- polr(rating ~ service_quality + price + location,
                      data = restaurant_data, Hess = TRUE)

library(ordinal)
# More flexible implementation
clm_model <- clm(rating ~ service_quality + price + location,
                 data = restaurant_data)

# Test proportional odds assumption
library(brant)
brant(ordered_model)
```

**Business Application:** Customer satisfaction analysis
```r
# Example: Product rating prediction
rating_data <- data.frame(
  rating = ordered(c("Poor", "Fair", "Good", "Excellent")),
  price = c(25, 35, 45, 55, 30, 40, 50, 60),
  quality = c(2, 3, 4, 5, 3, 4, 5, 4),
  service = c(1, 2, 4, 5, 2, 3, 4, 5)
)

satisfaction_model <- polr(rating ~ price + quality + service, data = rating_data)

# Predicted probabilities
predict(satisfaction_model, type = "probs")
```

## Advanced Diagnostic Methods

### Quantile Residuals
**Better residuals for non-normal GLMs:**
```r
library(statmod)

# Quantile residuals (more normally distributed)
qres <- qresid(model)
qqnorm(qres)
qqline(qres)

# Compare with deviance residuals
dev_res <- residuals(model, type = "deviance")
par(mfrow = c(1,2))
qqnorm(dev_res, main = "Deviance Residuals")
qqnorm(qres, main = "Quantile Residuals")
```

### Influence Diagnostics
```r
# Cook's distance for GLMs
cooksd <- cooks.distance(model)
plot(cooksd, type = "h")
abline(h = 4/nrow(data), col = "red")  # Threshold

# DFBETAS
dfbetas_vals <- dfbetas(model)
plot(dfbetas_vals[,2])  # For second coefficient
```

### Model Validation
```r
# Cross-validation for GLMs
library(boot)

# Define cost function
cost_function <- function(y, yhat) mean((y - yhat)^2)

# K-fold CV
cv_result <- cv.glm(data, model, cost = cost_function, K = 10)
cat("CV Error:", cv_result$delta[1])
```

## Time Series Extensions

### Autoregressive GLMs
**Purpose:** Handle temporal dependence in GLM responses.

#### Framework
**GLM with AR(1) errors:**
$$g(E[Y_t]) = X_t'\beta + \phi \cdot \epsilon_{t-1}$$

#### Implementation
```r
library(mgcv)

# GAM with AR(1) correlation
ar_model <- gamm(y ~ s(x1) + x2, correlation = corAR1(form = ~time))

# Alternative: GLM with lagged response
lag_model <- glm(y ~ lag(y, 1) + x1 + x2, family = poisson)
```

### Seasonal Effects
```r
# Include seasonal components
seasonal_model <- glm(y ~ x1 + x2 + factor(month) + sin(2*pi*time/365) + cos(2*pi*time/365),
                      family = poisson)
```

## Business Applications and ROI Analysis

### Marketing Response Modeling
**Scenario:** Digital advertising campaign optimization

```r
# Campaign performance data
campaign_data <- data.frame(
  conversions = rpois(365, 50),
  ad_spend = rnorm(365, 1000, 200),
  competitor_spend = rnorm(365, 800, 150),
  seasonality = sin(2*pi*(1:365)/365),
  day_of_week = factor(rep(1:7, length.out = 365))
)

# Negative binomial model for overdispersed count data
campaign_model <- glm.nb(conversions ~ log(ad_spend) + log(competitor_spend) +
                         seasonality + day_of_week, data = campaign_data)

# ROI calculation
predicted_conversions <- predict(campaign_model, type = "response")
conversion_value <- 25  # $ per conversion
ad_costs <- campaign_data$ad_spend

roi <- (predicted_conversions * conversion_value - ad_costs) / ad_costs
cat("Average ROI:", mean(roi))

# Optimal spending analysis
spend_levels <- seq(500, 2000, by = 100)
roi_by_spend <- sapply(spend_levels, function(spend) {
  pred_conv <- predict(campaign_model,
                       newdata = transform(campaign_data, ad_spend = spend),
                       type = "response")
  mean((pred_conv * conversion_value - spend) / spend)
})

optimal_spend <- spend_levels[which.max(roi_by_spend)]
cat("Optimal daily spend: $", optimal_spend)
```

### Quality Control Applications
**Scenario:** Manufacturing defect prediction

```r
# Manufacturing quality data
quality_data <- data.frame(
  defects = rpois(200, 3),
  machine_age = rnorm(200, 5, 2),
  operator_experience = rnorm(200, 3, 1),
  shift = factor(rep(c("Day", "Night"), 100)),
  temperature = rnorm(200, 72, 5)
)

# Poisson regression with overdispersion check
defect_model <- glm(defects ~ machine_age + operator_experience + shift + temperature,
                    family = poisson, data = quality_data)

# Check for overdispersion
dispersion <- sum(residuals(defect_model, type = "pearson")^2) / df.residual(defect_model)

if(dispersion > 1.5) {
  # Use negative binomial
  defect_model <- glm.nb(defects ~ machine_age + operator_experience + shift + temperature,
                         data = quality_data)
}

# Cost analysis
defect_cost <- 50  # $ per defect
predicted_defects <- predict(defect_model, type = "response")
daily_defect_cost <- predicted_defects * defect_cost

# Preventive maintenance ROI
maintenance_cost <- 200  # $ per day
defect_reduction <- 0.3  # 30% reduction with maintenance
savings <- daily_defect_cost * defect_reduction
net_benefit <- savings - maintenance_cost

cat("Average daily savings from maintenance: $", mean(net_benefit))
```

## Advanced Topics

### Bayesian GLMs
**Framework:** Incorporate prior information and uncertainty quantification.

```r
library(rstanarm)

# Bayesian logistic regression
bayes_logit <- stan_glm(outcome ~ x1 + x2, family = binomial,
                        data = data, prior_intercept = normal(0, 5))

# Posterior intervals
posterior_interval(bayes_logit, prob = 0.95)

# Posterior predictive checking
pp_check(bayes_logit)
```

### Machine Learning Integration
**Regularized GLMs:**

```r
library(glmnet)

# LASSO for GLMs
x_matrix <- model.matrix(~ . - y, data = data)
lasso_model <- cv.glmnet(x_matrix, data$y, family = "poisson", alpha = 1)

# Elastic net
elastic_model <- cv.glmnet(x_matrix, data$y, family = "poisson", alpha = 0.5)

# Extract coefficients
coef(lasso_model, s = "lambda.min")
```

### Non-parametric Extensions
**Generalized Additive Models (GAMs):**

```r
library(mgcv)

# Smooth functions of covariates
gam_model <- gam(y ~ s(x1) + s(x2) + factor(group), family = poisson)

# 2D smooths
gam_2d <- gam(y ~ s(x1, x2) + factor(group), family = poisson)

# Plot smooth effects
plot(gam_model, pages = 1)
```

## Model Selection and Comparison

### Information Criteria
```r
# Compare multiple models
models <- list(
  model1 = glm(y ~ x1, family = poisson),
  model2 = glm(y ~ x1 + x2, family = poisson),
  model3 = glm.nb(y ~ x1 + x2),
  model4 = glm(y ~ x1 + x2 + x1:x2, family = poisson)
)

# AIC comparison
aic_values <- sapply(models, AIC)
best_model <- names(aic_values)[which.min(aic_values)]

# BIC comparison
bic_values <- sapply(models, BIC)

# Create comparison table
comparison <- data.frame(
  Model = names(models),
  AIC = aic_values,
  BIC = bic_values,
  Deviance = sapply(models, deviance)
)
print(comparison)
```

### Cross-Validation
```r
# K-fold cross-validation function for GLMs
cv_glm <- function(formula, family, data, k = 10) {
  n <- nrow(data)
  folds <- sample(1:k, n, replace = TRUE)
  errors <- numeric(k)

  for(i in 1:k) {
    train <- data[folds != i, ]
    test <- data[folds == i, ]

    model <- glm(formula, family = family, data = train)
    pred <- predict(model, newdata = test, type = "response")

    if(family$family == "poisson") {
      errors[i] <- mean((test$y - pred)^2)  # MSE
    } else if(family$family == "binomial") {
      errors[i] <- mean(abs(test$y - (pred > 0.5)))  # Misclassification rate
    }
  }

  return(mean(errors))
}

# Compare models using CV
cv_poisson <- cv_glm(y ~ x1 + x2, poisson, data)
cv_nb <- cv_glm(y ~ x1 + x2, negative.binomial(1), data)
```

## Troubleshooting Common Issues

### Convergence Problems
```r
# Increase maximum iterations
model <- glm(y ~ x1 + x2, family = poisson,
             control = glm.control(maxit = 100))

# Try different starting values
model <- glm(y ~ x1 + x2, family = poisson,
             start = c(0, 0, 0))

# Scale predictors
data$x1_scaled <- scale(data$x1)
model <- glm(y ~ x1_scaled + x2, family = poisson, data = data)
```

### Perfect Separation in Logistic Regression
```r
library(logistf)
# Firth's penalized likelihood
firth_model <- logistf(y ~ x1 + x2, data = data)
```

### Zero-inflated Data
```r
library(pscl)
# Zero-inflated Poisson
zip_model <- zeroinfl(y ~ x1 + x2 | z1, dist = "poisson")

# Hurdle models
hurdle_model <- hurdle(y ~ x1 + x2 | z1, dist = "poisson")
```

## Key Takeaways

1. **Advanced GLMs** extend the basic framework to handle complex data structures
2. **Overdispersion** is common in count data and requires specialized approaches
3. **Mixed effects GLMs** handle dependent observations in clustered/longitudinal data
4. **Specialized distributions** (negative binomial, beta, Student-t) address specific data characteristics
5. **Ordinal models** properly handle ordered categorical responses
6. **Diagnostic methods** beyond basic GLMs are essential for model validation
7. **Business applications** require careful consideration of costs, benefits, and ROI
8. **Model selection** should balance complexity with predictive performance

Advanced GLMs provide the flexibility to model complex real-world phenomena while maintaining the interpretability and theoretical foundation of the GLM framework. These extensions are essential for practical data science applications in business and research contexts.