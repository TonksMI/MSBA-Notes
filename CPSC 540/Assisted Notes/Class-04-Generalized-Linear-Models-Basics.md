# Class 4: Generalized Linear Models - Basics
**Date:** September 8, 2025 (Monday)  
**Topics:** Basics of GLMs  
**Reading:** Regression and Other Stories Chapters 8 and 13

## Overview
Introduction to Generalized Linear Models (GLMs), extending linear regression to handle non-normal response variables and non-linear relationships through link functions.

## Motivation: Beyond Linear Regression

### Limitations of Ordinary Linear Regression
1. **Normality assumption**: Response must be normally distributed
2. **Constant variance**: Homoscedasticity requirement
3. **Unbounded predictions**: Can predict negative counts or probabilities > 1
4. **Linear relationships only**: Cannot model curved relationships directly

**Examples where linear regression fails:**
- **Binary outcomes**: Success/failure, Yes/No
- **Count data**: Number of events, always non-negative
- **Proportion data**: Bounded between 0 and 1
- **Positive continuous data**: Reaction times, prices

## Exponential Family of Distributions

### Definition
A distribution is in the exponential family if its PDF/PMF can be written as:
$$f(y|\theta) = \exp\left\{\frac{y\theta - b(\theta)}{a(\phi)} + c(y,\phi)\right\}$$

Where:
- **θ**: Natural/canonical parameter
- **φ**: Dispersion parameter  
- **b(θ)**: Cumulant function
- **a(φ)**: Often φ/w where w is a known weight
- **c(y,φ)**: Normalization term

### Key Properties
```
E[Y] = μ = b'(θ)
Var(Y) = a(φ)b''(θ)
```

### Common Exponential Family Distributions

#### Normal Distribution
```
Y ~ N(μ, σ²)
f(y) = exp{[yμ - μ²/2]/σ² - (y²/2σ² + log(√2πσ²))}

θ = μ (natural parameter)
φ = σ² (dispersion)
b(θ) = θ²/2
```

#### Bernoulli Distribution  
```
Y ~ Bernoulli(p)
f(y) = exp{y log(p/(1-p)) + log(1-p)}

θ = log(p/(1-p)) = logit(p)
φ = 1 (no dispersion)
b(θ) = log(1 + e^θ)
μ = e^θ/(1 + e^θ) = p
```

#### Poisson Distribution
```
Y ~ Poisson(λ)  
f(y) = exp{y log(λ) - λ - log(y!)}

θ = log(λ)
φ = 1 (no dispersion)
b(θ) = e^θ
μ = e^θ = λ
```

#### Gamma Distribution
```
Y ~ Gamma(α, β) with mean μ = α/β
f(y) = exp{(-y/μ - log(μ))/φ + ...}

θ = -1/μ
φ = 1/α (dispersion)
b(θ) = -log(-θ)
```

## Generalized Linear Model Framework

### Three Components

#### 1. Random Component
Response variable Y follows exponential family distribution:
```
Y ~ f(y|θ,φ)
```

#### 2. Systematic Component  
Linear predictor combines covariates:
```
η = X'β = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ
```

#### 3. Link Function
Connects mean response to linear predictor:
```
g(μ) = η = X'β
μ = g⁻¹(η) = g⁻¹(X'β)
```

**Properties of link functions:**
- Monotonic (one-to-one mapping)
- Differentiable
- Maps appropriate range

### Common Link Functions

#### Identity Link (Linear Regression)
```
g(μ) = μ
μ = η = X'β
Domain: μ ∈ (-∞, ∞)
```

#### Logit Link (Logistic Regression)
```
g(μ) = log(μ/(1-μ)) = logit(μ)
μ = e^η/(1 + e^η) = 1/(1 + e^(-η))
Domain: μ ∈ (0, 1)
```

#### Log Link (Poisson Regression)
```
g(μ) = log(μ)
μ = e^η = e^(X'β)
Domain: μ ∈ (0, ∞)
```

#### Inverse Link (Gamma Regression)
```
g(μ) = 1/μ
μ = 1/η = 1/(X'β)
Domain: μ ∈ (0, ∞)
```

### Canonical Links
When θ = η (natural parameter equals linear predictor):
- **Normal**: Identity link (θ = μ)
- **Bernoulli**: Logit link (θ = log(p/(1-p)))  
- **Poisson**: Log link (θ = log(λ))
- **Gamma**: Inverse link (θ = -1/μ)

**Advantages of canonical links:**
- Sufficient statistics
- Simpler likelihood equations
- Better numerical properties

## Maximum Likelihood Estimation

### Log-Likelihood for GLM
```
ℓ(β) = Σᵢ [yᵢθᵢ - b(θᵢ)]/a(φ) + Σᵢ c(yᵢ,φ)
```

Where θᵢ is related to μᵢ = g⁻¹(xᵢ'β)

### Score Function (First Derivative)
```
U(β) = ∂ℓ/∂β = X'W^(1/2)D(μ-y)
```

Where:
- **W**: Diagonal weight matrix
- **D**: Diagonal matrix of derivatives dμ/dη

### Information Matrix (Second Derivative)
```
I(β) = -∂²ℓ/∂β∂β' = X'WDX
```

### Fisher Scoring Algorithm
Iteratively solve:
```
β^(t+1) = β^(t) + [X'W^(t)X]⁻¹X'W^(t)(y - μ^(t))
```

**Alternative form (IRLS):**
```
β^(t+1) = [X'W^(t)X]⁻¹X'W^(t)z^(t)
```

Where z^(t) = η^(t) + W^(t-1)(y - μ^(t)) (working response)

## Model Interpretation

### Coefficient Interpretation

#### Linear Regression (Identity Link)
```
μ = β₀ + β₁x₁ + β₂x₂
```
- β₁: Change in mean response per unit increase in x₁

#### Logistic Regression (Logit Link)  
```
log(p/(1-p)) = β₀ + β₁x₁ + β₂x₂
p = e^(β₀+β₁x₁+β₂x₂)/(1 + e^(β₀+β₁x₁+β₂x₂))
```
- e^β₁: Odds ratio for unit increase in x₁
- β₁: Change in log-odds per unit increase in x₁

#### Poisson Regression (Log Link)
```
log(λ) = β₀ + β₁x₁ + β₂x₂  
λ = e^(β₀+β₁x₁+β₂x₂)
```
- e^β₁: Rate ratio for unit increase in x₁
- β₁: Change in log-rate per unit increase in x₁

## Examples

### Example 1: Logistic Regression
**Problem:** Predict probability of college admission based on GPA and SAT score.

```r
# Simulate data
set.seed(123)
n <- 1000
gpa <- rnorm(n, 3.0, 0.5)
sat <- rnorm(n, 1200, 200)

# Linear predictor
eta <- -10 + 2*gpa + 0.003*sat
prob <- 1/(1 + exp(-eta))  # Inverse logit
admit <- rbinom(n, 1, prob)

# Fit model
model <- glm(admit ~ gpa + sat, family = binomial(link = "logit"))
summary(model)

# Interpretation
exp(coef(model))  # Odds ratios
```

**Output interpretation:**
- Intercept odds: e^β₀
- GPA odds ratio: e^β₁ (odds multiply by this for each GPA point)
- SAT odds ratio: e^β₂ (odds multiply by this for each SAT point)

### Example 2: Poisson Regression
**Problem:** Model number of customer complaints per day based on temperature and day of week.

```r
# Simulate data
set.seed(456)
n <- 365
temp <- rnorm(n, 70, 15)
weekend <- rbinom(n, 1, 2/7)  # Weekend indicator

# Linear predictor  
eta <- 1 + 0.02*temp - 0.5*weekend
lambda <- exp(eta)
complaints <- rpois(n, lambda)

# Fit model
model <- glm(complaints ~ temp + weekend, family = poisson(link = "log"))
summary(model)

# Interpretation
exp(coef(model))  # Rate ratios
```

**Output interpretation:**
- Baseline rate: e^β₀ complaints when temp=0, weekday
- Temperature effect: e^β₁ rate ratio per degree
- Weekend effect: e^β₂ rate ratio for weekends vs weekdays

## Model Assessment

### Deviance
Generalization of sum of squared errors:
```
D = 2[ℓ(saturated) - ℓ(fitted)]
```

**Null deviance:** Compare to intercept-only model
**Residual deviance:** Goodness of fit measure

### Pearson Chi-Square
```
X² = Σᵢ (yᵢ - μᵢ)²/V(μᵢ)
```

Where V(μᵢ) is the variance function.

### Residuals

#### Pearson Residuals
```
rᵢᴾ = (yᵢ - μᵢ)/√V(μᵢ)
```

#### Deviance Residuals  
```
rᵢᴰ = sign(yᵢ - μᵢ)√dᵢ
```

Where dᵢ is the contribution to deviance.

#### Standardized Residuals
```
rᵢˢ = rᵢ/√(1 - hᵢᵢ)
```

Where hᵢᵢ is the i-th diagonal element of hat matrix.

## Diagnostics and Model Checking

### Residual Plots
```r
# Fit model
model <- glm(y ~ x1 + x2, family = poisson)

# Diagnostic plots
par(mfrow = c(2,2))

# Residuals vs fitted
plot(fitted(model), residuals(model, type = "pearson"))
abline(h = 0)

# Q-Q plot of residuals  
qqnorm(residuals(model, type = "deviance"))
qqline(residuals(model, type = "deviance"))

# Scale-location plot
plot(fitted(model), sqrt(abs(residuals(model, type = "pearson"))))

# Residuals vs leverage
plot(hatvalues(model), residuals(model, type = "pearson"))
```

### Overdispersion
When Var(Y) > E[Y] in Poisson models:

**Detection:**
```r
# Pearson chi-square test
pearson_chisq <- sum(residuals(model, type = "pearson")^2)
df <- df.residual(model)
dispersion <- pearson_chisq / df
dispersion  # Should be ≈ 1 for Poisson
```

**Solutions:**
- Quasi-Poisson: Allow dispersion parameter φ ≠ 1
- Negative binomial: Add random effect

## Information Criteria

### AIC (Akaike Information Criterion)
```
AIC = -2ℓ + 2p
```

### BIC (Bayesian Information Criterion)  
```
BIC = -2ℓ + p log(n)
```

**Model selection:** Lower values indicate better fit.

## Advantages and Limitations

### Advantages
1. **Flexible**: Handles various response distributions
2. **Interpretable**: Clear coefficient meanings
3. **Well-established**: Mature theory and software
4. **Unified framework**: Common approach for different problems

### Limitations  
1. **Linear predictor**: Still assumes linear relationships
2. **Distribution assumptions**: Must specify correct family
3. **Independence**: Assumes independent observations
4. **Outlier sensitivity**: Can be affected by extreme values

## Key Takeaways

1. **GLMs extend linear regression** to exponential family distributions
2. **Three components**: Random, systematic, and link function
3. **Link functions** ensure predictions stay in valid range
4. **Canonical links** have nice theoretical properties
5. **Maximum likelihood** provides parameter estimates
6. **Interpretation** depends on link function choice
7. **Model checking** is crucial for valid inference

GLMs provide a powerful framework for modeling non-normal responses while maintaining many advantages of linear models. Next, we'll explore specific GLM applications and extensions.