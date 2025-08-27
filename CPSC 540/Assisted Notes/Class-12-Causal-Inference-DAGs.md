# Class 12: Causal Inference - DAGs
**Date:** October 6, 2025 (Monday)  
**Topics:** DAGs + Common Issues + Experiments  
**Reading:** Causal Inference in R Chapter 5

## Overview
Deep dive into causal inference using Directed Acyclic Graphs (DAGs), understanding confounding, collider bias, and principles of experimental design for causal discovery.

## Causal vs. Statistical Thinking

### The Fundamental Problem
**Correlation ≠ Causation**

Statistical methods excel at finding associations, but causation requires additional assumptions and careful study design.

**Example:** Ice cream sales and drowning deaths are correlated, but ice cream doesn't cause drowning. Both are caused by hot weather (confounding).

### Counterfactual Framework
**Potential Outcomes Model (Rubin Causal Model):**

For each unit i and treatment t:
- Y_i(1): Potential outcome if treated  
- Y_i(0): Potential outcome if not treated
- **Individual causal effect:** Y_i(1) - Y_i(0)

**The Fundamental Problem:** We can only observe one potential outcome for each unit.

**Average Treatment Effect (ATE):**
```
ATE = E[Y_i(1) - Y_i(0)] = E[Y_i(1)] - E[Y_i(0)]
```

## Causal Graphs and DAGs

### Representing Causal Relationships
A DAG represents causal assumptions:
- **Nodes**: Variables
- **Directed edges**: Direct causal relationships  
- **Absence of edges**: No direct causal relationship

**Example: Education and Income**
```
Ability → Education → Income
   ↘                 ↗
    └─── Family SES ─┘
```

### Types of Causal Relationships

#### 1. Chains (Mediation)
```
X → M → Y
```
- X affects Y through mediator M
- **Total effect** = Direct effect + Indirect effect
- **Controlling for M** blocks the causal path

**Example: Education → Skills → Income**
- Education affects income partly through improving skills
- Controlling for skills removes the mediated effect

#### 2. Forks (Confounding)
```
    Z
   ↙ ↘
  X   Y  
```
- Z is a common cause of both X and Y
- Creates **spurious association** between X and Y
- **Must control for Z** to get unbiased causal effect

**Example: Hospital Quality Study**
```
  Severity
    ↙  ↘
Hospital → Mortality
Quality
```

#### 3. Colliders (Selection Bias)
```
  X   Y
   ↘ ↙
    Z
```
- X and Y both cause Z
- X and Y are **independent marginally**
- X and Y become **dependent when conditioning on Z**

**Example: Beauty and Talent in Hollywood**
```
Beauty   Talent
   ↘      ↙
   Celebrity
```
Among celebrities (conditioning on collider), beauty and talent are negatively correlated!

### d-Separation Rules

#### Path Blocking
A path is blocked if it contains:
1. **Chain or Fork**: A node that is conditioned on (in adjustment set)
2. **Collider**: A collider that is NOT conditioned on (and no descendant is conditioned on)

#### d-Separation Algorithm
Path P is d-separated by set S if:
- **Every non-collider** on P is in S, OR
- **Some collider** on P is not in S and no descendant of that collider is in S

**If all paths between X and Y are d-separated by S, then X ⊥ Y | S**

### Example: Simpson's Paradox
**Scenario:** Drug appears harmful overall but beneficial in each subgroup.

**DAG:**
```
Gender → Drug → Recovery
   ↘                ↗
    └─── Disease ───┘
    Severity
```

**Data:**
```
Overall: Drug group has 30% recovery, Control has 40%
Males: Drug 60% vs Control 50% 
Females: Drug 20% vs Control 10%
```

**Explanation:** Gender affects both drug assignment (men more likely to get drug) and baseline recovery (men recover better). Must stratify by gender.

## Identifying Causal Effects

### The Backdoor Criterion
To identify causal effect of X on Y, find adjustment set S such that:
1. **No descendant of X** is in S
2. **S blocks all backdoor paths** from X to Y (paths with arrow into X)

**Backdoor paths:** Paths from X to Y that begin with arrow into X (X ← ... → Y)

#### Example: Education and Income
```
Ability → Education → Income
   ↘                  ↗
    └─── Motivation ──┘
```

**Backdoor paths:**
- Education ← Ability → Motivation → Income
- Education ← Ability → Income (if direct arrow exists)

**Adjustment set:** {Ability} blocks all backdoor paths.

### The Frontdoor Criterion
When backdoor criterion fails due to unmeasured confounding:

**Requirements:**
1. **S intercepts all directed paths** from X to Y
2. **No backdoor paths** from X to S
3. **All backdoor paths** from S to Y are blocked by X

**Example: Smoking and Cancer**
```
      Genotype (unmeasured)
         ↙         ↘
   Smoking → Tar → Cancer
```
- Can't measure genotype (confounder)
- Can measure tar deposits (mediator)
- Use frontdoor criterion with mediator set {Tar}

### Instrumental Variables
When no valid adjustment set exists:

**IV Requirements:**
1. **Relevance**: Z affects X (Cov(Z,X) ≠ 0)
2. **Exclusion**: Z affects Y only through X
3. **Exchangeability**: Z is randomized (or as-good-as-random)

**Example: Returns to Education**
```
      Ability (unmeasured)
         ↙         ↘
   Education → Income
      ↑
 Compulsory
 School Laws
```

**IV Estimand:**
```
β_IV = Cov(Y,Z) / Cov(X,Z)
```

## Common Causal Fallacies

### 1. Confounding Bias
**Problem:** Omitting common causes

**Example:** Coffee and heart disease
```
    Smoking
    ↙    ↘
Coffee → Heart Disease
```
Coffee appears harmful until controlling for smoking.

### 2. Collider Bias
**Problem:** Conditioning on colliders

**Example:** Hospital mortality study
```
Disease   Treatment
Severity   Quality
    ↘       ↙
   Mortality
```
Comparing hospitals based on patients who died creates selection bias.

### 3. Overcontrol Bias  
**Problem:** Controlling for mediators or descendants

**Example:** Discrimination study
```
Race → Hiring → Performance
  ↘             ↙
   Education ──┘
```
Controlling for education (mediator) underestimates discrimination effect.

### 4. M-bias
**Problem:** Controlling for common effect of unmeasured variables

```
   U₁ → X → Y ← U₂
    ↘         ↙
      └─ C ─┘
```
Controlling for C creates bias even though C is not on causal path.

## Experimental Design

### Randomized Controlled Trials (RCTs)

#### Why Randomization Works
**Random assignment** breaks confounding paths:
```
Before:     After randomization:
Z → X → Y      Z   X → Y
  ↘ ↗           (no arrow from Z to X)
   Y
```

#### Types of Randomization

**Simple Randomization:**
```r
treatment <- rbinom(n, 1, 0.5)
```

**Block Randomization:**
Randomize within blocks of similar units:
```r
# Block by gender
males <- which(gender == "M")
females <- which(gender == "F")

treatment[males] <- sample(c(rep(1, length(males)/2), 
                           rep(0, length(males)/2)))
treatment[females] <- sample(c(rep(1, length(females)/2), 
                             rep(0, length(females)/2)))
```

**Stratified Randomization:**
Ensure balance on important covariates:
```r
library(randomizr)
treatment <- block_ra(blocks = age_group, prob = 0.5)
```

### Natural Experiments
When randomization is not possible/ethical:

#### Regression Discontinuity
Assignment based on cutoff:
```
Treatment = 1 if Score ≥ Cutoff, 0 otherwise
```

**Example:** Class size reduction
- Schools with enrollment ≥ 40 get extra teacher
- Compare schools just above/below cutoff

#### Difference-in-Differences
Compare treatment vs. control, before vs. after:
```
Y_it = α + β_1 Treatment_i + β_2 Post_t + β_3 (Treatment_i × Post_t) + ε_it
```

**Example:** Minimum wage increase
- Treatment: States that raised minimum wage
- Control: States that didn't
- Compare employment changes

#### Instrumental Variables
Use quasi-random variation:
- **Distance to college** as IV for education
- **Quarter of birth** as IV for school completion
- **Judge assignment** as IV for incarceration

## Implementation in R

### DAG Analysis with dagitty
```r
library(dagitty)
library(ggdag)

# Define DAG
dag <- dagitty("dag {
    X -> Y
    Z -> X
    Z -> Y
    W -> Y
}")

# Visualize
ggdag(dag) + theme_dag()

# Find adjustment sets
adjustmentSets(dag, exposure = "X", outcome = "Y")

# Check implications
impliedConditionalIndependencies(dag)
```

### Testing DAG Implications
```r
# Simulate data according to DAG
simulate_dag_data <- function(n = 1000) {
    Z <- rnorm(n)
    W <- rnorm(n)
    X <- 0.5 * Z + rnorm(n, 0, 0.5)
    Y <- 0.3 * X + 0.7 * Z + 0.2 * W + rnorm(n, 0, 0.3)
    
    data.frame(X = X, Y = Y, Z = Z, W = W)
}

data <- simulate_dag_data()

# Test implied independence
# Should find: W ⊥ X | Z
cor.test(data$W, data$X)  # Should be significant
library(ppcor)
pcor.test(data$W, data$X, data$Z)  # Should be non-significant
```

### Causal Effect Estimation
```r
# Naive (biased) estimate
naive_model <- lm(Y ~ X, data = data)

# Adjusted (unbiased) estimate  
adjusted_model <- lm(Y ~ X + Z, data = data)

# Compare estimates
cbind(
    naive = coef(naive_model)["X"],
    adjusted = coef(adjusted_model)["X"],
    true = 0.3
)
```

### Sensitivity Analysis
```r
library(sensemakr)

# Fit model
model <- lm(Y ~ X + Z, data = data)

# Sensitivity analysis
sensitivity <- sensemakr(model, treatment = "X", benchmark_covariates = "Z")

# Plot sensitivity
plot(sensitivity)
```

## Advanced Topics

### Mediation Analysis
**Direct vs. Indirect Effects:**

**Total Effect:** X → Y
**Direct Effect:** X → Y (controlling for mediator)
**Indirect Effect:** X → M → Y

```r
library(mediation)

# Fit models
mediator_model <- lm(M ~ X + Z, data = data)
outcome_model <- lm(Y ~ X + M + Z, data = data)

# Mediation analysis
mediation_result <- mediate(mediator_model, outcome_model, 
                          treat = "X", mediator = "M")
summary(mediation_result)
```

### Time-Varying Confounding
When confounders change over time and are affected by prior treatment:

**G-methods:**
- G-computation
- Inverse probability weighting
- G-estimation

### Principal Stratification
When treatment affects who we observe (e.g., survival):
- **Always-takers**: Would always take treatment
- **Never-takers**: Would never take treatment  
- **Compliers**: Follow assigned treatment

## Model Selection for Causal Inference

### Causal vs. Predictive Modeling
**Predictive models:** Include all variables that improve prediction
**Causal models:** Include only variables required by causal theory

**Example:**
```
# For prediction: include everything
predict_model <- lm(Y ~ X + Z + W + V + U)

# For causation: only include adjustment set
causal_model <- lm(Y ~ X + Z)  # If {Z} satisfies backdoor criterion
```

### Cross-Validation Warning
**Don't use CV** to select adjustment sets for causal inference:
- May select models that don't satisfy causal assumptions
- Prediction accuracy ≠ causal validity

### Robustness Checks
1. **Try different adjustment sets** (if multiple valid sets exist)
2. **Test sensitivity** to unmeasured confounding
3. **Use multiple methods** (matching, weighting, regression)
4. **Check balance** after adjustment

## Key Takeaways

1. **Causal inference requires assumptions** beyond statistical models
2. **DAGs help formalize** causal assumptions and guide analysis
3. **Three main structures**: chains, forks, colliders
4. **d-separation determines** conditional independence relationships
5. **Backdoor criterion** identifies valid adjustment sets
6. **Common biases**: confounding, collider bias, overcontrol
7. **Randomization eliminates** confounding but isn't always possible
8. **Natural experiments** can provide causal leverage
9. **Sensitivity analysis** tests robustness of conclusions

Understanding causal inference is crucial for making valid policy recommendations and scientific discoveries from observational data. The DAG framework provides a principled approach to causal reasoning in statistical machine learning.