# Class 10: MCMC - Monte Carlo Methods
**Date:** September 29, 2025 (Monday)  
**Topics:** Monte Carlo, Metropolis-Hastings  
**Reading:** Bayes Rules! Chapter 7

## Overview
Introduction to Monte Carlo methods and Markov Chain Monte Carlo (MCMC) for Bayesian inference, focusing on sampling from complex posterior distributions.

## The Need for MCMC

### Bayesian Inference Challenge
In Bayesian inference, we want the posterior distribution:
```
p(θ|y) = p(y|θ)p(θ) / p(y)
```

**The problem:** p(y) = ∫ p(y|θ)p(θ)dθ is often intractable.

**Examples where integration is difficult:**
- High-dimensional parameter spaces
- Non-conjugate priors
- Complex likelihood functions
- Hierarchical models

### Analytical Solutions (Limited Cases)
**Conjugate priors** allow closed-form solutions:
- Beta-Binomial
- Normal-Normal
- Gamma-Poisson

**Example: Beta-Binomial**
```
Prior:     θ ~ Beta(α, β)
Likelihood: y ~ Binomial(n, θ)
Posterior:  θ|y ~ Beta(α + y, β + n - y)
```

But most real problems require numerical methods.

## Monte Carlo Methods

### Basic Monte Carlo Integration
To estimate integral I = ∫ g(x)f(x)dx:

1. **Generate samples** x₁, x₂, ..., xₙ from distribution f(x)
2. **Estimate integral** as: Î = (1/n)Σᵢ g(xᵢ)

**Law of Large Numbers:** Î → I as n → ∞

**Standard error:** SE(Î) ≈ √(Var[g(X)]/n)

### Example: Estimating π
```
# Estimate π using unit circle
# π = 4 × P(X² + Y² ≤ 1) where X,Y ~ Uniform(-1,1)

n <- 10000
x <- runif(n, -1, 1)
y <- runif(n, -1, 1)
inside <- (x^2 + y^2 <= 1)
pi_estimate <- 4 * mean(inside)
print(pi_estimate)  # Should be ≈ 3.14159
```

### Importance Sampling
When sampling directly from f(x) is difficult:

1. **Choose proposal distribution** g(x) (easy to sample from)
2. **Compute weights** wᵢ = f(xᵢ)/g(xᵢ) for samples xᵢ ~ g
3. **Estimate integral** as: Î = Σᵢ wᵢg(xᵢ) / Σᵢ wᵢ

**Key requirement:** g(x) > 0 whenever f(x)g(x) > 0

## Markov Chain Monte Carlo (MCMC)

### Basic Idea
Instead of independent samples, create a **Markov chain** whose stationary distribution is the target distribution p(θ|y).

### Markov Chain Properties

#### Definition
A sequence θ⁽⁰⁾, θ⁽¹⁾, θ⁽²⁾, ... where:
```
P(θ⁽ᵗ⁺¹⁾|θ⁽ᵗ⁾, θ⁽ᵗ⁻¹⁾, ..., θ⁽⁰⁾) = P(θ⁽ᵗ⁺¹⁾|θ⁽ᵗ⁾)
```

#### Transition Kernel
Probability of moving from state θ to state φ:
```
K(θ → φ) = P(θ⁽ᵗ⁺¹⁾ = φ | θ⁽ᵗ⁾ = θ)
```

#### Stationary Distribution
Distribution π is stationary if:
```
π(φ) = ∫ π(θ)K(θ → φ)dθ
```

### Detailed Balance Condition
Sufficient condition for π to be stationary:
```
π(θ)K(θ → φ) = π(φ)K(φ → θ) for all θ,φ
```

**Interpretation:** Flow from θ to φ equals flow from φ to θ.

### Ergodic Theory Results

#### Convergence
Under regularity conditions:
```
(1/n)Σₜ g(θ⁽ᵗ⁾) → ∫ g(θ)π(θ)dθ as n → ∞
```

#### Central Limit Theorem for MCMC
```
√n[(1/n)Σₜ g(θ⁽ᵗ⁾) - E_π[g(θ)]] →ᵈ N(0, σ²)
```

Where σ² accounts for autocorrelation.

## Metropolis-Hastings Algorithm

### General Algorithm
**Goal:** Sample from target distribution π(θ) ∝ p(θ|y)

**Steps:**
1. **Initialize** θ⁽⁰⁾
2. **For t = 0, 1, 2, ...**:
   - **Propose** φ ~ q(·|θ⁽ᵗ⁾)
   - **Compute acceptance ratio** α = min(1, (π(φ)q(θ⁽ᵗ⁾|φ))/(π(θ⁽ᵗ⁾)q(φ|θ⁽ᵗ⁾)))
   - **Accept** with probability α:
     - If accept: θ⁽ᵗ⁺¹⁾ = φ
     - If reject: θ⁽ᵗ⁺¹⁾ = θ⁽ᵗ⁾

### Key Components

#### Proposal Distribution q(φ|θ)
**Random walk:** φ = θ + ε, where ε ~ N(0, σ²)
- **Small σ**: High acceptance, slow mixing
- **Large σ**: Low acceptance, fast mixing when accepted
- **Optimal acceptance rate**: ≈ 23% for high dimensions

#### Acceptance Ratio
For Bayesian posterior π(θ) ∝ p(y|θ)p(θ):
```
α = min(1, [p(y|φ)p(φ)q(θ|φ)] / [p(y|θ)p(θ)q(φ|θ)])
```

### Special Cases

#### Random Walk Metropolis
Symmetric proposal: q(φ|θ) = q(θ|φ)
```
α = min(1, π(φ)/π(θ))
```

#### Independence Sampler
Proposal independent of current state: q(φ|θ) = g(φ)
```
α = min(1, [π(φ)g(θ)] / [π(θ)g(φ)])
```

### Example: Normal Distribution
**Target:** θ ~ N(μ, σ²)
**Proposal:** Random walk with step size s

```r
# Metropolis sampler for Normal(5, 2^2)
metropolis_normal <- function(n_iter, mu = 5, sigma = 2, step_size = 1) {
    samples <- numeric(n_iter)
    current <- 0  # Starting value
    n_accept <- 0
    
    for (i in 1:n_iter) {
        # Propose new value
        proposal <- current + rnorm(1, 0, step_size)
        
        # Compute acceptance ratio (log scale for stability)
        log_alpha <- dnorm(proposal, mu, sigma, log = TRUE) - 
                     dnorm(current, mu, sigma, log = TRUE)
        alpha <- exp(min(0, log_alpha))
        
        # Accept or reject
        if (runif(1) < alpha) {
            current <- proposal
            n_accept <- n_accept + 1
        }
        samples[i] <- current
    }
    
    list(samples = samples, acceptance_rate = n_accept/n_iter)
}

# Run sampler
result <- metropolis_normal(10000, step_size = 2)
cat("Acceptance rate:", result$acceptance_rate, "\n")

# Check results
hist(result$samples, breaks = 50, freq = FALSE)
curve(dnorm(x, 5, 2), add = TRUE, col = "red", lwd = 2)
```

## Advanced MCMC Methods

### Gibbs Sampling
**Special case** when we can sample from full conditionals.

For multivariate θ = (θ₁, θ₂, ..., θₚ):
1. **Sample** θ₁⁽ᵗ⁺¹⁾ ~ p(θ₁|θ₂⁽ᵗ⁾, θ₃⁽ᵗ⁾, ..., θₚ⁽ᵗ⁾, y)
2. **Sample** θ₂⁽ᵗ⁺¹⁾ ~ p(θ₂|θ₁⁽ᵗ⁺¹⁾, θ₃⁽ᵗ⁾, ..., θₚ⁽ᵗ⁾, y)
3. **Continue** for all parameters

**Advantage:** Always accepts (no rejection)
**Disadvantage:** Requires conjugate structure or numerical integration

#### Example: Bivariate Normal
**Target:** (θ₁, θ₂) ~ N(μ, Σ) with correlation ρ

**Full conditionals:**
```
θ₁|θ₂ ~ N(μ₁ + ρ(σ₁/σ₂)(θ₂ - μ₂), σ₁²(1 - ρ²))
θ₂|θ₁ ~ N(μ₂ + ρ(σ₂/σ₁)(θ₁ - μ₁), σ₂²(1 - ρ²))
```

### Block Updating
Update multiple parameters simultaneously:
- **Better mixing** when parameters are correlated
- **More complex proposals** but potentially more efficient

### Adaptive MCMC
Adjust proposal distribution during sampling:
```r
# Example: Adaptive random walk
if (t %% 100 == 0) {  # Every 100 iterations
    if (acceptance_rate > 0.44) {
        step_size <- step_size * 1.1
    } else if (acceptance_rate < 0.23) {
        step_size <- step_size * 0.9
    }
}
```

## Hamiltonian Monte Carlo (HMC)

### Basic Idea
Use **gradient information** to propose distant states with high acceptance probability.

### Hamiltonian Dynamics
Introduce **momentum variables** p:
```
H(θ, p) = -log π(θ) + (1/2)p'M⁻¹p
```

**Hamilton's equations:**
```
dθ/dt = ∂H/∂p = M⁻¹p
dp/dt = -∂H/∂θ = ∇log π(θ)
```

### Leapfrog Integration
Numerical integration of Hamiltonian dynamics:
```
p_{t+ε/2} = p_t + (ε/2)∇log π(θ_t)
θ_{t+ε} = θ_t + ε M⁻¹ p_{t+ε/2}
p_{t+ε} = p_{t+ε/2} + (ε/2)∇log π(θ_{t+ε})
```

### No-U-Turn Sampler (NUTS)
**Automatically tunes** step size and number of leapfrog steps.
- **Implemented in Stan**: Popular probabilistic programming language
- **Much more efficient** than random walk for high-dimensional problems

## MCMC Diagnostics

### Trace Plots
Plot θ⁽ᵗ⁾ vs. t to visualize:
- **Convergence**: Chain reaches stationary distribution
- **Mixing**: Chain explores parameter space efficiently

```r
# Trace plot
plot(samples, type = "l", main = "Trace Plot")
abline(h = true_value, col = "red", lwd = 2)
```

### Running Averages
Plot (1/t)Σₛ₌₁ᵗ θ⁽ˢ⁾ to check convergence:
```r
running_mean <- cumsum(samples) / seq_along(samples)
plot(running_mean, type = "l", main = "Running Average")
```

### Effective Sample Size
Account for autocorrelation:
```
n_eff = n / (1 + 2Σₖ₌₁^∞ ρₖ)
```

Where ρₖ is lag-k autocorrelation.

### R̂ (R-hat) Statistic
Compare **within-chain** and **between-chain** variance:
```
R̂ = √[(n-1)/n + (1/n)(B/W)]
```

**Rule of thumb:** R̂ < 1.1 indicates convergence.

### Monte Carlo Standard Error
Standard error of MCMC estimate:
```
MCSE = √(Var(θ)/n_eff)
```

## Practical Implementation

### Using RStan
```r
library(rstan)

# Stan model code
stan_code <- "
data {
    int<lower=0> N;
    vector[N] y;
    vector[N] x;
}
parameters {
    real alpha;
    real beta;
    real<lower=0> sigma;
}
model {
    // Priors
    alpha ~ normal(0, 10);
    beta ~ normal(0, 10);
    sigma ~ cauchy(0, 5);
    
    // Likelihood
    y ~ normal(alpha + beta * x, sigma);
}
"

# Prepare data
stan_data <- list(N = length(y), y = y, x = x)

# Fit model
fit <- stan(model_code = stan_code, data = stan_data, 
           chains = 4, iter = 2000, warmup = 1000)

# Diagnostics
print(fit)
traceplot(fit)
stan_diag(fit)
```

### Tuning Parameters

#### Warmup/Burn-in Period
- **Remove initial samples** before convergence
- **Typical choice**: 50% of total iterations

#### Number of Chains
- **Multiple chains** help detect convergence problems
- **Typical choice**: 3-4 chains

#### Thinning
- **Keep every k-th sample** to reduce autocorrelation
- **Trade-off**: Reduces computational storage vs. information loss

## Common Problems and Solutions

### Poor Mixing
**Symptoms:**
- High autocorrelation
- Low effective sample size
- Slow convergence

**Solutions:**
- Adjust proposal variance
- Use block updating
- Try different parameterization
- Use HMC/NUTS

### Non-convergence
**Symptoms:**
- R̂ > 1.1
- Trace plots show trends
- Different chains in different regions

**Solutions:**
- Run longer chains
- Check for label switching
- Improve parameterization
- Use better starting values

### Label Switching
In mixture models, chains can switch between equivalent parameterizations.

**Solution:**
- Post-process to align labels
- Use identifiability constraints

## Key Takeaways

1. **MCMC enables Bayesian inference** for complex models
2. **Metropolis-Hastings** is the fundamental algorithm
3. **Proposal tuning** is crucial for efficiency
4. **Gibbs sampling** works when full conditionals are available
5. **HMC/NUTS** are much more efficient for smooth, high-dimensional problems
6. **Diagnostics** are essential to ensure valid inference
7. **Modern software** (Stan, JAGS) makes MCMC accessible

MCMC has revolutionized Bayesian statistics by making complex models tractable. Understanding these fundamentals prepares you for advanced Bayesian modeling in statistical machine learning.