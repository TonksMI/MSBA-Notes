# Class 1: Math Review + Introduction
**Date:** August 27, 2025 (Wednesday)  
**Quiz:** Yes (covering basic math concepts)

## Overview
Comprehensive review of essential mathematical concepts: linear algebra, probability theory, and statistics fundamentals needed for statistical machine learning.

## Linear Algebra Review

### Vector Spaces and Operations

#### Vector Basics
A vector $\mathbf{v}$ in $\mathbb{R}^n$ represents a point or direction in n-dimensional space:
$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$$

**Key Operations:**
- **Addition**: $\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1+v_1 \\ u_2+v_2 \\ \vdots \\ u_n+v_n \end{bmatrix}$
- **Scalar multiplication**: $c\mathbf{v} = \begin{bmatrix} cv_1 \\ cv_2 \\ \vdots \\ cv_n \end{bmatrix}$
- **Dot product**: $\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i$

**Example in ML Context:**
```
Feature vector: x = [age, income, education_years]ᵀ = [25, 50000, 16]ᵀ
Weight vector:  w = [0.1, 0.0001, 0.5]ᵀ
Prediction:     ŷ = w·x = 0.1(25) + 0.0001(50000) + 0.5(16) = 15.5
```

#### Vector Norms
**L2 Norm (Euclidean length):**
$$\|\mathbf{v}\|_2 = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2} = \sqrt{\mathbf{v} \cdot \mathbf{v}}$$

**L1 Norm (Manhattan distance):**
$$\|\mathbf{v}\|_1 = |v_1| + |v_2| + \cdots + |v_n|$$

**ML Application:** Regularization
- **Ridge regression**: Minimize $\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda\|\boldsymbol{\beta}\|_2^2$
- **Lasso regression**: Minimize $\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda\|\boldsymbol{\beta}\|_1$

### Matrix Operations

#### Matrix-Vector Multiplication
For matrix A (m×n) and vector x (n×1):
```
Ax = [a₁₁x₁ + a₁₂x₂ + ... + a₁ₙxₙ]
     [a₂₁x₁ + a₂₂x₂ + ... + a₂ₙxₙ]
     [⋮                           ]
     [aₘ₁x₁ + aₘ₂x₂ + ... + aₘₙxₙ]
```

**Linear Regression Example:**
```
Design matrix X (n×p), coefficient vector β (p×1)
Predictions: ŷ = Xβ

X = [1  x₁₁  x₁₂]     β = [β₀]     ŷ = [β₀ + β₁x₁₁ + β₂x₁₂]
    [1  x₂₁  x₂₂]         [β₁]         [β₀ + β₁x₂₁ + β₂x₂₂]
    [⋮  ⋮    ⋮  ]         [β₂]         [⋮                  ]
    [1  xₙ₁  xₙ₂]                      [β₀ + β₁xₙ₁ + β₂xₙ₂]
```

#### Matrix Multiplication
For A (m×k) and B (k×n), product C = AB is (m×n):
```
cᵢⱼ = Σₖ aᵢₖbₖⱼ
```

**Computational Complexity:** O(mkn)

### Special Matrices and Properties

#### Identity Matrix
```
I = [1  0  0]
    [0  1  0]  ⟹ AI = IA = A
    [0  0  1]
```

#### Matrix Transpose
```
If A = [1  2  3]  then Aᵀ = [1  4]
       [4  5  6]            [2  5]
                            [3  6]
```

**Properties:**
- (Aᵀ)ᵀ = A
- (AB)ᵀ = BᵀAᵀ
- (A + B)ᵀ = Aᵀ + Bᵀ

#### Matrix Inverse
For square matrix A, inverse A⁻¹ satisfies AA⁻¹ = A⁻¹A = I

**Normal Equation in Regression:**
```
β̂ = (XᵀX)⁻¹Xᵀy
```

**Warning:** XᵀX must be invertible (full rank condition)

### Eigenvalues and Eigenvectors

#### Definition
For square matrix A, eigenvector v and eigenvalue λ satisfy:
```
Av = λv
```

**Geometric Interpretation:** A stretches v by factor λ without changing direction.

#### Computing Eigenvalues
Solve characteristic equation:
```
det(A - λI) = 0
```

**Example:**
```
A = [3  1]
    [0  2]

det([3-λ   1  ]) = (3-λ)(2-λ) = λ² - 5λ + 6 = 0
   ([0   2-λ])

λ₁ = 3, λ₂ = 2
```

#### Principal Component Analysis (PCA)
PCA finds eigenvectors of covariance matrix:
```
Cov(X) = (1/(n-1))XᵀX
```
- **Principal components** = eigenvectors
- **Explained variance** = eigenvalues

### Matrix Decompositions

#### Eigenvalue Decomposition
For symmetric matrix A:
```
A = QΛQᵀ
```
where Q contains eigenvectors, Λ contains eigenvalues.

#### Singular Value Decomposition (SVD)
For any matrix A (m×n):
```
A = UΣVᵀ
```
where:
- U (m×m): Left singular vectors (orthogonal)
- Σ (m×n): Diagonal matrix of singular values
- V (n×n): Right singular vectors (orthogonal)

**ML Applications:**
- **Dimensionality reduction**: Keep top k singular values
- **Matrix completion**: Reconstruct missing entries
- **Pseudoinverse**: A⁺ = VΣ⁺Uᵀ

## Probability Theory Review

### Basic Probability

#### Sample Space and Events
- **Sample space Ω**: All possible outcomes
- **Event A**: Subset of Ω
- **Probability P(A)**: Measure of likelihood, 0 ≤ P(A) ≤ 1

#### Conditional Probability
```
P(A|B) = P(A ∩ B) / P(B)
```

**Chain Rule:**
```
P(A₁, A₂, ..., Aₙ) = P(A₁)P(A₂|A₁)P(A₃|A₁,A₂)...P(Aₙ|A₁,...,Aₙ₋₁)
```

### Bayes' Theorem

#### Formula
```
P(H|D) = P(D|H)P(H) / P(D)
```

**Components:**
- **P(H|D)**: Posterior probability (what we want)
- **P(D|H)**: Likelihood (how well hypothesis explains data)
- **P(H)**: Prior probability (initial belief)
- **P(D)**: Evidence (normalizing constant)

#### Medical Diagnosis Example
```
Disease prevalence: P(Disease) = 0.001
Test sensitivity: P(+|Disease) = 0.99
Test specificity: P(-|No Disease) = 0.95

P(Disease|+) = P(+|Disease)P(Disease) / P(+)
             = (0.99)(0.001) / [(0.99)(0.001) + (0.05)(0.999)]
             = 0.00099 / (0.00099 + 0.04995)
             = 0.0194
```

**Surprising Result:** Only 1.94% chance of having disease despite positive test!

### Random Variables and Distributions

#### Discrete Random Variables
**Probability Mass Function (PMF):**
```
p(x) = P(X = x)
```

**Bernoulli Distribution:**
```
X ~ Bernoulli(p)
P(X = 1) = p, P(X = 0) = 1-p
E[X] = p, Var(X) = p(1-p)
```

**Binomial Distribution:**
```
X ~ Binomial(n, p)
P(X = k) = C(n,k)pᵏ(1-p)ⁿ⁻ᵏ
E[X] = np, Var(X) = np(1-p)
```

#### Continuous Random Variables
**Probability Density Function (PDF):**
```
P(a ≤ X ≤ b) = ∫ₐᵇ f(x)dx
```

**Normal Distribution:**
```
X ~ N(μ, σ²)
f(x) = (1/√(2πσ²))exp(-(x-μ)²/(2σ²))
```

**Standard Normal:** Z ~ N(0,1)
**Standardization:** Z = (X - μ)/σ

### Expectation and Variance

#### Expected Value
**Discrete:** E[X] = Σₓ x·P(X = x)
**Continuous:** E[X] = ∫ x·f(x)dx

**Properties:**
- E[aX + b] = aE[X] + b (linearity)
- E[X + Y] = E[X] + E[Y] (additivity)

#### Variance
```
Var(X) = E[(X - μ)²] = E[X²] - (E[X])²
```

**Properties:**
- Var(aX + b) = a²Var(X)
- If X ⊥ Y: Var(X + Y) = Var(X) + Var(Y)

### Central Limit Theorem

#### Statement
For independent, identically distributed X₁, X₂, ..., Xₙ with mean μ and variance σ²:
```
(X̄ - μ)/(σ/√n) →ᵈ N(0,1) as n → ∞
```

**Practical Implication:** Sample means are approximately normal for large n, regardless of population distribution.

**Example:** Rolling dice
```
Population: Uniform{1,2,3,4,5,6}
μ = 3.5, σ² = 35/12

Sample mean of n=100 rolls:
X̄ ~ N(3.5, 35/(12×100)) = N(3.5, 0.0292)
```

## Statistics Fundamentals

### Maximum Likelihood Estimation

#### Concept
Find parameter values that make observed data most likely.

**Likelihood Function:**
```
L(θ) = ∏ᵢ f(xᵢ|θ)
```

**Log-likelihood:**
```
ℓ(θ) = log L(θ) = Σᵢ log f(xᵢ|θ)
```

#### Normal Distribution Example
For X₁, ..., Xₙ ~ N(μ, σ²):
```
ℓ(μ,σ²) = -n/2 log(2πσ²) - (1/(2σ²))Σᵢ(xᵢ-μ)²

∂ℓ/∂μ = 0 ⟹ μ̂ = (1/n)Σᵢxᵢ = x̄
∂ℓ/∂σ² = 0 ⟹ σ̂² = (1/n)Σᵢ(xᵢ-μ̂)²
```

### Confidence Intervals

#### For Population Mean (Known σ)
```
x̄ ± z_{α/2} × (σ/√n)
```

#### For Population Mean (Unknown σ)
```
x̄ ± t_{α/2,n-1} × (s/√n)
```

**Interpretation:** We are (1-α)×100% confident that true μ lies in this interval.

### Hypothesis Testing

#### Framework
1. **Null Hypothesis H₀**: Status quo
2. **Alternative H₁**: What we want to prove  
3. **Test statistic**: Measures evidence against H₀
4. **P-value**: P(observe data as extreme | H₀ true)
5. **Decision**: Reject H₀ if p-value < α

#### t-test Example
```
H₀: μ = μ₀
H₁: μ ≠ μ₀

Test statistic: t = (x̄ - μ₀)/(s/√n)
Under H₀: t ~ t(n-1)
```

## Applications in Statistical ML

### Linear Models
**Matrix formulation:**
```
y = Xβ + ε, where ε ~ N(0, σ²I)
```

**Assumptions:**
1. Linearity: E[y|X] = Xβ
2. Independence: Cov(εᵢ, εⱼ) = 0 for i ≠ j
3. Homoscedasticity: Var(εᵢ) = σ² for all i
4. Normality: ε ~ N(0, σ²I)

### Regularization
**Ridge Regression (L2 penalty):**
```
β̂ᵣᵢ𝒹𝑔ₑ = argmin_β ||y - Xβ||₂² + λ||β||₂²
β̂ᵣᵢ𝒹𝑔ₑ = (XᵀX + λI)⁻¹Xᵀy
```

**Lasso Regression (L1 penalty):**
```
β̂ₗₐₛₛₒ = argmin_β ||y - Xβ||₂² + λ||β||₁
```

### Principal Component Analysis
1. **Center data**: X̃ = X - X̄
2. **Compute covariance**: C = (1/(n-1))X̃ᵀX̃
3. **Find eigenvectors**: C = QΛQᵀ
4. **Transform data**: Y = X̃Q

## Computational Considerations

### Matrix Inversion
- **Computational complexity**: O(n³)
- **Numerical stability**: Use QR decomposition or SVD instead
- **Condition number**: κ(A) = σₘₐₓ/σₘᵢₙ (high = ill-conditioned)

### Large-Scale Problems
- **Gradient descent**: Iterative optimization
- **Stochastic methods**: Use subsets of data
- **Matrix factorizations**: Reduce dimensionality

## Common Pitfalls

### Linear Algebra
1. **Non-invertible matrices**: Check rank before inverting
2. **Numerical precision**: Floating-point errors accumulate
3. **Memory requirements**: O(n²) for n×n matrices

### Probability
1. **Independence assumption**: Often violated in real data
2. **Distribution assumptions**: Check with diagnostic plots
3. **Sample size**: CLT requires "large" n (usually n ≥ 30)

### Statistics
1. **Multiple testing**: Adjust for multiple comparisons
2. **Correlation ≠ Causation**: Need experimental design
3. **Outliers**: Can dramatically affect results

## Key Takeaways

1. **Linear algebra** provides the computational framework
2. **Probability** quantifies uncertainty and randomness
3. **Statistics** enables inference from data to population
4. **Matrix operations** are fundamental to ML algorithms
5. **Assumptions matter** - always validate before proceeding
6. **Computational efficiency** becomes critical with large data

These mathematical foundations underpin all statistical machine learning methods we'll encounter in CPSC 540.