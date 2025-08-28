# Class 3: Probability Theory + Graph Theory
**Date:** September 3, 2025 (Wednesday)  
**Quiz:** Yes (probability and graph basics)  
**Topics:** Graphs, DAGs, R programming

## Overview
Deep dive into probability theory applications and introduction to graph theory, particularly Directed Acyclic Graphs (DAGs) for representing probabilistic relationships and causal structures.

## Advanced Probability Theory

### Joint Distributions

#### Discrete Joint Distributions
For discrete random variables X and Y:
```
Joint PMF: $p(x,y) = P(X = x, Y = y)$
```

**Properties:**
- $p(x,y) \geq 0$ for all $x,y$
- $\sum_x \sum_y p(x,y) = 1$

**Example: Two Dice**
```
X = first die result, Y = second die result
p(x,y) = 1/36 for x,y ∈ {1,2,3,4,5,6}

Joint distribution table:
    Y  1   2   3   4   5   6
X  1  1/36 1/36 1/36 1/36 1/36 1/36
   2  1/36 1/36 1/36 1/36 1/36 1/36
   ⋮   ⋮    ⋮    ⋮    ⋮    ⋮    ⋮
   6  1/36 1/36 1/36 1/36 1/36 1/36
```

#### Continuous Joint Distributions
For continuous random variables:
```
P((X,Y) ∈ A) = ∬_A f(x,y) dxdy
```

**Bivariate Normal Distribution:**
```
f(x,y) = 1/(2πσₓσᵧ√(1-ρ²)) × exp(-Q/2)

where Q = 1/(1-ρ²)[(x-μₓ)²/σₓ² - 2ρ(x-μₓ)(y-μᵧ)/(σₓσᵧ) + (y-μᵧ)²/σᵧ²]
```

### Marginal and Conditional Distributions

#### Marginal Distributions
**Discrete:**
```
pₓ(x) = Σᵧ p(x,y)
pᵧ(y) = Σₓ p(x,y)
```

**Continuous:**
```
fₓ(x) = ∫ f(x,y) dy
fᵧ(y) = ∫ f(x,y) dx
```

#### Conditional Distributions
**Discrete:**
```
p(y|x) = p(x,y) / pₓ(x)
```

**Continuous:**
```
f(y|x) = f(x,y) / fₓ(x)
```

**Example: Heights and Weights**
If (X,Y) ~ Bivariate Normal, then:
```
Y|X = x ~ N(μᵧ + ρ(σᵧ/σₓ)(x - μₓ), σᵧ²(1-ρ²))
```

### Independence and Conditional Independence

#### Statistical Independence
X and Y are independent (X ⊥ Y) if:
```
f(x,y) = fₓ(x)fᵧ(y) for all x,y
```

**Equivalent conditions:**
- f(y|x) = fᵧ(y) for all x
- f(x|y) = fₓ(x) for all y
- E[XY] = E[X]E[Y]
- Cov(X,Y) = 0 (for finite variances)

#### Conditional Independence
X and Y are conditionally independent given Z (X ⊥ Y | Z) if:
```
f(x,y|z) = f(x|z)f(y|z)
```

**Key Insight:** X and Y may be dependent marginally but independent given Z.

**Example: Student Performance**
```
X = Math test score
Y = Reading test score  
Z = General intelligence

X and Y are correlated (both depend on intelligence)
But X ⊥ Y | Z (given intelligence level, math and reading scores independent)
```

## Graph Theory Fundamentals

### Basic Definitions

#### Graph Components
- **Vertices (Nodes)**: V = {v₁, v₂, ..., vₙ}
- **Edges**: E = {(vᵢ, vⱼ) : connection between vertices}
- **Graph**: G = (V, E)

#### Graph Types
```
Undirected Graph:    Directed Graph:
   A --- B              A → B
   |     |              ↑   ↓
   C --- D              C ← D
```

### Directed Acyclic Graphs (DAGs)

#### Definition
A directed graph with no directed cycles.
```
Valid DAG:           Invalid (has cycle):
A → B → D              A → B → D
↓   ↓                  ↑   ↓   ↓
C → E                  C ← E ← F
```

#### Topological Ordering
A linear ordering of vertices such that for every directed edge (u,v), u comes before v.

**Algorithm (Kahn's):**
1. Find vertices with no incoming edges
2. Remove vertex and its outgoing edges
3. Repeat until all vertices processed

### Probabilistic Graphical Models

#### Bayesian Networks
A DAG where:
- **Nodes** represent random variables
- **Edges** represent direct dependencies
- **Joint distribution** factorizes according to graph structure

**Factorization:**
```
P(X₁, X₂, ..., Xₙ) = ∏ᵢ P(Xᵢ | Parents(Xᵢ))
```

#### Example: Student Performance Network
```
Intelligence (I) → SAT Score (S) → College GPA (G)
       ↓              ↑
   Difficulty (D) ─────┘
```

**Joint distribution:**
```
P(I,D,S,G) = P(I)P(D)P(S|I,D)P(G|S)
```

### Causal Graphs

#### Causal vs. Statistical Relationships
- **Statistical**: X and Y are associated
- **Causal**: X directly influences Y

#### Causal Structures

**1. Chain (Mediation):**
```
X → M → Y
```
- X affects Y through M
- Controlling for M blocks the path

**2. Fork (Confounding):**
```
    Z
   ↙ ↘
  X   Y
```
- Z is a common cause of X and Y
- X and Y are associated but not causally related

**3. Collider (Selection bias):**
```
  X   Y
   ↘ ↙
    Z
```
- X and Y both affect Z
- X and Y are independent marginally
- X and Y become dependent when conditioning on Z

### d-Separation

#### Definition
A path is d-separated (blocked) by a set S if:
1. **Chain or Fork**: Path contains a node in S
2. **Collider**: Path contains a collider not in S, and no descendant of the collider is in S

#### Rules for Causal Inference
1. **Backdoor Criterion**: To estimate causal effect of X on Y, control for variables that block all backdoor paths
2. **Frontdoor Criterion**: When backdoor criterion fails, use mediating variables

**Example: Education and Income**
```
     Ability
    ↙      ↘
Education → Income
```
- Want causal effect of Education on Income
- Ability confounds the relationship
- Need to control for Ability (backdoor criterion)

## Implementation in R

### Basic Graph Operations

#### Creating Graphs
```r
library(igraph)
library(dagitty)

# Create directed graph
g <- graph_from_literal(A -+ B -+ D, A -+ C -+ D, B -+ C)

# Plot graph
plot(g, layout = layout_with_graphopt(g))

# Check if DAG
is_dag(g)  # Should return TRUE
```

#### DAG with dagitty
```r
# Define causal DAG
dag <- dagitty("dag {
    Intelligence -> SAT
    Intelligence -> GPA
    SAT -> GPA
    Difficulty -> SAT
}")

# Plot DAG
plot(dag)

# Find adjustment sets
adjustmentSets(dag, exposure = "SAT", outcome = "GPA")
```

### Simulating from Graphical Models

#### Linear Gaussian Model
```r
# Simulate data following DAG structure
set.seed(123)
n <- 1000

# Exogenous variables
intelligence <- rnorm(n, 100, 15)
difficulty <- rbinom(n, 1, 0.5)

# Endogenous variables  
sat_score <- 400 + 2*intelligence + 50*difficulty + rnorm(n, 0, 100)
gpa <- 1 + 0.01*sat_score + rnorm(n, 0, 0.5)

# Create data frame
data <- data.frame(
    intelligence = intelligence,
    difficulty = difficulty,
    sat_score = sat_score,
    gpa = gpa
)
```

#### Analyzing Relationships
```r
# Marginal correlation (confounded)
cor(data$sat_score, data$gpa)

# Partial correlation (controlling for intelligence)
library(ppcor)
pcor.test(data$sat_score, data$gpa, data$intelligence)

# Regression analysis
# Naive model (biased)
model1 <- lm(gpa ~ sat_score, data = data)

# Adjusted model (less biased)
model2 <- lm(gpa ~ sat_score + intelligence, data = data)

# Compare results
summary(model1)$coefficients["sat_score", ]
summary(model2)$coefficients["sat_score", ]
```

## Advanced Topics

### Markov Properties

#### Local Markov Property
Each variable is independent of its non-descendants given its parents:
```
Xᵢ ⊥ NonDescendants(Xᵢ) | Parents(Xᵢ)
```

#### Global Markov Property
If sets A and B are d-separated by set C, then:
```
XₐA ⊥ X_B | X_C
```

### Faithfulness Assumption
The conditional independence relations in the data match those implied by the graph structure.

**When it fails:**
- **Deterministic relationships**: Y = X₁ + X₂ exactly
- **Exact cancellation**: Effects cancel out precisely
- **Non-linear relationships**: Graph represents linear dependencies only

### Information Theory and Graphs

#### Mutual Information
```
I(X;Y) = ∫∫ p(x,y) log(p(x,y)/(p(x)p(y))) dx dy
```

**Properties:**
- I(X;Y) ≥ 0
- I(X;Y) = 0 iff X ⊥ Y
- I(X;Y) = H(X) - H(X|Y)

#### Conditional Mutual Information
```
I(X;Y|Z) = ∫∫∫ p(x,y,z) log(p(x,y|z)/(p(x|z)p(y|z))) dx dy dz
```

## Applications in Statistical ML

### Structure Learning
**Problem:** Given data, learn the graph structure.

**Approaches:**
1. **Score-based**: Search over DAGs, optimize score (BIC, AIC)
2. **Constraint-based**: Test conditional independencies
3. **Hybrid**: Combine both approaches

### Causal Discovery
**Goal:** Distinguish causation from correlation.

**Methods:**
1. **PC Algorithm**: Uses conditional independence tests
2. **FCI**: Handles latent confounders
3. **LiNGAM**: Assumes linear non-Gaussian additive noise

### Feature Selection
Use graph structure to identify relevant variables:
```r
# Find Markov blanket (parents + children + spouses)
markov_blanket <- function(dag, target) {
    parents <- parents(dag, target)
    children <- children(dag, target)
    spouses <- spouses_of_children(dag, target)
    return(c(parents, children, spouses))
}
```

## Practical Considerations

### Model Selection
**Information Criteria:**
```
AIC = -2 log L + 2k
BIC = -2 log L + k log n
```

Where:
- L: Maximum likelihood
- k: Number of parameters
- n: Sample size

### Sample Size Requirements
- **Rule of thumb**: Need ≥ 5-10 samples per parameter
- **Weak edges**: Require more data to detect
- **Complex structures**: Need exponentially more data

### Robustness Checks
1. **Bootstrap**: Resample data, check stability
2. **Cross-validation**: Test predictive performance
3. **Sensitivity analysis**: Vary assumptions

## Common Pitfalls

1. **Confusing correlation with causation**
2. **Ignoring selection bias** (collider conditioning)
3. **Assuming faithfulness** when it doesn't hold
4. **Over-interpreting** weak relationships
5. **Forgetting measurement error**

## Key Takeaways

1. **Joint distributions** capture complex dependencies
2. **Conditional independence** is key to causal reasoning
3. **DAGs** provide intuitive causal representations
4. **d-separation** determines statistical relationships
5. **Graphical models** enable principled inference
6. **Structure learning** is challenging but crucial
7. **Causal discovery** requires strong assumptions

These concepts form the foundation for understanding causal inference methods that we'll explore later in the course, particularly when studying causal DAGs and experimental design.