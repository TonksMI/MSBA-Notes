# CPSC 540: Statistical Machine Learning I - Assisted Notes

## Overview
This folder contains comprehensive assisted notes for CPSC 540, designed to provide detailed explanations, mathematical foundations, and practical examples for each topic covered in the course.

## How to Use These Notes
1. **Start with background files** (00-Background-*) to refresh foundational concepts
2. **Follow class sequence** for building understanding progressively  
3. **Use mathematical examples** to solidify theoretical concepts
4. **Reference diagrams and visualizations** for intuitive understanding
5. **Practice with R code examples** provided throughout

## Background Knowledge Files
These files cover essential prerequisites - review as needed:

- **[00-Background-Linear-Algebra.md](./00-Background-Linear-Algebra.md)**
  - Vectors, matrices, eigenvalues, SVD
  - Applications in machine learning
  - Matrix decompositions and their uses

- **[00-Background-Probability-Theory.md](./00-Background-Probability-Theory.md)**
  - Probability distributions and their properties
  - Bayes' theorem and conditional probability
  - Central Limit Theorem and Law of Large Numbers
  - Information theory basics

- **[00-Background-Statistics-Review.md](./00-Background-Statistics-Review.md)**  
  - Descriptive and inferential statistics
  - Hypothesis testing and confidence intervals
  - Regression analysis fundamentals
  - ANOVA and experimental design

## Class Notes by Topic

### Course Introduction and Foundations

#### [Class-00-Course-Structure-and-Intro.md](./Class-00-Course-Structure-and-Intro.md)
**Date:** August 25, 2025
- Course philosophy and structure
- Statistical vs. algorithmic machine learning
- Assessment overview and expectations
- Key topics roadmap

#### [Class-01-Math-Review-and-Intro.md](./Class-01-Math-Review-and-Intro.md)
**Date:** August 27, 2025 | **Quiz:** Math basics
- Comprehensive linear algebra review
- Probability theory fundamentals  
- Statistical concepts for ML
- Maximum likelihood estimation
- Applications in statistical ML

### Graph Theory and Causal Modeling

#### [Class-03-Probability-Theory-and-Graph-Theory.md](./Class-03-Probability-Theory-and-Graph-Theory.md)
**Date:** September 3, 2025 | **Quiz:** Probability and graphs
- Joint and conditional distributions
- Graph theory fundamentals
- Directed Acyclic Graphs (DAGs)
- Probabilistic graphical models
- d-separation and conditional independence

### Generalized Linear Models

#### [Class-04-Generalized-Linear-Models-Basics.md](./Class-04-Generalized-Linear-Models-Basics.md)
**Date:** September 8, 2025
- Exponential family distributions
- GLM framework: random, systematic, link components  
- Common link functions and their interpretations
- Maximum likelihood estimation for GLMs
- Model diagnostics and assessment

### Bayesian Methods

#### [Class-10-MCMC-Monte-Carlo-Methods.md](./Class-10-MCMC-Monte-Carlo-Methods.md)
**Date:** September 29, 2025 | **Reading:** Bayes Rules! Chapter 7
- Monte Carlo integration principles
- Markov Chain Monte Carlo fundamentals
- Metropolis-Hastings algorithm
- Gibbs sampling and advanced methods
- Hamiltonian Monte Carlo (HMC)
- MCMC diagnostics and troubleshooting

### Causal Inference

#### [Class-12-Causal-Inference-DAGs.md](./Class-12-Causal-Inference-DAGs.md)
**Date:** October 6, 2025 | **Reading:** Causal Inference in R Chapter 5
- Causal vs. statistical thinking
- Counterfactual framework
- Causal graph structures: chains, forks, colliders
- Backdoor and frontdoor criteria
- Common causal fallacies
- Experimental design principles

## Mathematical Notation Guide

### Common Symbols
- **θ**: Parameters (theta)
- **β**: Regression coefficients (beta)  
- **μ**: Mean (mu)
- **σ²**: Variance (sigma squared)
- **Σ**: Summation (capital sigma)
- **∏**: Product (capital pi)
- **∇**: Gradient (nabla)
- **⊥**: Independence
- **|**: Conditional on
- **~**: Distributed as
- **→**: Causal arrow
- **⟹**: Implies

### Matrix Notation
- **X**: Design matrix (capital letters for matrices)
- **x**: Vector (lowercase letters for vectors)
- **X'** or **X^T**: Matrix transpose
- **X⁻¹**: Matrix inverse
- **tr(X)**: Matrix trace
- **det(X)**: Matrix determinant

## R Packages Used
Key packages referenced throughout the notes:

### Core Analysis
```r
library(tidyverse)    # Data manipulation and visualization
library(broom)        # Tidy model outputs
```

### Statistical Modeling  
```r
library(lme4)         # Mixed effect models
library(mgcv)         # Generalized additive models
library(survival)     # Survival analysis
```

### Bayesian Analysis
```r
library(rstan)        # Stan interface
library(brms)         # Bayesian regression models
library(bayesplot)    # Bayesian visualization
```

### Causal Inference
```r
library(dagitty)      # DAG analysis
library(ggdag)        # DAG visualization
library(marginaleffects) # Effect estimation
```

## Study Tips

### For Mathematical Content
1. **Work through derivations** step by step
2. **Connect formulas to intuition** using provided examples
3. **Practice with R code** to see concepts in action
4. **Draw diagrams** for probability and causal relationships

### For Conceptual Understanding
1. **Focus on assumptions** behind each method
2. **Understand when methods apply** and when they don't
3. **Connect topics** across different classes
4. **Think about real-world applications**

### For Practical Application
1. **Run R code examples** and modify them
2. **Work with simulated data** to see how methods behave
3. **Practice model diagnostics** and interpretation
4. **Try sensitivity analyses** for robustness

## Exam Preparation

### Key Concepts to Master
- **Linear algebra**: Matrix operations, eigenvalues, SVD
- **Probability**: Joint distributions, conditional independence
- **GLMs**: Link functions, exponential family, interpretation
- **Causal inference**: DAGs, confounding, experimental design
- **Bayesian methods**: Prior/posterior, MCMC, model comparison

### Problem-Solving Approach
1. **Identify the type of problem** (prediction, inference, causation)
2. **Check assumptions** and data characteristics
3. **Choose appropriate method** based on problem type
4. **Implement solution** with proper diagnostics
5. **Interpret results** in context with uncertainty quantification

## Additional Resources

### Recommended Reading
- **Math for ML**: Mathematics for Machine Learning (Deisenroth et al.)
- **Regression**: Regression and Other Stories (Gelman et al.)
- **Bayesian**: Bayes Rules! An Introduction to Applied Bayesian Modeling
- **Causal**: Causal Inference in R (Cunningham)

### Online Resources
- **Stan Documentation**: mc-stan.org
- **dagitty**: dagitty.net for DAG analysis
- **R for Data Science**: r4ds.had.co.nz

### Software Installation
Ensure you have installed the required R packages as listed in the syllabus:
```r
install.packages(c("tidyverse", "rstan", "brms", "lme4", 
                  "mgcv", "dagitty", "tidybayes", "bayesplot", 
                  "marginaleffects"))
```

## Note Structure
Each class note follows a consistent structure:
- **Overview**: Topic summary and learning objectives
- **Theory**: Mathematical foundations with derivations
- **Examples**: Worked examples with interpretations  
- **R Implementation**: Code examples and practical applications
- **Diagnostics**: Model checking and validation techniques
- **Common Pitfalls**: Mistakes to avoid
- **Key Takeaways**: Essential points to remember

## Updates and Corrections
These notes are comprehensive but may contain occasional errors or could benefit from clarification. When studying:
- **Cross-reference** with official course materials
- **Ask questions** in class about unclear concepts
- **Check R code** by running examples yourself
- **Verify mathematical derivations** when possible

---

**Happy studying!** These notes are designed to support your learning throughout CPSC 540. Use them as a comprehensive reference alongside lectures, readings, and assignments.