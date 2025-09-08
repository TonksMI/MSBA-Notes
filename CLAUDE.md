# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Architecture

This is an academic repository containing comprehensive course materials for multiple university courses, primarily focused on statistical machine learning, business analytics, and quantitative methods. The repository is organized by course codes with a sophisticated assisted notes system.

### Core Structure

- `CPSC 540/`: Statistical Machine Learning I - Original comprehensive materials with LaTeX math
- `BUS 659/`: Machine Learning for Managers - Business-focused ML applications  
- `BUS 671/`: Managing Data for Analysis - SQL and data management systems
- `BUS 672/`: Data Visualization for Business - Visualization theory and tools
- `sample files/`: Template R Markdown files for projects

### Business Course Architecture

Each BUS course follows a standardized structure:
- `Assisted Notes/`: Comprehensive topic coverage with business applications
- `Assignments/`: Student work and presentation materials  
- `Class Folder/`: Course materials, datasets, slides, problem sets
- `Jupyter NoteBooks/`: Python-based computational work
- `Notes/`: Daily class notes and observations

## CPSC 540: Statistical Machine Learning I

This is the primary focus of the repository, containing a comprehensive assisted notes system with proper LaTeX mathematical notation.

### Assisted Notes Architecture

The `Assisted Notes/` directory contains two types of content:

#### Background Knowledge Files (00-Background-*)
- **00-Background-Linear-Algebra.md**: Complete linear algebra foundations
- **00-Background-Probability-Theory.md**: Probability theory with business applications  
- **00-Background-Statistics-Review.md**: Statistical concepts and methods

#### Class-by-Class Notes (Class-XX-*)
Sequential notes following the course curriculum:
- Class-00: Course structure and philosophy
- Class-01: Mathematical review and ML introduction
- Class-03: Probability theory and graph theory
- Class-04: Generalized Linear Models
- Class-10: MCMC and Monte Carlo methods
- Class-12: Causal inference with DAGs

#### Advanced Topic Extensions (Advanced-XX-*)
Six comprehensive advanced topics beyond the core curriculum:
- Advanced-01: Deep Learning Fundamentals
- Advanced-02: Natural Language Processing & LLMs
- Advanced-03: Computer Vision Applications
- Advanced-04: Reinforcement Learning Business Cases
- Advanced-05: Time Series Forecasting Methods
- Advanced-06: Ensemble Methods and Model Optimization

### Mathematical Notation Standards

All mathematical formulas use proper LaTeX notation:
- Inline math: `$formula$`
- Display math: `$$formula$$`
- Vectors: `$\mathbf{v}$` with bold notation
- Matrices: `$\mathbf{A}$` with bold capital letters
- Statistical distributions: `$p(\theta|y) = \frac{p(y|\theta)p(\theta)}{p(y)}$`

## Working with R Content

### R Markdown Files
- Use `.Rmd` extension for R Markdown documents
- Standard YAML header structure for academic projects
- Code chunks configured with `tidy=TRUE` and `size="vsmall"`
- Set working seed for reproducibility: `set.seed(1818)`

### R Package Ecosystem
Key packages used throughout materials:

**Core Analysis:**
```r
library(tidyverse)    # Data manipulation and visualization
library(broom)        # Tidy model outputs
```

**Statistical Modeling:**
```r
library(lme4)         # Mixed effect models
library(mgcv)         # Generalized additive models
```

**Bayesian Analysis:**
```r
library(rstan)        # Stan interface
library(brms)         # Bayesian regression models
library(bayesplot)    # Bayesian visualization
```

**Causal Inference:**
```r
library(dagitty)      # DAG analysis
library(ggdag)        # DAG visualization
library(marginaleffects) # Effect estimation
```

## Content Guidelines

### Note Structure Pattern
Each assisted note follows this architecture:
1. **Overview**: Learning objectives and context
2. **Mathematical Foundations**: Rigorous derivations with LaTeX
3. **Business Applications**: Real-world examples with ROI analysis
4. **Implementation**: Detailed code examples
5. **Advanced Topics**: Extensions beyond basic concepts

### Mathematical Formula Conversion
When working with mathematical content:
- Convert all formulas to proper LaTeX notation
- Use `$$` for display equations, `$` for inline
- Maintain consistent notation: `$\mathbf{X}$` for matrices, `$\mathbf{x}$` for vectors
- Include parameter definitions and variable explanations

### Business Context Integration
Advanced notes include comprehensive business applications:
- ROI calculations with realistic scenarios
- Industry-specific implementations
- Cost-benefit analysis
- Professional development pathways
- Workplace troubleshooting scenarios

## Development Workflow

### Jupyter Notebooks
- Located in `Jupyter NoteBooks/` subdirectories
- Primarily for computational assignments and data analysis
- Use standard data science libraries (pandas, numpy, scikit-learn)

### R Analysis
- R Markdown files for statistical analysis and reporting
- Use consistent chunk options for academic presentation
- Include proper documentation and interpretation

### Content Updates
When modifying assisted notes:
- Maintain LaTeX mathematical notation standards
- Preserve the comprehensive structure (theory + business + code)
- Ensure cross-references between related topics
- Update the README-Index.md when adding new sections

## Business Course Specifications

### BUS 659: Machine Learning for Managers
**Focus**: Business applications of ML without heavy technical prerequisites
**Topics**: Uncertainty/bias-variance, linear/logistic regression, regularization, decision trees, random forests, interpretable ML, SQL
**Tools**: R or Python (assumes no prior coding experience)
**Assisted Notes**: 5 comprehensive topics with business ROI analysis and implementation guidance

### BUS 671: Managing Data for Analysis  
**Focus**: Database systems and data pipeline management for business
**Topics**: SQL fundamentals, data extraction/transformation/loading (ETL), NoSQL variants
**Tools**: SQL databases, modern data storage systems
**Assisted Notes**: Covers SQL basics through advanced data retrieval with business applications

### BUS 672: Data Visualization for Business
**Focus**: Creating business-ready visualizations and presentations
**Topics**: Visualization theory, audience analysis, dashboard design, presentation skills
**Tools**: Tableau, ggplot, seaborn, matplotlib, plotly
**Assignments**: Includes personal story development and presentation optimization

## Python Environment

### Data Science Stack
The repository includes a comprehensive Python environment (`BUS 659/Jupyter NoteBooks/requirements.txt`):
- **Core**: numpy, pandas, scipy, scikit-learn, statsmodels
- **Visualization**: matplotlib, seaborn, plotly  
- **Jupyter**: Complete notebook ecosystem with extensions
- **Business Tools**: openpyxl for Excel integration

### Development Commands
```bash
# Install Python environment
pip install -r "BUS 659/Jupyter NoteBooks/requirements.txt"

# Launch Jupyter for analysis
jupyter lab

# For business presentations
streamlit run app.py  # If using Streamlit dashboards
```

## Academic Context

This repository supports advanced graduate coursework in:
- Statistical machine learning theory and application (CPSC 540)
- Business-focused machine learning applications (BUS 659)
- Enterprise data management and SQL (BUS 671)
- Professional data visualization and presentation (BUS 672)

The assisted notes system provides comprehensive coverage beyond typical course materials, with particular emphasis on business applications, ROI analysis, and professional development guidance for data science careers.