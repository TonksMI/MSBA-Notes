# Background: Linear Algebra Fundamentals

## Overview
This document covers essential linear algebra concepts needed for CPSC 540: Statistical Machine Learning I. These concepts form the mathematical foundation for understanding machine learning algorithms.

## 1. Vectors

### Definition
A vector is an ordered collection of numbers (scalars).

**Column Vector:**
```
v = [v₁]
    [v₂]
    [⋮ ]
    [vₙ]
```

**Row Vector:**
```
v^T = [v₁, v₂, ..., vₙ]
```

### Vector Operations

#### Addition
```
a + b = [a₁ + b₁]
        [a₂ + b₂]
        [⋮      ]
        [aₙ + bₙ]
```

#### Scalar Multiplication
```
ca = [ca₁]
     [ca₂]
     [⋮  ]
     [caₙ]
```

#### Dot Product (Inner Product)
For vectors a and b:
```
a · b = a^T b = Σᵢ aᵢbᵢ = a₁b₁ + a₂b₂ + ... + aₙbₙ
```

**Example:**
```
a = [2, 3, 1]
b = [4, 1, 2]
a · b = 2×4 + 3×1 + 1×2 = 8 + 3 + 2 = 13
```

#### Vector Norm (Length)
**L2 Norm (Euclidean):**
```
||a||₂ = √(a · a) = √(a₁² + a₂² + ... + aₙ²)
```

**L1 Norm (Manhattan):**
```
||a||₁ = |a₁| + |a₂| + ... + |aₙ|
```

## 2. Matrices

### Definition
A matrix is a rectangular array of numbers arranged in rows and columns.

```
A = [a₁₁  a₁₂  ...  a₁ₙ]
    [a₂₁  a₂₂  ...  a₂ₙ]
    [⋮    ⋮    ⋱   ⋮  ]
    [aₘ₁  aₘ₂  ...  aₘₙ]
```

### Matrix Operations

#### Addition
Matrices of the same dimensions can be added element-wise:
```
C = A + B where cᵢⱼ = aᵢⱼ + bᵢⱼ
```

#### Scalar Multiplication
```
cA = [ca₁₁  ca₁₂  ...  ca₁ₙ]
     [ca₂₁  ca₂₂  ...  ca₂ₙ]
     [⋮     ⋮     ⋱    ⋮   ]
     [caₘ₁  caₘ₂  ...  caₘₙ]
```

#### Matrix Multiplication
For A (m×p) and B (p×n), the product C = AB is (m×n):
```
cᵢⱼ = Σₖ aᵢₖbₖⱼ
```

**Example:**
```
A = [1  2]    B = [5  6]
    [3  4]        [7  8]

AB = [1×5 + 2×7  1×6 + 2×8] = [19  22]
     [3×5 + 4×7  3×6 + 4×8]   [43  50]
```

#### Matrix Transpose
The transpose A^T switches rows and columns:
```
If A = [1  2  3]  then A^T = [1  4]
       [4  5  6]              [2  5]
                              [3  6]
```

#### Matrix Inverse
For a square matrix A, the inverse A⁻¹ satisfies:
```
AA⁻¹ = A⁻¹A = I
```

For a 2×2 matrix:
```
A = [a  b]    A⁻¹ = 1/(ad-bc) [ d  -b]
    [c  d]                     [-c   a]
```

## 3. Special Matrices

### Identity Matrix
The identity matrix I has 1s on the diagonal and 0s elsewhere:
```
I = [1  0  0]
    [0  1  0]
    [0  0  1]
```

### Diagonal Matrix
A diagonal matrix has non-zero elements only on the main diagonal:
```
D = [d₁  0   0 ]
    [0   d₂  0 ]
    [0   0   d₃]
```

### Symmetric Matrix
A matrix where A = A^T:
```
A = [1  2  3]
    [2  4  5]
    [3  5  6]
```

## 4. Linear Independence and Rank

### Linear Independence
Vectors v₁, v₂, ..., vₖ are linearly independent if:
```
c₁v₁ + c₂v₂ + ... + cₖvₖ = 0
```
only when c₁ = c₂ = ... = cₖ = 0.

### Rank
The rank of a matrix is the maximum number of linearly independent rows (or columns).

**Properties:**
- rank(A) ≤ min(m, n) for an m×n matrix
- Full rank means rank(A) = min(m, n)
- A square matrix is invertible iff it has full rank

## 5. Eigenvalues and Eigenvectors

### Definition
For a square matrix A, if there exists a non-zero vector v and scalar λ such that:
```
Av = λv
```
Then λ is an eigenvalue and v is the corresponding eigenvector.

### Geometric Interpretation
When A acts on eigenvector v, it only stretches or shrinks v by factor λ, without changing direction.

### Computing Eigenvalues
Solve the characteristic equation:
```
det(A - λI) = 0
```

**Example for 2×2 matrix:**
```
A = [4  1]
    [2  3]

det([4-λ   1  ]) = (4-λ)(3-λ) - 2 = λ² - 7λ + 10 = 0
   ([2   3-λ])

λ₁ = 5, λ₂ = 2
```

## 6. Matrix Decompositions

### Eigenvalue Decomposition
For a symmetric matrix A:
```
A = QΛQ^T
```
where Q contains eigenvectors and Λ is diagonal matrix of eigenvalues.

### Singular Value Decomposition (SVD)
For any matrix A:
```
A = UΣV^T
```
where:
- U: left singular vectors (orthogonal)
- Σ: diagonal matrix of singular values
- V: right singular vectors (orthogonal)

**Applications:**
- Dimensionality reduction (PCA)
- Matrix approximation
- Solving linear systems

## 7. Quadratic Forms

### Definition
A quadratic form in vector x is:
```
x^TAx
```

**Example:**
```
x = [x₁]    A = [2  1]    x^TAx = 2x₁² + 2x₁x₂ + 3x₂²
    [x₂]        [1  3]
```

### Positive Definite Matrices
A symmetric matrix A is positive definite if:
```
x^TAx > 0 for all x ≠ 0
```

**Properties:**
- All eigenvalues > 0
- Determinant > 0
- Useful in optimization (convex functions)

## 8. Applications in Machine Learning

### Linear Regression
The normal equation:
```
β = (X^TX)⁻¹X^Ty
```

### Principal Component Analysis (PCA)
Find eigenvectors of covariance matrix:
```
C = (1/n)X^TX
```

### Least Squares Solution
For overdetermined system Ax = b:
```
x = (A^TA)⁻¹A^Tb
```

## Deep Dive: Why Linear Algebra Matters in the Workplace

### The Business Impact of Linear Algebra

Linear algebra isn't just abstract mathematics—it's the computational engine behind virtually every data-driven decision in modern business. Understanding these concepts deeply allows you to:

1. **Design better algorithms** that scale with business growth
2. **Debug model performance** when things go wrong
3. **Optimize computational resources** and reduce costs
4. **Communicate technical concepts** to non-technical stakeholders
5. **Make informed trade-offs** between accuracy and efficiency

### Real-World Workplace Applications

#### 1. Recommender Systems (E-commerce/Streaming)

**The Challenge:** Netflix has 230 million users and 15,000 titles. How do you efficiently recommend movies each user might like?

**Linear Algebra Solution:** Matrix Factorization
```
Rating Matrix R (users × movies) ≈ U × V^T
where U (users × factors), V (movies × factors)
```

**Business Impact:**
- **Increased engagement**: Better recommendations → more viewing time
- **Reduced churn**: Satisfied customers stay subscribed
- **Computational efficiency**: O(k(m+n)) instead of O(mn) operations

**Workplace Implementation:**
```python
# Simplified collaborative filtering
from scipy.sparse.linalg import svds
import pandas as pd

# Load user-item ratings (sparse matrix)
ratings_matrix = load_user_ratings()  # 1M users × 50K items

# Apply SVD with k factors
U, sigma, Vt = svds(ratings_matrix, k=100)

# Generate recommendations for user i
user_vector = U[i, :]
recommendations = user_vector @ Vt  # Matrix multiplication!

# Business value: Predict ratings for unseen movies
predicted_ratings = U @ np.diag(sigma) @ Vt
```

**Why This Matters to Management:**
- **Cost savings**: Process 50 billion interactions daily with manageable compute
- **Revenue increase**: 80% of Netflix viewing comes from recommendations
- **Competitive advantage**: Better algorithms = better user experience

#### 2. Principal Component Analysis in Finance

**The Challenge:** A hedge fund tracks 500+ economic indicators but needs to understand the main drivers of market movements for risk management.

**Linear Algebra Solution:** PCA via Eigenvalue Decomposition
```
Covariance Matrix C = (1/n)X^T X
Principal Components = Eigenvectors of C
Explained Variance = Eigenvalues of C
```

**Workplace Scenario:**
```r
# Financial risk management example
library(prcomp)

# Load economic indicators: interest rates, inflation, GDP growth, etc.
economic_data <- read.csv("economic_indicators.csv")  # 500 variables

# Apply PCA
pca_result <- prcomp(economic_data, scale = TRUE)

# First 3 components explain 70% of market variance
summary(pca_result)  # Variance explained

# Identify key risk factors
loadings <- pca_result$rotation[, 1:3]
print("Top risk factors:")
print(abs(loadings) > 0.3)  # Significant loadings
```

**Business Impact:**
- **Risk reduction**: Focus monitoring on 3-5 key factors instead of 500
- **Faster decisions**: Real-time risk assessment possible
- **Regulatory compliance**: Meet Basel III capital requirements efficiently
- **Cost savings**: Reduce data collection and processing overhead

**Practical Implementation:**
- **Portfolio construction**: Weight assets based on principal component loadings
- **Stress testing**: Simulate extreme scenarios using PC directions
- **Reporting**: Explain market movements to regulators and clients

#### 3. Supply Chain Optimization (Manufacturing)

**The Challenge:** A global manufacturer needs to optimize inventory across 100 warehouses, 500 suppliers, and 1000+ products while minimizing costs and meeting demand.

**Linear Algebra Solution:** Linear Programming (Matrix Form)
```
Minimize: c^T x  (cost vector × decision variables)
Subject to: Ax ≤ b  (constraint matrix × variables ≤ limits)
           x ≥ 0   (non-negativity constraints)
```

**Workplace Implementation:**
```python
from scipy.optimize import linprog
import numpy as np

# Supply chain optimization
def optimize_supply_chain():
    # Decision variables: [warehouse_1_stock, warehouse_2_stock, ...]
    n_warehouses = 100
    n_products = 1000
    
    # Cost vector (holding costs + transportation)
    c = calculate_holding_costs() + calculate_transport_costs()
    
    # Constraint matrix A
    # Row constraints: demand satisfaction, capacity limits, balance equations
    A_ub, b_ub = build_constraint_matrices()
    
    # Solve linear program
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
    
    return result.x.reshape(n_warehouses, n_products)

optimal_inventory = optimize_supply_chain()
```

**Business Impact:**
- **Cost reduction**: 15-20% reduction in inventory holding costs
- **Service improvement**: 99.5% order fulfillment rate
- **Capital efficiency**: Free up $50M+ in working capital
- **Risk management**: Optimize inventory against supply disruptions

#### 4. Image Recognition in Healthcare

**The Challenge:** A medical imaging company needs to detect cancer in X-rays with 99%+ accuracy to assist radiologists.

**Linear Algebra Solution:** Convolutional Neural Networks
```
Convolution Operation: (f * g)[n] = Σ f[m]g[n-m]
This is matrix multiplication: Y = W * X + b
where W contains learned filters (edge detectors, pattern recognizers)
```

**Deep Technical Implementation:**
```python
import torch
import torch.nn as nn

class MedicalImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Each convolution is a matrix multiplication
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)  # 3x3 filter matrices
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        
        # Fully connected layers are pure matrix multiplication
        self.fc1 = nn.Linear(128 * 56 * 56, 512)  # W @ x + b
        self.fc2 = nn.Linear(512, 2)  # Binary classification
    
    def forward(self, x):
        # Each operation involves matrix multiplication
        x = torch.relu(self.conv1(x))  # Filter @ image_patch
        x = torch.relu(self.conv2(x))  # Feature_maps @ filters
        x = x.view(x.size(0), -1)     # Reshape for matrix mult
        x = torch.relu(self.fc1(x))   # Dense_weights @ features
        return self.fc2(x)            # Output_weights @ hidden
```

**Business Impact:**
- **Lives saved**: Earlier cancer detection improves survival rates
- **Efficiency**: Process 10,000 images/day vs. 100 manual reviews
- **Cost reduction**: $500/analysis → $50/analysis
- **Scalability**: Deploy to underserved regions with fewer radiologists

### Why Matrix Operations Are Critical for Performance

#### Computational Complexity Understanding
```python
# Performance comparison for recommendation system

# Naive approach: O(users × items × features)
def naive_recommendations(users, items, features):
    recommendations = []
    for u in range(users):
        for i in range(items):
            score = 0
            for f in range(features):
                score += user_features[u][f] * item_features[i][f]
            recommendations.append(score)
    return recommendations
# Time complexity: O(10^6 × 10^4 × 100) = O(10^12) operations

# Matrix multiplication approach: O(users × features × items)  
def matrix_recommendations(user_matrix, item_matrix):
    return user_matrix @ item_matrix.T  # Single operation!
# Time complexity: O(10^6 × 100 × 10^4) = O(10^12) BUT:
# - Highly optimized BLAS libraries
# - Parallel GPU execution
# - Cache-friendly memory access
# Real performance: 1000x faster
```

#### Memory Efficiency in Production Systems
```python
# Sparse matrix example for large-scale systems
from scipy.sparse import csr_matrix
import numpy as np

def create_user_item_matrix():
    # Netflix: 230M users, 15K titles
    # Dense matrix: 230M × 15K × 8 bytes = 27 TB
    # Actual ratings: ~10B ratings = 27 GB (0.1% sparse)
    
    # Sparse representation saves 99.9% memory
    rows, cols, ratings = load_rating_data()
    sparse_matrix = csr_matrix((ratings, (rows, cols)), 
                              shape=(230_000_000, 15_000))
    
    # Matrix operations still work!
    return sparse_matrix

# Business impact: Fit in memory instead of requiring distributed storage
```

### Linear Algebra in Modern Business Domains

#### 1. Marketing Attribution (Digital Marketing)
**Problem**: Which marketing channels drive sales? $100M advertising budget needs optimal allocation.

```r
# Marketing Mix Modeling using linear regression
library(Matrix)

# Multi-touch attribution model
# Y = sales, X = marketing touchpoints (TV, digital, social, etc.)
marketing_data <- load_campaign_data()

# Design matrix with interaction terms
X <- model.matrix(~ TV_spend * seasonality + 
                    digital_impressions * competitor_activity + 
                    social_engagement * demographic_match, 
                  data = marketing_data)

# Ridge regression for coefficient estimation (handles multicollinearity)
library(glmnet)
attribution_model <- cv.glmnet(X, marketing_data$sales, alpha = 0)

# Extract attribution coefficients
coefficients <- coef(attribution_model, s = "lambda.min")
```

**Business Decision Making:**
- **Budget allocation**: Shift 30% from TV to digital based on coefficients
- **ROI measurement**: $3.2 return per $1 spent on targeted social media
- **Optimization**: Real-time bid adjustments in programmatic advertising

#### 2. Fraud Detection (Financial Services)
**Problem**: Detect fraudulent transactions among billions of daily payments.

```python
# Anomaly detection using eigenvalue decomposition
import numpy as np
from sklearn.decomposition import PCA

def detect_payment_fraud():
    # Load transaction features: amount, time, location, merchant, etc.
    transactions = load_transaction_features()  # Shape: (1B transactions, 50 features)
    
    # PCA for anomaly detection
    pca = PCA(n_components=10)  # Reduce to key patterns
    normal_patterns = pca.fit_transform(normal_transactions)
    
    # Reconstruction error indicates anomalies
    def fraud_score(transaction):
        # Project onto principal components and back
        projected = pca.transform(transaction.reshape(1, -1))
        reconstructed = pca.inverse_transform(projected)
        
        # Large reconstruction error = anomaly = potential fraud
        return np.linalg.norm(transaction - reconstructed)
    
    return fraud_score
```

**Business Impact:**
- **Loss prevention**: Reduce fraud losses from $2B to $200M annually
- **Customer experience**: 99.9% legitimate transactions processed smoothly
- **Regulatory compliance**: Meet anti-money laundering requirements

#### 3. Predictive Maintenance (Manufacturing)
**Problem**: Predict when industrial equipment will fail to optimize maintenance schedules.

```python
# Sensor data analysis using matrix factorization
from sklearn.decomposition import NMF  # Non-negative Matrix Factorization

def predict_equipment_failure():
    # Sensor data: temperature, vibration, pressure over time
    sensor_data = load_sensor_readings()  # Shape: (machines, time_series_features)
    
    # NMF to find failure patterns
    # V ≈ W @ H where:
    # V = observed sensor patterns
    # W = failure mode signatures  
    # H = activation levels over time
    
    model = NMF(n_components=5, max_iter=200)
    failure_signatures = model.fit_transform(sensor_data)
    temporal_activations = model.components_
    
    # Predict failure when signature activation exceeds threshold
    current_signature = model.transform(current_readings)
    failure_probability = predict_from_signature(current_signature)
    
    return failure_probability
```

**Business Impact:**
- **Cost reduction**: Planned maintenance vs. emergency repairs (5x cheaper)
- **Uptime improvement**: 99.9% equipment availability
- **Safety**: Prevent catastrophic failures that could injure workers

### Advanced Workplace Applications

#### 1. Natural Language Processing for Customer Service
```python
# Document similarity using SVD (Latent Semantic Analysis)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def build_support_system():
    # Load customer support tickets and solutions
    tickets, solutions = load_support_data()
    
    # Convert text to matrix (TF-IDF)
    vectorizer = TfidfVectorizer(max_features=10000)
    doc_term_matrix = vectorizer.fit_transform(tickets)
    
    # SVD for dimensionality reduction and concept extraction
    svd = TruncatedSVD(n_components=100)
    concept_space = svd.fit_transform(doc_term_matrix)
    
    # Find similar tickets using cosine similarity (dot product in concept space)
    def find_similar_tickets(new_ticket):
        new_vector = vectorizer.transform([new_ticket])
        new_concepts = svd.transform(new_vector)
        
        # Cosine similarity = normalized dot product
        similarities = concept_space @ new_concepts.T
        return np.argsort(similarities, axis=0)[-5:]  # Top 5 similar tickets
```

**Business Impact:**
- **Response time**: Reduce from 24 hours to 2 hours
- **Accuracy**: 85% of auto-suggested solutions are accepted
- **Cost savings**: Handle 2x volume with same staff

#### 2. Portfolio Optimization in Asset Management
```r
# Modern Portfolio Theory using quadratic programming
library(quadprog)

optimize_portfolio <- function(returns, target_return = 0.10) {
    # Expected returns vector
    mu <- colMeans(returns)
    
    # Covariance matrix (risk model)
    Sigma <- cov(returns)
    
    # Quadratic programming formulation:
    # minimize: (1/2) * w^T * Sigma * w  (portfolio variance)
    # subject to: w^T * mu >= target_return (return constraint)
    #            w^T * 1 = 1             (fully invested)
    #            w >= 0                  (long-only)
    
    # Solve QP problem
    n <- ncol(returns)
    Dmat <- 2 * Sigma  # Quadratic term matrix
    dvec <- rep(0, n)  # Linear term vector
    
    # Constraint matrices
    Amat <- cbind(mu, rep(1, n), diag(n))  # return, sum, individual weights
    bvec <- c(target_return, 1, rep(0, n))  # constraint values
    
    solution <- solve.QP(Dmat, dvec, Amat, bvec, meq = 2)
    
    return(list(
        weights = solution$solution,
        risk = sqrt(solution$value),
        return = sum(solution$solution * mu)
    ))
}
```

**Business Impact:**
- **Risk management**: Quantify portfolio risk precisely
- **Return optimization**: Achieve 12% returns with 8% volatility
- **Client confidence**: Transparent, mathematically-driven investment process

### Troubleshooting Common Workplace Issues

#### Performance Problems
```python
# Common issue: Slow matrix operations
# Problem: Dense matrix operations on sparse data

# BAD: Dense operations
dense_matrix = np.zeros((1000000, 1000000))  # 8 TB memory!
result = dense_matrix @ another_dense_matrix  # Hours to compute

# GOOD: Sparse operations  
from scipy.sparse import csr_matrix
sparse_matrix = csr_matrix((data, (rows, cols)), shape=(1000000, 1000000))
result = sparse_matrix @ another_sparse_matrix  # Seconds to compute

# Workplace impact: Algorithm runs in production vs. crashes
```

#### Numerical Stability Issues
```python
# Common issue: Matrix inversion failures
# Problem: Poorly conditioned matrices

def robust_linear_solve():
    # BAD: Direct inversion
    try:
        beta = np.linalg.inv(X.T @ X) @ X.T @ y  # Can fail!
    except np.linalg.LinAlgError:
        return "Matrix is singular"
    
    # GOOD: Pseudoinverse or regularization
    beta = np.linalg.pinv(X) @ y  # Always works
    
    # BETTER: Ridge regression for stability
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=0.01)  # Regularization prevents singularity
    beta = model.fit(X, y).coef_
    
    return beta

# Business impact: Model works reliably vs. crashes in production
```

## Strategic Career Impact

### 1. Technical Leadership Opportunities
Understanding linear algebra deeply allows you to:
- **Debug complex ML systems** when others cannot
- **Optimize algorithms** for business constraints (memory, speed, cost)
- **Design new solutions** rather than just applying existing ones
- **Communicate effectively** with both technical teams and executives

### 2. Business Value Creation
- **Cost optimization**: Choose algorithms that scale efficiently
- **Revenue generation**: Build better recommendation/prediction systems
- **Risk mitigation**: Understand when models might fail
- **Strategic planning**: Evaluate feasibility of ML initiatives

### 3. Cross-Functional Collaboration
- **Product teams**: Explain what's possible with current data/compute
- **Business teams**: Translate technical constraints into business terms
- **Executive teams**: Provide realistic timelines and resource requirements
- **Engineering teams**: Design scalable production systems

## Key Takeaways for Professional Success

1. **Linear algebra is the computational backbone** of all modern AI/ML systems
2. **Matrix operations enable massive scale** - from millions to billions of data points
3. **Understanding complexity trade-offs** helps make informed business decisions
4. **Memory and computational efficiency** directly impact profitability
5. **Robust implementations** prevent costly production failures
6. **Domain expertise + linear algebra** creates unique competitive advantages

**The Bottom Line**: Linear algebra isn't just math—it's the language of data-driven business innovation. Mastering these concepts positions you to lead technical initiatives, optimize business operations, and drive strategic decisions in any data-intensive industry.

Every successful data scientist, ML engineer, or quantitative analyst relies on these fundamentals daily. The depth of your understanding directly correlates with your ability to solve complex business problems and advance your career.