# Topic 4: Decision Trees and Random Forests

## Learning Objectives
By the end of this topic, managers should understand:
- How decision trees work and when to use them for business problems
- The advantages of tree-based methods for interpretability
- How Random Forests improve upon single decision trees
- Business applications of ensemble methods for complex decision-making

## 1. Decision Trees: The Business Logic Approach

### What Makes Decision Trees Special?

**Key Advantages:**
- **Interpretability**: Mirror human decision-making processes
- **No Assumptions**: Don't require linear relationships or normal distributions
- **Handle Mixed Data**: Work with both numerical and categorical variables
- **Automatic Feature Selection**: Focus on most informative variables
- **Business Rules**: Can be easily translated into if-then rules

### How Decision Trees Work

#### The Splitting Process
**Goal**: Create homogeneous groups (pure nodes)
**Method**: Ask yes/no questions that best separate the data

**Business Example: Loan Approval Decision Tree**
```
Root Node: All loan applications (10,000 cases)
├─ Credit Score >= 650? 
   ├─ YES: Debt-to-Income < 0.4?
   │   ├─ YES: APPROVE (95% success rate)
   │   └─ NO: Employment > 2 years?
   │       ├─ YES: APPROVE (88% success rate)
   │       └─ NO: REJECT (45% success rate)
   └─ NO: Income >= $50k?
       ├─ YES: Previous Defaults = 0?
       │   ├─ YES: APPROVE (72% success rate)
       │   └─ NO: REJECT (25% success rate)
       └─ NO: REJECT (15% success rate)
```

#### Splitting Criteria

**For Classification: Gini Impurity**
$$\text{Gini} = 1 - \sum_{i=1}^{c} p_i^2$$

Where $p_i$ is the proportion of class $i$ in the node.

**Business Interpretation:**
- Gini = 0: Pure node (all same class)
- Gini = 0.5: Maximum impurity (equal mix of classes)

**For Regression: Mean Squared Error**
$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \bar{y})^2$$

**Information Gain:**
$$\text{Gain} = \text{Impurity}_{\text{parent}} - \sum \frac{n_{\text{child}}}{n_{\text{parent}}} \times \text{Impurity}_{\text{child}}$$

### Real Business Example: Customer Segmentation

#### Problem Setup
**Business Context:** E-commerce company wants to identify high-value customers
**Available Data:**
- Annual spending
- Number of orders
- Average order value
- Time as customer
- Geographic region
- Device preference (mobile/desktop)

#### Decision Tree Analysis
```python
# Python implementation example
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Prepare data
features = ['Annual_Spending', 'Num_Orders', 'Avg_Order_Value', 
           'Tenure_Months', 'Region', 'Device_Preference']
X = customer_data[features]
y = customer_data['Customer_Segment']  # High/Medium/Low value

# Convert categorical variables
X_encoded = pd.get_dummies(X, columns=['Region', 'Device_Preference'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# Train decision tree
tree_model = DecisionTreeClassifier(max_depth=5, min_samples_split=100, random_state=42)
tree_model.fit(X_train, y_train)

# Evaluate
train_accuracy = tree_model.score(X_train, y_train)
test_accuracy = tree_model.score(X_test, y_test)
```

#### Interpreting the Results
```
Resulting Business Rules:
1. IF Annual_Spending >= $2,000 AND Avg_Order_Value >= $150
   THEN Customer_Segment = "High Value"

2. IF Annual_Spending >= $500 AND Num_Orders >= 10 AND Tenure_Months >= 12
   THEN Customer_Segment = "Medium Value"
   
3. IF Annual_Spending < $500 OR Num_Orders < 3
   THEN Customer_Segment = "Low Value"

Business Applications:
- Personalized marketing campaigns
- Customer service prioritization  
- Loyalty program tier assignment
- Inventory recommendations
```

### Feature Importance in Trees

#### Understanding Variable Importance
**Calculation**: Based on how much each variable reduces impurity when used for splits
**Business Value**: Identifies which factors most influence outcomes

**Example Output:**
```
Feature Importance Rankings:
1. Annual_Spending: 0.35 (35% of total importance)
2. Avg_Order_Value: 0.28 (28% of total importance)  
3. Num_Orders: 0.20 (20% of total importance)
4. Tenure_Months: 0.12 (12% of total importance)
5. Region_West: 0.03 (3% of total importance)
6. Device_Mobile: 0.02 (2% of total importance)

Business Insights:
- Spending behavior more important than demographics
- Focus retention efforts on spending patterns
- Geographic and device factors less critical
```

## 2. Advantages and Limitations of Single Trees

### Business Advantages

#### 1. Transparency and Explainability
**Stakeholder Communication:**
```
Executive Summary: Customer Segmentation Rules
"Our analysis identified three key factors for high-value customers:
1. Annual spending above $2,000 (most important)
2. Average order value above $150 (second most important)
3. More than 10 orders per year (third most important)

These rules correctly classify 87% of our customers and can be 
implemented directly in our CRM system."
```

#### 2. Handle Missing Data Naturally
**Surrogate Splits**: Trees can use alternative questions when data is missing
**Business Benefit**: No need to exclude customers with incomplete information

#### 3. Capture Non-linear Relationships
**Example**: Customer satisfaction might have threshold effects
```
Traditional Linear Assumption:
Satisfaction = β₀ + β₁ × Response_Time

Tree-Based Reality:
IF Response_Time <= 5 minutes THEN Satisfaction = "High"
ELIF Response_Time <= 15 minutes THEN Satisfaction = "Medium"  
ELSE Satisfaction = "Low"
```

#### 4. Automatic Interaction Detection
**Trees Automatically Find**:
- Age effects that vary by income level
- Product preferences that differ by region
- Promotional effectiveness that depends on customer history

### Limitations of Single Trees

#### 1. High Variance (Instability)
**Problem**: Small changes in data can create very different trees
**Business Impact**: Inconsistent business rules across time periods
**Example**: 
```
Training Set A produces rule: "If Age > 35, then High Value"
Training Set B produces rule: "If Income > $50k, then High Value"
Same underlying population, different samples
```

#### 2. Overfitting Tendencies
**Problem**: Trees can become too specific to training data
**Signs**:
- Perfect accuracy on training data
- Poor performance on new customers
- Very deep trees with few examples per leaf

#### 3. Bias Toward Features with More Values
**Problem**: Categorical variables with many levels get selected more often
**Business Example**: ZIP code (hundreds of values) might dominate over region (4 values)

#### 4. Difficulty with Linear Relationships
**Problem**: Trees approximate smooth relationships with step functions
**Business Impact**: May require many splits to capture simple linear trends

## 3. Random Forests: The Ensemble Solution

### The Wisdom of Crowds Approach

**Core Concept**: Combine many imperfect models to create a superior predictor
**Analogy**: Like asking 100 experts instead of relying on one opinion

#### How Random Forests Work

**Step 1: Bootstrap Sampling**
```
Original Dataset: 10,000 customers
Tree 1: Random sample of 10,000 customers (with replacement)
Tree 2: Different random sample of 10,000 customers (with replacement)
...
Tree 100: Another different random sample of 10,000 customers
```

**Step 2: Random Feature Selection**
```
Available Features: 20 variables
Tree 1: Randomly select 4 variables for each split decision
Tree 2: Randomly select 4 different variables for each split decision
...
Each tree sees only a subset of variables
```

**Step 3: Ensemble Prediction**
```
Classification: Majority vote across all trees
Tree 1 predicts: "High Value"
Tree 2 predicts: "Medium Value"  
Tree 3 predicts: "High Value"
...
Tree 100 predicts: "High Value"

Final Prediction: "High Value" (60% of trees voted this way)
Confidence: 60% probability of "High Value"
```

### Business Benefits of Random Forests

#### 1. Improved Accuracy
**Typical Performance Gains:**
- 2-5% accuracy improvement over single trees
- More consistent performance across different datasets
- Better handling of complex patterns

**Business Value:**
```
Customer Churn Prediction Comparison:
Single Decision Tree: 78% accuracy
Random Forest: 84% accuracy
Business Impact: 6% improvement = 60 more customers saved per 1,000
```

#### 2. Automatic Overfitting Protection
**Built-in Regularization:**
- Each tree sees different data and features
- Averaging reduces model complexity
- More stable predictions over time

#### 3. Robust Feature Importance
**Aggregate Importance Scores:**
```
Feature Importance (averaged across 100 trees):
Annual_Spending: 0.31 ± 0.04 (stable across trees)
Avg_Order_Value: 0.25 ± 0.03 (consistent ranking)
Num_Orders: 0.22 ± 0.05 (reliable importance)
```

#### 4. Uncertainty Quantification
**Prediction Confidence:**
```python
# Get prediction probabilities from Random Forest
probabilities = rf_model.predict_proba(new_customer_data)

# Example output for customer segmentation:
High Value: 0.65 (65% of trees voted this way)
Medium Value: 0.30 (30% of trees voted this way)  
Low Value: 0.05 (5% of trees voted this way)

# Business interpretation: "65% confident this is a high-value customer"
```

### Practical Implementation Example

#### Business Problem: Marketing Campaign Response
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Prepare marketing campaign data
features = ['Age', 'Income', 'Previous_Purchases', 'Days_Since_Last_Purchase', 
           'Email_Opens_Last_Month', 'Website_Visits', 'Customer_Tenure']
X = campaign_data[features]
y = campaign_data['Responded']  # 1 = responded, 0 = did not respond

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    stratify=y, random_state=42)

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,        # 100 trees
    max_depth=10,           # Limit tree depth
    min_samples_split=50,   # Minimum samples to split
    min_samples_leaf=20,    # Minimum samples per leaf
    random_state=42
)

rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)

# Evaluate performance
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)
```

#### Business Results Interpretation
```
Campaign Response Model Results:

Performance Metrics:
- Accuracy: 82% (vs. 75% for single tree)
- Precision: 68% (of predicted responders, 68% actually respond)  
- Recall: 45% (caught 45% of actual responders)

Feature Importance:
1. Previous_Purchases: 0.28 (most predictive factor)
2. Customer_Tenure: 0.22 (loyalty matters)
3. Income: 0.18 (economic capacity)
4. Email_Opens_Last_Month: 0.15 (engagement level)
5. Age: 0.08 (demographic factor)

Business Recommendations:
1. Target customers with 3+ previous purchases
2. Focus on loyal customers (tenure > 12 months)
3. Prioritize engaged customers (recent email opens)
4. Income more important than age for targeting
```

## 4. Advanced Random Forest Applications

### Customer Lifetime Value Prediction

#### Regression with Random Forests
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Customer LTV prediction
features = ['Initial_Purchase_Amount', 'Frequency_First_3_Months', 
           'Customer_Service_Interactions', 'Referral_Source', 'Age_At_Signup']
X = customer_data[features]
y = customer_data['Lifetime_Value']

# Handle categorical variables
X_encoded = pd.get_dummies(X, columns=['Referral_Source'])

# Train Random Forest Regressor  
rf_regressor = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
rf_regressor.fit(X_train, y_train)

# Predictions and evaluation
y_pred = rf_regressor.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: ${mae:.2f}")
print(f"R-squared: {r2:.3f}")
```

#### Business Value Assessment
```
CLV Prediction Model Performance:
- Mean Absolute Error: $156 (average prediction error)
- R-squared: 0.73 (explains 73% of CLV variation)

Business Applications:
1. Customer Acquisition: Focus on high-predicted-CLV prospects
2. Marketing Spend: Allocate budget based on CLV potential
3. Retention Efforts: Prioritize high-CLV customers at risk
4. Product Development: Design features for high-CLV segments
```

### Multi-class Classification: Customer Risk Segmentation

#### Problem Setup
**Objective**: Classify customers into risk categories
**Categories**: Low Risk, Medium Risk, High Risk, Very High Risk
**Use Case**: Credit decisions, insurance pricing, fraud detection

```python
# Multi-class customer risk classification
risk_features = ['Credit_Score', 'Debt_To_Income', 'Payment_History_Score',
                'Employment_Stability', 'Savings_Ratio', 'Previous_Defaults']

# Prepare target variable (4 risk categories)
y_risk = customer_data['Risk_Category']  # 0=Low, 1=Medium, 2=High, 3=Very High

# Train multi-class Random Forest
rf_multiclass = RandomForestClassifier(n_estimators=150, max_depth=12, 
                                      class_weight='balanced', random_state=42)
rf_multiclass.fit(X_train, y_risk)

# Get prediction probabilities for all classes
risk_probabilities = rf_multiclass.predict_proba(X_test)
```

#### Business Implementation
```
Risk Classification Results:

Class Distribution in Predictions:
- Low Risk: 40% of customers
- Medium Risk: 35% of customers  
- High Risk: 20% of customers
- Very High Risk: 5% of customers

Business Rules Implementation:
- Low Risk: Standard approval process
- Medium Risk: Additional verification required
- High Risk: Manual review, higher collateral
- Very High Risk: Deny or special terms only

Model Confidence:
- 85% of predictions have >70% probability confidence
- Manual review triggered for ambiguous cases (50-70% confidence)
```

## 5. Hyperparameter Tuning for Business

### Key Random Forest Parameters

#### n_estimators (Number of Trees)
**Business Consideration**: More trees = better performance but slower computation
```
Performance vs. Speed Trade-off:
50 trees: Fast training, 80% accuracy
100 trees: Moderate speed, 83% accuracy  
200 trees: Slower training, 84% accuracy
500 trees: Much slower, 84.1% accuracy (diminishing returns)

Business Decision: Choose 100-200 trees for most applications
```

#### max_depth (Tree Depth)
**Business Impact**: Controls model complexity and overfitting
```
Depth Analysis:
Depth 5: 78% accuracy, highly interpretable
Depth 10: 82% accuracy, moderately interpretable
Depth 20: 83% accuracy, difficult to interpret
Unlimited: 85% training, 79% test (overfitting!)

Business Recommendation: max_depth=10-15 for good balance
```

#### min_samples_split and min_samples_leaf
**Purpose**: Prevent overfitting to small groups
**Business Logic**: Ensure business rules apply to meaningful customer segments

```python
# Business-oriented parameter selection
rf_business = RandomForestClassifier(
    n_estimators=100,           # Good performance-speed balance
    max_depth=12,              # Moderate complexity
    min_samples_split=100,     # Rules must apply to ≥100 customers
    min_samples_leaf=50,       # Each segment has ≥50 customers
    max_features='sqrt',       # Standard feature selection
    class_weight='balanced',   # Handle imbalanced classes
    random_state=42
)
```

### Business-Focused Hyperparameter Tuning

#### Grid Search with Business Constraints
```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# Define business metric (e.g., profit-focused scoring)
def business_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    profit = tp * 100 - fp * 20 - fn * 50  # Business-specific costs/benefits
    return profit

business_scorer = make_scorer(business_score, greater_is_better=True)

# Parameter grid focused on business needs
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [8, 12, 16],
    'min_samples_split': [50, 100, 200],
    'min_samples_leaf': [20, 50, 100]
}

# Grid search with business scoring
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring=business_scorer,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

## 6. Model Interpretation and Business Rules

### Extracting Business Rules from Random Forests

#### Individual Tree Inspection
```python
from sklearn.tree import export_text

# Extract rules from one representative tree
sample_tree = rf_model.estimators_[0]
tree_rules = export_text(sample_tree, feature_names=features, max_depth=4)

print("Sample Business Rules from Tree:")
print(tree_rules)
```

#### Feature Interaction Detection
```python
from sklearn.inspection import partial_dependence, plot_partial_dependence

# Analyze how features interact
fig, ax = plot_partial_dependence(
    rf_model, X_train, 
    features=['Annual_Spending', 'Customer_Tenure'],
    feature_names=features
)
plt.suptitle('Customer Value: Spending vs Tenure Interaction')
plt.show()
```

### Global Feature Importance Analysis
```python
# Comprehensive feature importance analysis
feature_importance_df = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_,
    'std': np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

# Business interpretation
print("Top Drivers of Customer Value:")
for idx, row in feature_importance_df.head(5).iterrows():
    print(f"{row['feature']}: {row['importance']:.3f} ± {row['std']:.3f}")
```

### SHAP Values for Individual Predictions
```python
import shap

# Initialize SHAP explainer
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test[:10])  # Explain first 10 predictions

# Visualize individual prediction explanations
shap.summary_plot(shap_values, X_test[:10], feature_names=features)
shap.waterfall_plot(explainer.expected_value[1], shap_values[1][0], X_test.iloc[0])
```

## Common Business Applications and Case Studies

### 1. Fraud Detection in Financial Services
```
Problem: Identify fraudulent transactions in real-time

Random Forest Advantages:
- Handles mixed data types (amounts, merchant categories, locations)
- Provides probability scores for risk assessment
- Robust to adversarial attempts to game the system
- Fast enough for real-time scoring

Key Features:
- Transaction amount vs. historical patterns
- Merchant category and location
- Time since last transaction
- Device and IP address patterns
- Customer behavior deviations

Business Rules Generated:
- Flag transactions >3x normal amount
- Alert on unusual merchant/location combinations
- Score based on device fingerprint changes
- Escalate customers with >2 recent anomalies
```

### 2. Supply Chain Optimization
```
Problem: Predict demand for inventory planning

Random Forest Application:
- Multiple seasonality patterns (weekly, monthly, annual)
- Weather and economic indicators
- Promotional and competitive effects
- Product lifecycle considerations

Features:
- Historical sales patterns
- Economic indicators
- Weather forecasts
- Promotional calendar
- Competitor activities
- Social media sentiment

Business Value:
- 15% reduction in stockouts
- 8% decrease in excess inventory
- Improved cash flow management
- Better customer satisfaction
```

### 3. Employee Retention Modeling
```
Problem: Predict which employees are likely to quit

HR Analytics with Random Forest:
- Survey responses and engagement scores
- Performance reviews and career progression
- Compensation and benefits utilization
- Work-life balance indicators
- Manager relationship scores

Actionable Insights:
- Compensation gaps drive 40% of departures
- Career development opportunities matter most for high performers
- Work-life balance critical for parents
- Manager quality affects retention more than salary

Intervention Strategies:
- Proactive salary adjustments for at-risk employees
- Targeted development programs
- Flexible work arrangements
- Manager coaching for low-scoring supervisors
```

## Key Takeaways for Managers

### 1. Decision Trees Provide Interpretability
- Easy to understand and communicate to stakeholders
- Generate clear business rules
- Handle complex data naturally
- Good starting point for any classification problem

### 2. Random Forests Improve Performance
- Consistently outperform single trees
- Provide uncertainty estimates
- More stable over time
- Still reasonably interpretable

### 3. Focus on Business-Relevant Metrics
- Accuracy isn't always the right goal
- Consider costs of different types of errors
- Tune models for business outcomes, not statistical metrics

### 4. Balance Complexity with Interpretability
- Random Forests offer good middle ground
- More complex than logistic regression
- Less complex than neural networks
- Suitable for most business applications

### 5. Feature Engineering Still Matters
- Good features lead to better models
- Domain knowledge improves results
- Focus on business-meaningful variables
- Regular feature importance review

## Practical Exercises

### Exercise 1: Business Problem Formulation
Choose a classification problem in your organization:
1. Define the business objective clearly
2. Identify available features and target variable
3. Determine success metrics beyond accuracy
4. Consider implementation constraints

### Exercise 2: Model Comparison
For your chosen problem:
1. Build a single decision tree
2. Build a random forest
3. Compare performance and interpretability
4. Determine which approach better suits your business needs

### Exercise 3: Feature Importance Analysis
Using your random forest model:
1. Analyze feature importance rankings
2. Translate findings into business insights
3. Identify features that could be improved or collected
4. Recommend actions based on key drivers

### Exercise 4: Business Rules Extraction
From your decision tree:
1. Extract the top 5 most important business rules
2. Validate these rules with domain experts
3. Assess feasibility of implementing rules operationally
4. Estimate business impact of following these rules

---

**Next Topic**: [Regularization and Feature Selection](./Topic-05-Regularization-and-Feature-Selection.md) - Learn how to build robust models by controlling complexity and selecting the most relevant features for business decisions.