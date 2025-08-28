# Topic 3: Logistic Regression and Classification

## Learning Objectives
By the end of this topic, managers should understand:
- When to use classification models versus regression models
- How logistic regression works and how to interpret results
- Key metrics for evaluating classification performance
- Business applications of classification in customer analytics, risk management, and operations

## 1. Introduction to Classification Problems

### Classification vs. Regression
**Regression**: Predicts continuous numerical values
- Sales revenue ($125,000)
- Customer lifetime value ($2,450)
- Processing time (4.7 minutes)

**Classification**: Predicts discrete categories or classes
- Will customer churn? (Yes/No)
- Email spam detection (Spam/Not Spam)
- Credit approval (Approve/Deny)
- Customer segment (Premium/Standard/Basic)

### Binary vs. Multi-class Classification

#### Binary Classification
**Definition**: Predicting between two possible outcomes
**Business Examples:**
- **Marketing**: Will customer respond to campaign? (Yes/No)
- **Finance**: Will loan default? (Default/No Default)
- **Operations**: Will machine fail? (Fail/Normal)
- **HR**: Will employee quit? (Leave/Stay)

#### Multi-class Classification
**Definition**: Predicting among three or more categories
**Business Examples:**
- **Customer Segmentation**: Premium/Standard/Basic
- **Product Categorization**: Electronics/Clothing/Books
- **Risk Rating**: Low/Medium/High/Critical
- **Service Priority**: Urgent/High/Normal/Low

## 2. Logistic Regression Fundamentals

### Why Not Linear Regression for Classification?

**Problem with Linear Regression:**
```
Customer Income vs. Purchase Decision:
Purchase │ 
(1=Yes)   │     ● ●
          │   ●     ●
          │ ●   ●     ●  ← Linear regression line
          │●      ●     ●
(0=No)    │  ●         ●
          └─────────────────────→ Income
```

**Issues:**
- Predictions can exceed [0,1] range
- Assumes constant effect across income levels
- Doesn't capture S-shaped relationship

### The Logistic Function

**Mathematical Form:**
$$p = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_p x_p)}}$$

**Alternative Form:**
$$\ln\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_p x_p$$

**Where:**
- $p$ = Probability of positive outcome
- $\frac{p}{1-p}$ = Odds ratio
- $\ln\left(\frac{p}{1-p}\right)$ = Log odds (logit)

### Business Interpretation of Coefficients

#### Odds Ratios
**Formula:** $OR = e^{\beta_i}$

**Example: Email Marketing Campaign**
```
Model: ln(p/(1-p)) = -2.5 + 1.2×(Age) + 0.8×(Previous_Purchases)

Coefficient Interpretations:
- Age coefficient (β₁ = 1.2):
  * Odds Ratio = e^1.2 = 3.32
  * "Each additional year of age increases odds of response by 3.32 times"
  
- Previous Purchases coefficient (β₂ = 0.8):
  * Odds Ratio = e^0.8 = 2.23  
  * "Each additional previous purchase increases odds of response by 2.23 times"
```

#### Probability Interpretation
**Converting to Business Language:**
```
For a 35-year-old customer with 2 previous purchases:
Logit = -2.5 + 1.2×35 + 0.8×2 = -2.5 + 42 + 1.6 = 41.1

Probability = 1/(1 + e^(-41.1)) ≈ 1.0 (essentially 100%)

Business Translation: "This customer profile has virtually certain campaign response"
```

### Practical Business Example: Customer Churn Prediction

#### Problem Setup
**Business Context:** Telecommunications company wants to predict customer churn
**Data Available:**
- Monthly charges
- Contract length  
- Customer service calls
- Total charges
- Payment method

#### Model Development
```r
# R Example: Churn Prediction Model
churn_model <- glm(Churn ~ MonthlyCharges + ContractLength + ServiceCalls + 
                          TotalCharges + PaymentMethod, 
                   data = telecom_data, 
                   family = binomial)

# View results
summary(churn_model)
```

#### Interpreting Results
```
Coefficient Analysis:
MonthlyCharges: β = 0.03, OR = 1.03
- "Each $1 increase in monthly charges increases churn odds by 3%"

ServiceCalls: β = 0.45, OR = 1.57  
- "Each additional service call increases churn odds by 57%"

ContractLength: β = -0.8, OR = 0.45
- "Each additional month of contract decreases churn odds by 55%"
```

## 3. Model Evaluation for Classification

### Confusion Matrix

**Structure:**
```
                 Predicted
                Yes    No
Actual  Yes    TP    FN
        No     FP    TN

Where:
TP = True Positives (correctly predicted Yes)
TN = True Negatives (correctly predicted No)  
FP = False Positives (incorrectly predicted Yes)
FN = False Negatives (incorrectly predicted No)
```

**Business Example: Fraud Detection**
```
                 Predicted
              Fraud  Normal
Actual Fraud   85     15    (100 actual fraud cases)
       Normal  20    880    (900 actual normal cases)

Business Translation:
- Caught 85% of actual fraud (85/100)
- 20 false alarms out of 900 normal transactions
- Overall accuracy: (85+880)/1000 = 96.5%
```

### Key Performance Metrics

#### Accuracy
**Formula:** $Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$
**Business Interpretation:** "What percentage of predictions were correct?"
**When Important:** When false positives and false negatives have similar costs

#### Precision (Positive Predictive Value)
**Formula:** $Precision = \frac{TP}{TP + FP}$
**Business Interpretation:** "Of predicted positives, how many were actually positive?"
**Critical When:** False positives are expensive
**Example:** "Of customers we predicted would respond to campaign, what percentage actually responded?"

#### Recall (Sensitivity, True Positive Rate)
**Formula:** $Recall = \frac{TP}{TP + FN}$
**Business Interpretation:** "Of actual positives, how many did we correctly identify?"
**Critical When:** False negatives are expensive
**Example:** "Of customers who actually churned, what percentage did we predict would churn?"

#### Specificity (True Negative Rate)
**Formula:** $Specificity = \frac{TN}{TN + FP}$
**Business Interpretation:** "Of actual negatives, how many did we correctly identify?"
**Example:** "Of customers who stayed loyal, what percentage did we correctly predict would stay?"

### The Precision-Recall Tradeoff

#### Business Context: Medical Diagnosis
```
Conservative Approach (High Recall):
- Lower threshold for positive prediction
- Catch more actual cases (fewer false negatives)
- More false alarms (lower precision)
- Better for life-threatening conditions

Selective Approach (High Precision):
- Higher threshold for positive prediction
- Fewer false alarms (higher precision)
- Miss more actual cases (lower recall)
- Better when follow-up is expensive
```

#### ROC Curve and AUC
**ROC Curve**: Plots True Positive Rate vs. False Positive Rate
**AUC (Area Under Curve)**: Single metric summarizing model performance
- **AUC = 0.5**: Random guessing
- **AUC = 0.7-0.8**: Acceptable performance
- **AUC = 0.8-0.9**: Excellent performance
- **AUC > 0.9**: Outstanding performance (check for overfitting)

**Business Interpretation:**
"AUC = 0.85 means the model correctly ranks a randomly chosen positive case higher than a randomly chosen negative case 85% of the time"

## 4. Advanced Classification Applications

### Customer Segmentation

#### RFM Analysis with Classification
**Variables:**
- **Recency**: Days since last purchase
- **Frequency**: Number of purchases in time period
- **Monetary**: Total spending amount

**Classification Goal:** Predict customer value tier
```
Business Segments:
Champions: High R, F, M (retain and reward)
Loyal Customers: High F, M, medium R (re-engage)
At Risk: Low R, high F, M (win-back campaigns)
Lost Customers: Low R, F, M (difficult to recover)
```

**Model Implementation:**
```python
# Python Example: Customer Segmentation
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Prepare features
features = ['Recency', 'Frequency', 'Monetary', 'Tenure', 'AvgOrderValue']
X = customer_data[features]
y = customer_data['ValueTier']  # High/Medium/Low

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train multi-class logistic regression
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_scaled, y)

# Predict probabilities for new customers
probabilities = model.predict_proba(new_customer_features)
```

### Credit Risk Assessment

#### Traditional Scorecard Approach
**Business Problem:** Determine loan approval and interest rate
**Key Variables:**
- Credit score
- Debt-to-income ratio
- Employment history
- Loan amount relative to income
- Previous defaults

**Model Structure:**
```
Default Risk = β₀ + β₁×CreditScore + β₂×DebtToIncome + 
               β₃×Employment + β₄×LoanRatio + β₅×PreviousDefaults

Business Rules:
- High Risk (p > 0.3): Deny loan
- Medium Risk (0.1 < p < 0.3): Approve with higher rate
- Low Risk (p < 0.1): Approve with standard rate
```

#### Regulatory Considerations
**Fair Lending Requirements:**
- Cannot discriminate based on protected characteristics
- Model must be explainable and defensible
- Need to demonstrate business necessity
- Regular monitoring for disparate impact

### Marketing Response Modeling

#### Campaign Targeting
**Business Objective:** Maximize ROI of marketing campaign
**Approach:** Predict probability of positive response

**Cost-Benefit Analysis:**
```
Campaign Economics:
Cost per contact: $5
Revenue per conversion: $50
Break-even response rate: $5/$50 = 10%

Model-Based Targeting:
- Target customers with predicted response rate > 15%
- Expected profit per contact: 0.15 × $50 - $5 = $2.50
- Campaign becomes profitable
```

**Lift Analysis:**
```
Model Performance vs. Random:
Random targeting: 8% response rate
Model targeting (top decile): 24% response rate
Lift = 24%/8% = 3.0x improvement

Business Impact:
- 3x better response rate
- Higher ROI per marketing dollar
- Reduced waste on unlikely responders
```

## 5. Implementation and Model Management

### Feature Engineering for Classification

#### Creating Predictive Variables
**Behavioral Features:**
```python
# Customer behavior features
customer_data['Days_Since_Last_Purchase'] = (today - last_purchase_date).days
customer_data['Purchase_Frequency'] = purchases / tenure_months
customer_data['Avg_Order_Value'] = total_spend / total_orders
customer_data['Trend_Spending'] = recent_3months / previous_3months
```

**Interaction Terms:**
```python
# Business logic interactions
customer_data['High_Value_Frequent'] = (avg_order_value > 100) & (frequency > 10)
customer_data['Risk_Score'] = (days_since_last * 0.3) + (complaint_count * 0.4)
```

#### Categorical Variable Encoding
**One-Hot Encoding:**
```python
# Convert categorical variables
payment_method_dummies = pd.get_dummies(customer_data['PaymentMethod'], 
                                       prefix='Payment')
customer_features = pd.concat([customer_data, payment_method_dummies], axis=1)
```

### Threshold Selection for Business Decisions

#### Cost-Sensitive Classification
**Framework:**
```
Business Decision Matrix:
                Predict No    Predict Yes
Actually No     TN (Correct)  FP (Cost = C₁)
Actually Yes    FN (Cost = C₂) TP (Benefit = B)

Optimal Threshold when: B × P(TP) - C₁ × P(FP) - C₂ × P(FN) is maximized
```

**Example: Customer Retention Campaign**
```
Costs and Benefits:
- Retention offer cost: $50 per customer
- Revenue from retained customer: $500
- Cost of losing customer: $300

Threshold Analysis:
- High threshold (0.8): Few campaigns, high precision
- Low threshold (0.2): Many campaigns, high recall
- Optimal threshold (0.4): Maximizes expected profit
```

### Model Monitoring and Maintenance

#### Performance Degradation Detection
**Monitoring Metrics:**
- Accuracy trends over time
- Precision/recall stability
- Population drift (changing customer base)
- Feature importance changes

**Alert Thresholds:**
```python
# Model performance monitoring
def monitor_model_performance(current_metrics, baseline_metrics, threshold=0.05):
    accuracy_drop = baseline_metrics['accuracy'] - current_metrics['accuracy']
    precision_drop = baseline_metrics['precision'] - current_metrics['precision']
    
    if accuracy_drop > threshold or precision_drop > threshold:
        alert_system.send_alert("Model performance degradation detected")
        return True
    return False
```

#### Model Retraining Strategy
**Triggers for Retraining:**
1. Performance drops below threshold
2. Significant changes in business environment
3. New data reveals previously unseen patterns
4. Regulatory requirements change

**Retraining Process:**
```python
def automated_retraining_pipeline():
    # 1. Check if retraining is needed
    if performance_check() or time_since_last_training > 90_days:
        
        # 2. Prepare fresh training data
        new_data = fetch_recent_data(months=12)
        
        # 3. Retrain model
        new_model = train_logistic_regression(new_data)
        
        # 4. Validate performance
        if validate_model(new_model) > current_model_performance():
            deploy_model(new_model)
            log_model_update()
```

## 6. Advanced Topics and Extensions

### Regularized Logistic Regression

#### When Standard Logistic Regression Fails
**Problems:**
- Too many features relative to sample size
- Multicollinearity among predictors
- Overfitting to training data

#### Ridge Regression (L2 Regularization)
**Objective Function:**
$$\text{Cost} = \text{Log-likelihood} + \lambda \sum_{i=1}^p \beta_i^2$$

**Business Benefits:**
- Prevents overfitting with many variables
- More stable predictions
- Better generalization to new data

**Implementation:**
```python
from sklearn.linear_model import LogisticRegression

# Ridge logistic regression
ridge_model = LogisticRegression(penalty='l2', C=1.0)  # C = 1/λ
ridge_model.fit(X_train, y_train)
```

#### Lasso Regression (L1 Regularization)
**Objective Function:**
$$\text{Cost} = \text{Log-likelihood} + \lambda \sum_{i=1}^p |\beta_i|$$

**Business Benefits:**
- Automatic feature selection
- Creates simpler, interpretable models
- Identifies most important variables

**Example: Customer Churn Model**
```python
# Lasso for feature selection
lasso_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)
lasso_model.fit(X_train, y_train)

# Identify selected features
selected_features = X.columns[lasso_model.coef_[0] != 0]
print(f"Important features: {selected_features.tolist()}")
```

### Ensemble Methods Preview

#### Why Single Models May Not Be Enough
**Limitations:**
- Linear relationship assumptions
- Sensitive to outliers
- May miss complex interactions
- Limited flexibility

**Ensemble Approach:**
- Combine multiple models
- Better performance than individual models
- More robust predictions
- Reduced overfitting risk

**Examples:**
- Random Forest (covered in next topic)
- Gradient Boosting
- Stacking multiple classifiers

## Common Business Pitfalls and Solutions

### 1. Class Imbalance
**Problem:** Rare events (fraud, churn) create imbalanced datasets
```
Typical Churn Dataset:
Stayed: 95% (19,000 customers)
Churned: 5% (1,000 customers)
```

**Solutions:**
- **Oversampling**: Generate synthetic examples of minority class
- **Undersampling**: Reduce majority class size
- **Cost-sensitive learning**: Penalize misclassifying minority class more
- **Threshold adjustment**: Lower threshold for positive prediction

### 2. Feature Leakage
**Problem:** Using information that wouldn't be available at prediction time
**Example:** Using "account closure date" to predict churn
**Solution:** Only use features available before the event occurs

### 3. Overfitting to Historical Patterns
**Problem:** Model captures noise rather than signal
**Signs:** Very high training accuracy, poor test performance
**Solutions:** 
- Cross-validation
- Regularization
- Simpler models
- More training data

### 4. Misinterpreting Probability Outputs
**Problem:** Treating predicted probabilities as certainties
**Example:** "Model says 85% probability of churn, so customer will definitely churn"
**Solution:** Communicate uncertainty and use probability ranges

## Key Takeaways for Managers

### 1. Classification is About Decision-Making
- Focus on business outcomes, not just accuracy
- Consider costs of different types of errors
- Set thresholds based on business logic

### 2. Model Performance is Context-Dependent
- High accuracy isn't always the goal
- Sometimes catching rare events is more important
- Consider precision vs. recall tradeoffs

### 3. Logistic Regression Provides Interpretability
- Coefficients have clear business meaning
- Can explain model decisions to stakeholders
- Good baseline before trying complex methods

### 4. Monitor and Maintain Models
- Performance degrades over time
- Business conditions change
- Regular retraining is necessary

### 5. Balance Sophistication with Practicality
- Start with simple, interpretable models
- Add complexity only when justified
- Ensure models can be implemented and maintained

## Practical Exercises

### Exercise 1: Business Problem Identification
Identify three classification problems in your organization:
1. Define the business objective
2. Specify the target variable (binary or multi-class)
3. List available predictive features
4. Describe how wrong predictions would affect the business

### Exercise 2: Metric Selection
For each classification problem above:
1. Determine whether precision or recall is more important
2. Justify your choice with business reasoning
3. Set appropriate alert thresholds for model monitoring

### Exercise 3: Model Interpretation
Given this customer churn model output:
```
Coefficient Estimates:
Intercept: -2.1
Monthly_Charges: 0.02
Service_Calls: 0.35
Contract_Length: -0.8
Payment_Auto: -0.6
```

1. Calculate odds ratios for each variable
2. Interpret each coefficient in business terms
3. Calculate churn probability for a specific customer profile
4. Recommend business actions based on the insights

---

**Next Topic**: [Decision Trees and Random Forests](./Topic-04-Decision-Trees-and-Random-Forests.md) - Learn how to use tree-based methods for classification and their advantages in business contexts.