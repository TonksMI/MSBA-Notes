# Cell 1
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
import statsmodels.formula.api as smf
from IPython.display import display

plt.style.use('seaborn-v0_8')
sns.set_theme(style='whitegrid', palette='muted')
warnings.filterwarnings('ignore')


# Cell 2
# Your code here
NOTEBOOK_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
DATA_DIR = NOTEBOOK_DIR / 'datasets'
career_df = pd.read_csv(DATA_DIR / 'career_outcomes_survey.csv')

print("Dataset Shape:", career_df.shape)
print("\n" + "="*80)
print("Data Types:")
print(career_df.dtypes)
print("\n" + "="*80)
print("Missing Values:")
print(career_df.isnull().sum())
print("\n" + "="*80)
print("First few rows:")
print(career_df.head())

# Check for impossible values
print("Data Quality Checks:")
print(f"Negative salaries: {(career_df['salary_current'] < 0).sum()}")

print(f"GPA > 4.0: {(career_df['gpa'] > 4.0).sum()}")

print(f"Negative years_experience: {(career_df['years_experience'] < 0).sum()}")

print(f"Salary starting > current: {(career_df['salary_starting'] > career_df['salary_current']).sum()}")


# Cell 4
career_df = career_df[career_df['salary_current'].notnull() & 
                      career_df['salary_starting'].notnull() & 
                      career_df['years_experience'].notnull()]

if 'gpa' in career_df.columns:
    career_df['gpa'] = career_df['gpa'].fillna(career_df['gpa'].median())

categorical_cols = career_df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    career_df[col] = career_df[col].fillna('Unknown')

print(f"\nRows after cleaning: {len(career_df)}")

# Cell 6
career_df['log_salary_current'] = np.log(career_df['salary_current'])
career_df['log_salary_starting'] = np.log(career_df['salary_starting'])
career_df['salary_growth'] = (career_df['salary_current'] - career_df['salary_starting']) / career_df['salary_starting']

# Create elite_university indicator
if 'university_tier' in career_df.columns:
    career_df['elite_university'] = career_df['university_tier'].isin(['Top 10', 'Top 50']).astype(int)
else:
    career_df['elite_university'] = 0

# Create stem_major indicator
if 'major' in career_df.columns:
    stem_majors = ['Engineering', 'Computer Science', 'Data Science', 'Mathematics']
    career_df['stem_major'] = career_df['major'].isin(stem_majors).astype(int)
else:
    career_df['stem_major'] = 0
print(career_df.columns)

# Cell 7
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Histogram of salary_current with normal curve overlay
axes[0, 0].hist(career_df['salary_current'], bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
mu, sigma = career_df['salary_current'].mean(), career_df['salary_current'].std()
x = np.linspace(career_df['salary_current'].min(), career_df['salary_current'].max(), 100)
axes[0, 0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label='Normal Distribution')
axes[0, 0].set_title('Distribution of Current Salary', fontweight='bold', fontsize=12)
axes[0, 0].set_xlabel('Salary ($)')
axes[0, 0].set_ylabel('Density')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Histogram of log_salary_current
axes[0, 1].hist(career_df['log_salary_current'], bins=50, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
mu_log, sigma_log = career_df['log_salary_current'].mean(), career_df['log_salary_current'].std()
x_log = np.linspace(career_df['log_salary_current'].min(), career_df['log_salary_current'].max(), 100)
axes[0, 1].plot(x_log, stats.norm.pdf(x_log, mu_log, sigma_log), 'r-', lw=2, label='Normal Distribution')
axes[0, 1].set_title('Distribution of Log(Current Salary)', fontweight='bold', fontsize=12)
axes[0, 1].set_xlabel('Log(Salary)')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

print(f"\nSalary Distribution Skewness:")
print(f"  Current Salary: {career_df['salary_current'].skew():.3f}")
print(f"  Log Salary: {career_df['log_salary_current'].skew():.3f}")
print(f"  More normal- {'Log-transformed' if abs(career_df['log_salary_current'].skew()) < abs(career_df['salary_current'].skew()) else 'Raw'}")

# Plot 3: Scatter plot - GPA vs salary_current
if 'gpa' in career_df.columns:
    axes[1, 0].scatter(career_df['gpa'], career_df['salary_current'], alpha=0.5, s=30)
    axes[1, 0].set_title('GPA vs Current Salary', fontweight='bold', fontsize=12)
    axes[1, 0].set_xlabel('GPA')
    axes[1, 0].set_ylabel('Current Salary ($)')
    axes[1, 0].grid(alpha=0.3)

# Plot 4: Experience vs salary_current
axes[1, 1].scatter(career_df['years_experience'], career_df['salary_current'], alpha=0.5, s=30, color='coral')
axes[1, 1].set_title('Years Experience vs Current Salary', fontweight='bold', fontsize=12)
axes[1, 1].set_xlabel('Years of Experience')
axes[1, 1].set_ylabel('Current Salary ($)')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Scatter plot matrix
if 'gpa' in career_df.columns and 'job_satisfaction' in career_df.columns:
    scatter_matrix_vars = ['gpa', 'years_experience', 'salary_current', 'job_satisfaction']
    scatter_df = career_df[scatter_matrix_vars].dropna()
    
    pd.plotting.scatter_matrix(scatter_df, figsize=(12, 12), alpha=0.5, diagonal='hist')
    plt.suptitle('Scatter Plot Matrix: Key Career Variables', fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.show()

# Cell 8

print("Summary Statistics:")
print(career_df[['salary_current', 'salary_starting', 'salary_growth', 
                'years_experience', 'gpa']].describe())

# Cell 10
career_df.columns

# Cell 11
import numpy as np
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

train_df, test_df = train_test_split(career_df, test_size=0.30, random_state=42)
print(f'Train size: {len(train_df):,}, Test size: {len(test_df):,}')

baseline_formula = 'salary_current ~ years_experience + gpa + internship_count + elite_university + stem_major'
baseline_model = smf.ols(baseline_formula, data=train_df).fit()
baseline_summary = baseline_model.summary2().tables[1]
print('Baseline model coefficients:')
display(baseline_summary)

baseline_preds = baseline_model.predict(test_df)
baseline_eval = {
    'Model': 'Baseline Linear',
    'R2': r2_score(test_df['salary_current'], baseline_preds),
    'RMSE': np.sqrt(mean_squared_error(test_df['salary_current'], baseline_preds)),
    'MAE': mean_absolute_error(test_df['salary_current'], baseline_preds)
}

# Cell 12
log_formula = ('log_salary_current ~ years_experience + I(years_experience**2) + gpa + internship_count + '
               'elite_university + stem_major + C(major) + C(industry)')
log_model = smf.ols(log_formula, data=train_df).fit()
log_summary = log_model.summary2().tables[1]
print()
print('Log-linear model coefficients:')
display(log_summary)

log_preds_log = log_model.predict(test_df)
log_preds = np.exp(log_preds_log)
log_eval = {
    'Model': 'Log-Linear',
    'R2': r2_score(test_df['salary_current'], log_preds),
    'RMSE': np.sqrt(mean_squared_error(test_df['salary_current'], log_preds)),
    'MAE': mean_absolute_error(test_df['salary_current'], log_preds)
}

# Cell 13
print()
print('Model comparison (test set):')
display(pd.DataFrame([baseline_eval, log_eval]))


# Cell 14
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(test_df['salary_current'], baseline_preds, alpha=0.4)
axes[0].plot([test_df['salary_current'].min(), test_df['salary_current'].max()],
             [test_df['salary_current'].min(), test_df['salary_current'].max()], color='red', linestyle='--')
axes[0].set_title('Baseline Model -  Actual vs. Predicted')
axes[0].set_xlabel('Actual Salary')
axes[0].set_ylabel('Predicted Salary')

axes[1].scatter(test_df['salary_current'], log_preds, alpha=0.4, color='purple')
axes[1].plot([test_df['salary_current'].min(), test_df['salary_current'].max()],
             [test_df['salary_current'].min(), test_df['salary_current'].max()], color='red', linestyle='--')
axes[1].set_title('Log-Linear Model -  Actual vs. Predicted')
axes[1].set_xlabel('Actual Salary')
axes[1].set_ylabel('Predicted Salary')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(baseline_preds, test_df['salary_current'] - baseline_preds, alpha=0.4)
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_title('Baseline Residuals vs. Fitted')
axes[0].set_xlabel('Fitted Salary')
axes[0].set_ylabel('Residual')

axes[1].scatter(log_preds, test_df['salary_current'] - log_preds, alpha=0.4, color='purple')
axes[1].axhline(0, color='red', linestyle='--')
axes[1].set_title('Log-Linear Residuals vs. Fitted')
axes[1].set_xlabel('Fitted Salary')
axes[1].set_ylabel('Residual')
plt.tight_layout()
plt.show()

# Store for later cells
q2_models = {
    'baseline_model': baseline_model,
    'log_model': log_model,
    'baseline_eval': baseline_eval,
    'log_eval': log_eval,
    'baseline_preds': baseline_preds,
    'log_preds': log_preds,
    'test_df': test_df,
    'train_df': train_df
}

# Cell 16
# Get the coefficient for years_experience
coef = log_model.params['years_experience']

# Exponentiate it
exp_coef = np.exp(coef)
print(f"Exponentiated coefficient for years_experience: {exp_coef:.4f}")    

# Cell 18
print(np.exp(0.05))

# Cell 20
# Get the coefficient for stem_major
coef = log_model.params['stem_major']

# Exponentiate it
exp_coef = np.exp(coef)
print(f"Exponentiated coefficient for stem_major: {exp_coef:.4f}")

# Cell 23
# Your code here

log_model = smf.ols('log_salary_current ~ np.log(years_experience + 1) + np.log(gpa) + np.log(internship_count + 1) + C(major) + C(industry)',
                       data=q2_models['train_df']).fit()

print('Log-log model coefficients:')
print(log_model.summary())

elasticities = log_model.params[['np.log(years_experience + 1)', 'np.log(gpa)', 'np.log(internship_count + 1)']]
elasticity_df = elasticities.rename({'np.log(years_experience + 1)': 'Experience elasticity',
                                     'np.log(gpa)': 'GPA elasticity',
                                     'np.log(internship_count + 1)': 'Internship elasticity'}).to_frame(name='elasticity')
print()
print('Salary elasticities (log-log model):')
display(elasticity_df)

# Cell 25
comparison_table = pd.DataFrame({
    'Model': ['Linear-Linear', 'Log-Linear', 'Log-Log'],
    'AIC': [q2_models['baseline_model'].aic, q2_models['log_model'].aic, log_model.aic],
    'BIC': [q2_models['baseline_model'].bic, q2_models['log_model'].bic, log_model.bic]
})

print('Model selection criteria (lower is better):')
display(comparison_table)

# Cell 27
mode_major = career_df['major'].mode()[0]
mode_industry = career_df['industry'].mode()[0]
mean_gpa = career_df['gpa'].mean()
mean_internships = career_df['internship_count'].mean()

marginal_rows = []
for years in [1, 5, 10]:
    row = {
        'years_experience': years,
        'gpa': mean_gpa,
        'internship_count': mean_internships,
        'major': mode_major,
        'industry': mode_industry
    }
    row_df = pd.DataFrame([row])
    pred_log_salary = log_model.predict(row_df)[0]
    predicted_salary = np.exp(pred_log_salary)
    elasticity = log_model.params['np.log(years_experience + 1)']
    marginal_effect = elasticity * predicted_salary / (years + 1)
    marginal_rows.append({'years_experience': years, 'predicted_salary': predicted_salary, 'marginal_effect': marginal_effect})

marginal_df = pd.DataFrame(marginal_rows)

print('Marginal salary impact of an additional year of experience:')
display(marginal_df)

# Cell 28
years_grid = np.arange(0, 21)
curve_df = []
for years in years_grid:
    base_row = {
        'years_experience': years,
        'gpa': mean_gpa,
        'internship_count': mean_internships,
        'major': mode_major,
        'industry': mode_industry
    }
    base_row_df = pd.DataFrame([base_row])
    salary = np.exp(log_model.predict(base_row_df)[0])
    marginal = elasticities['np.log(years_experience + 1)'] * salary / (years + 1)
    curve_df.append({'years_experience': years, 'marginal_effect': marginal})
curve_df = pd.DataFrame(curve_df)
curve_df = pd.DataFrame(curve_df)

plt.figure(figsize=(8, 5))
plt.plot(curve_df['years_experience'], curve_df['marginal_effect'], marker='o')
plt.title('Diminishing Marginal Returns to Experience (Log-Log Model)')
plt.xlabel('Years of experience')
plt.ylabel('Marginal salary gain (USD)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Cell 29
q2_models['log_model'] = log_model
q2_models['marginal_curve'] = curve_df

# Cell 31
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score,
    roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve
import statsmodels.formula.api as smf
# Calculate 75th percentile of current salary
salary_75th = career_df['salary_current'].quantile(0.75)
print(f"75th percentile of salary_current: ${salary_75th:,.2f}")

# Create binary target variable (recalculate to ensure consistency)
career_df['high_earner'] = (career_df['salary_current'] > salary_75th).astype(int)
# Calculate percentage of high earners
high_earner_pct = career_df['high_earner'].mean() * 100
print(f"\nPercentage of high earners: {high_earner_pct:.2f}%")

# Check class distribution
class_counts = career_df['high_earner'].value_counts().sort_index()
print("\nClass distribution:")
print(f"  Non-high earners (0): {class_counts[0]:,} ({class_counts[0]/len(career_df)*100:.1f}%)")
print(f"  High earners (1): {class_counts[1]:,} ({class_counts[1]/len(career_df)*100:.1f}%)")

# Assess imbalance
imbalance_ratio = class_counts[0] / class_counts[1]
print(f"\nImbalance ratio (majority/minority): {imbalance_ratio:.2f}:1")

if imbalance_ratio > 1.5:
    print("\n⚠️ Dataset is IMBALANCED (ratio > 1.5:1)")
    print("Consider: precision-recall curves, F1-score, and adjusting decision threshold")
else:
    print("\n✓ Dataset is relatively BALANCED")

# Cell 32
# Visualize class distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot
class_counts.plot(kind='bar', ax=axes[0], color=['#3498db', '#e74c3c'])
axes[0].set_title('High Earner Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('High Earner Status', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_xticklabels(['Non-High Earner (0)', 'High Earner (1)'], rotation=45, ha='right')
axes[0].grid(axis='y', alpha=0.3)

# Add count labels on bars
for i, v in enumerate(class_counts):
    axes[0].text(i, v + 20, f'{v:,}\n({v/len(career_df)*100:.1f}%)', 
                ha='center', va='bottom', fontweight='bold')

# Pie chart
colors = ['#3498db', '#e74c3c']
axes[1].pie(class_counts, labels=['Non-High Earner', 'High Earner'], 
           autopct='%1.1f%%', startangle=90, colors=colors,
           textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[1].set_title('Class Proportion', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# Cell 33
# Create derived features to match the formula
# Create elite_university indicator (Top 10 or Top 100)
career_df['elite_university'] = career_df['university_tier'].isin(['Top 10', 'Top 100']).astype(int)

# Create stem_major indicator
stem_majors = ['Engineering', 'Computer Science', 'Data Science']
career_df['stem_major'] = career_df['major'].isin(stem_majors).astype(int)
# List all features that will be used in the model
features = [
    'years_experience',
    'gpa',
    'internship_count',
    'elite_university',
    'stem_major',
    'technical_skills',
    'leadership_roles',
    'major',
    'industry'
]

print("Features used in the model:")
print("\nContinuous/Binary features:")
for feat in ['years_experience', 'gpa', 'internship_count', 'elite_university', 
             'stem_major', 'technical_skills', 'leadership_roles']:
    print(f"  - {feat}")

print("\nCategorical features:")
print(f"  - C(major)")
print(f"  - C(industry)")

print(f"\nDataset shape: {career_df.shape}")
print(f"\nUnique majors: {career_df['major'].nunique()}")
print(f"Unique industries: {career_df['industry'].nunique()}")

# Cell 34

# Train-test split (70/30) for the classifier
model_data = career_df[['high_earner', 'years_experience', 'gpa', 'internship_count',
                 'elite_university', 'stem_major', 'technical_skills',
                 'leadership_roles', 'major', 'industry']].copy()

print("Missing values check:")
print(model_data.isnull().sum())

if model_data.isnull().to_numpy().sum() > 0:
    print("
WARNING: Missing values detected. Filling numeric columns with median...")
    numeric_cols = ['years_experience', 'gpa', 'internship_count', 'technical_skills', 'leadership_roles']
    model_data[numeric_cols] = model_data[numeric_cols].fillna(model_data[numeric_cols].median())

logit_train_df, logit_test_df = train_test_split(
    model_data,
    test_size=0.30,
    random_state=1818,
    stratify=model_data['high_earner']
)

print(f"
Training set size: {len(logit_train_df):,} ({len(logit_train_df)/len(model_data)*100:.1f}%)")
print(f"Test set size: {len(logit_test_df):,} ({len(logit_test_df)/len(model_data)*100:.1f}%)")

print("
Class distribution in training set:")
print(logit_train_df['high_earner'].value_counts(normalize=True).sort_index())
print("
Class distribution in test set:")
print(logit_test_df['high_earner'].value_counts(normalize=True).sort_index())


# Cell 35
# Build logistic regression model using statsmodels
formula = 'high_earner ~ years_experience + gpa + internship_count + elite_university + stem_major + technical_skills + leadership_roles + C(major) + C(industry)'

print("Model Formula:")
print(formula)
print("\n" + "="*70)

# Fit the model on training data
logit_model = smf.logit(formula, data=logit_train_df).fit()

print("\n✓ Logistic Regression model trained successfully!")
print(f"\nModel Summary:")
print(f"  - Number of observations: {logit_model.nobs:.0f}")
print(f"  - Number of parameters: {len(logit_model.params)}")
print(f"  - Log-Likelihood: {logit_model.llf:.2f}")
print(f"  - AIC: {logit_model.aic:.2f}")
print(f"  - BIC: {logit_model.bic:.2f}")
print(f"  - Pseudo R-squared (McFadden): {logit_model.prsquared:.4f}")

# Cell 36
# Display full model summary
print("\n" + "="*70)
print("FULL MODEL SUMMARY")
print("="*70)
print(logit_model.summary())

# Extract and display coefficients and odds ratios for main effects
print("\n" + "="*70)
print("COEFFICIENTS AND ODDS RATIOS (Main Effects)")
print("="*70)

# Create coefficients dataframe
coef_df = pd.DataFrame({
    'Coefficient': logit_model.params,
    'Std_Error': logit_model.bse,
    'z_value': logit_model.tvalues,
    'p_value': logit_model.pvalues,
    'Odds_Ratio': np.exp(logit_model.params)
})

# Sort by absolute coefficient value
coef_df['abs_coef'] = np.abs(coef_df['Coefficient'])
coef_df_sorted = coef_df.sort_values('abs_coef', ascending=False)

# Display main numerical predictors
main_predictors = ['years_experience', 'gpa', 'internship_count', 'elite_university', 
                   'stem_major', 'technical_skills', 'leadership_roles']
print("\nNumerical and Binary Predictors:")
for pred in main_predictors:
    if pred in coef_df.index:
        row = coef_df.loc[pred]
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"  {pred:25} Coef: {row['Coefficient']:8.4f}  OR: {row['Odds_Ratio']:7.4f}  p: {row['p_value']:.4f} {sig}")

print("\n" + "="*70)
print("\nInterpretation:")
print("- Positive coefficient → increases probability of being a high earner")
print("- Negative coefficient → decreases probability of being a high earner")
print("- Odds Ratio > 1 → positive association with high earner status")
print("- Odds Ratio < 1 → negative association with high earner status")
print("- Significance: *** p<0.001, ** p<0.01, * p<0.05")

# Cell 37
# Visualize coefficients for main effects only
main_predictors = ['years_experience', 'gpa', 'internship_count', 'elite_university', 
                   'stem_major', 'technical_skills', 'leadership_roles']

# Filter to main predictors
main_coefs = coef_df.loc[[p for p in main_predictors if p in coef_df.index]]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Coefficients bar plot
coef_sorted = main_coefs.sort_values('Coefficient')
colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in coef_sorted['Coefficient']]
axes[0].barh(coef_sorted.index, coef_sorted['Coefficient'], color=colors, alpha=0.7)
axes[0].axvline(x=0, color='black', linestyle='--', linewidth=1)
axes[0].set_xlabel('Coefficient Value', fontsize=12)
axes[0].set_title('Logistic Regression Coefficients (Main Effects)', fontsize=14, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

# Add significance stars
for i, (idx, row) in enumerate(coef_sorted.iterrows()):
    sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
    if sig:
        axes[0].text(row['Coefficient'], i, f" {sig}", va='center', fontsize=12, fontweight='bold')

# Odds ratios bar plot
or_sorted = main_coefs.sort_values('Odds_Ratio')
colors_or = ['#e74c3c' if x < 1 else '#2ecc71' for x in or_sorted['Odds_Ratio']]
axes[1].barh(or_sorted.index, or_sorted['Odds_Ratio'], color=colors_or, alpha=0.7)
axes[1].axvline(x=1, color='black', linestyle='--', linewidth=1, label='Odds Ratio = 1')
axes[1].set_xlabel('Odds Ratio', fontsize=12)
axes[1].set_title('Odds Ratios (exp(coefficient))', fontsize=14, fontweight='bold')
axes[1].legend(loc='best')
axes[1].grid(axis='x', alpha=0.3)

# Add significance stars
for i, (idx, row) in enumerate(or_sorted.iterrows()):
    sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
    if sig:
        axes[1].text(row['Odds_Ratio'], i, f" {sig}", va='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# Cell 38
# Generate predictions using 0.5 threshold
y_prob = logit_model.predict(logit_test_df)
y_pred_default = (y_prob >= 0.5).astype(int)

print(f"Predictions generated for {len(logit_test_df):,} test samples")
print(f"\nPredicted class distribution (threshold=0.5):")
print(pd.Series(y_pred_default).value_counts().sort_index())

# Cell 39
# Calculate baseline accuracy (always predict majority class)
y_test = logit_test_df['high_earner']
baseline_accuracy = y_test.value_counts().max() / len(y_test)

print(f"Baseline Accuracy (always predict majority class): {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
print(f"\nMajority class in test set: {y_test.value_counts().idxmax()}")
print(f"Majority class frequency: {y_test.value_counts().max()} / {len(y_test)}")

# Cell 40
# Calculate classification metrics
accuracy = accuracy_score(y_test, y_pred_default)
precision = precision_score(y_test, y_pred_default)
recall = recall_score(y_test, y_pred_default)
f1 = f1_score(y_test, y_pred_default)

print("\n" + "="*70)
print("CLASSIFICATION METRICS (Threshold = 0.5)")
print("="*70)
print(f"\nBaseline Accuracy:  {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
print(f"Model Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision:          {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:             {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score:           {f1:.4f}")
print("\n" + "="*70)

# Compare to baseline
improvement = (accuracy - baseline_accuracy) / baseline_accuracy * 100
print(f"\nModel improvement over baseline: {improvement:.2f}%")

# Cell 41
# Create confusion matrix
cm = confusion_matrix(y_test, y_pred_default)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-High Earner (0)', 'High Earner (1)'],
            yticklabels=['Non-High Earner (0)', 'High Earner (1)'],
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix (Threshold = 0.5)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)

# Add percentage annotations
for i in range(2):
    for j in range(2):
        pct = cm[i, j] / cm.sum() * 100
        plt.text(j + 0.5, i + 0.7, f'({pct:.1f}%)', 
                ha='center', va='center', fontsize=11, color='gray')

plt.tight_layout()
plt.show()

# Print confusion matrix breakdown
tn, fp, fn, tp = cm.ravel()
print("\nConfusion Matrix Breakdown:")
print(f"  True Negatives (TN):  {tn:,}  - Correctly predicted non-high earners")
print(f"  False Positives (FP): {fp:,}  - Incorrectly predicted as high earners")
print(f"  False Negatives (FN): {fn:,}  - Missed high earners")
print(f"  True Positives (TP):  {tp:,}  - Correctly predicted high earners")

# Cell 42
# Detailed classification report
print("\nDetailed Classification Report:")
print("="*70)
print(classification_report(y_test, y_pred_default, 
                          target_names=['Non-High Earner (0)', 'High Earner (1)'],
                          digits=4))

# Cell 43
# Calculate ROC curve
fpr, tpr, thresholds_roc = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print(f"ROC AUC Score: {roc_auc:.4f}")
print(f"\nNumber of threshold points: {len(thresholds_roc)}")

# Cell 44
# Plot ROC curve
fig1, ax = plt.subplots(figsize=(6,6))

# plot ROC curve
ax.plot(fpr, tpr, color="black", linewidth=2)
ax.plot([0,1], [0,1], linestyle="--", color="gray")  # diagonal baseline
ax.scatter(fpr, tpr, s=10, color="blue")  # optional: mark cutoff points

# label some cutoff points
for cutoff in [0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.05, 0.075, 0.025, 0.01, 0.001, 0.003, 0.0]:
    # Find closest threshold value
    if cutoff == 0.0:
        idx = len(thresholds_roc) - 1  # Last threshold is usually 0.0
    else:
        idx = np.argmin(np.abs(thresholds_roc - cutoff))
    
    # Add text with slight offset and background for visibility
    ax.text(fpr[idx] + 0.02, tpr[idx] + 0.02, f"{cutoff:.3f}", 
            fontsize=9, color='red', weight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f'ROC Curve (AUC = {roc_auc:.4f})', fontsize=14, fontweight='bold')
plt.show(fig1)

print("\nROC Curve Interpretation:")
if roc_auc >= 0.9:
    print(f"  Excellent model (AUC ≥ 0.9)")
elif roc_auc >= 0.8:
    print(f"  Good model (0.8 ≤ AUC < 0.9)")
elif roc_auc >= 0.7:
    print(f"  Fair model (0.7 ≤ AUC < 0.8)")
else:
    print(f"  Poor model (AUC < 0.7)")

# Cell 45
# Find optimal threshold using Youden's J statistic
j_stats = tpr - fpr
optimal_idx = np.argmax(j_stats)
optimal_threshold = thresholds_roc[optimal_idx]
optimal_tpr = tpr[optimal_idx]
optimal_fpr = fpr[optimal_idx]

print("\n" + "="*70)
print("OPTIMAL THRESHOLD ANALYSIS (Youden's J Statistic)")
print("="*70)
print(f"\nOptimal Threshold: {optimal_threshold:.4f}")
print(f"J-Statistic at optimal: {j_stats[optimal_idx]:.4f}")
print(f"TPR (Recall) at optimal: {optimal_tpr:.4f}")
print(f"FPR at optimal: {optimal_fpr:.4f}")
print("\n" + "="*70)

# Generate predictions with optimal threshold
y_pred_optimal = (y_prob >= optimal_threshold).astype(int)

# Calculate metrics at optimal threshold
accuracy_opt = accuracy_score(y_test, y_pred_optimal)
precision_opt = precision_score(y_test, y_pred_optimal)
recall_opt = recall_score(y_test, y_pred_optimal)
f1_opt = f1_score(y_test, y_pred_optimal)

# Cell 46
# Plot ROC curve with optimal threshold point
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='#e74c3c', linewidth=2.5, 
         label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linewidth=1.5, 
         linestyle='--', label='Random Classifier')
plt.scatter(optimal_fpr, optimal_tpr, color='#2ecc71', s=200, 
           zorder=5, edgecolors='black', linewidths=2,
           label=f'Optimal Point (threshold={optimal_threshold:.3f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate (Recall)', fontsize=14)
plt.title('ROC Curve with Optimal Threshold', fontsize=16, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Cell 47
# Compare metrics at default vs optimal threshold
comparison_df = pd.DataFrame({
    'Metric': ['Threshold', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Default (0.5)': [0.5, accuracy, precision, recall, f1],
    'Optimal (Youden)': [optimal_threshold, accuracy_opt, precision_opt, recall_opt, f1_opt],
    'Change': [
        optimal_threshold - 0.5,
        accuracy_opt - accuracy,
        precision_opt - precision,
        recall_opt - recall,
        f1_opt - f1
    ]
})

print("\n" + "="*90)
print("METRICS COMPARISON: Default (0.5) vs Optimal Threshold")
print("="*90)
print(comparison_df.to_string(index=False))
print("\n" + "="*90)

# Highlight improvements
print("\nKey Changes:")
for idx, row in comparison_df.iterrows():
    if idx > 0:  # Skip threshold row
        change = row['Change']
        symbol = '↑' if change > 0 else '↓' if change < 0 else '→'
        print(f"  {row['Metric']:15} {symbol} {abs(change):.4f} ({abs(change)*100:.2f}%)")

# Cell 48
# Visualize metric comparison
fig, ax = plt.subplots(figsize=(12, 6))

metrics_to_plot = comparison_df[comparison_df['Metric'] != 'Threshold']
x = np.arange(len(metrics_to_plot))
width = 0.35

bars1 = ax.bar(x - width/2, metrics_to_plot['Default (0.5)'], width, 
              label='Default (0.5)', color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, metrics_to_plot['Optimal (Youden)'], width,
              label=f'Optimal ({optimal_threshold:.3f})', color='#2ecc71', alpha=0.8)

ax.set_xlabel('Metrics', fontsize=14)
ax.set_ylabel('Score', fontsize=14)
ax.set_title('Metrics Comparison: Default vs Optimal Threshold', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_to_plot['Metric'], fontsize=12)
ax.legend(fontsize=12)
ax.set_ylim([0, 1.05])
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# Cell 49
# Plot Precision-Recall Curve (important for imbalanced data)
precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, y_prob)
avg_precision = average_precision_score(y_test, y_prob)

plt.figure(figsize=(10, 8))
plt.plot(recall_vals, precision_vals, color='#9b59b6', linewidth=2.5,
         label=f'PR Curve (AP = {avg_precision:.4f})')
plt.axhline(y=y_test.mean(), color='gray', linestyle='--', linewidth=1.5,
           label=f'Baseline (No Skill) = {y_test.mean():.3f}')
plt.xlabel('Recall (Sensitivity)', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
plt.legend(loc='best', fontsize=12)
plt.grid(alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.tight_layout()
plt.show()

print(f"\nAverage Precision Score: {avg_precision:.4f}")
print(f"Baseline (random classifier): {y_test.mean():.4f}")
print("\nNote: For imbalanced datasets, PR curves are more informative than ROC curves.")

# Cell 50

# Persist classifier diagnostics for later use
from sklearn.metrics import brier_score_loss

fpr, tpr, thresholds_roc = roc_curve(y_test, y_prob)
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="quantile")
brier = brier_score_loss(y_test, y_prob)

metrics_default = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "Baseline_accuracy": baseline_accuracy,
}

q2_models["logit_model"] = logit_model
q2_models["logit_metrics"] = {
    "default": metrics_default,
    "brier_score": brier,
    "roc_curve": (fpr, tpr, thresholds_roc),
    "calibration": {"prob_true": prob_true, "prob_pred": prob_pred},
}
q2_models["logit_coefficients"] = coef_df
q2_models["logit_data"] = {
    "train": logit_train_df,
    "test": logit_test_df,
    "y_prob": y_prob,
    "y_pred_default": y_pred_default,
    "y_test": y_test,
}


# Cell 52
print(f"\n✓ Baseline Model")
print(f"  R²: {baseline_model.rsquared:.4f}")
print(f"  Adjusted R²: {baseline_model.rsquared_adj:.4f}")
print(f"  AIC: {baseline_model.aic:.2f}")
print(f"\n✓ Log-Log Model")
print(f"  R²: {log_model.rsquared:.4f}")
print(f"  Adjusted R²: {log_model.rsquared_adj:.4f}")
print(f"  AIC: {log_model.aic:.2f}")

# Cell 53
# train_df, test_df = train_test_split(
#     model_data, test_size=0.30, random_state=1818, stratify=model_data['high_earner']
# )
print(f"\nTraining set: {len(train_df):,} samples")
print(f"Test set: {len(test_df):,} samples")

# Cell 54
test_df.columns 

# Cell 55
# Generate predictions on test set
y_test = test_df['salary']

# Baseline model predictions
y_pred_baseline = baseline_model.predict(test_df)

# Log-log model predictions (need to transform back to original scale)
log_pred_loglog = log_model.predict(test_df)
y_pred_loglog = np.exp(log_pred_loglog)

print("✓ Predictions generated for both models")

# Cell 56
# Calculate metrics for both models
def calculate_regression_metrics(y_true, y_pred, model_name):
    """Calculate R², RMSE, and MAE for a regression model"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    return {
        'Model': model_name,
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae
    }

# Calculate metrics
metrics_baseline = calculate_regression_metrics(y_test, y_pred_baseline, 'Baseline Linear')
metrics_loglog = calculate_regression_metrics(y_test, y_pred_loglog, 'Log-Log')

# Create comparison table
comparison_df = pd.DataFrame([metrics_baseline, metrics_loglog])

print("\n" + "="*80)
print("MODEL PERFORMANCE COMPARISON")
print("="*80)
print(comparison_df.to_string(index=False))
print("\n" + "="*80)

# Add percentage differences
print("\nPerformance Differences (Log-Log vs Baseline):")
r2_diff = (metrics_loglog['R²'] - metrics_baseline['R²']) / metrics_baseline['R²'] * 100
rmse_diff = (metrics_loglog['RMSE'] - metrics_baseline['RMSE']) / metrics_baseline['RMSE'] * 100
mae_diff = (metrics_loglog['MAE'] - metrics_baseline['MAE']) / metrics_baseline['MAE'] * 100

print(f"  R² difference: {r2_diff:+.2f}%")
print(f"  RMSE difference: {rmse_diff:+.2f}% (lower is better)")
print(f"  MAE difference: {mae_diff:+.2f}% (lower is better)")

# Cell 57
# Your code here

import numpy as np
comparison_results = pd.DataFrame([
    q2_models['baseline_eval'],
    q2_models['log_eval']
])
print('Linear model performance summary:')
display(comparison_results)

residuals_baseline = q2_models['test_df']['salary_current'] - q2_models['baseline_preds']
residuals_log = q2_models['test_df']['salary_current'] - q2_models['log_preds']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(residuals_baseline, bins=40, color='steelblue', alpha=0.7)
axes[0].set_title('Baseline Model Residuals')
axes[0].set_xlabel('Error (USD)')
axes[0].set_ylabel('Count')

axes[1].hist(residuals_log, bins=40, color='purple', alpha=0.7)
axes[1].set_title('Log-Linear Model Residuals')
axes[1].set_xlabel('Error (USD)')
axes[1].set_ylabel('Count')
plt.tight_layout()
plt.show()

scenario_inputs = pd.DataFrame([
    {
        'scenario': 'New grad (STEM elite)',
        'years_experience': 0,
        'gpa': 3.5,
        'internship_count': 2,
        'elite_university': 1,
        'stem_major': 1,
        'major': 'Computer Science',
        'industry': 'Technology'
    },
    {
        'scenario': 'New grad (Liberal Arts regional)',
        'years_experience': 0,
        'gpa': 3.8,
        'internship_count': 1,
        'elite_university': 0,
        'stem_major': 0,
        'major': 'Liberal Arts',
        'industry': 'Education'
    },
    {
        'scenario': '5-year Business professional',
        'years_experience': 5,
        'gpa': 3.2,
        'internship_count': 3,
        'elite_university': 0,
        'stem_major': 0,
        'major': 'Business',
        'industry': 'Finance'
    }
])

scenario_inputs['predicted_log_salary'] = q2_models['log_model'].predict(scenario_inputs)
scenario_inputs['predicted_salary'] = np.exp(scenario_inputs['predicted_log_salary'])
salary_scenarios = scenario_inputs[['scenario', 'predicted_salary']]
print()
print('Scenario salary projections (log-linear model):')
display(salary_scenarios)

# Cell 58
salary_gap = scenario_inputs.loc[0, 'predicted_salary'] - scenario_inputs.loc[1, 'predicted_salary']
print(f"Salary gap between STEM elite vs. Liberal Arts regional new grad: ${salary_gap:,.0f}")

linear_top_factors = (q2_models['log_model'].params[['stem_major', 'elite_university', 'years_experience', 'gpa', 'internship_count']]
                      .to_frame(name='log_coefficient'))
linear_top_factors['percent_effect'] = (np.exp(linear_top_factors['log_coefficient']) - 1) * 100
print()
print('Top linear model drivers (converted to % impact when applicable):')
display(linear_top_factors)

logit_metrics = q2_models['logit_metrics']
print()
print('High earner classifier metrics summary:')
display(pd.DataFrame([
    {'Scenario': 'Default 0.50 threshold', **logit_metrics['default']},
    {'Scenario': 'Optimal threshold', **logit_metrics['optimal']}
]))


# Cell 59
logit_coef_df = q2_models['logit_coefficients'].reset_index()
logit_coef_df.columns = ['feature', 'coefficient']
logit_coef_df['odds_ratio'] = np.exp(logit_coef_df['coefficient'])
print('Top positive and negative odds ratios:')
display(pd.concat([
    logit_coef_df.sort_values('coefficient', ascending=False).head(5),
    logit_coef_df.sort_values('coefficient', ascending=True).head(5)
]))


def salary_calculator(years_experience, gpa, internship_count, elite_university, stem_major, major, industry):
    input_df = pd.DataFrame([{
        'years_experience': years_experience,
        'gpa': gpa,
        'internship_count': internship_count,
        'elite_university': elite_university,
        'stem_major': stem_major,
        'major': major,
        'industry': industry
    }])
    log_salary = q2_models['log_model'].predict(input_df)[0]
    return float(np.exp(log_salary))

print()
print('Example salary calculator outputs:')
for _, row in scenario_inputs.iterrows():
    calc_salary = salary_calculator(row['years_experience'], row['gpa'], row['internship_count'],
                                    row['elite_university'], row['stem_major'], row['major'], row['industry'])
    print(f"{row['scenario']}: ${calc_salary:,.0f}")

q2_models['salary_scenarios'] = salary_scenarios
q2_models['logit_coef_df'] = logit_coef_df

# Cell 63
# Enhanced Bonus: Interactive Plotly dashboard for student career planning with KPIs
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score

# Ensure we have the latest career dataset with engineered fields
career_dashboard_df = career_df.copy()

majors = sorted(career_dashboard_df['major'].unique())
industries = sorted(career_dashboard_df['industry'].unique())
university_tiers = sorted(career_dashboard_df['university_tier'].unique())
graduate_degrees = sorted(career_dashboard_df['graduate_degree'].unique())
stem_set = {'Engineering', 'Computer Science', 'Data Science', 'Mathematics'}

def is_elite(tier):
    return 1 if tier in {'Top 10', 'Top 50'} else 0

# Input widgets so students can describe their profile
major_dd = widgets.Dropdown(options=majors, description='Major:', value=majors[0], style={'description_width': 'initial'})
industry_dd = widgets.Dropdown(options=industries, description='Target industry:', value=industries[0], style={'description_width': 'initial'})
university_dd = widgets.Dropdown(options=university_tiers, description='University tier:', value=university_tiers[0], style={'description_width': 'initial'})
gpa_slider = widgets.FloatSlider(value=3.2, min=2.0, max=4.0, step=0.05, description='GPA:', style={'description_width': 'initial'})
years_slider = widgets.IntSlider(value=0, min=0, max=15, step=1, description='Years exp:', style={'description_width': 'initial'})
internships_slider = widgets.IntSlider(value=2, min=0, max=8, step=1, description='Internships:', style={'description_width': 'initial'})
tech_slider = widgets.FloatSlider(value=6.0, min=0.0, max=10.0, step=0.5, description='Tech skills:', style={'description_width': 'initial'})
leader_slider = widgets.IntSlider(value=1, min=0, max=5, step=1, description='Leadership roles:', style={'description_width': 'initial'})
grad_dd = widgets.Dropdown(options=graduate_degrees, description='Grad degree:', value=graduate_degrees[0], style={'description_width': 'initial'})

controls_left = widgets.VBox([major_dd, industry_dd, university_dd, grad_dd])
controls_right = widgets.VBox([gpa_slider, years_slider, internships_slider, tech_slider, leader_slider])
controls_box = widgets.HBox([controls_left, controls_right])

# Output widgets
kpi_out = widgets.Output()
fig_salary_out = widgets.Output()
fig_progression_out = widgets.Output()
fig_roi_out = widgets.Output()
fig_traj_out = widgets.Output()

def build_profile():
    elite_flag = is_elite(university_dd.value)
    stem_flag = 1 if major_dd.value in stem_set else 0
    profile = {
        'years_experience': years_slider.value,
        'gpa': gpa_slider.value,
        'internship_count': internships_slider.value,
        'elite_university': elite_flag,
        'stem_major': stem_flag,
        'technical_skills': tech_slider.value,
        'leadership_roles': leader_slider.value,
        'major': major_dd.value,
        'industry': industry_dd.value
    }
    return profile

def predict_salary(df_rows):
    log_preds = q2_models['log_model'].predict(df_rows)
    return np.exp(log_preds)

def predict_high_earner(df_row):
    return float(q2_models['log_model'].predict(df_row)[0])

def update_dashboard(*args):
    profile = build_profile()
    scenario = pd.DataFrame([profile])
    salary_pred = float(predict_salary(scenario)[0])
    high_prob = predict_high_earner(scenario)
    
    # Calculate benchmark comparisons
    avg_all = career_dashboard_df['salary_current'].mean()
    avg_major = career_dashboard_df[career_dashboard_df['major'] == profile['major']]['salary_current'].mean()
    avg_industry = career_dashboard_df[career_dashboard_df['industry'] == profile['industry']]['salary_current'].mean()
    
    # Calculate Career Success Score (0-100)
    success_score = 0
    # Major (40 points)
    if profile['stem_major'] == 1 or profile['major'] in ['Finance', 'Economics']:
        success_score += 40
    elif profile['major'] in ['Business', 'Marketing']:
        success_score += 25
    else:
        success_score += 10
    # Internships (30 points)
    success_score += min(profile['internship_count'] * 7.5, 30)
    # GPA (15 points)
    success_score += (profile['gpa'] - 2.0) * 7.5
    # University tier (10 points)
    success_score += 10 if profile['elite_university'] == 1 else 5
    # Leadership (5 points)
    success_score += min(profile['leadership_roles'], 5)
    
    # Calculate model performance metrics on-the-fly
    try:
        # For linear regression model
        log_salary_preds = q2_models['log_model'].predict(career_dashboard_df)
        salary_preds = np.exp(log_salary_preds)
        actual_salaries = career_dashboard_df['salary_current']
        
        linear_r2 = r2_score(actual_salaries, salary_preds)
        linear_rmse = np.sqrt(mean_squared_error(actual_salaries, salary_preds))
        
        # For logistic regression model
        high_earner_preds = q2_models['log_model'].predict(career_dashboard_df)
        actual_high_earner = career_dashboard_df['high_earner']
        
        logistic_accuracy = accuracy_score(actual_high_earner, high_earner_preds)
        logistic_f1 = f1_score(actual_high_earner, high_earner_preds)
        
        metrics_available = True
    except Exception as e:
        print(f"Warning: Could not calculate metrics - {e}")
        metrics_available = False
    
    # Display KPI Dashboard
    with kpi_out:
        kpi_out.clear_output(wait=True)
        print("=" * 80)
        print("📊 CAREER SUCCESS DASHBOARD - KEY PERFORMANCE INDICATORS")
        print("=" * 80)
        print(f"\n🎯 CAREER SUCCESS SCORE: {success_score:.0f}/100")
        print(f"   {'█' * int(success_score/2)}{'░' * (50-int(success_score/2))}")
        
        print(f"\n💰 SALARY PREDICTIONS:")
        print(f"   Current Profile Prediction:        ${salary_pred:>12,.0f}")
        print(f"   vs. Overall Average:               ${avg_all:>12,.0f}  ({(salary_pred/avg_all-1)*100:+.1f}%)")
        print(f"   vs. {profile['major'][:20]} Average: ${avg_major:>12,.0f}  ({(salary_pred/avg_major-1)*100:+.1f}%)")
        print(f"   vs. {profile['industry'][:20]} Industry: ${avg_industry:>12,.0f}  ({(salary_pred/avg_industry-1)*100:+.1f}%)")
        
        print(f"\n🎓 HIGH EARNER PROBABILITY: {high_prob:.1%}")
        print(f"   Likelihood of earning >$120,000/year")
        if high_prob > 0.7:
            print("   ✅ EXCELLENT - Strong high-earner profile")
        elif high_prob > 0.4:
            print("   ⚠️  MODERATE - Consider boosting key factors")
        else:
            print("   ❌ LOW - Recommend major career interventions")
        
        if metrics_available:
            print(f"\n📈 MODEL PERFORMANCE METRICS:")
            print(f"   Linear Model R²:                   {linear_r2:.3f}")
            print(f"   Linear Model RMSE:                 ${linear_rmse:,.0f}")
            print(f"   Logistic Model Accuracy:           {logistic_accuracy:.1%}")
            print(f"   Logistic Model F1 Score:           {logistic_f1:.3f}")
        
        # Get coefficient impacts from the model
        try:
            model_params = q2_models['log_model'].params
            top_coefs = model_params.sort_values(ascending=False).head(6)
            
            print(f"\n🔑 KEY COEFFICIENT IMPACTS (from Linear Model):")
            for feat, coef in top_coefs.items():
                if 'Intercept' not in str(feat):
                    impact_pct = (np.exp(coef) - 1) * 100
                    feat_str = str(feat)[:35]
                    print(f"   {feat_str:<35} {impact_pct:>+7.1f}%")
        except Exception as e:
            print(f"\n🔑 KEY FACTORS: Major, Industry, Experience, Internships, GPA")
        
        print("=" * 80)

    # Salary by industry comparison
    industries_df = pd.DataFrame({
        'industry': industries,
        'years_experience': profile['years_experience'],
        'gpa': profile['gpa'],
        'internship_count': profile['internship_count'],
        'elite_university': profile['elite_university'],
        'stem_major': profile['stem_major'],
        'major': profile['major']
    })
    industries_df['predicted_salary'] = predict_salary(industries_df)
    industries_df['selected'] = np.where(industries_df['industry'] == profile['industry'], 'Your Choice', 'Alternative')
    industries_df = industries_df.sort_values('predicted_salary', ascending=False)

    with fig_salary_out:
        fig_salary_out.clear_output(wait=True)
        fig = px.bar(
            industries_df,
            x='industry',
            y='predicted_salary',
            color='selected',
            color_discrete_map={'Your Choice': '#2ecc71', 'Alternative': '#95a5a6'},
            title=f'💼 Predicted Salary Across Industries ({profile["major"]} Major)',
            text='predicted_salary'
        )
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig.update_layout(
            yaxis_title='Predicted Salary (USD)', 
            xaxis_title='Industry',
            showlegend=True,
            height=500
        )
        fig.show()

    # Salary progression over years (NEW CHART)
    years_range = list(range(0, 16))
    progression_df = pd.DataFrame({
        'years_experience': years_range,
        'gpa': profile['gpa'],
        'internship_count': profile['internship_count'],
        'elite_university': profile['elite_university'],
        'stem_major': profile['stem_major'],
        'major': profile['major'],
        'industry': profile['industry']
    })
    progression_df['predicted_salary'] = predict_salary(progression_df)
    
    with fig_progression_out:
        fig_progression_out.clear_output(wait=True)
        fig_prog = go.Figure()
        
        # Main trajectory
        fig_prog.add_trace(go.Scatter(
            x=progression_df['years_experience'],
            y=progression_df['predicted_salary'],
            mode='lines+markers',
            name='Your Projected Path',
            line=dict(color='#3498db', width=3),
            marker=dict(size=8)
        ))
        
        # Add current position marker
        current_salary = progression_df[progression_df['years_experience'] == profile['years_experience']]['predicted_salary'].values[0]
        fig_prog.add_trace(go.Scatter(
            x=[profile['years_experience']],
            y=[current_salary],
            mode='markers+text',
            name='Current Position',
            marker=dict(size=15, color='#e74c3c', symbol='star'),
            text=[f"You are here<br>${current_salary:,.0f}"],
            textposition='top center'
        ))
        
        # Add 5-year and 10-year projections
        if profile['years_experience'] + 5 <= 15:
            salary_5yr = progression_df[progression_df['years_experience'] == profile['years_experience'] + 5]['predicted_salary'].values[0]
            fig_prog.add_trace(go.Scatter(
                x=[profile['years_experience'] + 5],
                y=[salary_5yr],
                mode='markers+text',
                name='+5 Years',
                marker=dict(size=12, color='#f39c12', symbol='diamond'),
                text=[f"+5yr: ${salary_5yr:,.0f}"],
                textposition='top center'
            ))
        
        if profile['years_experience'] + 10 <= 15:
            salary_10yr = progression_df[progression_df['years_experience'] == profile['years_experience'] + 10]['predicted_salary'].values[0]
            fig_prog.add_trace(go.Scatter(
                x=[profile['years_experience'] + 10],
                y=[salary_10yr],
                mode='markers+text',
                name='+10 Years',
                marker=dict(size=12, color='#9b59b6', symbol='diamond'),
                text=[f"+10yr: ${salary_10yr:,.0f}"],
                textposition='top center'
            ))
        
        fig_prog.update_layout(
            title=f'📈 Expected Salary Progression Over Career ({profile["major"]} in {profile["industry"]})',
            xaxis_title='Years of Experience',
            yaxis_title='Expected Salary (USD)',
            hovermode='x unified',
            height=500,
            showlegend=True
        )
        fig_prog.show()

    # Graduate degree ROI
    roi_base = career_dashboard_df.copy()
    roi_filtered = roi_base[roi_base['major'] == profile['major']]
    if roi_filtered.empty:
        roi_filtered = roi_base
    roi_summary = (roi_filtered
                   .groupby('graduate_degree')
                   .agg(avg_salary=('salary_current', 'mean'),
                        avg_growth=('salary_growth', 'mean'),
                        sample=('graduate_id', 'count'))
                   .reset_index())
    roi_summary['avg_growth_pct'] = roi_summary['avg_growth'] * 100

    with fig_roi_out:
        fig_roi_out.clear_output(wait=True)
        fig_roi = make_subplots(specs=[[{"secondary_y": True}]])
        fig_roi.add_trace(
            go.Bar(
                x=roi_summary['graduate_degree'], 
                y=roi_summary['avg_salary'],
                name='Avg Salary',
                marker_color='#2ecc71',
                text=roi_summary['avg_salary'],
                texttemplate='$%{text:,.0f}',
                textposition='outside'
            ), 
            secondary_y=False
        )
        fig_roi.add_trace(
            go.Scatter(
                x=roi_summary['graduate_degree'], 
                y=roi_summary['avg_growth_pct'],
                mode='markers+lines+text', 
                name='Salary Growth %',
                marker=dict(size=10, color='#e74c3c'),
                line=dict(width=2),
                text=roi_summary['avg_growth_pct'],
                texttemplate='%{text:.1f}%',
                textposition='top center'
            ),
            secondary_y=True
        )
        fig_roi.update_yaxes(title_text='Average Salary (USD)', secondary_y=False)
        fig_roi.update_yaxes(title_text='Average Salary Growth (%)', secondary_y=True)
        fig_roi.update_layout(
            title=f'🎓 Graduate Degree ROI for {profile["major"]} Majors',
            hovermode='x unified',
            height=500
        )
        fig_roi.show()

    # Career trajectory for similar alumni
    similar = career_dashboard_df[
        (career_dashboard_df['major'] == profile['major']) &
        (career_dashboard_df['industry'] == profile['industry'])
    ]
    if similar.empty:
        similar = career_dashboard_df[career_dashboard_df['major'] == profile['major']]
    if similar.empty:
        similar = career_dashboard_df

    traj_summary = (similar.groupby(['years_experience', 'graduate_degree'])
                    .agg(avg_salary=('salary_current', 'mean'), count=('graduate_id', 'count'))
                    .reset_index())
    traj_summary = traj_summary[traj_summary['count'] >= 3]  # Only show if enough samples

    with fig_traj_out:
        fig_traj_out.clear_output(wait=True)
        fig_traj = px.line(
            traj_summary, 
            x='years_experience', 
            y='avg_salary', 
            color='graduate_degree',
            markers=True,
            title=f'👥 Career Salary Trajectory - Similar Alumni ({profile["major"]} in {profile["industry"]})',
            labels={'avg_salary': 'Average Salary (USD)', 'years_experience': 'Years of Experience'}
        )
        fig_traj.update_layout(height=500)
        fig_traj.show()

# Attach observers to all widgets
for widget in [major_dd, industry_dd, university_dd, grad_dd,
               gpa_slider, years_slider, internships_slider, tech_slider, leader_slider]:
    widget.observe(update_dashboard, names='value')

# Initial dashboard render
update_dashboard()

# Display the complete dashboard
print("\n" + "="*80)
print("🎓 INTERACTIVE CAREER PLANNING DASHBOARD")
print("="*80)
print("Adjust the controls below to see personalized predictions and insights")
print("="*80 + "\n")

display(controls_box, kpi_out, fig_salary_out, fig_progression_out, fig_roi_out, fig_traj_out)

