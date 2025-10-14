# CPSC 540 Midterm Exam Study Guide

## Exam Logistics
- **Duration:** 75 minutes
- **Number of Questions:** 4-8 (varies by question length)
- **NOT on exam:** Code, Journal Club articles
- **Note:** This review is a sample of important topics, not exhaustive

---

## 1. Design Matrix (Slide G1)

**Main Idea:** The design matrix X provides a compact mathematical notation for linear regression, representing the relationship y = Xβ where y is the response vector, X contains predictors, and β contains coefficients.

**Why Important:** Simplifies mathematical derivations and computational implementations of regression models.

**Key Points:**
- Y is typically n×1 (n observations)
- X is n×(p+1) (n observations, p predictors + intercept)
- Equivalent to: y = β₀ + β₁x₁ + ... + βₙxₙ
- The "+1" accounts for the intercept term

---

## 2. Generalized Linear Models (GLM) Framework (Slide G4)

**Main Idea:** GLMs extend linear regression by allowing different response distributions through a link function g() that connects the linear predictor to the expected value.

**Why Important:** Enables modeling of non-normal responses (binary, count data, etc.) while maintaining the linear modeling framework.

**Components:**
- **Link function g():** Transforms the mean μ to the linear scale
- **Expected value:** E(y|X) = μ = g⁻¹(Xβ)
- **Likelihood function π():** Specifies the probability distribution of y given x

---

## 3. Choosing a Likelihood Function (Slide G3)

**Main Idea:** The likelihood function should match the distribution of your response variable based on its characteristics.

**Why Important:** Using the wrong likelihood leads to inappropriate models, inefficient estimates, and invalid inference.

**Common Choices:**
- **Linear regression:** Continuous, normally distributed responses
- **Logistic regression:** Binary responses (0/1), Bernoulli distribution

---

## 4. Robust Student t Regression (Slide G2)

**Main Idea:** Student's t distribution has heavier tails than the normal distribution, making regression more resistant to outliers and influential points.

**Why Important:** Real data often contains outliers that can severely distort ordinary least squares estimates.

**Visual Understanding:**
- Blue line (normal): Heavily influenced by outliers
- Red line (robust t): Less affected by extreme values
- Influential points (circled) have disproportionate effect on normal regression

---

## 5. Maximum Likelihood Estimation (MLE) (Slide G1)

**Main Idea:** MLE finds parameter values (θ) that maximize the probability of observing the actual data.

**Why Important:** Provides a principled, general method for parameter estimation with desirable statistical properties (consistency, efficiency).

**Key Concept:**
- Higher likelihood → More evidence for that parameter value
- The peak of the likelihood curve indicates the best-fitting θ
- Example shows three distributions with different means; the one centered on observed data (vertical line) has highest likelihood

---

## 6. Zero-Inflated Beta Regression (ZOIB) (Slide G3)

**Main Idea:** ZOIB is a mixture model combining three components to handle responses that are proportions with excess zeros and/or ones.

**Why Important:** Many real-world proportions (e.g., percentages, rates) have point masses at boundaries that standard beta regression cannot handle.

**Three Components:**
1. **Logistic regression:** Predicts if y is 0/1 vs. in (0,1)
2. **Logistic regression:** For 0/1 values, predicts which (0 or 1)
3. **Beta regression:** For values in (0,1), models the continuous proportion

---

## 7. Ordered Beta Regression (Slide G2)

**Main Idea:** Uses cumulative logit approach with a latent continuous variable to model ordered categorical outcomes coded as [0,1].

**Why Important:** Provides a principled way to model ordered categories while respecting their ordinal nature.

**Key Concept:**
- Underlying continuous "latent" variable determines observed category
- Cutpoints (thresholds) divide the latent distribution
- Shaded areas represent probabilities of different outcomes

---

## 8. Wiggliness and Basis Functions (Slide G4)

**Main Idea:** The number of basis functions controls model flexibility (wiggliness) - more functions allow more complex shapes.

**Why Important:** Controls the bias-variance tradeoff in nonparametric regression.

**Visual Progression:**
- **3 basis functions:** Smooth, simple fit (high bias, low variance)
- **7 basis functions:** Moderate flexibility
- **12 basis functions:** Very flexible, can overfit (low bias, high variance)

---

## 9. Regularization (Slide G1)

**Main Idea:** Regularization parameter λ penalizes model complexity to balance fitting the data (low bias) with generalization (low variance).

**Why Important:** Prevents overfitting, especially with many predictors or flexible models.

**Three Scenarios:**
- **λ too large:** Underfit, high bias (too smooth)
- **λ too small:** Overfit, high variance (too wiggly)
- **λ just right:** Optimal bias-variance tradeoff

---

## 10. Kaplan-Meier Curves (Slide G2)

**Main Idea:** Non-parametric method to estimate survival functions, showing the probability of survival over time.

**Why Important:** Fundamental tool for survival analysis; handles censored data and allows comparison between groups.

**Formula:** S(dₖ) = ∏(rⱼ - qⱼ)/rⱼ from j=1 to k

**Interpretation:**
- Y-axis: Survival probability (0 to 1)
- X-axis: Time (months)
- Step-down pattern shows events occurring
- Shaded bands: Confidence intervals
- Different colored lines compare groups (e.g., treatment effects)

---

## 11. Mixed Effect Regularization (Slide G3)

**Main Idea:** Combines information from group-level data with overall population mean, with weight w determining the balance.

**Why Important:** Partial pooling improves estimates for groups with small sample sizes by borrowing strength from the population.

**Formula:** β₀[j] ≈ w × (ȳⱼ - βx̄ⱼ) + (1-w) × μ₀

**Weight w depends on:**
- **w is large when:**
  - nⱼ is large (more data in group)
  - σᵧ² is small compared to σᵦ₀² (group variance small)
- **Effect:** More data = trust group estimate more; less data = shrink toward population mean

---

## 12. Gaussian Processes - Conditioning on Observed Data (Slide G4)

**Main Idea:** GP prior encodes smoothness assumptions; conditioning on observed data updates predictions to pass through known points with uncertainty bands.

**Why Important:** Provides principled uncertainty quantification for predictions and interpolation.

**Formula:** X|Y ~ N(μₓ + Σₓᵧ Σᵧᵧ⁻¹(Y - μᵧ), Σₓₓ - Σₓᵧ Σᵧᵧ⁻¹ Σᵧₓ)

**Visual:**
- Gray lines: Possible functions from prior
- Blue line: Posterior mean (best prediction)
- Blue band: Uncertainty (wider far from data)
- Red points: Observed data

**Key Insight:** Uncertainty shrinks near observations, grows in unobserved regions

---

## 13. Accept/Reject Sampling (Slide G1)

**Main Idea:** Sample from a complex distribution f(x) by sampling from a simpler proposal g(x) and accepting/rejecting based on a ratio.

**Why Important:** Enables sampling from distributions we can't sample from directly.

**Algorithm:**
1. Sample x from g(x)
2. Accept with probability f(x)/(M×g(x))
3. M must satisfy: M×g(x) ≥ f(x) for all x

**Visual:** Shows envelope M×g(x) must bound target f(x)

---

## 14. Metropolis-Hastings Algorithm (Slide G2)

**Main Idea:** MCMC method that uses a proposal distribution with bias correction to sample from posterior distributions.

**Why Important:** Core algorithm for Bayesian inference when direct sampling is impossible.

**Acceptance Ratio:** A(xₙ → x*) = min(1, [f(x*)/f(xₙ)] × [q(xₙ|x*)/q(x*|xₙ)])

**Key Components:**
- **f(x*)/f(xₙ):** Likelihood ratio (do we prefer new state?)
- **q(xₙ|x*)/q(x*|xₙ):** Proposal correction (accounts for asymmetric proposals)
- When q(xₙ|x*) > q(x*|xₙ): Correction makes us MORE likely to accept x*

---

## 15. Trace Plots and R̂ Diagnostics (Slide G3)

**Main Idea:** Trace plots visualize MCMC chain behavior; R̂ quantifies convergence by comparing within-chain and between-chain variance.

**Why Important:** Ensures MCMC has converged before using samples for inference.

**Good vs. Bad:**
- **Top plot:** "Fuzzy caterpillar" - poor mixing, chains haven't converged
- **Bottom plot:** Good mixing, chains exploring full posterior
- **R̂ ≈ 1.0:** Convergence achieved (typically R̂ < 1.01 desired)
- **R̂ >> 1.0:** Chains haven't converged, need more iterations

---

## 16. Hamiltonian Monte Carlo (HMC) (Slide G4)

**Main Idea:** Uses physics simulation (momentum + gradient information) to propose distant moves efficiently in the posterior.

**Why Important:** Much more efficient than random-walk Metropolis, especially in high dimensions.

**Three Steps:**
1. Generate random momentum vector (kinetic energy)
2. Simulate physics using position x₀ and momentum w₀ to get new state (xₜ, wₜ)
3. Accept with A = min(1, exp(-H(xₜ,wₜ))/exp(-H(x₀,w₀)))

**Key Insight:** HMC tends to have high acceptance rates because energy is (theoretically) conserved during simulation

---

## 17. Core DAG Paths (Slide G2)

**Main Idea:** Three fundamental causal structures determine when conditioning is needed to identify causal effects.

**Why Important:** Understanding these paths is essential for proper causal inference and confounder adjustment.

**Three Types:**
- **Fork (z → x, z → y):** z is a **confounder** - must control for it
- **Chain (z → y → x):** z is a **mediator** - control for it blocks indirect effect
- **Collider (x → z ← y):** z is a **collider** - do NOT control for it (creates spurious association)

---

## 18. Adjustment Sets (Slide G1)

**Main Idea:** To identify causal effect of podcast on exam, find variables to control for by blocking non-causal paths (forks, chains) without opening colliders.

**Why Important:** Incorrect adjustment leads to biased causal effect estimates.

**Process:**
1. Find all paths from treatment to outcome
2. Identify forks and chains (need blocking)
3. Identify colliders (avoid conditioning)
4. Minimal adjustment set blocks all backdoor paths

**Example:** Look for mood → podcast and mood → exam (fork); humor → podcast (chain); prepared is on causal path

---

## 19. Difference-in-Differences (Slide G3)

**Main Idea:** Uses parallel trends assumption to identify causal effects by comparing treatment vs. control group changes over time.

**Why Important:** Natural experiments often provide strongest real-world causal evidence.

**Parallel Trends Assumption:** Without treatment, both groups would have followed the same trend (shown by pre-treatment period)

**Visual (Cholera Example):**
- Pre-1849: Both companies have similar trends
- Post-1849: Lambeth (red) drops sharply after treatment
- NonLambeth (blue): Continues previous trend
- **Causal effect:** Difference between observed and counterfactual (dashed red line)

---

## 20. Semi-Partial Correlation in GLMs (Slide G4)

**Main Idea:** Coefficient β₁ represents the unique contribution of x1 to y after removing the shared influence of x2.

**Why Important:** Clarifies what each predictor adds beyond other predictors; essential for interpretation in multiple regression.

**Visual (Venn Diagram):**
- Y circle: Total outcome variance
- X1 circle: Predictor 1 variance
- X2 circle: Predictor 2 variance
- **Arrow points to:** Overlap between Y and X1 AFTER removing X2's influence
- This is what β₁ estimates

---

## Bonus: The "Divergent Transitions" Meme

**Explanation:** In Bayesian statistics (especially Stan/MCMC), "divergent transitions" are a diagnostic warning indicating the sampler is having trouble exploring the posterior distribution, often due to difficult geometry. The meme jokes that when you get this error, you need to deeply investigate your model's structure and parameterization - "the target distribution must be explored" - which can be a frustrating debugging process. It's funny because it's a common pain point for Bayesian modelers!

---

## Study Tips

1. **For each concept:** Be able to explain the main idea in 1-2 sentences
2. **Practice interpreting:** Graphs, formulas, and visual representations
3. **Understand connections:** How regularization relates to basis functions, how GLMs generalize linear models
4. **Know when to use:** Each method (e.g., robust regression for outliers, ZOIB for bounded proportions)
5. **Focus on intuition:** Why methods work, not just mechanical application

Good luck on your exam! 🎯