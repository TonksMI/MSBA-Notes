# Analysis

- Imported the provided `marketingcampaign.csv` data and verified types (5000 customers, 7 variables).
- Ran descriptive checks for key fields (demographics, past purchases, campaign spend) and visualized distributions with histograms and box plots.
- Modeled `past_purchases` with a Poisson regression using age, gender, campaign variant, ad source, and email signup as predictors to address Question 1 (count outcome, log link).
- Modeled `campaign_spend` with an OLS regression on `log1p(campaign_spend)` using the same predictors plus past purchases to address Question 2 (stabilizes skew, enables interpretation of multiplicative effects).
- Generated marginal interpretations (incidence rate ratios and exponentiated log-spend estimates) and scenario predictions for a “typical” customer.

# Results

## Question 1: Which customers make more purchases?

- Overall customers averaged 41.1 purchases across two years (sd ≈ 19.1).
- Age is the strongest driver: each additional year of age corresponds to ~1.9% fewer purchases (IRR = 0.981, p < 0.001).
- Compared to men, women (+32%, IRR = 1.32), non-binary customers (+41%, IRR = 1.41), and customers selecting “Other” (+7%, IRR = 1.07) all make statistically more purchases on average.
- Advertising source effects are modest: customers arriving from Google Search (IRR = 0.97), Instagram (0.97), and TikTok (0.98) make slightly fewer purchases than those from Facebook (p < 0.01 for all three).
- Campaign variant and email signup status do not show meaningful differences once other factors are controlled (p > 0.05). Descriptively, email subscribers average 41.0 purchases vs. 41.2 for non-subscribers, confirming the model result.

**Takeaway:** Younger customers and those identifying as women or non-binary are the most active repeat purchasers. Acquisition channel matters a little (Facebook customers buy more), while the campaign variant itself did not change existing purchasing patterns.

## Question 2: Which campaign variant should Company X use?

- Average campaign spend by variant: A = $148, B = $217, C = $202. Medians (A = $141, B = $205, C = $188) tell the same story.
- The log-linear model finds Variant B customers spend ~56% more than Variant A (coefficient 0.442 ⇒ exp(0.442) ≈ 1.56, p < 0.001) after adjusting for demographics, past purchases, channel, and email signup. Variant C spends ~42% more than Variant A (exp(0.353) ≈ 1.42, p < 0.001).
- Holding other factors at typical values (age 40.5, 41 prior purchases, most common gender = “Man”, channel = Facebook, not email subscribed) predicted spend is ~$86 for Variant A, $134 for Variant B, and $122 for Variant C.
- Variant B leads every ad channel: e.g., Google Search customers spend $266 under B versus $178 under A and $248 under C.

**Recommendation:** Adopt Variant B as the default campaign. It consistently delivers the highest spend both on average and after controlling for customer mix. Variant C is a reasonable alternative, clearly outperforming Variant A, so it could be retained for mental-health-focused messaging or targeted segments.

# Discussion

- **Business impact:** Switching from Variant A to B yields an expected ~$69 uplift per customer (+46%) under typical conditions. Rolling out B broadly could increase revenue by roughly $350k per 5000 customers at current mix. Variant C provides a smaller but still meaningful uplift if B cannot be deployed in specific contexts.
- **Next steps:**
  - Track campaign profitability by incorporating campaign costs and contribution margins to ensure higher spend translates to higher profit.
  - Collect more behavioral features (loyalty tier, product categories, browsing data) to explain the remaining variance in purchases and refine targeting.
  - Run an A/B retest with Variant B vs. C, focusing on long-term retention and customer satisfaction metrics, since both materially outperform the legacy creative.
- **If repeating the assignment:** I would explore models that capture over-dispersion in purchases (e.g., negative binomial) and evaluate zero-inflated specifications for spend, though the current data has few zero spenders (0.8%). Additionally, automating report generation via Quarto would streamline future updates.
