# Topic 1: Data Types and Chart Selection

## Learning Objectives
By the end of this topic, students should understand:
- How to classify different types of business data
- The relationship between data types and appropriate visualizations
- Best practices for chart selection in business contexts
- Common mistakes in chart selection and how to avoid them

## 1. Understanding Business Data Types

### The Data Type Framework

**Data Classification Hierarchy:**
```
Business Data
├── Quantitative (Numerical)
│   ├── Continuous (Revenue, Time, Weight)
│   └── Discrete (Customers, Units, Transactions)
└── Qualitative (Categorical)
    ├── Nominal (Regions, Products, Methods)
    └── Ordinal (Ratings, Priorities, Sizes)
```

### Quantitative Data in Business

#### Continuous Data
**Definition**: Data that can take any value within a range
**Business Examples:**
- **Financial**: Revenue ($1,234.56), profit margins (15.3%), stock prices
- **Temporal**: Response times (2.7 seconds), project duration (45.5 days)
- **Customer**: Satisfaction scores (7.8/10), lifetime value ($2,150.75)
- **Operational**: Machine efficiency (87.3%), temperature (72.4°F)

**Visualization Considerations:**
- Can show precise values and trends
- Suitable for scatter plots, line charts, histograms
- Allow for mathematical operations and statistical analysis

#### Discrete Data
**Definition**: Data that can only take specific, separate values
**Business Examples:**
- **Counts**: Number of customers (1,247), units sold (500), employees (43)
- **Events**: Website visits (12,453), support tickets (23), orders (156)
- **Rankings**: Position in market (3rd), quarter number (Q2)

**Visualization Considerations:**
- Often displayed as whole numbers
- Suitable for bar charts, dot plots, frequency tables
- Cannot meaningfully interpolate between values

### Qualitative Data in Business

#### Nominal Data
**Definition**: Categories without inherent order or ranking
**Business Examples:**
- **Geographic**: Countries, states, cities, regions
- **Categorical**: Product categories, customer segments, departments
- **Descriptive**: Colors, models, methods, status (active/inactive)

**Visualization Guidelines:**
- Use distinct colors or symbols for each category
- Alphabetical or frequency-based ordering
- Avoid implying order where none exists

#### Ordinal Data
**Definition**: Categories with meaningful order but unequal intervals
**Business Examples:**
- **Ratings**: Poor < Fair < Good < Excellent
- **Sizes**: Small < Medium < Large < Extra Large
- **Priorities**: Low < Medium < High < Critical
- **Education**: High School < Bachelor's < Master's < PhD

**Visualization Guidelines:**
- Maintain natural order in displays
- Use color gradients or progressive symbols
- Consider the non-uniform spacing between categories

## 2. The Chart Selection Framework

### Business Question Types and Chart Selection

#### Comparison Questions
**"How do categories compare to each other?"**

**Bar Charts** - Best for comparing discrete categories
```
When to Use Bar Charts:
✓ Sales by product line
✓ Performance by department
✓ Customer satisfaction by region
✓ Market share comparison

Design Best Practices:
- Start bars at zero
- Order categories logically (alphabetical, by value, or by importance)
- Use consistent colors unless highlighting specific categories
- Keep bar width consistent
```

**Column Charts** - Vertical bars for emphasis or space constraints
```
Business Applications:
- Monthly sales comparison (time categories)
- Survey responses by age group
- Budget allocation by department

When to Choose Columns over Bars:
- Category labels are short
- Emphasizing height/magnitude
- Dashboard space is wider than tall
```

#### Distribution Questions  
**"How is the data spread out?"**

**Histograms** - For continuous data distribution
```
Business Applications:
- Customer age distribution
- Order value frequency
- Response time patterns
- Employee salary ranges

Key Design Considerations:
- Choose appropriate bin width
- Label axes clearly with units
- Consider logarithmic scales for skewed data
```

**Box Plots** - For comparing distributions across groups
```
Business Use Cases:
- Sales performance by quarter (showing variability)
- Customer satisfaction by service tier
- Processing times by facility

Interpretation Guide:
- Box shows middle 50% of data (IQR)
- Whiskers show typical range
- Dots indicate outliers needing investigation
```

#### Relationship Questions
**"How do two or more variables relate?"**

**Scatter Plots** - For correlation analysis
```
Business Applications:
- Marketing spend vs. revenue
- Price vs. demand analysis
- Employee experience vs. productivity
- Customer age vs. lifetime value

Design Guidelines:
- Use point size for third variable (bubble chart)
- Add trend lines when appropriate
- Consider log scales for wide value ranges
- Annotate outliers with business context
```

**Line Charts** - For trends over time
```
When to Use Line Charts:
✓ Revenue trends over months/quarters
✓ Stock price movements
✓ Customer acquisition over time
✓ Performance metrics tracking

Critical Design Elements:
- Start time axis at meaningful point
- Use consistent time intervals
- Annotate significant business events
- Multiple lines for comparison (limit to 5-7)
```

#### Composition Questions
**"What are the parts of the whole?"**

**Pie Charts** - For simple part-to-whole relationships
```
Appropriate Uses:
✓ Market share (3-6 competitors)
✓ Budget allocation (major categories)
✓ Survey responses (limited options)

Strict Guidelines:
- Maximum 6-7 slices
- Start largest slice at 12 o'clock
- Use "Other" category for small slices (<5%)
- Avoid 3D effects and explosion
```

**Stacked Bar Charts** - For comparing composition across categories
```
Business Applications:
- Revenue composition by product line and region
- Customer segments by satisfaction level
- Expense breakdown by department and quarter

Design Considerations:
- Order segments consistently across bars
- Use distinctive colors for each segment
- Consider 100% stacked for proportion comparison
```

## 3. Advanced Chart Selection for Business

### Multi-Variable Visualization

#### Three-Variable Relationships
**Bubble Charts** - Size encodes third variable
```
Business Example: Market Portfolio Analysis
X-axis: Market Growth Rate
Y-axis: Market Share
Bubble Size: Revenue
Color: Product Category

Interpretation Benefits:
- Identify high-opportunity products (high growth, low share, large revenue)
- Spot underperforming categories
- Strategic positioning insights
```

**Small Multiples** - Separate charts for each category
```
Use Case: Sales Performance Across Regions
- Same chart type for each region
- Consistent scales for easy comparison
- Facilitates pattern recognition across groups

When to Choose Small Multiples:
- Need to compare patterns across categories
- Single chart would be too crowded
- Audience needs to focus on individual segments
```

#### Geographic Data Visualization

**Choropleth Maps** - Color-coded regions
```
Business Applications:
- Sales performance by state/country
- Customer density mapping
- Market penetration analysis
- Demographic distribution

Design Essentials:
- Choose appropriate color scale
- Include clear legend
- Consider population normalization
- Provide alternative view for accessibility
```

**Symbol Maps** - Proportional symbols on geography
```
Use Cases:
- Store locations with revenue (circle size)
- Distribution centers with capacity
- Market size by metropolitan area

Advantages over Choropleth:
- Shows exact locations
- Better for point data
- Combines location with magnitude
```

### Time Series Specializations

#### Seasonal Patterns
**Line Charts with Seasonal Decomposition**
```
Business Application: Retail Sales Analysis
Component 1: Overall trend (growing/declining)
Component 2: Seasonal pattern (holiday peaks)  
Component 3: Cyclical behavior (economic cycles)
Component 4: Random variation (unexplained)

Value for Business:
- Separate seasonal from underlying trends
- Better forecasting accuracy
- Identify unusual patterns
```

#### Comparative Time Series
**Small Multiple Line Charts**
```
Example: Customer Acquisition by Channel
- Separate chart for each marketing channel
- Same time scale for comparison
- Easy to spot channel-specific patterns

Alternative: Stacked Area Chart
- Shows total and composition over time
- Good for cumulative metrics
- Can be harder to read individual trends
```

## 4. Industry-Specific Chart Applications

### Financial Services

#### Performance Dashboards
```
Primary Charts:
- Line charts: Portfolio performance, stock prices
- Bar charts: Quarterly earnings comparison
- Gauge charts: Risk indicators, compliance scores
- Heat maps: Correlation matrices, risk exposure

Specialized Needs:
- Candlestick charts for trading
- Waterfall charts for P&L analysis
- Bullet charts for target vs. actual
```

#### Risk Management
```
Key Visualizations:
- Box plots: Value-at-risk distributions
- Scatter plots: Risk-return analysis
- Heat maps: Correlation and concentration risk
- Histograms: Loss distributions

Critical Design Elements:
- Color coding for risk levels
- Clear threshold markers
- Real-time update capabilities
```

### Retail and E-commerce

#### Sales Analytics
```
Essential Charts:
- Time series: Sales trends, seasonal patterns
- Geographical: Regional performance maps
- Tree maps: Product category contribution
- Funnel charts: Conversion process

Key Metrics Visualization:
- Revenue trends (line charts)
- Product performance (bar/column charts)
- Customer segmentation (scatter plots)
- Geographic distribution (maps)
```

#### Inventory Management
```
Specialized Visualizations:
- Stock level monitoring (gauge charts)
- ABC analysis (scatter plots)
- Turnover rates (bar charts)
- Seasonal demand patterns (line charts with forecasting)
```

### Manufacturing

#### Quality Control
```
Statistical Process Control Charts:
- Control charts: Process monitoring
- Histograms: Quality distributions  
- Scatter plots: Parameter relationships
- Pareto charts: Defect prioritization

Real-time Requirements:
- Immediate alert visualization
- Trend detection capabilities
- Historical comparison views
```

#### Operations Monitoring
```
Dashboard Components:
- Gauge charts: Efficiency metrics
- Line charts: Production rates
- Heat maps: Equipment status
- Geographic views: Multi-facility monitoring
```

## 5. Chart Selection Decision Framework

### The 5-Step Selection Process

#### Step 1: Define the Business Question
```
Question Categories:
- Comparison: "Which region performs best?"
- Trend: "How have sales changed over time?"
- Distribution: "What's the typical order size?"
- Relationship: "Does price affect demand?"
- Composition: "What makes up our revenue?"
```

#### Step 2: Identify Data Characteristics
```
Data Assessment Checklist:
□ Number of variables (1, 2, 3+)
□ Data types (continuous, discrete, categorical, ordinal)
□ Time element (static snapshot vs. time series)
□ Sample size (small dataset vs. big data)
□ Update frequency (static, daily, real-time)
```

#### Step 3: Consider Audience Needs
```
Audience Factors:
- Technical expertise (executives vs. analysts)
- Decision-making level (strategic vs. operational)
- Interaction requirements (static vs. interactive)
- Device/platform (desktop, tablet, mobile)
- Cultural considerations (global vs. local)
```

#### Step 4: Apply Chart Selection Matrix
```
Chart Selection Matrix:
┌─────────────────┬─────────────┬─────────────┬─────────────┐
│ Purpose         │ 1 Variable  │ 2 Variables │ 3+ Variables│
├─────────────────┼─────────────┼─────────────┼─────────────┤
│ Comparison      │ Bar Chart   │ Grouped Bar │ Small Multi │
│ Distribution    │ Histogram   │ Scatter     │ Bubble      │
│ Trend           │ Line Chart  │ Multi-line  │ Faceted     │
│ Composition     │ Pie Chart   │ Stacked Bar │ Tree Map    │
│ Relationship    │ Box Plot    │ Scatter     │ Matrix      │
└─────────────────┴─────────────┴─────────────┴─────────────┘
```

#### Step 5: Validate and Iterate
```
Validation Questions:
□ Does the chart clearly answer the business question?
□ Can the audience interpret it correctly?
□ Are there alternative views that might be clearer?
□ Does it support the required business decisions?
□ Is it accessible to all intended users?
```

### Common Selection Mistakes and Solutions

#### Mistake 1: Using Pie Charts for Too Many Categories
**Problem**: Pie chart with 10+ slices becomes unreadable
```
Poor Choice:
Market share pie chart with 12 competitors
(Small slices are indistinguishable)

Better Solution:
Horizontal bar chart showing all competitors
+ Focus on top 5 with "Others" category for remainder
```

#### Mistake 2: Line Charts for Categorical Data
**Problem**: Connecting unordered categories with lines
```
Poor Choice:
Line chart connecting performance across departments
(No meaningful progression from Sales to HR to Finance)

Better Solution:
Bar chart comparing departments
+ Ordered by performance level or alphabetically
```

#### Mistake 3: 3D Charts for Business Data
**Problem**: 3D effects distort data perception
```
Problems with 3D:
- Difficult to read exact values
- Perspective distorts comparisons
- Unnecessary complexity

Solution:
Use clean 2D charts with clear labeling
Add depth through color and typography instead
```

#### Mistake 4: Wrong Scale or Axis Treatment
**Problem**: Truncated axes or inappropriate scales
```
Common Issues:
- Bar charts not starting at zero
- Time series with irregular intervals  
- Log scales without clear labeling

Solutions:
- Always start bar/column charts at zero
- Use consistent time intervals
- Clearly label scale transformations
```

## 6. Interactive Chart Considerations

### When to Add Interactivity

#### Appropriate Use Cases
```
Beneficial Interactivity:
✓ Drill-down from summary to detail
✓ Filter by time period or category
✓ Hover for additional context
✓ Toggle between different metrics
✓ Zoom into specific data regions

Avoid Unnecessary Interactivity:
✗ Interactions that don't add insight
✗ Complex controls that confuse users
✗ Features that slow down decision-making
```

#### Interactive Design Principles
```
User Experience Guidelines:
1. Progressive Disclosure: Start simple, add detail on demand
2. Consistent Navigation: Same interactions across similar charts
3. Clear Affordances: Visual cues for interactive elements
4. Graceful Degradation: Core message visible without interaction
5. Performance Consideration: Smooth interactions even with large data
```

### Mobile-Responsive Chart Design

#### Adaptation Strategies
```
Desktop → Mobile Transformations:

Multi-column Dashboard → Single column with priority ordering
Complex Multi-series Chart → Small multiples or tabs
Large Data Table → Summary cards with drill-down
Hover Interactions → Touch-friendly taps and swipes
Small Text Labels → Larger, fewer labels
```

## Key Takeaways for Business Users

### 1. Match Chart Type to Business Question
- Start with what you need to communicate
- Choose charts that directly support the decision
- Avoid chart types that don't fit your data structure

### 2. Consider Your Audience
- Executive dashboards need different charts than operational monitors
- Technical users can handle complexity; executives need simplicity
- Cultural and accessibility needs affect design choices

### 3. Prioritize Clarity Over Complexity
- Simple, clear charts beat impressive but confusing visualizations
- Add complexity only when it genuinely adds insight
- Test chart interpretability with actual users

### 4. Validate Chart Effectiveness
- Can users quickly find the key insight?
- Do the visuals support the necessary business decisions?
- Are there simpler alternatives that work as well?

### 5. Iterate Based on Usage
- Monitor which charts drive action
- Gather feedback from dashboard users
- Continuously improve based on business outcomes

## Practical Exercises

### Exercise 1: Chart Type Selection
Given the following business scenarios, choose the most appropriate chart type and justify your selection:

1. **Scenario**: Compare quarterly sales across five product categories
2. **Scenario**: Show the relationship between marketing spend and lead generation
3. **Scenario**: Display customer satisfaction distribution for different service levels
4. **Scenario**: Illustrate how website traffic changes throughout a typical day

### Exercise 2: Design Evaluation
Evaluate these chart choices and suggest improvements:

1. **Given**: 3D pie chart showing budget allocation across 8 departments
2. **Given**: Line chart connecting customer satisfaction scores across different product categories
3. **Given**: Bar chart with y-axis starting at 95% instead of 0% to show small differences

### Exercise 3: Business Context Application
For your organization (or a case study company):

1. Identify three key business questions that require visualization
2. Determine what data types are involved
3. Select appropriate chart types using the decision framework
4. Consider what interactivity would be valuable
5. Plan how to validate the effectiveness of your choices

---

**Next Topic**: [Design Principles and Visual Perception](./Topic-02-Design-Principles-and-Visual-Perception.md) - Learn how to apply design principles and understanding of human perception to create more effective business visualizations.