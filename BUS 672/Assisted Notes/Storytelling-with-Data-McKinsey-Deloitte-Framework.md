# Storytelling with Data: McKinsey and Deloitte Framework for Business Presentations and Dashboards

## Overview
This document provides a comprehensive guide to data storytelling using proven frameworks from top-tier consulting firms McKinsey & Company and Deloitte. These methodologies are designed to create compelling, actionable business presentations and dashboards that drive decision-making at the executive level.

## Table of Contents
1. [Consulting Firm Storytelling Philosophy](#consulting-firm-storytelling-philosophy)
2. [The McKinsey Pyramid Principle](#the-mckinsey-pyramid-principle)
3. [Deloitte's SCRAP Framework](#deloittes-scrap-framework)
4. [Executive Dashboard Design](#executive-dashboard-design)
5. [Presentation Structure Templates](#presentation-structure-templates)
6. [Data Visualization Best Practices](#data-visualization-best-practices)
7. [Stakeholder Communication Strategies](#stakeholder-communication-strategies)
8. [Implementation Toolkit](#implementation-toolkit)

## Consulting Firm Storytelling Philosophy

### Core Principles

#### 1. **Answer First, Evidence Second**
**McKinsey Approach:**
- Lead with the conclusion
- Support with data and analysis
- Structure arguments deductively

**Business Application:**
```
Executive Summary Structure:
┌─────────────────────────────────────────────────────────┐
│ RECOMMENDATION: Increase marketing spend by 40%        │
│ EXPECTED OUTCOME: $2.5M additional revenue in Q1      │
│ INVESTMENT REQUIRED: $800K                             │
│ ROI: 312% over 6 months                               │
├─────────────────────────────────────────────────────────┤
│ Supporting Evidence:                                    │
│ • Customer acquisition cost down 25%                   │
│ • Market share opportunity: 15% untapped              │
│ • Competitive analysis shows pricing advantage         │
└─────────────────────────────────────────────────────────┘
```

#### 2. **So What? Factor**
**Deloitte Approach:**
- Every data point must answer "So what?"
- Connect insights to business impact
- Quantify implications wherever possible

**Example Application:**
```
Data Point: Customer churn rate increased from 5% to 8%
So What: $2.1M annual revenue at risk
Therefore: Implement retention program worth $300K investment
Expected Outcome: Save $1.5M net annual revenue
```

### The MECE Principle (Mutually Exclusive, Collectively Exhaustive)

#### Structure for Complex Business Problems
```
Problem: Declining Market Share
├─ Internal Factors (Mutually Exclusive)
│  ├─ Product Quality Issues
│  ├─ Pricing Strategy Problems
│  └─ Sales Force Effectiveness
└─ External Factors (Mutually Exclusive)
   ├─ Competitive Pressure
   ├─ Market Dynamics
   └─ Economic Conditions

Each category is complete (Collectively Exhaustive)
```

#### Dashboard Application
```
Executive KPI Dashboard (MECE Structure):
┌─────────────────────────────────────────────────────────┐
│ FINANCIAL PERFORMANCE                                   │
│ ├─ Revenue Growth: +12% YoY                            │
│ ├─ Profit Margin: 18.5% (Target: 20%)                 │
│ └─ Cash Flow: $2.1M positive                          │
├─────────────────────────────────────────────────────────┤
│ OPERATIONAL EXCELLENCE                                  │
│ ├─ Customer Satisfaction: 4.2/5.0                     │
│ ├─ Employee Engagement: 78% (Industry: 72%)           │
│ └─ Process Efficiency: 95% target achievement         │
├─────────────────────────────────────────────────────────┤
│ STRATEGIC INITIATIVES                                   │
│ ├─ New Product Launch: 85% of timeline                │
│ ├─ Market Expansion: 3 new regions                    │
│ └─ Digital Transformation: Phase 2 complete           │
└─────────────────────────────────────────────────────────┘
```

## The McKinsey Pyramid Principle

### Three-Level Structure

#### Level 1: Main Argument (Apex)
**Template:**
```
Primary Recommendation/Conclusion
"Company X should [SPECIFIC ACTION] to achieve [QUANTIFIED OUTCOME] by [TIMEFRAME]"

Example:
"TechCorp should acquire CloudStart to increase cloud revenue by $50M annually within 18 months"
```

#### Level 2: Supporting Arguments (Second Level)
**Three Key Pillars Framework:**
```
Pillar 1: Strategic Fit
├─ Market positioning alignment
├─ Technology synergies
└─ Cultural compatibility

Pillar 2: Financial Rationale
├─ Revenue synergies: $50M
├─ Cost synergies: $15M
└─ ROI: 24% IRR

Pillar 3: Implementation Feasibility
├─ Integration timeline: 12 months
├─ Regulatory approvals: 90% probability
└─ Management bandwidth: Available
```

#### Level 3: Data and Evidence (Base)
**Supporting Visualizations:**

```python
# McKinsey-style chart code examples
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Revenue Synergy Waterfall Chart
def create_mckinsey_waterfall():
    categories = ['Current Revenue', 'Cross-selling', 'Upselling', 'New Markets', 'Combined Revenue']
    values = [100, 15, 20, 15, 150]

    fig, ax = plt.subplots(figsize=(12, 6))

    # McKinsey color scheme
    colors = ['#1f77b4', '#2ca02c', '#2ca02c', '#2ca02c', '#1f77b4']

    bars = ax.bar(categories, values, color=colors)

    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'${value}M', ha='center', va='bottom', fontweight='bold')

    ax.set_title('Revenue Synergy Analysis: Path to $150M Combined Revenue',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Revenue ($M)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig

# 2. McKinsey 2x2 Matrix
def create_strategic_matrix():
    fig, ax = plt.subplots(figsize=(10, 8))

    # Sample business units
    units = {
        'Cloud Services': (8.5, 7.2),
        'Legacy Software': (3.2, 8.8),
        'Mobile Apps': (7.8, 4.1),
        'Analytics Platform': (6.5, 6.8),
        'Security Solutions': (9.1, 5.5)
    }

    # Plot points
    for unit, (growth, profit) in units.items():
        ax.scatter(growth, profit, s=200, alpha=0.7)
        ax.annotate(unit, (growth, profit), xytext=(5, 5),
                   textcoords='offset points', fontweight='bold')

    # Quadrant lines
    ax.axhline(y=6, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=6, color='gray', linestyle='--', alpha=0.5)

    # Quadrant labels
    ax.text(8.5, 8.5, 'STARS\n(Invest)', ha='center', va='center',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))

    ax.text(3, 8.5, 'CASH COWS\n(Maintain)', ha='center', va='center',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))

    ax.text(8.5, 3, 'QUESTION MARKS\n(Selective)', ha='center', va='center',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.7))

    ax.text(3, 3, 'DOGS\n(Divest)', ha='center', va='center',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))

    ax.set_xlabel('Market Growth Rate', fontweight='bold', fontsize=12)
    ax.set_ylabel('Profit Margin', fontweight='bold', fontsize=12)
    ax.set_title('Business Unit Portfolio Analysis', fontweight='bold', fontsize=14, pad=20)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    plt.tight_layout()
    return fig
```

### Pyramid Implementation in Presentations

#### Slide Structure Template
```
Slide 1: Executive Summary
├─ Main recommendation (30-word headline)
├─ Three supporting reasons
└─ Call to action

Slides 2-4: Supporting Arguments
├─ One slide per supporting argument
├─ Data visualization + key insight
└─ "Therefore" statement linking to main recommendation

Slides 5-7: Detailed Evidence
├─ Methodology and assumptions
├─ Detailed analysis and calculations
└─ Risk mitigation and next steps
```

## Deloitte's SCRAP Framework

### S - Situation
**Purpose:** Set the business context and establish stakes

**Template:**
```
Situation Statement:
"[Company/Industry] is facing [specific challenge/opportunity] that impacts [business metric] by [quantified amount] and threatens/enables [strategic objective]."

Example:
"RetailCorp is facing a 15% decline in in-store sales over the past 18 months, threatening the company's goal of $500M annual revenue and requiring immediate digital transformation initiatives."
```

**Dashboard Application:**
```
Situation Dashboard Panel:
┌─────────────────────────────────────────────────────────┐
│ CURRENT BUSINESS CONTEXT                                │
│                                                         │
│ Market Position: #3 in regional market (down from #2)  │
│ Revenue Trend: ↓ 15% YoY                               │
│ Customer Base: 2.1M active (↓ 8% retention)           │
│ Competitive Threat: New entrant gained 12% share      │
│                                                         │
│ Key Challenge: Digital customer experience gap         │
│ Strategic Goal: $500M annual revenue by 2025          │
└─────────────────────────────────────────────────────────┘
```

### C - Complication
**Purpose:** Explain why the situation is problematic or urgent

**Framework:**
```
Complication Analysis:
├─ Root Causes
│  ├─ Internal factors (within control)
│  └─ External factors (environmental)
├─ Impact Assessment
│  ├─ Financial implications
│  ├─ Operational consequences
│  └─ Strategic risks
└─ Urgency Drivers
   ├─ Competitive dynamics
   ├─ Market timing
   └─ Resource constraints
```

**Example Visualization:**
```python
# Complication Impact Chart
def create_impact_assessment():
    factors = ['Customer\nExperience', 'Operational\nEfficiency', 'Market\nPosition', 'Financial\nPerformance']
    current_state = [6.2, 7.8, 5.4, 6.9]
    target_state = [8.5, 8.2, 7.8, 8.0]

    x = range(len(factors))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar([i - width/2 for i in x], current_state, width,
           label='Current State', color='#ff7f7f', alpha=0.8)
    ax.bar([i + width/2 for i in x], target_state, width,
           label='Target State', color='#90EE90', alpha=0.8)

    # Gap arrows
    for i, (current, target) in enumerate(zip(current_state, target_state)):
        if target > current:
            ax.annotate('', xy=(i, target), xytext=(i, current),
                       arrowprops=dict(arrowstyle='<->', color='red', lw=2))
            gap = target - current
            ax.text(i, (current + target)/2, f'Gap: {gap:.1f}',
                   ha='center', va='center', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

    ax.set_xlabel('Business Dimensions', fontweight='bold')
    ax.set_ylabel('Performance Score (1-10)', fontweight='bold')
    ax.set_title('Performance Gap Analysis: Current vs Target State',
                 fontweight='bold', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(factors)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig
```

### R - Resolution
**Purpose:** Present the solution and its benefits

**McKinsey-Deloitte Hybrid Approach:**
```
Resolution Framework:
├─ Primary Solution
│  ├─ What: Specific initiative/program
│  ├─ How: Implementation approach
│  └─ When: Timeline and milestones
├─ Expected Outcomes
│  ├─ Quantified benefits
│  ├─ Risk mitigation
│  └─ Success metrics
└─ Alternative Options
   ├─ Option B (with trade-offs)
   ├─ Do nothing scenario
   └─ Recommendation rationale
```

### A - Action
**Purpose:** Define specific next steps and ownership

**Action Planning Template:**
```
Next Steps Framework:
┌─────────────────────────────────────────────────────────┐
│ IMMEDIATE ACTIONS (Next 30 Days)                       │
│ ├─ Form steering committee (Owner: CEO)                │
│ ├─ Conduct detailed feasibility study (Owner: Strategy)│
│ └─ Secure initial funding approval (Owner: CFO)        │
├─────────────────────────────────────────────────────────┤
│ SHORT-TERM ACTIONS (90 Days)                           │
│ ├─ Complete vendor selection process                   │
│ ├─ Finalize implementation roadmap                     │
│ └─ Begin pilot program in 2 markets                   │
├─────────────────────────────────────────────────────────┤
│ LONG-TERM ACTIONS (6-12 Months)                       │
│ ├─ Full program rollout across all markets            │
│ ├─ Measure and optimize performance                    │
│ └─ Scale successful initiatives                        │
└─────────────────────────────────────────────────────────┘
```

### P - Payoff
**Purpose:** Quantify the business value and ROI

**Financial Impact Framework:**
```python
# ROI Visualization
def create_payoff_analysis():
    months = range(1, 25)
    investment = [-2000, -1500, -1000, -500] + [0] * 20
    returns = [0, 0, 200, 500] + [800 + i*50 for i in range(20)]
    cumulative = []
    total = 0

    for inv, ret in zip(investment, returns):
        total += inv + ret
        cumulative.append(total)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot cumulative cash flow
    ax.plot(months, cumulative, linewidth=3, color='#2E86AB', marker='o')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)

    # Highlight breakeven point
    breakeven_month = next(i for i, val in enumerate(cumulative) if val > 0) + 1
    ax.axvline(x=breakeven_month, color='green', linestyle='--', alpha=0.7)
    ax.text(breakeven_month + 1, 1000, f'Breakeven:\nMonth {breakeven_month}',
            fontweight='bold', ha='left', va='bottom',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))

    # Add investment phase
    ax.fill_between(months[:4], 0, cumulative[:4], alpha=0.3, color='red', label='Investment Phase')
    ax.fill_between(months[3:], 0, cumulative[3:], alpha=0.3, color='green', label='Return Phase')

    ax.set_xlabel('Months from Start', fontweight='bold')
    ax.set_ylabel('Cumulative Cash Flow ($000)', fontweight='bold')
    ax.set_title('Project ROI Timeline: Investment to Payoff', fontweight='bold', fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add key metrics
    final_roi = cumulative[-1]
    ax.text(20, final_roi/2, f'24-Month ROI:\n${final_roi:,.0f}K\n\nROI: {((final_roi + 5000)/5000 - 1)*100:.0f}%',
            fontsize=12, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    return fig
```

## Executive Dashboard Design

### McKinsey Dashboard Principles

#### 1. **Top-Down Information Architecture**
```
Executive Dashboard Hierarchy:
Level 1: Business Health (Red/Yellow/Green)
├─ Financial Performance
├─ Operational Excellence
└─ Strategic Progress

Level 2: Key Metrics (Trends and Targets)
├─ Revenue, Profit, Cash Flow
├─ Customer, Employee, Process metrics
└─ Initiative status and milestones

Level 3: Detailed Analytics (Drill-down)
├─ Regional/product breakdowns
├─ Root cause analysis
└─ Predictive insights
```

#### 2. **Action-Oriented Design**
```python
# Executive KPI Card Template
def create_executive_kpi_card(metric_name, current_value, target_value,
                             trend_direction, business_impact):
    """
    Creates McKinsey-style KPI cards for executive dashboards
    """
    # Color coding based on performance
    if current_value >= target_value:
        status_color = '#2E8B57'  # Green
        status_text = 'ON TRACK'
    elif current_value >= target_value * 0.9:
        status_color = '#FFD700'  # Yellow
        status_text = 'AT RISK'
    else:
        status_color = '#DC143C'  # Red
        status_text = 'OFF TRACK'

    return f"""
    KPI Card: {metric_name}
    ┌─────────────────────────────────────┐
    │ {current_value} │ Target: {target_value} │ {status_text} │
    │ Trend: {trend_direction}            │
    │ Impact: {business_impact}           │
    │ [Action Required] [Drill Down]     │
    └─────────────────────────────────────┘
    """

# Example usage
revenue_card = create_executive_kpi_card(
    metric_name="Q4 Revenue",
    current_value="$42.3M",
    target_value="$45.0M",
    trend_direction="↗ +8% vs Q3",
    business_impact="$2.7M gap to target"
)
```

### Deloitte Dashboard Framework

#### 1. **Insight-Driven Layout**
```
Deloitte Dashboard Structure:
┌─────────────────────────────────────────────────────────┐
│ EXECUTIVE SUMMARY                                       │
│ • 3 Key Insights (What happened?)                      │
│ • 2 Recommended Actions (What to do?)                  │
│ • 1 Critical Decision (What to decide?)                │
├─────────────────────────────────────────────────────────┤
│ PERFORMANCE METRICS                                     │
│ [KPI Cards with Traffic Light Status]                  │
├─────────────────────────────────────────────────────────┤
│ TREND ANALYSIS                                          │
│ [Time Series Charts with Annotations]                  │
├─────────────────────────────────────────────────────────┤
│ DEEP DIVE ANALYTICS                                     │
│ [Interactive Filters and Drill-down Options]           │
└─────────────────────────────────────────────────────────┘
```

#### 2. **Predictive Elements**
```python
# Predictive Dashboard Component
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_predictive_dashboard():
    """
    Creates Deloitte-style predictive analytics dashboard
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Revenue Forecast', 'Risk Assessment',
                       'Scenario Planning', 'Action Impact'),
        specs=[[{"secondary_y": True}, {}],
               [{}, {}]]
    )

    # Historical and predicted data
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
    historical = [100, 105, 110, 108, 115, 120, None, None, None]
    predicted = [None, None, None, None, None, 120, 125, 130, 135]
    confidence_upper = [None, None, None, None, None, 125, 132, 140, 145]
    confidence_lower = [None, None, None, None, None, 115, 118, 120, 125]

    # Add historical data
    fig.add_trace(
        go.Scatter(x=months, y=historical, name='Historical',
                  line=dict(color='blue', width=3)),
        row=1, col=1
    )

    # Add prediction with confidence intervals
    fig.add_trace(
        go.Scatter(x=months, y=predicted, name='Forecast',
                  line=dict(color='red', width=3, dash='dash')),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=months, y=confidence_upper,
                  fill=None, mode='lines', line_color='rgba(0,0,0,0)',
                  showlegend=False),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=months, y=confidence_lower,
                  fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)',
                  name='Confidence Interval', fillcolor='rgba(255,0,0,0.2)'),
        row=1, col=1
    )

    fig.update_layout(
        title_text="Executive Predictive Analytics Dashboard",
        height=600,
        showlegend=True
    )

    return fig
```

## Presentation Structure Templates

### McKinsey Presentation Template

#### Slide 1: Executive Summary
```
Template Structure:
┌─────────────────────────────────────────────────────────┐
│ [COMPANY LOGO]              [DATE]                     │
│                                                         │
│ EXECUTIVE SUMMARY                                       │
│ [Project/Analysis Title]                               │
│                                                         │
│ RECOMMENDATION:                                         │
│ [One sentence, specific action with quantified outcome]│
│                                                         │
│ KEY SUPPORTING ARGUMENTS:                               │
│ 1. [Supporting point 1 with key metric]               │
│ 2. [Supporting point 2 with key metric]               │
│ 3. [Supporting point 3 with key metric]               │
│                                                         │
│ EXPECTED IMPACT:                                        │
│ • Financial: [Specific dollar amount and timeframe]   │
│ • Strategic: [Competitive/market position impact]     │
│ • Operational: [Process/efficiency improvements]      │
│                                                         │
│ NEXT STEPS:                                            │
│ 1. [Immediate action with owner and date]             │
│ 2. [Follow-up action with timeline]                   │
└─────────────────────────────────────────────────────────┘
```

#### Slide Types and Templates

**Analysis Slide Template:**
```
[SLIDE TITLE: Insight-driven headline]

Key Insight: [One sentence summary of finding]

[MAIN VISUALIZATION: Chart supporting the insight]

Supporting Evidence:
• [Data point 1]
• [Data point 2]
• [Data point 3]

Implication: [So what? Business impact]

Source: [Data sources and methodology]
```

**Recommendation Slide Template:**
```
[SLIDE TITLE: Action-oriented headline]

Recommendation: [Specific action to take]

Rationale:
1. [Reason 1 with supporting data]
2. [Reason 2 with supporting data]
3. [Reason 3 with supporting data]

Implementation:
• Timeline: [Specific dates and milestones]
• Resources: [Investment and team requirements]
• Success Metrics: [How to measure progress]

Risk Mitigation: [Key risks and mitigation strategies]
```

### Deloitte Presentation Framework

#### The "Story Arc" Structure
```
Presentation Story Arc:
┌─────────────────────────────────────────────────────────┐
│ Act I: Setup (Slides 1-3)                              │
│ ├─ Current Situation                                   │
│ ├─ Challenge/Opportunity                               │
│ └─ Stakes/Urgency                                      │
├─────────────────────────────────────────────────────────┤
│ Act II: Conflict (Slides 4-8)                          │
│ ├─ Root Cause Analysis                                 │
│ ├─ Impact Assessment                                   │
│ ├─ Options Evaluation                                  │
│ └─ Trade-offs and Risks                               │
├─────────────────────────────────────────────────────────┤
│ Act III: Resolution (Slides 9-12)                      │
│ ├─ Recommended Solution                                │
│ ├─ Implementation Roadmap                              │
│ ├─ Expected Outcomes                                   │
│ └─ Call to Action                                      │
└─────────────────────────────────────────────────────────┘
```

#### Slide Transition Framework
```python
# Deloitte Slide Transition Template
def create_slide_transitions():
    """
    Template for smooth narrative transitions between slides
    """
    transitions = {
        'setup_to_conflict': "While this situation appears manageable, deeper analysis reveals...",
        'conflict_to_analysis': "To understand the root causes, we examined...",
        'analysis_to_options': "Based on this analysis, we identified three potential solutions...",
        'options_to_recommendation': "After evaluating all options, we recommend...",
        'recommendation_to_implementation': "To execute this recommendation successfully, we propose...",
        'implementation_to_payoff': "This approach will deliver the following benefits..."
    }
    return transitions

# Usage in presentation
transition = create_slide_transitions()
slide_text = f"""
Previous slide conclusion: Market share declined 15%
{transition['setup_to_conflict']}
Current slide: Root cause analysis showing...
"""
```

## Data Visualization Best Practices

### McKinsey Chart Standards

#### Color Palette and Branding
```python
# McKinsey-inspired color palette
mckinsey_colors = {
    'primary_blue': '#1F4E79',
    'secondary_blue': '#5B9BD5',
    'accent_green': '#70AD47',
    'warning_orange': '#FFC000',
    'error_red': '#C55A5A',
    'neutral_gray': '#7F7F7F',
    'light_gray': '#D9D9D9'
}

# Chart styling function
def apply_mckinsey_style(fig, ax):
    """Apply McKinsey-style formatting to matplotlib charts"""

    # Remove chart junk
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(mckinsey_colors['neutral_gray'])
    ax.spines['bottom'].set_color(mckinsey_colors['neutral_gray'])

    # Grid styling
    ax.grid(True, alpha=0.3, color=mckinsey_colors['light_gray'])
    ax.set_axisbelow(True)

    # Font styling
    ax.tick_params(colors=mckinsey_colors['neutral_gray'])
    ax.xaxis.label.set_color(mckinsey_colors['neutral_gray'])
    ax.yaxis.label.set_color(mckinsey_colors['neutral_gray'])

    # Title styling
    ax.title.set_fontweight('bold')
    ax.title.set_fontsize(14)
    ax.title.set_color(mckinsey_colors['primary_blue'])

    return fig, ax
```

#### Chart Type Selection Matrix
```
Chart Selection Framework:
┌─────────────────┬─────────────────┬─────────────────┐
│ Data Type       │ Relationship    │ Recommended     │
│                 │ to Show         │ Chart Type      │
├─────────────────┼─────────────────┼─────────────────┤
│ Time Series     │ Trends          │ Line Chart      │
│ Categories      │ Comparison      │ Bar Chart       │
│ Part-to-Whole   │ Composition     │ Stacked Bar     │
│ Two Variables   │ Correlation     │ Scatter Plot    │
│ Geographic      │ Spatial         │ Map/Choropleth  │
│ Hierarchical    │ Structure       │ Treemap         │
│ Process Flow    │ Sequence        │ Waterfall       │
│ Distribution    │ Frequency       │ Histogram       │
└─────────────────┴─────────────────┴─────────────────┘
```

### Deloitte Visualization Standards

#### Interactive Dashboard Elements
```python
# Deloitte-style interactive dashboard components
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def create_deloitte_dashboard(data):
    """
    Create interactive dashboard with Deloitte design principles
    """

    # Color scheme
    deloitte_colors = ['#86BC25', '#62B5E5', '#00A8CC', '#FFB000', '#C4C4C4']

    fig = make_subplots(
        rows=3, cols=3,
        specs=[[{"colspan": 2}, None, {}],
               [{"colspan": 3}, None, None],
               [{}, {}, {}]],
        subplot_titles=('Executive Summary', 'Key Metrics',
                       'Trend Analysis', 'Regional Performance',
                       'Customer Segments', 'Action Items'),
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )

    # Executive KPI cards (top-left)
    kpi_values = [85, 92, 78]
    kpi_labels = ['Customer Satisfaction', 'Operational Efficiency', 'Market Share']

    for i, (value, label) in enumerate(zip(kpi_values, kpi_labels)):
        color = deloitte_colors[0] if value >= 80 else deloitte_colors[3] if value >= 70 else deloitte_colors[4]

        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=value,
                domain={'x': [0, 0.3], 'y': [0.7, 1]},
                title={'text': label},
                delta={'reference': 80},
                gauge={'axis': {'range': [None, 100]},
                      'bar': {'color': color},
                      'steps': [{'range': [0, 70], 'color': "lightgray"},
                               {'range': [70, 90], 'color': "gray"}],
                      'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}
            )
        )

    # Add interactivity
    fig.update_layout(
        title={
            'text': "Executive Performance Dashboard",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#003F5C'}
        },
        height=800,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig

# Filter component for interactivity
def create_dashboard_filters():
    """Create interactive filter components"""
    filters = {
        'date_range': {
            'type': 'date_picker',
            'label': 'Date Range',
            'default': 'Last 90 Days'
        },
        'business_unit': {
            'type': 'dropdown',
            'label': 'Business Unit',
            'options': ['All', 'North America', 'Europe', 'Asia-Pacific'],
            'default': 'All'
        },
        'metric_view': {
            'type': 'radio_buttons',
            'label': 'View Type',
            'options': ['Summary', 'Detailed', 'Trends'],
            'default': 'Summary'
        }
    }
    return filters
```

### Chart Annotation Best Practices

#### McKinsey Annotation Style
```python
def add_mckinsey_annotations(fig, ax, key_insights):
    """
    Add McKinsey-style annotations to highlight key insights
    """

    # Insight callout box
    insight_style = dict(
        boxstyle="round,pad=0.3",
        facecolor='lightblue',
        edgecolor='navy',
        alpha=0.8
    )

    # Add primary insight
    ax.annotate(
        key_insights['primary'],
        xy=key_insights['primary_location'],
        xytext=(0.7, 0.9),
        xycoords='data',
        textcoords='axes fraction',
        fontsize=12,
        fontweight='bold',
        bbox=insight_style,
        arrowprops=dict(arrowstyle='->', color='navy', lw=2)
    )

    # Add secondary insights
    for i, insight in enumerate(key_insights['secondary']):
        ax.text(
            0.02, 0.98 - i*0.05,
            f"• {insight}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7)
        )

    # Add source note
    ax.text(
        0.99, 0.01,
        f"Source: {key_insights['source']}",
        transform=ax.transAxes,
        fontsize=8,
        horizontalalignment='right',
        verticalalignment='bottom',
        style='italic',
        color='gray'
    )

    return fig, ax

# Usage example
key_insights = {
    'primary': "Revenue growth accelerating:\n+15% vs target of +10%",
    'primary_location': (6, 150),  # x, y coordinates on chart
    'secondary': [
        "Digital channels driving 60% of growth",
        "Customer acquisition cost down 25%",
        "Market share increased from 18% to 22%"
    ],
    'source': "Company financial reports, Q1-Q3 2024"
}
```

## Stakeholder Communication Strategies

### Audience Segmentation Framework

#### Executive Level (C-Suite)
**Communication Style:**
- **Time**: 5-10 minutes maximum
- **Focus**: Strategic implications and ROI
- **Format**: High-level summary with 3 key points
- **Metrics**: Financial impact, competitive advantage, risk mitigation

**Dashboard Requirements:**
```
C-Suite Dashboard Elements:
┌─────────────────────────────────────────────────────────┐
│ RED/YELLOW/GREEN STATUS INDICATORS                      │
│ ├─ Overall business health                             │
│ ├─ Strategic initiative progress                       │
│ └─ Critical issues requiring attention                 │
├─────────────────────────────────────────────────────────┤
│ KEY FINANCIAL METRICS                                   │
│ ├─ Revenue vs target                                   │
│ ├─ Profit margin trends                               │
│ └─ Cash flow projection                               │
├─────────────────────────────────────────────────────────┤
│ STRATEGIC INITIATIVES                                   │
│ ├─ Project timeline and milestones                    │
│ ├─ Budget vs actual spending                          │
│ └─ Expected ROI and payback period                    │
└─────────────────────────────────────────────────────────┘
```

#### Management Level (VPs, Directors)
**Communication Style:**
- **Time**: 15-30 minutes
- **Focus**: Operational performance and tactical decisions
- **Format**: Detailed analysis with actionable recommendations
- **Metrics**: Departmental KPIs, process efficiency, team performance

#### Operational Level (Managers, Analysts)
**Communication Style:**
- **Time**: 30-60 minutes
- **Focus**: Detailed implementation and troubleshooting
- **Format**: Comprehensive analysis with technical details
- **Metrics**: Granular operational data, individual performance, quality measures

### Presentation Customization Templates

```python
def customize_presentation_for_audience(content, audience_level):
    """
    Customize presentation content based on audience level
    """

    templates = {
        'executive': {
            'slide_count': 5,
            'detail_level': 'high_level',
            'focus_areas': ['financial_impact', 'strategic_alignment', 'risk_assessment'],
            'visual_style': 'simple_charts',
            'time_allocation': {
                'setup': 0.2,
                'analysis': 0.3,
                'recommendations': 0.4,
                'next_steps': 0.1
            }
        },
        'management': {
            'slide_count': 15,
            'detail_level': 'moderate',
            'focus_areas': ['operational_metrics', 'implementation_plan', 'resource_requirements'],
            'visual_style': 'detailed_charts',
            'time_allocation': {
                'setup': 0.15,
                'analysis': 0.45,
                'recommendations': 0.25,
                'next_steps': 0.15
            }
        },
        'operational': {
            'slide_count': 25,
            'detail_level': 'detailed',
            'focus_areas': ['technical_analysis', 'methodology', 'detailed_findings'],
            'visual_style': 'complex_visualizations',
            'time_allocation': {
                'setup': 0.1,
                'analysis': 0.6,
                'recommendations': 0.2,
                'next_steps': 0.1
            }
        }
    }

    return templates[audience_level]

# Example usage
exec_template = customize_presentation_for_audience(content, 'executive')
```

## Implementation Toolkit

### Dashboard Development Checklist

#### Pre-Development Phase
```
□ Stakeholder Requirements Gathering
  □ Define primary users and use cases
  □ Identify key metrics and KPIs
  □ Determine refresh frequency and data sources
  □ Establish success criteria

□ Data Architecture Planning
  □ Map data sources and quality
  □ Design ETL/data pipeline
  □ Plan for real-time vs batch updates
  □ Consider data governance and security

□ Design and UX Planning
  □ Create wireframes and mockups
  □ Define color scheme and branding
  □ Plan responsive design for mobile
  □ Consider accessibility requirements
```

#### Development Phase
```
□ Technical Implementation
  □ Set up development environment
  □ Implement data connections
  □ Build visualization components
  □ Add interactivity and filters

□ Content Development
  □ Create chart templates
  □ Implement business logic
  □ Add annotations and insights
  □ Build alert and notification systems

□ Testing and Validation
  □ Test with real data
  □ Validate calculations and metrics
  □ Test performance with large datasets
  □ Conduct user acceptance testing
```

#### Deployment Phase
```
□ Production Deployment
  □ Deploy to production environment
  □ Configure security and access controls
  □ Set up monitoring and alerting
  □ Create user documentation

□ Training and Adoption
  □ Conduct user training sessions
  □ Create quick reference guides
  □ Establish support processes
  □ Monitor usage and gather feedback

□ Maintenance and Evolution
  □ Schedule regular reviews
  □ Plan feature enhancements
  □ Monitor performance and optimization
  □ Update based on user feedback
```

### Presentation Development Templates

#### McKinsey Slide Template (PowerPoint/Google Slides)
```
Slide Specifications:
┌─────────────────────────────────────────────────────────┐
│ Layout: 16:9 aspect ratio                              │
│ Fonts: Helvetica or similar sans-serif                 │
│ Title: 24pt, bold, dark blue (#1F4E79)                │
│ Body text: 18pt, regular, dark gray (#404040)         │
│ Margins: 1 inch on all sides                          │
│                                                         │
│ Header: Company logo (left), Date (right)             │
│ Footer: Page number (center), Confidential (right)    │
│                                                         │
│ Chart specifications:                                   │
│ ├─ Maximum 1 chart per slide                          │
│ ├─ Clear title and axis labels                        │
│ ├─ Source citation at bottom                          │
│ └─ Color scheme: Blue primary, green accent           │
└─────────────────────────────────────────────────────────┘
```

#### Deloitte Presentation Template
```
Slide Structure:
┌─────────────────────────────────────────────────────────┐
│ [Slide Number] [Section Title]                         │
│                                                         │
│ INSIGHT-DRIVEN HEADLINE                               │
│ [One-sentence summary of key finding]                  │
│                                                         │
│ [MAIN VISUALIZATION]                                    │
│ [Chart with clear title and annotations]               │
│                                                         │
│ Key Takeaways:                                         │
│ • [Takeaway 1]                                         │
│ • [Takeaway 2]                                         │
│ • [Takeaway 3]                                         │
│                                                         │
│ So What: [Business implication]                        │
│                                                         │
│ Source: [Data sources and methodology]                 │
└─────────────────────────────────────────────────────────┘
```

### Code Templates and Snippets

#### Dashboard Framework (Python/Streamlit)
```python
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta

def create_executive_dashboard():
    """
    Create McKinsey/Deloitte-style executive dashboard using Streamlit
    """

    # Page configuration
    st.set_page_config(
        page_title="Executive Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for McKinsey-style appearance
    st.markdown("""
    <style>
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1F4E79;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1F4E79;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
        margin-bottom: 0.5rem;
    }
    .insight-box {
        background-color: #E8F4FD;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #5B9BD5;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar filters
    st.sidebar.title("Dashboard Filters")
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(datetime.now() - timedelta(days=90), datetime.now()),
        max_value=datetime.now()
    )

    business_unit = st.sidebar.selectbox(
        "Business Unit",
        ["All", "North America", "Europe", "Asia-Pacific"]
    )

    # Main dashboard
    st.title("Executive Performance Dashboard")
    st.markdown("---")

    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Revenue (YTD)</div>
            <div class="metric-value">$42.3M</div>
            <div style="color: green;">↗ +12% vs target</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Customer Satisfaction</div>
            <div class="metric-value">4.2/5.0</div>
            <div style="color: orange;">→ Target: 4.5</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Market Share</div>
            <div class="metric-value">22.1%</div>
            <div style="color: green;">↗ +2.1% YoY</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Operating Margin</div>
            <div class="metric-value">18.5%</div>
            <div style="color: red;">↘ -0.5% vs Q3</div>
        </div>
        """, unsafe_allow_html=True)

    # Key Insights Section
    st.markdown("""
    <div class="insight-box">
        <h3>📈 Key Insights</h3>
        <ul>
            <li><strong>Revenue Growth Acceleration:</strong> Q4 revenue up 12% vs target, driven by digital channel expansion</li>
            <li><strong>Customer Experience Gap:</strong> Satisfaction score 0.3 points below target, requires immediate attention</li>
            <li><strong>Market Position Strengthening:</strong> Gained 2.1% market share, outpacing competitors</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Charts Section
    col1, col2 = st.columns(2)

    with col1:
        # Revenue trend chart
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        revenue = [3.2, 3.4, 3.8, 3.6, 4.1, 4.3, 4.0, 4.5, 4.2, 4.7, 4.4, 4.6]
        target = [3.5, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months, y=revenue, mode='lines+markers',
                                name='Actual Revenue', line=dict(color='#1F4E79', width=3)))
        fig.add_trace(go.Scatter(x=months, y=target, mode='lines',
                                name='Target', line=dict(color='#5B9BD5', dash='dash')))

        fig.update_layout(
            title="Monthly Revenue Performance",
            xaxis_title="Month",
            yaxis_title="Revenue ($M)",
            height=400,
            plot_bgcolor='white'
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Customer satisfaction by region
        regions = ['North America', 'Europe', 'Asia-Pacific', 'Latin America']
        satisfaction = [4.3, 4.1, 4.0, 4.4]
        colors = ['#1F4E79', '#5B9BD5', '#70AD47', '#FFC000']

        fig = go.Figure(data=[go.Bar(x=regions, y=satisfaction, marker_color=colors)])
        fig.add_hline(y=4.5, line_dash="dash", line_color="red",
                     annotation_text="Target: 4.5")

        fig.update_layout(
            title="Customer Satisfaction by Region",
            xaxis_title="Region",
            yaxis_title="Satisfaction Score",
            height=400,
            plot_bgcolor='white',
            yaxis=dict(range=[3.5, 5.0])
        )

        st.plotly_chart(fig, use_container_width=True)

    # Action Items Section
    st.markdown("---")
    st.subheader("🎯 Priority Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **Immediate (Next 30 Days)**
        - Launch customer experience improvement initiative
        - Conduct satisfaction survey deep dive
        - Implement customer feedback system
        """)

    with col2:
        st.markdown("""
        **Short-term (Next 90 Days)**
        - Expand digital marketing in underperforming regions
        - Optimize pricing strategy for Q1
        - Develop customer retention program
        """)

    with col3:
        st.markdown("""
        **Long-term (6+ Months)**
        - Evaluate market expansion opportunities
        - Invest in advanced analytics capabilities
        - Consider strategic partnerships
        """)

if __name__ == "__main__":
    create_executive_dashboard()
```

#### Chart Template Library
```python
class ConsultingChartTemplates:
    """
    Chart templates following McKinsey and Deloitte standards
    """

    @staticmethod
    def mckinsey_colors():
        return {
            'primary': '#1F4E79',
            'secondary': '#5B9BD5',
            'accent': '#70AD47',
            'warning': '#FFC000',
            'danger': '#C55A5A',
            'neutral': '#7F7F7F'
        }

    @staticmethod
    def deloitte_colors():
        return {
            'primary': '#86BC25',
            'secondary': '#62B5E5',
            'accent': '#00A8CC',
            'warning': '#FFB000',
            'neutral': '#C4C4C4'
        }

    def create_waterfall_chart(self, categories, values, title="Waterfall Analysis"):
        """McKinsey-style waterfall chart"""

        colors = self.mckinsey_colors()
        fig = go.Figure()

        # Calculate cumulative values
        cumulative = [values[0]]
        for i in range(1, len(values)):
            cumulative.append(cumulative[-1] + values[i])

        # Add bars
        for i, (cat, val) in enumerate(zip(categories, values)):
            color = colors['primary'] if i == 0 or i == len(values)-1 else colors['accent'] if val > 0 else colors['danger']

            fig.add_trace(go.Bar(
                x=[cat],
                y=[val],
                name=cat,
                marker_color=color,
                showlegend=False
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Categories",
            yaxis_title="Value",
            plot_bgcolor='white',
            height=500
        )

        return fig

    def create_bcg_matrix(self, business_units, growth_rates, market_shares, title="BCG Matrix Analysis"):
        """McKinsey/BCG-style 2x2 matrix"""

        colors = self.mckinsey_colors()
        fig = go.Figure()

        # Create scatter plot
        fig.add_trace(go.Scatter(
            x=growth_rates,
            y=market_shares,
            mode='markers+text',
            text=business_units,
            textposition="middle center",
            marker=dict(
                size=60,
                color=colors['primary'],
                opacity=0.7,
                line=dict(width=2, color='white')
            ),
            showlegend=False
        ))

        # Add quadrant lines
        fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=10, line_dash="dash", line_color="gray", opacity=0.5)

        # Add quadrant labels
        fig.add_annotation(x=20, y=75, text="STARS", showarrow=False,
                          font=dict(size=14, color=colors['accent']))
        fig.add_annotation(x=5, y=75, text="CASH COWS", showarrow=False,
                          font=dict(size=14, color=colors['accent']))
        fig.add_annotation(x=20, y=25, text="QUESTION MARKS", showarrow=False,
                          font=dict(size=14, color=colors['warning']))
        fig.add_annotation(x=5, y=25, text="DOGS", showarrow=False,
                          font=dict(size=14, color=colors['danger']))

        fig.update_layout(
            title=title,
            xaxis_title="Market Growth Rate (%)",
            yaxis_title="Market Share (%)",
            xaxis=dict(range=[0, 25]),
            yaxis=dict(range=[0, 100]),
            plot_bgcolor='white',
            height=600
        )

        return fig

    def create_gantt_timeline(self, tasks, start_dates, durations, title="Implementation Timeline"):
        """Project timeline chart"""

        colors = self.deloitte_colors()
        fig = go.Figure()

        for i, (task, start, duration) in enumerate(zip(tasks, start_dates, durations)):
            fig.add_trace(go.Bar(
                x=[duration],
                y=[task],
                orientation='h',
                name=task,
                marker_color=colors['primary'],
                showlegend=False
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Timeline (weeks)",
            plot_bgcolor='white',
            height=400
        )

        return fig

# Usage example
chart_templates = ConsultingChartTemplates()

# Create waterfall chart
categories = ['Starting Revenue', 'New Customers', 'Upselling', 'Churn', 'Ending Revenue']
values = [100, 15, 10, -8, 117]
waterfall_fig = chart_templates.create_waterfall_chart(categories, values)
```

### ROI Calculation Templates

#### Financial Impact Framework
```python
class ROICalculator:
    """
    ROI calculation templates for business cases
    """

    def __init__(self, discount_rate=0.10):
        self.discount_rate = discount_rate

    def calculate_simple_roi(self, investment, annual_benefit, years):
        """Simple ROI calculation"""
        total_benefit = annual_benefit * years
        roi = (total_benefit - investment) / investment * 100
        payback_period = investment / annual_benefit

        return {
            'roi_percentage': round(roi, 1),
            'payback_years': round(payback_period, 1),
            'total_benefit': total_benefit,
            'net_benefit': total_benefit - investment
        }

    def calculate_npv_irr(self, initial_investment, cash_flows):
        """NPV and IRR calculation"""
        import numpy as np

        # NPV calculation
        npv = -initial_investment
        for i, cf in enumerate(cash_flows):
            npv += cf / (1 + self.discount_rate) ** (i + 1)

        # IRR calculation (simplified)
        def npv_at_rate(rate):
            npv_calc = -initial_investment
            for i, cf in enumerate(cash_flows):
                npv_calc += cf / (1 + rate) ** (i + 1)
            return npv_calc

        # Find IRR using binary search
        low, high = 0, 1
        while npv_at_rate(high) > 0:
            high *= 2

        for _ in range(50):  # Binary search iterations
            mid = (low + high) / 2
            if npv_at_rate(mid) > 0:
                low = mid
            else:
                high = mid

        irr = mid

        return {
            'npv': round(npv, 2),
            'irr': round(irr * 100, 1),
            'discount_rate': self.discount_rate * 100
        }

    def create_business_case_summary(self, project_name, investment, benefits, costs, timeline):
        """Create comprehensive business case summary"""

        net_benefits = [b - c for b, c in zip(benefits, costs)]
        total_investment = investment + sum(costs)
        total_benefits = sum(benefits)

        simple_roi = self.calculate_simple_roi(total_investment, sum(net_benefits)/len(net_benefits), len(benefits))
        npv_analysis = self.calculate_npv_irr(investment, net_benefits)

        summary = f"""
        BUSINESS CASE SUMMARY: {project_name}

        Investment Required: ${total_investment:,.0f}
        Expected Benefits: ${total_benefits:,.0f}
        Net Benefit: ${total_benefits - total_investment:,.0f}

        Financial Returns:
        • ROI: {simple_roi['roi_percentage']}%
        • Payback Period: {simple_roi['payback_years']} years
        • NPV: ${npv_analysis['npv']:,.0f}
        • IRR: {npv_analysis['irr']}%

        Timeline: {timeline} months
        Risk Level: Medium (based on implementation complexity)

        Recommendation: {'PROCEED' if simple_roi['roi_percentage'] > 15 else 'REVIEW'}
        """

        return summary

# Usage example
roi_calc = ROICalculator(discount_rate=0.12)

# Example business case
business_case = roi_calc.create_business_case_summary(
    project_name="Digital Transformation Initiative",
    investment=2000000,
    benefits=[500000, 750000, 1000000, 1200000],
    costs=[200000, 150000, 100000, 100000],
    timeline=24
)

print(business_case)
```

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Examine BUS 672 assisted notes structure and existing content", "status": "completed", "activeForm": "Examining BUS 672 assisted notes structure and existing content"}, {"content": "Research McKinsey and Deloitte storytelling frameworks", "status": "completed", "activeForm": "Researching McKinsey and Deloitte storytelling frameworks"}, {"content": "Create comprehensive storytelling with data document", "status": "completed", "activeForm": "Creating comprehensive storytelling with data document"}, {"content": "Include practical examples and templates", "status": "completed", "activeForm": "Including practical examples and templates"}, {"content": "Add business presentation best practices", "status": "completed", "activeForm": "Adding business presentation best practices"}]