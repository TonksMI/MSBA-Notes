"""
Visualization Playbook — Good / Bad / Important (Improved Context Edition)
=========================================================================

An enhanced teaching handout with better examples, clearer annotations,
and more meaningful visualizations for each data category.

Installation:
pip install -U plotly kaleido reportlab networkx numpy pandas

Author: BUS 672 Course Materials (Improved Version)
Date: 2025
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import networkx as nx
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import warnings
warnings.filterwarnings('ignore')

# Set global random seed for reproducibility
np.random.seed(42)

class ImprovedVisualizationPlaybook:
    def __init__(self):
        # Enhanced color schemes
        self.colors = {
            'good': '#2e7d32',      # Green
            'bad': '#b71c1c',       # Red
            'important': '#f9a825', # Gold
            'neutral': '#1e88e5',   # Blue
            'light_good': '#c8e6c9',
            'light_bad': '#ffcdd2',
            'light_important': '#fff3c4'
        }

        # Create output directory
        self.output_dir = 'output_playbook'
        os.makedirs(self.output_dir, exist_ok=True)

        # Generate comprehensive datasets
        self._generate_all_data()

    def _generate_all_data(self):
        """Generate all datasets with realistic business scenarios"""
        # Categorical: Department performance analysis
        self.dept_names = ['Sales', 'Marketing', 'IT', 'HR', 'Finance', 'Operations', 'Support', 'Legal']
        self.dept_performance = [92, 88, 76, 84, 95, 78, 71, 82]
        self.dept_budget = [150000, 120000, 200000, 80000, 100000, 180000, 90000, 70000]
        self.dept_headcount = [25, 15, 30, 12, 18, 35, 20, 8]

        # Classify departments
        sorted_idx = np.argsort(self.dept_performance)[::-1]
        self.good_depts = [self.dept_names[i] for i in sorted_idx[:3]]
        self.bad_depts = [self.dept_names[i] for i in sorted_idx[-3:]]
        self.important_depts = [self.dept_names[i] for i in sorted_idx[3:5]]

        # Ordinal: Employee satisfaction survey
        self.satisfaction_levels = ['Very Poor', 'Poor', 'Fair', 'Good', 'Excellent']
        self.satisfaction_counts = [8, 25, 45, 85, 37]
        self.education_levels = ['High School', 'Associates', 'Bachelor', 'Master', 'PhD']
        self.salary_2022 = [35000, 42000, 65000, 85000, 120000]
        self.salary_2023 = [38000, 46000, 70000, 92000, 128000]

        # Quantitative: Sales performance analysis
        n = 150
        self.sales_kpi = np.random.normal(75, 20, n)
        self.sales_cost = 3000 + 0.4 * self.sales_kpi + np.random.normal(0, 800, n)
        self.sales_revenue = 5000 + 1.2 * self.sales_kpi + np.random.normal(0, 1200, n)

        # Temporal: Monthly business metrics
        self.dates = pd.date_range('2022-01-01', '2023-12-31', freq='M')
        base_revenue = np.linspace(100000, 120000, len(self.dates))
        seasonal = 15000 * np.sin(2 * np.pi * np.arange(len(self.dates)) / 12)
        noise = np.random.normal(0, 3000, len(self.dates))
        self.monthly_revenue = base_revenue + seasonal + noise
        self.monthly_costs = 0.7 * self.monthly_revenue + np.random.normal(0, 2000, len(self.dates))

        # Calculate metrics
        self.profit_margin = ((self.monthly_revenue - self.monthly_costs) / self.monthly_revenue) * 100
        self.yoy_growth = []
        for i in range(len(self.dates)):
            if i >= 12:
                yoy = ((self.monthly_revenue[i] - self.monthly_revenue[i-12]) / self.monthly_revenue[i-12]) * 100
            else:
                yoy = 0
            self.yoy_growth.append(yoy)

        # Network: Customer-Product relationships
        self.network_graph = nx.barabasi_albert_graph(25, 4)
        self.network_pos = nx.spring_layout(self.network_graph, seed=42)
        self.betweenness = nx.betweenness_centrality(self.network_graph)
        self.degree = dict(self.network_graph.degree())

        # Classify nodes
        degrees = list(self.degree.values())
        q33, q67 = np.percentile(degrees, [33, 67])
        self.good_nodes = [n for n, d in self.degree.items() if d >= q67]
        self.bad_nodes = [n for n, d in self.degree.items() if d <= q33]
        self.important_nodes = [n for n, d in self.degree.items() if q33 < d < q67]

        # Spatial: Regional sales data
        self.states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI',
                      'NJ', 'VA', 'WA', 'AZ', 'MA', 'TN', 'IN', 'MO', 'MD', 'WI']
        self.state_population = np.random.uniform(2000000, 40000000, len(self.states))
        self.state_sales = np.random.uniform(50000000, 2000000000, len(self.states))
        self.sales_per_capita = self.state_sales / self.state_population

        # Classify states by performance
        q33, q67 = np.percentile(self.sales_per_capita, [33, 67])
        self.good_states = [self.states[i] for i, r in enumerate(self.sales_per_capita) if r >= q67]
        self.bad_states = [self.states[i] for i, r in enumerate(self.sales_per_capita) if r <= q33]
        self.important_states = [self.states[i] for i, r in enumerate(self.sales_per_capita) if q33 < r < q67]

    def create_categorical_figure(self, focus):
        """Enhanced categorical visualization with better context"""
        focus_color = self.colors[focus]

        # Create subplot with secondary y-axis for the bottom plot
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{}, {}], [{"secondary_y": True, "colspan": 2}, None]],
            subplot_titles=(
                f'{focus.title()}: Top Performers vs Target',
                f'{focus.title()}: Performance Zones',
                f'{focus.title()}: Budget Analysis (Pareto Chart)'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        # Get focus departments
        if focus == 'good':
            focus_depts = self.good_depts
            focus_desc = "High Performers (>90%)"
        elif focus == 'bad':
            focus_depts = self.bad_depts
            focus_desc = "Needs Improvement (<80%)"
        else:
            focus_depts = self.important_depts
            focus_desc = "Watch List (80-90%)"

        # Top-left: Performance vs Target
        focus_perf = [self.dept_performance[self.dept_names.index(d)] for d in focus_depts]
        sorted_idx = np.argsort(focus_perf)[::-1]
        sorted_depts = [focus_depts[i] for i in sorted_idx]
        sorted_perf = [focus_perf[i] for i in sorted_idx]

        fig.add_trace(
            go.Bar(x=sorted_depts, y=sorted_perf,
                   marker_color=focus_color,
                   text=[f'{p:.0f}%' for p in sorted_perf],
                   textposition='outside',
                   name=focus_desc, showlegend=False),
            row=1, col=1
        )

        # Add target line
        target = 85
        fig.add_hline(y=target, line_dash="dash", line_color="black",
                     annotation_text=f"Target: {target}%", row=1, col=1)

        # Top-right: Performance zones
        if len(sorted_depts) > 0:
            fig.add_shape(type="rect", x0=-0.5, y0=90, x1=len(sorted_depts)-0.5, y1=100,
                         fillcolor=self.colors['light_good'], opacity=0.3, layer="below", row=1, col=2)
            fig.add_shape(type="rect", x0=-0.5, y0=80, x1=len(sorted_depts)-0.5, y1=90,
                         fillcolor=self.colors['light_important'], opacity=0.3, layer="below", row=1, col=2)
            fig.add_shape(type="rect", x0=-0.5, y0=0, x1=len(sorted_depts)-0.5, y1=80,
                         fillcolor=self.colors['light_bad'], opacity=0.3, layer="below", row=1, col=2)

        fig.add_trace(
            go.Bar(x=sorted_depts, y=sorted_perf,
                   marker_color=focus_color,
                   text=[f'{p:.0f}%' for p in sorted_perf],
                   textposition='outside', showlegend=False),
            row=1, col=2
        )

        # Add zone labels
        if len(sorted_depts) > 0:
            fig.add_annotation(x=len(sorted_depts)/2, y=95, text="Excellence Zone",
                              showarrow=False, font_size=10, row=1, col=2)
            fig.add_annotation(x=len(sorted_depts)/2, y=85, text="Target Zone",
                              showarrow=False, font_size=10, row=1, col=2)
            fig.add_annotation(x=len(sorted_depts)/2, y=75, text="Improvement Zone",
                              showarrow=False, font_size=10, row=1, col=2)

        # Bottom: Pareto Analysis
        all_perf = self.dept_performance.copy()
        all_depts = self.dept_names.copy()
        all_budget = self.dept_budget.copy()

        # Sort by budget (for Pareto analysis)
        sorted_all_idx = np.argsort(all_budget)[::-1]
        sorted_all_depts = [all_depts[i] for i in sorted_all_idx]
        sorted_all_budget = [all_budget[i] for i in sorted_all_idx]

        # Calculate cumulative contribution
        total_budget = sum(all_budget)
        cumulative_pct = np.cumsum(sorted_all_budget) / total_budget * 100

        # Color bars based on focus
        bar_colors = []
        for dept in sorted_all_depts:
            if dept in focus_depts:
                bar_colors.append(focus_color)
            else:
                bar_colors.append('#E0E0E0')

        # Add budget bars on primary y-axis
        fig.add_trace(
            go.Bar(x=sorted_all_depts, y=sorted_all_budget,
                   marker_color=bar_colors,
                   name="Budget Allocation",
                   text=[f'${b/1000:.0f}K' for b in sorted_all_budget],
                   textposition='outside',
                   showlegend=False),
            row=2, col=1, secondary_y=False
        )

        # Add cumulative percentage line on secondary y-axis
        fig.add_trace(
            go.Scatter(x=sorted_all_depts, y=cumulative_pct,
                      mode='lines+markers',
                      line=dict(color=self.colors['neutral'], width=3),
                      marker=dict(size=8),
                      name="Cumulative %",
                      text=[f'{pct:.0f}%' for pct in cumulative_pct],
                      textposition='top center',
                      showlegend=False),
            row=2, col=1, secondary_y=True
        )

        # Add 80% line for Pareto principle
        fig.add_hline(y=80, line_dash="dot", line_color="red",
                     annotation_text="80% Rule", row=2, col=1, secondary_y=True)

        # Update layout
        fig.update_layout(
            title=f"Categorical Analysis - {focus.title()} Focus",
            height=700,
            margin=dict(l=50, r=50, t=100, b=50),
            font=dict(size=11)
        )

        # Update axes
        fig.update_xaxes(showgrid=False, title_text="Departments", tickangle=45)
        fig.update_yaxes(title_text="Performance %", row=1, col=1)
        fig.update_yaxes(title_text="Performance %", row=1, col=2)
        fig.update_yaxes(title_text="Budget ($)", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Cumulative %", row=2, col=1, secondary_y=True)

        return fig

    def create_ordinal_figure(self, focus):
        """Enhanced ordinal visualization with clearer examples"""
        focus_color = self.colors[focus]
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{}, {}], [{"colspan": 2}, None]],
            subplot_titles=(
                f'{focus.title()}: Education-Salary Relationship',
                f'{focus.title()}: Satisfaction Distribution',
                f'{focus.title()}: Year-over-Year Salary Changes'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        # Top-left: Education levels and salaries
        if focus == 'good':
            focus_edu = self.education_levels[-2:]  # Top education
            focus_sal = self.salary_2023[-2:]
            desc = "Advanced Degrees"
        elif focus == 'bad':
            focus_edu = self.education_levels[:2]   # Lower education
            focus_sal = self.salary_2023[:2]
            desc = "Basic Education"
        else:
            focus_edu = self.education_levels[2:4]  # Middle education
            focus_sal = self.salary_2023[2:4]
            desc = "Standard Degrees"

        fig.add_trace(
            go.Bar(x=focus_edu, y=focus_sal,
                   marker_color=focus_color,
                   text=[f'${s:,.0f}' for s in focus_sal],
                   textposition='outside',
                   name=desc, showlegend=False),
            row=1, col=1
        )

        # Add median line
        median_salary = np.median(self.salary_2023)
        fig.add_hline(y=median_salary, line_dash="dash", line_color="black",
                     annotation_text=f"Median: ${median_salary:,.0f}", row=1, col=1)

        # Top-right: Satisfaction Likert scale
        neg_count = sum(self.satisfaction_counts[:2])  # Very Poor + Poor
        neutral_count = self.satisfaction_counts[2]     # Fair
        pos_count = sum(self.satisfaction_counts[3:])   # Good + Excellent

        if focus == 'good':
            highlight_val = pos_count
            highlight_color = focus_color
        elif focus == 'bad':
            highlight_val = neg_count
            highlight_color = focus_color
        else:
            highlight_val = neutral_count
            highlight_color = focus_color

        # Diverging bar chart
        fig.add_trace(
            go.Bar(x=[-neg_count], y=["Employee Satisfaction"],
                   orientation='h',
                   marker_color=self.colors['bad'] if focus == 'bad' else '#ffcdd2',
                   name=f"Negative ({neg_count})",
                   text=[f'{neg_count}'], textposition='inside',
                   showlegend=False),
            row=1, col=2
        )

        fig.add_trace(
            go.Bar(x=[neutral_count], y=["Employee Satisfaction"],
                   orientation='h', base=[0],
                   marker_color=highlight_color if focus == 'important' else '#E0E0E0',
                   name=f"Neutral ({neutral_count})",
                   text=[f'{neutral_count}'], textposition='inside',
                   showlegend=False),
            row=1, col=2
        )

        fig.add_trace(
            go.Bar(x=[pos_count], y=["Employee Satisfaction"],
                   orientation='h', base=[neutral_count],
                   marker_color=focus_color if focus == 'good' else '#c8e6c9',
                   name=f"Positive ({pos_count})",
                   text=[f'{pos_count}'], textposition='inside',
                   showlegend=False),
            row=1, col=2
        )

        # Bottom: Slope chart for salary changes
        for i, edu_level in enumerate(self.education_levels):
            sal_2022 = self.salary_2022[i]
            sal_2023 = self.salary_2023[i]
            change_pct = ((sal_2023 - sal_2022) / sal_2022) * 100

            # Determine if this fits the focus
            if focus == 'good' and change_pct > 7:
                color = focus_color
                arrow = "▲"
            elif focus == 'bad' and change_pct < 5:
                color = focus_color
                arrow = "▼"
            elif focus == 'important' and 5 <= change_pct <= 7:
                color = focus_color
                arrow = "•"
            else:
                continue

            fig.add_trace(
                go.Scatter(x=[1, 2], y=[sal_2022, sal_2023],
                          mode='lines+markers',
                          line=dict(color=color, width=3),
                          marker=dict(color=color, size=8),
                          name=edu_level,
                          customdata=[sal_2022, sal_2023, change_pct],
                          hovertemplate=f'{edu_level}<br>2022: $%{{customdata[0]:,.0f}}<br>2023: $%{{customdata[1]:,.0f}}<br>Change: %{{customdata[2]:.1f}}%<extra></extra>',
                          showlegend=False),
                row=2, col=1
            )

            # Add change percentage annotation
            mid_y = (sal_2022 + sal_2023) / 2
            fig.add_annotation(x=1.5, y=mid_y,
                              text=f"{edu_level}<br>{arrow} {change_pct:.1f}%",
                              showarrow=False, font_size=9,
                              bgcolor="white", bordercolor=color,
                              row=2, col=1)

        # Add year labels
        fig.add_annotation(x=1, y=max(self.salary_2023)*1.1, text="2022",
                          showarrow=False, font_size=12, font_weight="bold", row=2, col=1)
        fig.add_annotation(x=2, y=max(self.salary_2023)*1.1, text="2023",
                          showarrow=False, font_size=12, font_weight="bold", row=2, col=1)

        # Update layout
        fig.update_layout(
            title=f"Ordinal Analysis - {focus.title()} Focus",
            height=700,
            margin=dict(l=50, r=50, t=100, b=50),
            font=dict(size=11)
        )

        fig.update_xaxes(showgrid=False, title_text="")
        fig.update_yaxes(title_text="Salary ($)", row=1, col=1)
        fig.update_yaxes(title_text="", row=1, col=2)
        fig.update_yaxes(title_text="Salary ($)", row=2, col=1)

        return fig

    def create_quantitative_figure(self, focus):
        """Enhanced quantitative visualization with business context"""
        focus_color = self.colors[focus]

        # Define performance regions
        good_kpi_threshold = 85
        good_cost_threshold = 4000
        bad_kpi_threshold = 60
        bad_cost_threshold = 5500

        # Create masks for different performance levels
        good_mask = (self.sales_kpi >= good_kpi_threshold) & (self.sales_cost <= good_cost_threshold)
        bad_mask = (self.sales_kpi <= bad_kpi_threshold) | (self.sales_cost >= bad_cost_threshold)
        important_mask = ~(good_mask | bad_mask)

        if focus == 'good':
            mask = good_mask
            desc = "High Performance, Low Cost"
        elif focus == 'bad':
            mask = bad_mask
            desc = "Low Performance or High Cost"
        else:
            mask = important_mask
            desc = "Average Performance Range"

        focus_kpi = self.sales_kpi[mask]
        focus_cost = self.sales_cost[mask]
        focus_revenue = self.sales_revenue[mask]

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{}, {}], [{"colspan": 2}, None]],
            subplot_titles=(
                f'{focus.title()}: Performance vs Cost Analysis',
                f'{focus.title()}: Revenue Distribution',
                f'{focus.title()}: Cost Distribution & Outliers'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        # Top-left: Scatter plot with performance regions
        # Add performance zones
        fig.add_shape(type="rect",
                     x0=good_kpi_threshold, y0=0,
                     x1=120, y1=good_cost_threshold,
                     fillcolor=self.colors['light_good'], opacity=0.3, layer="below",
                     row=1, col=1)

        fig.add_shape(type="rect",
                     x0=0, y0=bad_cost_threshold,
                     x1=bad_kpi_threshold, y1=8000,
                     fillcolor=self.colors['light_bad'], opacity=0.3, layer="below",
                     row=1, col=1)

        fig.add_trace(
            go.Scatter(x=focus_kpi, y=focus_cost,
                      mode='markers',
                      marker=dict(color=focus_color, size=8, opacity=0.7),
                      name=desc,
                      customdata=focus_revenue,
                      hovertemplate='KPI: %{x:.1f}<br>Cost: $%{y:,.0f}<br>Revenue: $%{customdata:,.0f}<extra></extra>',
                      showlegend=False),
            row=1, col=1
        )

        # Add trend line
        if len(focus_kpi) > 1:
            z = np.polyfit(focus_kpi, focus_cost, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(focus_kpi.min(), focus_kpi.max(), 100)
            corr = np.corrcoef(focus_kpi, focus_cost)[0, 1]

            fig.add_trace(
                go.Scatter(x=x_trend, y=p(x_trend),
                          mode='lines',
                          line=dict(color=self.colors['neutral'], width=2, dash='dash'),
                          name=f'Trend (r={corr:.2f})',
                          showlegend=False),
                row=1, col=1
            )

        # Add zone labels
        fig.add_annotation(x=100, y=3000, text="Sweet Spot<br>(High KPI, Low Cost)",
                          showarrow=False, font_size=9, bgcolor="white",
                          bordercolor=self.colors['good'], row=1, col=1)

        # Top-right: Revenue histogram with percentiles
        bin_count = max(12, int(np.sqrt(len(focus_revenue))))
        fig.add_trace(
            go.Histogram(x=focus_revenue, nbinsx=bin_count,
                        marker_color=focus_color, opacity=0.7,
                        name="Revenue Distribution", showlegend=False),
            row=1, col=2
        )

        # Add percentile lines
        p25, p50, p75, p90 = np.percentile(focus_revenue, [25, 50, 75, 90])

        for pct, val, label in [(25, p25, '25th'), (50, p50, 'Median'), (75, p75, '75th'), (90, p90, '90th')]:
            line_style = 'solid' if pct == 50 else 'dash'
            fig.add_vline(x=val, line_dash=line_style, line_color="black",
                         annotation_text=f"{label}: ${val:,.0f}", row=1, col=2)

        # Bottom: Box plot with outlier analysis
        q1, q3 = np.percentile(focus_cost, [25, 75])
        iqr = q3 - q1
        outliers = focus_cost[(focus_cost < q1 - 1.5*iqr) | (focus_cost > q3 + 1.5*iqr)]

        fig.add_trace(
            go.Box(y=focus_cost,
                   marker_color=focus_color,
                   name=f'{desc}<br>Outliers: {len(outliers)} ({len(outliers)/len(focus_cost)*100:.1f}%)',
                   boxpoints='outliers',
                   pointpos=0,
                   showlegend=False),
            row=2, col=1
        )

        # Add statistical annotations
        mean_cost = np.mean(focus_cost)
        std_cost = np.std(focus_cost)
        fig.add_annotation(x=0.2, y=mean_cost,
                          text=f"Mean: ${mean_cost:,.0f}<br>Std: ${std_cost:,.0f}",
                          showarrow=True, arrowhead=2,
                          bgcolor="white", bordercolor=focus_color,
                          row=2, col=1)

        # Update layout
        fig.update_layout(
            title=f"Quantitative Analysis - {focus.title()} Focus",
            height=700,
            margin=dict(l=50, r=50, t=100, b=50),
            font=dict(size=11)
        )

        fig.update_xaxes(title_text="Performance KPI", row=1, col=1)
        fig.update_yaxes(title_text="Cost ($)", row=1, col=1)
        fig.update_xaxes(title_text="Revenue ($)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Cost ($)", row=2, col=1)

        return fig

    def create_temporal_figure(self, focus):
        """Enhanced temporal visualization with business insights"""
        focus_color = self.colors[focus]

        # Define temporal performance criteria
        if focus == 'good':
            # Periods with high growth or high margin
            mask = (np.array(self.yoy_growth) > 8) | (np.array(self.profit_margin) > 25)
            desc = "High Growth/Margin Periods"
        elif focus == 'bad':
            # Periods with negative growth or low margin
            mask = (np.array(self.yoy_growth) < -2) | (np.array(self.profit_margin) < 15)
            desc = "Challenging Performance Periods"
        else:
            # Stable periods
            mask = (np.array(self.yoy_growth) >= -2) & (np.array(self.yoy_growth) <= 8) & \
                   (np.array(self.profit_margin) >= 15) & (np.array(self.profit_margin) <= 25)
            desc = "Stable Performance Periods"

        focus_dates = [d for i, d in enumerate(self.dates) if mask[i]]
        focus_revenue = [r for i, r in enumerate(self.monthly_revenue) if mask[i]]
        focus_margin = [m for i, m in enumerate(self.profit_margin) if mask[i]]
        focus_yoy = [y for i, y in enumerate(self.yoy_growth) if mask[i] and y != 0]

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{}, {}], [{"colspan": 2}, None]],
            subplot_titles=(
                f'{focus.title()}: Revenue Trend with Moving Average',
                f'{focus.title()}: Profit Margin Over Time',
                f'{focus.title()}: Year-over-Year Growth Analysis'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        if len(focus_dates) > 0:
            # Top-left: Revenue with moving average
            fig.add_trace(
                go.Scatter(x=focus_dates, y=focus_revenue,
                          mode='lines+markers',
                          line=dict(color=focus_color, width=2),
                          marker=dict(size=6),
                          name="Monthly Revenue",
                          hovertemplate='%{x}<br>Revenue: $%{y:,.0f}<extra></extra>',
                          showlegend=False),
                row=1, col=1
            )

            # Add 3-month moving average if enough data
            if len(focus_revenue) >= 3:
                ma_3 = pd.Series(focus_revenue).rolling(window=3, center=True).mean()
                fig.add_trace(
                    go.Scatter(x=focus_dates, y=ma_3,
                              mode='lines',
                              line=dict(color=self.colors['neutral'], width=2, dash='dot'),
                              name="3-Month MA",
                              hovertemplate='%{x}<br>3M Average: $%{y:,.0f}<extra></extra>',
                              showlegend=False),
                    row=1, col=1
                )

            # Add trend annotation
            if len(focus_revenue) > 1:
                trend_slope = (focus_revenue[-1] - focus_revenue[0]) / len(focus_revenue)
                trend_direction = "↗ Rising" if trend_slope > 0 else "↘ Declining" if trend_slope < 0 else "→ Stable"
                fig.add_annotation(x=focus_dates[len(focus_dates)//2], y=max(focus_revenue)*0.9,
                                  text=f"Trend: {trend_direction}",
                                  showarrow=False, bgcolor="white",
                                  bordercolor=focus_color, row=1, col=1)

            # Top-right: Profit margin area chart
            fig.add_trace(
                go.Scatter(x=focus_dates, y=focus_margin,
                          fill='tonexty',
                          mode='lines',
                          line=dict(color=focus_color, width=2),
                          fillcolor=focus_color,
                          opacity=0.6,
                          name="Profit Margin %",
                          hovertemplate='%{x}<br>Margin: %{y:.1f}%<extra></extra>',
                          showlegend=False),
                row=1, col=2
            )

            # Add margin target lines
            fig.add_hline(y=20, line_dash="dash", line_color="green",
                         annotation_text="Target: 20%", row=1, col=2)
            fig.add_hline(y=15, line_dash="dash", line_color="red",
                         annotation_text="Minimum: 15%", row=1, col=2)

            # Bottom: YoY growth analysis
            if len(focus_yoy) > 0:
                # Filter out zeros and create corresponding dates
                focus_yoy_dates = [d for i, d in enumerate(focus_dates) if i < len(focus_yoy) and focus_yoy[i] != 0]
                focus_yoy_clean = [y for y in focus_yoy if y != 0]

                if len(focus_yoy_clean) > 0:
                    fig.add_trace(
                        go.Scatter(x=focus_yoy_dates, y=focus_yoy_clean,
                                  mode='lines+markers',
                                  line=dict(color=focus_color, width=2),
                                  marker=dict(size=8),
                                  name="YoY Growth %",
                                  hovertemplate='%{x}<br>YoY Growth: %{y:.1f}%<extra></extra>',
                                  showlegend=False),
                        row=2, col=1
                    )

                    # Add growth target line
                    fig.add_hline(y=0, line_dash="solid", line_color="black",
                                 annotation_text="Break-even", row=2, col=1)
                    fig.add_hline(y=5, line_dash="dash", line_color="green",
                                 annotation_text="Growth Target: 5%", row=2, col=1)

                    # Add key events (example)
                    events = {
                        '2022-06-01': 'Product Launch',
                        '2022-12-01': 'Holiday Season',
                        '2023-03-01': 'Market Expansion'
                    }

                    for event_date, event_name in events.items():
                        event_dt = pd.to_datetime(event_date)
                        if event_dt in focus_yoy_dates:
                            fig.add_vline(x=event_dt, line_dash="dot", line_color="gray", row=2, col=1)
                            fig.add_annotation(x=event_dt, y=max(focus_yoy_clean)*0.8,
                                              text=event_name, showarrow=False,
                                              textangle=90, font_size=9, row=2, col=1)

        # Update layout
        fig.update_layout(
            title=f"Temporal Analysis - {focus.title()} Focus",
            height=700,
            margin=dict(l=50, r=50, t=100, b=50),
            font=dict(size=11)
        )

        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Revenue ($)", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_yaxes(title_text="Profit Margin (%)", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="YoY Growth (%)", row=2, col=1)

        return fig

    def create_relational_figure(self, focus):
        """Enhanced relational visualization with network insights"""
        focus_color = self.colors[focus]

        if focus == 'good':
            focus_nodes = self.good_nodes
            desc = "High-Connectivity Nodes (Hubs)"
        elif focus == 'bad':
            focus_nodes = self.bad_nodes
            desc = "Low-Connectivity Nodes (Isolated)"
        else:
            focus_nodes = self.important_nodes
            desc = "Medium-Connectivity Nodes (Bridges)"

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{}, {"type": "domain"}], [{"colspan": 2}, None]],
            subplot_titles=(
                f'{focus.title()}: Network Structure',
                f'{focus.title()}: Flow Analysis',
                f'{focus.title()}: Centrality Analysis'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        # Top-left: Network visualization
        G = self.network_graph
        pos = self.network_pos

        # Create subgraph edges
        subgraph = G.subgraph(focus_nodes)
        edge_x, edge_y = [], []
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # Add edges
        fig.add_trace(
            go.Scatter(x=edge_x, y=edge_y,
                      mode='lines',
                      line=dict(width=1, color='lightgray'),
                      showlegend=False, hoverinfo='none'),
            row=1, col=1
        )

        # Add nodes
        node_x = [pos[node][0] for node in focus_nodes]
        node_y = [pos[node][1] for node in focus_nodes]
        node_size = [self.betweenness[node] * 2000 + 15 for node in focus_nodes]
        node_text = [f'Node {node}<br>Degree: {self.degree[node]}<br>Centrality: {self.betweenness[node]:.3f}'
                    for node in focus_nodes]

        fig.add_trace(
            go.Scatter(x=node_x, y=node_y,
                      mode='markers',
                      marker=dict(size=node_size, color=focus_color,
                                 line=dict(width=2, color='white')),
                      text=[f'N{node}' for node in focus_nodes],
                      textposition='middle center',
                      textfont=dict(color='white', size=10),
                      hovertext=node_text,
                      hoverinfo='text',
                      name=desc, showlegend=False),
            row=1, col=1
        )

        # Top-right: Sankey diagram (simplified flow)
        if len(focus_nodes) >= 4:
            source_nodes = focus_nodes[:len(focus_nodes)//2]
            target_nodes = focus_nodes[len(focus_nodes)//2:]

            sankey_source = []
            sankey_target = []
            sankey_value = []

            for i, source in enumerate(source_nodes):
                for j, target in enumerate(target_nodes):
                    if source != target and G.has_edge(source, target):
                        sankey_source.append(i)
                        sankey_target.append(len(source_nodes) + j)
                        sankey_value.append(self.degree[source] + self.degree[target])

            if sankey_source:
                all_labels = [f'N{n}' for n in source_nodes + target_nodes]
                fig.add_trace(
                    go.Sankey(
                        node=dict(
                            pad=15,
                            thickness=20,
                            line=dict(color="black", width=0.5),
                            label=all_labels,
                            color=focus_color
                        ),
                        link=dict(
                            source=sankey_source,
                            target=sankey_target,
                            value=sankey_value,
                            color=focus_color,
                            hovertemplate='Flow: %{value}<br>%{source.label} → %{target.label}<extra></extra>'
                        )
                    ),
                    row=1, col=2
                )

        # Bottom: Centrality analysis
        # Show all nodes but highlight focus nodes
        all_node_x = [pos[node][0] for node in G.nodes()]
        all_node_y = [pos[node][1] for node in G.nodes()]
        all_node_colors = [focus_color if node in focus_nodes else '#E0E0E0' for node in G.nodes()]
        all_node_sizes = [self.betweenness[node] * 2000 + 10 for node in G.nodes()]

        # Add all edges (faded)
        all_edge_x, all_edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            all_edge_x.extend([x0, x1, None])
            all_edge_y.extend([y0, y1, None])

        fig.add_trace(
            go.Scatter(x=all_edge_x, y=all_edge_y,
                      mode='lines',
                      line=dict(width=0.5, color='lightgray'),
                      showlegend=False, hoverinfo='none'),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=all_node_x, y=all_node_y,
                      mode='markers',
                      marker=dict(size=all_node_sizes, color=all_node_colors,
                                 line=dict(width=1, color='white')),
                      hovertext=[f'Node {node}<br>Centrality: {self.betweenness[node]:.3f}' for node in G.nodes()],
                      hoverinfo='text',
                      showlegend=False),
            row=2, col=1
        )

        # Add stars for top centrality nodes
        top_central = sorted(focus_nodes, key=lambda x: self.betweenness[x], reverse=True)[:3]
        if top_central:
            star_x = [pos[node][0] for node in top_central]
            star_y = [pos[node][1] for node in top_central]

            fig.add_trace(
                go.Scatter(x=star_x, y=star_y,
                          mode='text',
                          text=['★'] * len(top_central),
                          textfont=dict(size=20, color='gold'),
                          hovertext=[f'Top Hub: Node {node}' for node in top_central],
                          hoverinfo='text',
                          showlegend=False),
                row=2, col=1
            )

        # Update layout
        fig.update_layout(
            title=f"Network Analysis - {focus.title()} Focus",
            height=700,
            margin=dict(l=50, r=50, t=100, b=50),
            font=dict(size=11)
        )

        # Hide axes for network plots
        for row, col in [(1, 1), (2, 1)]:
            fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, row=row, col=col)
            fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=row, col=col)

        return fig

    def create_spatial_figure(self, focus):
        """Enhanced spatial visualization with geographic insights"""
        focus_color = self.colors[focus]

        if focus == 'good':
            focus_states = self.good_states
            desc = "Top Performing States"
        elif focus == 'bad':
            focus_states = self.bad_states
            desc = "Underperforming States"
        else:
            focus_states = self.important_states
            desc = "Average Performing States"

        # Get focus data
        focus_indices = [self.states.index(state) for state in focus_states if state in self.states]
        focus_sales_per_capita = [self.sales_per_capita[i] for i in focus_indices]
        focus_total_sales = [self.state_sales[i] for i in focus_indices]

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "geo"}, {"type": "geo"}],
                   [{"type": "xy", "colspan": 2}, None]],
            subplot_titles=(
                f'{focus.title()}: Sales per Capita by State',
                f'{focus.title()}: Total Sales Volume',
                f'{focus.title()}: Regional Performance Heatmap'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        # Top-left: Choropleth map (sales per capita)
        fig.add_trace(
            go.Choropleth(
                locations=focus_states,
                z=focus_sales_per_capita,
                locationmode='USA-states',
                colorscale=[[0, 'white'], [1, focus_color]],
                showscale=True,
                colorbar=dict(title="Sales per Capita", x=0.45),
                hovertemplate='%{locations}<br>Per Capita: $%{z:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Top-right: Proportional symbol map (total sales)
        # Approximate state coordinates (simplified)
        state_coords = {
            'CA': (36.7783, -119.4179), 'TX': (31.9686, -99.9018), 'FL': (27.7663, -82.6404),
            'NY': (42.3601, -74.0060), 'PA': (41.2033, -77.1945), 'IL': (40.6331, -89.3985),
            'OH': (40.4173, -82.9071), 'GA': (32.1656, -82.9001), 'NC': (35.7596, -79.0193),
            'MI': (44.3148, -85.6024), 'NJ': (40.0583, -74.4057), 'VA': (37.4316, -78.6569),
            'WA': (47.7511, -120.7401), 'AZ': (34.0489, -111.0937), 'MA': (42.4072, -71.3824),
            'TN': (35.5175, -86.5804), 'IN': (40.2732, -86.1349), 'MO': (37.9643, -91.8318),
            'MD': (39.0458, -76.6413), 'WI': (43.7844, -88.7879)
        }

        focus_lats = [state_coords.get(state, (0, 0))[0] for state in focus_states if state in state_coords]
        focus_lons = [state_coords.get(state, (0, 0))[1] for state in focus_states if state in state_coords]

        # Adjust sales data to match available coordinates
        coord_sales = [focus_total_sales[i] for i, state in enumerate(focus_states) if state in state_coords]

        if focus_lats and focus_lons and coord_sales:
            fig.add_trace(
                go.Scattergeo(
                    lat=focus_lats,
                    lon=focus_lons,
                    mode='markers',
                    marker=dict(
                        size=[s/1e8 * 30 + 10 for s in coord_sales],  # Scale for visibility
                        color=focus_color,
                        opacity=0.7,
                        line=dict(width=2, color='white')
                    ),
                    text=[state for state in focus_states if state in state_coords],
                    hovertemplate='%{text}<br>Total Sales: $%{customdata:,.0f}M<extra></extra>',
                    customdata=[s/1e6 for s in coord_sales],
                    showlegend=False
                ),
                row=1, col=2
            )

        # Bottom: Regional performance heatmap
        # Create synthetic regional data (simplified 2D representation)
        x_regions = ['West', 'Southwest', 'Midwest', 'Southeast', 'Northeast']
        y_metrics = ['Market Share', 'Growth Rate', 'Customer Satisfaction', 'Profitability']

        # Generate performance matrix based on focus
        np.random.seed(42)
        if focus == 'good':
            base_performance = np.random.uniform(0.7, 1.0, (len(y_metrics), len(x_regions)))
        elif focus == 'bad':
            base_performance = np.random.uniform(0.2, 0.6, (len(y_metrics), len(x_regions)))
        else:
            base_performance = np.random.uniform(0.4, 0.8, (len(y_metrics), len(x_regions)))

        fig.add_trace(
            go.Heatmap(
                z=base_performance,
                x=x_regions,
                y=y_metrics,
                colorscale=[[0, 'white'], [1, focus_color]],
                showscale=True,
                colorbar=dict(title="Performance Score", x=1.02),
                text=[[f'{val:.2f}' for val in row] for row in base_performance],
                texttemplate="%{text}",
                textfont={"size": 12},
                hovertemplate='%{y}<br>%{x}<br>Score: %{z:.2f}<extra></extra>'
            ),
            row=2, col=1
        )

        # Add performance threshold contour
        threshold = 0.7 if focus == 'good' else 0.4 if focus == 'bad' else 0.6
        fig.add_trace(
            go.Contour(
                z=base_performance,
                x=x_regions,
                y=y_metrics,
                contours=dict(
                    start=threshold,
                    end=threshold,
                    size=0.01,
                    coloring='lines'
                ),
                line=dict(color='black', width=2),
                showscale=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )

        # Update geo layout
        fig.update_geos(scope="usa", showlakes=False, showrivers=False,
                       projection_type="albers usa")

        # Update layout
        fig.update_layout(
            title=f"Spatial Analysis - {focus.title()} Focus",
            height=700,
            margin=dict(l=50, r=50, t=100, b=50),
            font=dict(size=11)
        )

        fig.update_xaxes(title_text="Region", row=2, col=1)
        fig.update_yaxes(title_text="Metrics", row=2, col=1)

        return fig

    def save_figure_as_png(self, fig, filename):
        """Save figure as PNG with proper sizing"""
        filepath = os.path.join(self.output_dir, filename)
        fig.write_image(filepath, width=1400, height=900, scale=1.5)
        return filepath

    def create_pdf_report(self):
        """Create enhanced PDF report with improved visualizations"""
        pdf_path = os.path.join(self.output_dir, "Visualization_Playbook_Enhanced_Edition.pdf")
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=28,
            spaceAfter=30,
            alignment=1,
            textColor=colors.darkblue
        )

        page_title_style = ParagraphStyle(
            'PageTitle',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=20,
            alignment=1,
            textColor=colors.black
        )

        notes_style = ParagraphStyle(
            'Notes',
            parent=styles['Normal'],
            fontSize=10,
            leftIndent=20,
            rightIndent=20,
            leading=12
        )

        # Enhanced cover page
        story.append(Paragraph("Visualization Playbook", title_style))
        story.append(Paragraph("Enhanced Context Edition with Business Examples", styles['Heading2']))
        story.append(Spacer(1, 40))

        # Enhanced legend
        story.append(Paragraph("Focus Areas & Color Coding:", styles['Heading3']))
        story.append(Paragraph("• <font color='#2e7d32'><b>Good Focus:</b></font> High-performing patterns, best practices, success stories", styles['Normal']))
        story.append(Paragraph("• <font color='#b71c1c'><b>Bad Focus:</b></font> Poor-performing patterns, common pitfalls, areas needing improvement", styles['Normal']))
        story.append(Paragraph("• <font color='#f9a825'><b>Important Focus:</b></font> Moderate patterns, watch areas, balanced performance", styles['Normal']))
        story.append(Paragraph("• <font color='#1e88e5'><b>Reference Lines:</b></font> Benchmarks, targets, trends, and statistical measures", styles['Normal']))
        story.append(Spacer(1, 30))

        # Enhanced description
        story.append(Paragraph("Key Features:", styles['Heading3']))
        story.append(Paragraph("• Each visualization shows realistic business scenarios with meaningful data", styles['Normal']))
        story.append(Paragraph("• Clear annotations, targets, and performance zones for context", styles['Normal']))
        story.append(Paragraph("• Statistical measures (correlations, percentiles, outliers) clearly marked", styles['Normal']))
        story.append(Paragraph("• Business insights and actionable recommendations embedded in charts", styles['Normal']))

        # Enhanced categories and notes
        categories = [
            ('Categorical', 'Department performance analysis with targets and zones'),
            ('Ordinal', 'Employee satisfaction and education-salary relationships'),
            ('Quantitative', 'Sales performance with cost analysis and outlier detection'),
            ('Temporal', 'Revenue trends with growth analysis and business events'),
            ('Relational', 'Network connectivity analysis with centrality measures'),
            ('Spatial', 'Regional performance with geographic and heatmap analysis')
        ]

        enhanced_notes = {
            'Categorical': "Features department performance vs targets, budget allocation analysis (Pareto), and performance zones. Shows clear examples of high performers vs underperformers with actionable insights.",
            'Ordinal': "Demonstrates education-salary relationships, satisfaction surveys with Likert scales, and year-over-year progression analysis. Includes median benchmarks and growth indicators.",
            'Quantitative': "Shows performance vs cost analysis with optimal zones, revenue distribution with percentiles, and cost outlier analysis. Includes correlation analysis and statistical boundaries.",
            'Temporal': "Revenue trends with moving averages, profit margin analysis with targets, and year-over-year growth with business events. Features seasonality and trend identification.",
            'Relational': "Network connectivity analysis with centrality measures, flow diagrams, and hub identification. Shows relationship patterns and influence mapping in business networks.",
            'Spatial': "Geographic performance analysis with choropleth maps, proportional symbols for volume, and regional performance heatmaps. Includes per-capita normalization and threshold analysis."
        }

        # Create enhanced figures and add to PDF
        focuses = ['good', 'bad', 'important']
        focus_names = ['Good', 'Bad', 'Important']
        png_paths = []

        for category, description in categories:
            for focus, focus_name in zip(focuses, focus_names):
                # Create appropriate figure
                if category == 'Categorical':
                    fig = self.create_categorical_figure(focus)
                elif category == 'Ordinal':
                    fig = self.create_ordinal_figure(focus)
                elif category == 'Quantitative':
                    fig = self.create_quantitative_figure(focus)
                elif category == 'Temporal':
                    fig = self.create_temporal_figure(focus)
                elif category == 'Relational':
                    fig = self.create_relational_figure(focus)
                elif category == 'Spatial':
                    fig = self.create_spatial_figure(focus)

                # Save as PNG
                png_filename = f"{category}_{focus}_enhanced.png"
                png_path = self.save_figure_as_png(fig, png_filename)
                png_paths.append(png_path)

                # Add page to PDF
                story.append(Paragraph(f"{category} Data Analysis - {focus_name} Focus", page_title_style))
                story.append(Paragraph(f"<i>{description}</i>", styles['Italic']))
                story.append(Spacer(1, 0.3*inch))

                # Add image
                img = Image(png_path, width=7*inch, height=5*inch)
                story.append(img)
                story.append(Spacer(1, 15))

                # Add enhanced notes
                story.append(Paragraph("Analysis Notes:", styles['Heading4']))
                focus_description = f"This page focuses on {focus_name.lower()} patterns in {category.lower()} data. "
                full_notes = focus_description + enhanced_notes[category]
                story.append(Paragraph(full_notes, notes_style))
                story.append(Spacer(1, 30))

        # Build PDF
        doc.build(story)

        print(f"Enhanced PDF created: {pdf_path}")
        print(f"PNG files created: {len(png_paths)} images in {self.output_dir}/")
        for png_path in png_paths:
            print(f"  - {os.path.basename(png_path)}")

        return pdf_path, png_paths

def main():
    """Main execution function for enhanced playbook"""
    print("Creating Enhanced Visualization Playbook...")
    print("Features: Better examples, clearer annotations, business context")
    print("Dependencies: plotly, kaleido, reportlab, networkx, numpy, pandas")
    print()

    # Create enhanced playbook
    playbook = ImprovedVisualizationPlaybook()

    # Generate enhanced PDF and PNGs
    pdf_path, png_paths = playbook.create_pdf_report()

    print()
    print("Enhanced Visualization Playbook created successfully!")
    print(f"Enhanced PDF: {pdf_path}")
    print(f"Enhanced PNGs: {len(png_paths)} files in output_playbook/")
    print()
    print("Key improvements:")
    print("- Realistic business scenarios with meaningful data")
    print("- Clear annotations, targets, and performance zones")
    print("- Statistical measures and business insights")
    print("- Enhanced readability and professional appearance")

if __name__ == "__main__":
    main()