"""
Visualization Playbook — Good / Bad / Important (Context Edition)
================================================================

A comprehensive teaching handout that generates focused visualization examples
across six data categories with three focus views each.

Installation:
pip install -U plotly kaleido reportlab networkx numpy pandas

Author: BUS 672 Course Materials
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

class VisualizationPlaybook:
    def __init__(self):
        # Color semantics
        self.colors = {
            'good': '#2e7d32',      # Green
            'bad': '#b71c1c',       # Red
            'important': '#f9a825', # Gold
            'neutral': '#1e88e5'    # Blue
        }

        # Create output directory
        self.output_dir = 'output_playbook'
        os.makedirs(self.output_dir, exist_ok=True)

        # Data generation
        self.categorical_data = self._generate_categorical_data()
        self.ordinal_data = self._generate_ordinal_data()
        self.quantitative_data = self._generate_quantitative_data()
        self.temporal_data = self._generate_temporal_data()
        self.network_data = self._generate_network_data()
        self.spatial_data = self._generate_spatial_data()

    def _generate_categorical_data(self):
        """Generate categorical data with focus classifications"""
        departments = ['Sales', 'Marketing', 'IT', 'HR', 'Finance', 'Operations', 'Support', 'Legal']
        performance = np.random.normal(75, 15, len(departments))
        performance = np.clip(performance, 40, 100)

        # Classify into focus groups
        sorted_idx = np.argsort(performance)[::-1]
        good_depts = [departments[i] for i in sorted_idx[:3]]
        bad_depts = [departments[i] for i in sorted_idx[-3:]]
        important_depts = [departments[i] for i in sorted_idx[3:5]]

        return {
            'departments': departments,
            'performance': performance,
            'good': good_depts,
            'bad': bad_depts,
            'important': important_depts,
            'target': 80
        }

    def _generate_ordinal_data(self):
        """Generate ordinal data with focus classifications"""
        education_levels = ['High School', 'Associates', 'Bachelor', 'Master', 'PhD']
        salary_2022 = [35000, 42000, 65000, 85000, 120000]
        salary_2023 = [37000, 45000, 68000, 90000, 125000]

        # Likert scale data
        likert_labels = ['Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree']
        likert_counts = [5, 12, 25, 45, 13]

        return {
            'education_levels': education_levels,
            'salary_2022': salary_2022,
            'salary_2023': salary_2023,
            'likert_labels': likert_labels,
            'likert_counts': likert_counts,
            'median_salary': 68000
        }

    def _generate_quantitative_data(self):
        """Generate quantitative data with focus classifications"""
        n = 200
        kpi = np.random.normal(75, 20, n)
        cost = 5000 + 0.3 * kpi + np.random.normal(0, 500, n)

        # Classify by performance regions
        good_mask = (kpi > 80) & (cost < 6000)
        bad_mask = (kpi < 60) | (cost > 7000)
        important_mask = ~(good_mask | bad_mask)

        return {
            'kpi': kpi,
            'cost': cost,
            'good_mask': good_mask,
            'bad_mask': bad_mask,
            'important_mask': important_mask,
            'correlation': np.corrcoef(kpi, cost)[0, 1]
        }

    def _generate_temporal_data(self):
        """Generate temporal data with seasonality and events"""
        dates = pd.date_range('2022-01-01', '2023-12-31', freq='M')
        base_trend = np.linspace(1000, 1200, len(dates))
        seasonal = 200 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
        noise = np.random.normal(0, 50, len(dates))
        sales = base_trend + seasonal + noise

        # Calculate YoY change
        yoy_change = []
        for i in range(len(dates)):
            if i >= 12:
                yoy = ((sales[i] - sales[i-12]) / sales[i-12]) * 100
            else:
                yoy = 0
            yoy_change.append(yoy)

        return {
            'dates': dates,
            'sales': sales,
            'yoy_change': yoy_change,
            'events': {
                '2022-06-01': 'Promo Launch',
                '2023-03-01': 'Pricing Update'
            }
        }

    def _generate_network_data(self):
        """Generate network data with centrality measures"""
        G = nx.barabasi_albert_graph(20, 3)
        pos = nx.spring_layout(G, seed=42)

        # Calculate centrality measures
        betweenness = nx.betweenness_centrality(G)
        degree = dict(G.degree())

        # Classify nodes by degree quantiles
        degrees = list(degree.values())
        q33, q67 = np.percentile(degrees, [33, 67])

        good_nodes = [n for n, d in degree.items() if d >= q67]
        bad_nodes = [n for n, d in degree.items() if d <= q33]
        important_nodes = [n for n, d in degree.items() if q33 < d < q67]

        return {
            'graph': G,
            'pos': pos,
            'betweenness': betweenness,
            'degree': degree,
            'good_nodes': good_nodes,
            'bad_nodes': bad_nodes,
            'important_nodes': important_nodes
        }

    def _generate_spatial_data(self):
        """Generate spatial data with state-level metrics"""
        states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI',
                 'NJ', 'VA', 'WA', 'AZ', 'MA', 'TN', 'IN', 'MO', 'MD', 'WI']

        population = np.random.uniform(2000000, 40000000, len(states))
        sales = np.random.uniform(50000000, 2000000000, len(states))
        rate_per_100k = (sales / population) * 100000

        # Classify by rate quantiles
        q33, q67 = np.percentile(rate_per_100k, [33, 67])

        good_states = [states[i] for i, r in enumerate(rate_per_100k) if r >= q67]
        bad_states = [states[i] for i, r in enumerate(rate_per_100k) if r <= q33]
        important_states = [states[i] for i, r in enumerate(rate_per_100k) if q33 < r < q67]

        return {
            'states': states,
            'population': population,
            'sales': sales,
            'rate_per_100k': rate_per_100k,
            'good_states': good_states,
            'bad_states': bad_states,
            'important_states': important_states
        }

    def create_categorical_figure(self, focus):
        """Create categorical visualization figure for given focus"""
        data = self.categorical_data
        focus_color = self.colors[focus]

        if focus == 'good':
            focus_depts = data['good']
        elif focus == 'bad':
            focus_depts = data['bad']
        else:
            focus_depts = data['important']

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{}, {}],
                   [{"colspan": 2}, None]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # Top-left: Sorted bar chart with target line
        focus_performance = [data['performance'][data['departments'].index(dept)] for dept in focus_depts]
        sorted_idx = np.argsort(focus_performance)[::-1]
        sorted_depts = [focus_depts[i] for i in sorted_idx]
        sorted_perf = [focus_performance[i] for i in sorted_idx]

        fig.add_trace(
            go.Bar(x=sorted_depts, y=sorted_perf, marker_color=focus_color, showlegend=False),
            row=1, col=1
        )
        fig.add_hline(y=data['target'], line_dash="dash", line_color="black", row=1, col=1)
        fig.add_annotation(x=len(sorted_depts)-0.5, y=data['target']+2, text=f"Target={data['target']}",
                          showarrow=False, row=1, col=1)

        # Top-right: Grouped spotlight with context bands
        fig.add_shape(type="rect", x0=-0.5, y0=data['target'], x1=len(sorted_depts)-0.5, y1=100,
                     fillcolor="lightgreen", opacity=0.2, layer="below", row=1, col=2)
        fig.add_shape(type="rect", x0=-0.5, y0=0, x1=len(sorted_depts)-0.5, y1=data['target'],
                     fillcolor="lightcoral", opacity=0.2, layer="below", row=1, col=2)
        fig.add_trace(
            go.Bar(x=sorted_depts, y=sorted_perf, marker_color=focus_color, showlegend=False),
            row=1, col=2
        )
        fig.add_hline(y=data['target'], line_dash="dash", line_color="black", row=1, col=2)

        # Bottom: Pareto chart
        all_perf = data['performance']
        all_depts = data['departments']
        sorted_all_idx = np.argsort(all_perf)[::-1]
        sorted_all_depts = [all_depts[i] for i in sorted_all_idx]
        sorted_all_perf = [all_perf[i] for i in sorted_all_idx]

        cumulative_pct = np.cumsum(sorted_all_perf) / np.sum(sorted_all_perf) * 100

        # Color bars based on focus
        bar_colors = [focus_color if dept in focus_depts else 'lightgray' for dept in sorted_all_depts]

        fig.add_trace(
            go.Bar(x=sorted_all_depts, y=sorted_all_perf, marker_color=bar_colors,
                   name="Performance", showlegend=False),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=sorted_all_depts, y=cumulative_pct, mode='lines+markers',
                      line_color=self.colors['neutral'], name="Cumulative %",
                      yaxis="y4", showlegend=False),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            title=None,
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        fig.update_xaxes(showticklabels=False, title_text=None)
        fig.update_yaxes(showticklabels=False, title_text=None)

        return fig

    def create_ordinal_figure(self, focus):
        """Create ordinal visualization figure for given focus"""
        data = self.ordinal_data
        focus_color = self.colors[focus]

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{}, {}],
                   [{"colspan": 2}, None]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # Top-left: Ordered bar with median line
        salaries = data['salary_2023']
        if focus == 'good':
            focus_salaries = salaries[-2:]  # Top education levels
        elif focus == 'bad':
            focus_salaries = salaries[:2]   # Bottom education levels
        else:
            focus_salaries = salaries[2:4]  # Middle education levels

        focus_levels = data['education_levels'][-2:] if focus == 'good' else \
                      data['education_levels'][:2] if focus == 'bad' else \
                      data['education_levels'][2:4]

        fig.add_trace(
            go.Bar(x=focus_levels, y=focus_salaries, marker_color=focus_color, showlegend=False),
            row=1, col=1
        )
        fig.add_hline(y=data['median_salary'], line_dash="dash", line_color="black", row=1, col=1)
        fig.add_annotation(x=len(focus_levels)-0.5, y=data['median_salary']+5000,
                          text=f"Median", showarrow=False, row=1, col=1)

        # Top-right: Diverging Likert
        likert_data = data['likert_counts']
        negative = sum(likert_data[:2])
        neutral = likert_data[2]
        positive = sum(likert_data[3:])

        if focus == 'good':
            focus_value = positive
        elif focus == 'bad':
            focus_value = negative
        else:
            focus_value = neutral

        fig.add_trace(
            go.Bar(x=[-negative], y=["Satisfaction"], orientation='h',
                   marker_color='lightcoral', showlegend=False),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=[neutral], y=["Satisfaction"], orientation='h',
                   marker_color='lightgray', showlegend=False),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=[positive], y=["Satisfaction"], orientation='h',
                   marker_color=focus_color if focus == 'good' else 'lightgreen', showlegend=False),
            row=1, col=2
        )

        # Bottom: Slope chart
        education_levels = data['education_levels']
        salary_2022 = data['salary_2022']
        salary_2023 = data['salary_2023']

        for i, level in enumerate(education_levels):
            if focus == 'good' and salary_2023[i] > salary_2022[i]:
                color = focus_color
                arrow = "▲"
            elif focus == 'bad' and salary_2023[i] < salary_2022[i]:
                color = focus_color
                arrow = "▼"
            elif focus == 'important' and salary_2023[i] == salary_2022[i]:
                color = focus_color
                arrow = "•"
            else:
                continue

            fig.add_trace(
                go.Scatter(x=[1, 2], y=[salary_2022[i], salary_2023[i]],
                          mode='lines+markers', line_color=color, marker_color=color,
                          showlegend=False, customdata=[salary_2022[i], salary_2023[i]],
                          hovertemplate=f'{level}<br>T1: %{{customdata[0]:$,.0f}} → T2: %{{customdata[1]:$,.0f}}<extra></extra>'),
                row=2, col=1
            )
            fig.add_annotation(x=1.5, y=(salary_2022[i] + salary_2023[i])/2,
                              text=f"{level} {arrow}", showarrow=False, row=2, col=1)

        # Update layout
        fig.update_layout(
            title=None,
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        fig.update_xaxes(showticklabels=False, title_text=None)
        fig.update_yaxes(showticklabels=False, title_text=None)

        return fig

    def create_quantitative_figure(self, focus):
        """Create quantitative visualization figure for given focus"""
        data = self.quantitative_data
        focus_color = self.colors[focus]

        if focus == 'good':
            mask = data['good_mask']
        elif focus == 'bad':
            mask = data['bad_mask']
        else:
            mask = data['important_mask']

        focus_kpi = data['kpi'][mask]
        focus_cost = data['cost'][mask]

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{}, {}],
                   [{"colspan": 2}, None]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # Top-left: Scatter with good region and trend line
        fig.add_shape(type="rect", x0=80, y0=0, x1=120, y1=6000,
                     fillcolor="lightgreen", opacity=0.2, layer="below", row=1, col=1)
        fig.add_trace(
            go.Scatter(x=focus_kpi, y=focus_cost, mode='markers',
                      marker_color=focus_color, showlegend=False,
                      hovertemplate='KPI: %{x:.1f}<br>Cost: $%{y:,.0f}<extra></extra>'),
            row=1, col=1
        )

        # Add trend line
        z = np.polyfit(focus_kpi, focus_cost, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(focus_kpi.min(), focus_kpi.max(), 100)
        fig.add_trace(
            go.Scatter(x=x_trend, y=p(x_trend), mode='lines',
                      line_color=self.colors['neutral'], name=f'r={data["correlation"]:.2f}',
                      showlegend=False),
            row=1, col=1
        )

        # Top-right: Histogram with 90th percentile
        bin_count = max(12, int(np.sqrt(len(focus_kpi))))
        fig.add_trace(
            go.Histogram(x=focus_kpi, nbinsx=bin_count, marker_color=focus_color, showlegend=False),
            row=1, col=2
        )
        pct_90 = np.percentile(focus_kpi, 90)
        fig.add_vline(x=pct_90, line_dash="dash", line_color="black", row=1, col=2)
        fig.add_annotation(x=pct_90+2, y=10, text="90th pct", showarrow=False, row=1, col=2)

        # Bottom: Boxplot with outlier count
        q1, q3 = np.percentile(focus_cost, [25, 75])
        iqr = q3 - q1
        outliers = focus_cost[(focus_cost < q1 - 1.5*iqr) | (focus_cost > q3 + 1.5*iqr)]

        fig.add_trace(
            go.Box(y=focus_cost, marker_color=focus_color,
                   name=f'Outliers: {len(outliers)}', showlegend=False),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            title=None,
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        fig.update_xaxes(showticklabels=False, title_text=None)
        fig.update_yaxes(showticklabels=False, title_text=None)

        return fig

    def create_temporal_figure(self, focus):
        """Create temporal visualization figure for given focus"""
        data = self.temporal_data
        focus_color = self.colors[focus]

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{}, {}],
                   [{"colspan": 2}, None]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # Filter data based on focus
        if focus == 'good':
            # Show periods with positive growth
            mask = np.array(data['yoy_change']) > 5
        elif focus == 'bad':
            # Show periods with negative growth
            mask = np.array(data['yoy_change']) < -5
        else:
            # Show stable periods
            mask = (np.array(data['yoy_change']) >= -5) & (np.array(data['yoy_change']) <= 5)

        focus_dates = [d for i, d in enumerate(data['dates']) if mask[i]]
        focus_sales = [s for i, s in enumerate(data['sales']) if mask[i]]

        # Top-left: Line with moving average
        fig.add_trace(
            go.Scatter(x=focus_dates, y=focus_sales, mode='lines',
                      line_color=focus_color, name='Sales', showlegend=False),
            row=1, col=1
        )

        # 3-month moving average
        if len(focus_sales) >= 3:
            ma_3 = pd.Series(focus_sales).rolling(window=3, center=True).mean()
            fig.add_trace(
                go.Scatter(x=focus_dates, y=ma_3, mode='lines',
                          line=dict(color=self.colors['neutral'], dash='dot'),
                          name='3M MA', showlegend=False),
                row=1, col=1
            )

        # Top-right: Area chart for seasonality
        fig.add_trace(
            go.Scatter(x=focus_dates, y=focus_sales, fill='tonexty',
                      mode='lines', line_color=focus_color, fillcolor=focus_color,
                      opacity=0.6, showlegend=False),
            row=1, col=2
        )

        # Bottom: YoY % change with events
        focus_yoy = [y for i, y in enumerate(data['yoy_change']) if mask[i]]
        fig.add_trace(
            go.Scatter(x=focus_dates, y=focus_yoy, mode='lines+markers',
                      line_color=focus_color, marker_color=focus_color, showlegend=False),
            row=2, col=1
        )

        # Add event markers
        for event_date, event_name in data['events'].items():
            event_dt = pd.to_datetime(event_date)
            if event_dt in focus_dates:
                fig.add_vline(x=event_dt, line_dash="dash", line_color="gray", row=2, col=1)
                fig.add_annotation(x=event_dt, y=max(focus_yoy)*0.8, text=event_name,
                                  showarrow=False, textangle=90, row=2, col=1)

        # Update layout
        fig.update_layout(
            title=None,
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        fig.update_xaxes(showticklabels=False, title_text=None)
        fig.update_yaxes(showticklabels=False, title_text=None)

        return fig

    def create_relational_figure(self, focus):
        """Create relational visualization figure for given focus"""
        data = self.network_data
        focus_color = self.colors[focus]

        if focus == 'good':
            focus_nodes = data['good_nodes']
        elif focus == 'bad':
            focus_nodes = data['bad_nodes']
        else:
            focus_nodes = data['important_nodes']

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{}, {"type": "domain"}],
                   [{"colspan": 2}, None]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # Top-left: Network with focus nodes
        G = data['graph']
        pos = data['pos']
        betweenness = data['betweenness']

        # Extract subgraph for focus nodes
        subgraph = G.subgraph(focus_nodes)

        # Create edges
        edge_x, edge_y = [], []
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        fig.add_trace(
            go.Scatter(x=edge_x, y=edge_y, mode='lines',
                      line=dict(width=1, color='lightgray'),
                      showlegend=False, hoverinfo='none'),
            row=1, col=1
        )

        # Add nodes
        node_x = [pos[node][0] for node in focus_nodes]
        node_y = [pos[node][1] for node in focus_nodes]
        node_size = [betweenness[node] * 1000 + 10 for node in focus_nodes]
        node_text = [f'N{node}' for node in focus_nodes]

        fig.add_trace(
            go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text,
                      textposition='middle center', textfont_size=8,
                      marker=dict(size=node_size, color=focus_color),
                      showlegend=False, hoverinfo='text'),
            row=1, col=1
        )

        # Top-right: Sankey diagram
        # Simple flow between focus nodes
        source_nodes = focus_nodes[:len(focus_nodes)//2] if len(focus_nodes) > 1 else focus_nodes[:1]
        target_nodes = focus_nodes[len(focus_nodes)//2:] if len(focus_nodes) > 1 else focus_nodes[:1]

        if len(source_nodes) > 0 and len(target_nodes) > 0:
            sankey_source = []
            sankey_target = []
            sankey_value = []

            for i, source in enumerate(source_nodes):
                for j, target in enumerate(target_nodes):
                    if source != target:
                        sankey_source.append(i)
                        sankey_target.append(len(source_nodes) + j)
                        sankey_value.append(10)

            if sankey_source:  # Only add if there are connections
                fig.add_trace(
                    go.Sankey(
                        node=dict(
                            pad=15,
                            thickness=20,
                            line=dict(color="black", width=0.5),
                            label=[f'N{n}' for n in source_nodes + target_nodes],
                            color=focus_color
                        ),
                        link=dict(
                            source=sankey_source,
                            target=sankey_target,
                            value=sankey_value,
                            color=focus_color
                        )
                    ),
                    row=1, col=2
                )

        # Bottom: Hub analysis with stars for top hubs
        all_betweenness = [(node, betweenness[node]) for node in focus_nodes]
        all_betweenness.sort(key=lambda x: x[1], reverse=True)
        top_hubs = [node for node, _ in all_betweenness[:3]]

        fig.add_trace(
            go.Scatter(x=edge_x, y=edge_y, mode='lines',
                      line=dict(width=1, color='lightgray'),
                      showlegend=False, hoverinfo='none'),
            row=2, col=1
        )

        # Regular nodes
        regular_nodes = [n for n in focus_nodes if n not in top_hubs]
        if regular_nodes:
            reg_x = [pos[node][0] for node in regular_nodes]
            reg_y = [pos[node][1] for node in regular_nodes]
            reg_size = [betweenness[node] * 1000 + 10 for node in regular_nodes]

            fig.add_trace(
                go.Scatter(x=reg_x, y=reg_y, mode='markers',
                          marker=dict(size=reg_size, color=focus_color),
                          showlegend=False, hoverinfo='text'),
                row=2, col=1
            )

        # Top hubs with stars
        if top_hubs:
            hub_x = [pos[node][0] for node in top_hubs]
            hub_y = [pos[node][1] for node in top_hubs]
            hub_text = ['★' for _ in top_hubs]

            fig.add_trace(
                go.Scatter(x=hub_x, y=hub_y, mode='text', text=hub_text,
                          textfont=dict(size=20, color='gold'),
                          showlegend=False, hoverinfo='text'),
                row=2, col=1
            )

        # Update layout
        fig.update_layout(
            title=None,
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        fig.update_xaxes(showticklabels=False, title_text=None)
        fig.update_yaxes(showticklabels=False, title_text=None)

        return fig

    def create_spatial_figure(self, focus):
        """Create spatial visualization figure for given focus"""
        data = self.spatial_data
        focus_color = self.colors[focus]

        if focus == 'good':
            focus_states = data['good_states']
        elif focus == 'bad':
            focus_states = data['bad_states']
        else:
            focus_states = data['important_states']

        # Get focus data
        focus_indices = [data['states'].index(state) for state in focus_states]
        focus_rates = [data['rate_per_100k'][i] for i in focus_indices]
        focus_sales = [data['sales'][i] for i in focus_indices]

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "geo"}, {"type": "geo"}],
                   [{"type": "xy", "colspan": 2}, None]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # Top-left: Choropleth for focus states
        fig.add_trace(
            go.Choropleth(
                locations=focus_states,
                z=focus_rates,
                locationmode='USA-states',
                colorscale=[[0, focus_color], [1, focus_color]],
                showscale=False,
                hovertemplate='%{locations}<br>Rate: %{z:,.0f}/100k<extra></extra>'
            ),
            row=1, col=1
        )

        # Top-right: Proportional symbol map
        # Use approximate coordinates for states (simplified)
        state_coords = {
            'CA': (36.7783, -119.4179), 'TX': (31.9686, -99.9018), 'FL': (27.7663, -82.6404),
            'NY': (40.7128, -74.0060), 'PA': (41.2033, -77.1945), 'IL': (40.6331, -89.3985),
            'OH': (40.4173, -82.9071), 'GA': (32.1656, -82.9001), 'NC': (35.7596, -79.0193),
            'MI': (44.3148, -85.6024), 'NJ': (40.0583, -74.4057), 'VA': (37.4316, -78.6569),
            'WA': (47.7511, -120.7401), 'AZ': (34.0489, -111.0937), 'MA': (42.4072, -71.3824),
            'TN': (35.5175, -86.5804), 'IN': (40.2732, -86.1349), 'MO': (37.9643, -91.8318),
            'MD': (39.0458, -76.6413), 'WI': (43.7844, -88.7879)
        }

        focus_lats = [state_coords.get(state, (0, 0))[0] for state in focus_states]
        focus_lons = [state_coords.get(state, (0, 0))[1] for state in focus_states]

        fig.add_trace(
            go.Scattergeo(
                lat=focus_lats,
                lon=focus_lons,
                mode='markers',
                marker=dict(
                    size=[s/1e8 * 50 for s in focus_sales],  # Scale for visibility
                    color=focus_color,
                    opacity=0.7
                ),
                text=focus_states,
                hovertemplate='%{text}<br>Sales: $%{customdata:,.0f}M<extra></extra>',
                customdata=[s/1e6 for s in focus_sales],
                showlegend=False
            ),
            row=1, col=2
        )

        # Bottom: Heatmap with contour
        # Create 2D synthetic field
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 10, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X/2) * np.cos(Y/2) + np.random.normal(0, 0.1, X.shape)

        # Set threshold based on focus
        if focus == 'good':
            threshold = np.percentile(Z, 75)
        elif focus == 'bad':
            threshold = np.percentile(Z, 25)
        else:
            threshold = np.percentile(Z, 50)

        fig.add_trace(
            go.Heatmap(
                z=Z,
                colorscale=[[0, 'white'], [1, focus_color]],
                showscale=False,
                hovertemplate='X: %{x}<br>Y: %{y}<br>Value: %{z:.2f}<extra></extra>'
            ),
            row=2, col=1
        )

        # Add contour line at threshold
        fig.add_trace(
            go.Contour(
                z=Z,
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
        fig.update_geos(scope="usa", showlakes=False, showrivers=False)

        # Update layout
        fig.update_layout(
            title=None,
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        fig.update_xaxes(showticklabels=False, title_text=None)
        fig.update_yaxes(showticklabels=False, title_text=None)

        return fig

    def save_figure_as_png(self, fig, filename):
        """Save figure as PNG using kaleido"""
        filepath = os.path.join(self.output_dir, filename)
        fig.write_image(filepath, width=1700, height=980, scale=1)
        return filepath

    def create_pdf_report(self):
        """Create comprehensive PDF report with all visualizations"""
        pdf_path = os.path.join(self.output_dir, "Visualization_Playbook_Focused_Views_NoAxes.pdf")
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center alignment
        )

        page_title_style = ParagraphStyle(
            'PageTitle',
            parent=styles['Heading2'],
            fontSize=18,
            spaceAfter=20,
            alignment=1
        )

        notes_style = ParagraphStyle(
            'Notes',
            parent=styles['Normal'],
            fontSize=10,
            leftIndent=20,
            rightIndent=20
        )

        # Cover page
        story.append(Paragraph("Visualization Playbook", title_style))
        story.append(Paragraph("Good / Bad / Important Context Edition", styles['Heading2']))
        story.append(Spacer(1, 40))

        # Legend
        story.append(Paragraph("Color Legend:", styles['Heading3']))
        story.append(Paragraph("• <font color='#2e7d32'>Green: Good practices and positive patterns</font>", styles['Normal']))
        story.append(Paragraph("• <font color='#b71c1c'>Red: Bad practices and negative patterns</font>", styles['Normal']))
        story.append(Paragraph("• <font color='#f9a825'>Gold: Important/neutral patterns requiring attention</font>", styles['Normal']))
        story.append(Paragraph("• <font color='#1e88e5'>Blue: Neutral accents and reference lines</font>", styles['Normal']))
        story.append(Spacer(1, 40))

        story.append(Paragraph("Each visualization category is presented in three focused views, showing only patterns matching the current focus. Context is provided through colors, annotations, and notes rather than axis labels.", styles['Normal']))

        # Categories and their descriptions
        categories = [
            ('Categorical', 'Bar / Grouped / Pareto'),
            ('Ordinal', 'Ordered Bars / Diverging Likert / Slope'),
            ('Quantitative', 'Scatter / Histogram / Boxplot'),
            ('Temporal', 'Line+MA / Seasonal Area / YoY% + Events'),
            ('Relational', 'Network / Sankey / Hubs'),
            ('Spatial', 'Choropleth / Symbols / Heatmap')
        ]

        base_notes = {
            'Categorical': "Best: bar, grouped, Pareto; Use: compare categories, 80/20; Pitfalls: too many cats, truncated axes, color overload.",
            'Ordinal': "Best: ordered bars, diverging Likert, slope; Use: survey scales, rank order, two-point change; Pitfalls: wrong order, clutter, labels.",
            'Quantitative': "Best: scatter + fit, histogram, boxplot; Use: correlation, distribution, outliers; Pitfalls: bad bins, overplotting, unlabeled axes.",
            'Temporal': "Best: line + MA, YoY%, events, seasonal area; Use: trends, seasonality; Pitfalls: bars for continuous, uneven intervals.",
            'Relational': "Best: node-link + centrality, Sankey (%); Use: networks, flows, bottlenecks; Pitfalls: hairball—filter/highlight.",
            'Spatial': "Best: rates per 100k choropleth, proportional symbols, heatmap; Use: geo comparisons, density; Pitfalls: projections, rainbow scales, no normalization."
        }

        # Create figures and add to PDF
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
                png_filename = f"{category}_{focus}_visualization.png"
                png_path = self.save_figure_as_png(fig, png_filename)
                png_paths.append(png_path)

                # Add page to PDF
                story.append(Paragraph(f"{category} Data - {focus_name} Focus", page_title_style))
                story.append(Spacer(1, 0.5*inch))  # 0.5 inch space before image

                # Add image
                img = Image(png_path, width=7*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 20))

                # Add notes
                story.append(Paragraph("Notes:", styles['Heading4']))
                focus_preface = f"This page spotlights {focus_name.lower()} patterns only. "
                full_notes = focus_preface + base_notes[category]
                story.append(Paragraph(full_notes, notes_style))
                story.append(Spacer(1, 40))

        # Build PDF
        doc.build(story)

        print(f"PDF created: {pdf_path}")
        print(f"PNG files created: {len(png_paths)} images in {self.output_dir}/")
        for png_path in png_paths:
            print(f"  - {os.path.basename(png_path)}")

        return pdf_path, png_paths

def main():
    """Main execution function"""
    print("Creating Visualization Playbook - Good/Bad/Important Context Edition...")
    print("Dependencies required: plotly, kaleido, reportlab, networkx, numpy, pandas")
    print("Install with: pip install -U plotly kaleido reportlab networkx numpy pandas")
    print()

    # Create playbook
    playbook = VisualizationPlaybook()

    # Generate PDF and PNGs
    pdf_path, png_paths = playbook.create_pdf_report()

    print()
    print("Visualization Playbook created successfully!")
    print(f"Final PDF: {pdf_path}")
    print(f"Individual PNGs: {len(png_paths)} files in output_playbook/")

if __name__ == "__main__":
    main()