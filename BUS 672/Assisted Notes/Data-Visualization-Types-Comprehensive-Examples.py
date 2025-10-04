"""
BUS 672 - Data Visualization Types Comprehensive Examples
========================================================

This script demonstrates various data visualization types using Plotly with:
- Good practices (effective color schemes and design)
- Bad practices (common pitfalls to avoid)
- Importance identification (highlighting key insights)

Data Types Covered:
- Categorical Data: Bar charts, grouped bars, stacked bars, small multiples
- Ordinal Data: Ordered bar charts, slope charts
- Continuous Data: Scatter plots, histograms, box plots
- Temporal Data: Line charts, area charts
- Relational Data: Network diagrams, node-link charts
- Spatial Data: Choropleth maps, heatmaps, proportional symbol maps

Author: BUS 672 Course Materials
Date: 2025
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
import random
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import base64

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class DataVisualizationExamples:
    def __init__(self):
        self.colors = {
            'good': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],
            'bad': ['#FF0000', '#00FF00', '#0000FF', '#FFFF00'],
            'importance': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        }

    def generate_categorical_data(self):
        """Generate spoofed categorical data for examples"""
        return {
            'departments': ['Sales', 'Marketing', 'IT', 'HR', 'Finance'],
            'performance': [85, 92, 78, 88, 95],
            'budget': [120000, 80000, 150000, 60000, 100000],
            'employees': [25, 15, 30, 12, 20],
            'satisfaction': [4.2, 4.5, 3.8, 4.1, 4.6]
        }

    def generate_ordinal_data(self):
        """Generate spoofed ordinal data for examples"""
        return {
            'education_levels': ['High School', 'Associates', 'Bachelor', 'Master', 'PhD'],
            'salary_2022': [35000, 42000, 65000, 85000, 120000],
            'salary_2023': [37000, 45000, 68000, 90000, 125000],
            'satisfaction_scale': ['Very Dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very Satisfied'],
            'responses': [5, 12, 25, 45, 13]
        }

    def generate_continuous_data(self):
        """Generate spoofed continuous data for examples"""
        n = 200
        return {
            'income': np.random.normal(50000, 15000, n),
            'spending': np.random.normal(30000, 10000, n) + 0.6 * np.random.normal(50000, 15000, n) * 0.001,
            'age': np.random.randint(22, 65, n),
            'experience': np.random.randint(0, 40, n),
            'sales_amounts': np.random.gamma(2, 1000, n)
        }

    def generate_temporal_data(self):
        """Generate spoofed temporal data for examples"""
        dates = pd.date_range('2022-01-01', '2023-12-31', freq='M')
        return {
            'dates': dates,
            'sales': 1000 + np.cumsum(np.random.normal(50, 200, len(dates))) + 500 * np.sin(np.arange(len(dates)) * 2 * np.pi / 12),
            'website_visits': 5000 + np.cumsum(np.random.normal(100, 500, len(dates))) + 1000 * np.sin(np.arange(len(dates)) * 2 * np.pi / 12),
            'marketing_spend': 2000 + 500 * np.sin(np.arange(len(dates)) * 2 * np.pi / 12) + np.random.normal(0, 100, len(dates))
        }

    def generate_network_data(self):
        """Generate spoofed network/relational data"""
        G = nx.barabasi_albert_graph(20, 3)
        pos = nx.spring_layout(G)

        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]

        return {
            'edge_x': edge_x, 'edge_y': edge_y,
            'node_x': node_x, 'node_y': node_y,
            'node_text': [f'Node {i}' for i in G.nodes()],
            'node_adjacencies': [len(list(G.neighbors(node))) for node in G.nodes()]
        }

    def generate_spatial_data(self):
        """Generate spoofed spatial data for examples"""
        states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI']
        return {
            'states': states,
            'sales_by_state': np.random.randint(50000, 500000, len(states)),
            'lat': np.random.uniform(25, 49, 50),
            'lon': np.random.uniform(-125, -70, 50),
            'values': np.random.gamma(2, 10, 50)
        }

    def create_categorical_examples(self):
        """Create categorical data visualization examples"""
        data = self.generate_categorical_data()

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Good: Clean Bar Chart', 'Bad: Cluttered Colors',
                          'Good: Grouped Comparison', 'Importance: Highlighted Key Metric'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Good example: Clean bar chart
        fig.add_trace(
            go.Bar(x=data['departments'], y=data['performance'],
                   marker_color=self.colors['good'][0],
                   name='Performance Score',
                   text=data['performance'],
                   textposition='outside'),
            row=1, col=1
        )

        # Bad example: Overuse of colors
        fig.add_trace(
            go.Bar(x=data['departments'], y=data['performance'],
                   marker_color=self.colors['bad'],
                   name='Poor Color Choice',
                   showlegend=False),
            row=1, col=2
        )

        # Good example: Grouped bars
        fig.add_trace(
            go.Bar(x=data['departments'], y=data['budget'],
                   marker_color=self.colors['good'][1],
                   name='Budget', yaxis='y3',
                   offsetgroup=1),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=data['departments'], y=[x*1000 for x in data['employees']],
                   marker_color=self.colors['good'][2],
                   name='Employees (×1000)', yaxis='y3',
                   offsetgroup=2),
            row=2, col=1
        )

        # Importance example: Highlighted key department
        colors_importance = [self.colors['importance'][0] if dept == 'Finance'
                           else '#CCCCCC' for dept in data['departments']]
        fig.add_trace(
            go.Bar(x=data['departments'], y=data['satisfaction'],
                   marker_color=colors_importance,
                   name='Satisfaction (Finance Highlighted)',
                   showlegend=False),
            row=2, col=2
        )

        fig.update_layout(
            title_text="Categorical Data Visualization Examples",
            showlegend=True,
            height=800
        )

        return fig

    def create_ordinal_examples(self):
        """Create ordinal data visualization examples"""
        data = self.generate_ordinal_data()

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Good: Ordered Bar Chart', 'Good: Slope Chart', 'Bad: Wrong Order'),
        )

        # Good example: Proper ordered bar chart
        fig.add_trace(
            go.Bar(x=data['education_levels'], y=data['salary_2023'],
                   marker_color=self.colors['good'][0],
                   name='2023 Salary by Education'),
            row=1, col=1
        )

        # Good example: Slope chart showing change
        fig.add_trace(
            go.Scatter(x=[1]*len(data['education_levels']), y=data['salary_2022'],
                      mode='markers+text', text=data['education_levels'],
                      textposition='middle left', name='2022',
                      marker=dict(color=self.colors['good'][1], size=8)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=[2]*len(data['education_levels']), y=data['salary_2023'],
                      mode='markers+text', text=data['education_levels'],
                      textposition='middle right', name='2023',
                      marker=dict(color=self.colors['good'][2], size=8)),
            row=1, col=2
        )

        # Add slope lines
        for i in range(len(data['education_levels'])):
            fig.add_trace(
                go.Scatter(x=[1, 2], y=[data['salary_2022'][i], data['salary_2023'][i]],
                          mode='lines', line=dict(color='gray', width=1),
                          showlegend=False),
                row=1, col=2
            )

        # Bad example: Wrong order (alphabetical instead of logical)
        wrong_order = sorted(data['education_levels'])
        wrong_salary = [data['salary_2023'][data['education_levels'].index(level)]
                       for level in wrong_order]
        fig.add_trace(
            go.Bar(x=wrong_order, y=wrong_salary,
                   marker_color='red',
                   name='Wrong Order (Alphabetical)'),
            row=1, col=3
        )

        fig.update_layout(
            title_text="Ordinal Data Visualization Examples",
            height=500
        )

        return fig

    def create_continuous_examples(self):
        """Create continuous data visualization examples"""
        data = self.generate_continuous_data()

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Good: Scatter Plot with Correlation', 'Bad: Overplotting',
                          'Good: Histogram with Proper Bins', 'Good: Box Plot for Distribution'),
        )

        # Good example: Clear scatter plot
        fig.add_trace(
            go.Scatter(x=data['income'], y=data['spending'],
                      mode='markers',
                      marker=dict(color=self.colors['good'][0], opacity=0.6),
                      name='Income vs Spending'),
            row=1, col=1
        )

        # Bad example: Overplotting
        overplot_x = np.random.normal(50000, 5000, 2000)
        overplot_y = np.random.normal(30000, 3000, 2000)
        fig.add_trace(
            go.Scatter(x=overplot_x, y=overplot_y,
                      mode='markers',
                      marker=dict(color='red', size=3),
                      name='Overplotted Data'),
            row=1, col=2
        )

        # Good example: Histogram with proper bins
        fig.add_trace(
            go.Histogram(x=data['sales_amounts'],
                        nbinsx=20,
                        marker_color=self.colors['good'][1],
                        name='Sales Distribution'),
            row=2, col=1
        )

        # Good example: Box plot
        fig.add_trace(
            go.Box(y=data['sales_amounts'],
                  marker_color=self.colors['good'][2],
                  name='Sales Box Plot'),
            row=2, col=2
        )

        fig.update_layout(
            title_text="Continuous Data Visualization Examples",
            height=800
        )

        return fig

    def create_temporal_examples(self):
        """Create temporal data visualization examples"""
        data = self.generate_temporal_data()

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Good: Line Chart for Trends', 'Bad: Bars for Continuous Time',
                          'Good: Area Chart for Volume', 'Good: Multiple Time Series'),
        )

        # Good example: Line chart for trends
        fig.add_trace(
            go.Scatter(x=data['dates'], y=data['sales'],
                      mode='lines',
                      line=dict(color=self.colors['good'][0], width=2),
                      name='Monthly Sales'),
            row=1, col=1
        )

        # Bad example: Bars for continuous time series
        fig.add_trace(
            go.Bar(x=data['dates'], y=data['sales'],
                   marker_color='red',
                   name='Sales (Wrong: Bars for Time Series)'),
            row=1, col=2
        )

        # Good example: Area chart
        fig.add_trace(
            go.Scatter(x=data['dates'], y=data['website_visits'],
                      fill='tonexty',
                      mode='lines',
                      line=dict(color=self.colors['good'][1]),
                      name='Website Visits'),
            row=2, col=1
        )

        # Good example: Multiple time series
        fig.add_trace(
            go.Scatter(x=data['dates'], y=data['sales'],
                      mode='lines',
                      line=dict(color=self.colors['good'][2]),
                      name='Sales'),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=data['dates'], y=data['marketing_spend'],
                      mode='lines',
                      line=dict(color=self.colors['good'][3]),
                      name='Marketing Spend',
                      yaxis='y4'),
            row=2, col=2
        )

        fig.update_layout(
            title_text="Temporal Data Visualization Examples",
            height=800
        )

        return fig

    def create_network_examples(self):
        """Create network/relational data visualization examples"""
        data = self.generate_network_data()

        fig = go.Figure()

        # Add edges
        fig.add_trace(go.Scatter(x=data['edge_x'], y=data['edge_y'],
                               mode='lines',
                               line=dict(width=0.5, color='#888'),
                               hoverinfo='none',
                               showlegend=False))

        # Add nodes
        fig.add_trace(go.Scatter(x=data['node_x'], y=data['node_y'],
                               mode='markers',
                               hoverinfo='text',
                               text=data['node_text'],
                               hovertext=[f'{text}<br>Connections: {adj}'
                                        for text, adj in zip(data['node_text'], data['node_adjacencies'])],
                               marker=dict(
                                   showscale=True,
                                   colorscale='Viridis',
                                   reversescale=True,
                                   color=data['node_adjacencies'],
                                   size=10,
                                   colorbar=dict(
                                       thickness=15,
                                       len=0.5,
                                       x=1.02,
                                       title="Node Connections"
                                   ),
                                   line=dict(width=2))))

        fig.update_layout(title=dict(text='Network Diagram Example', font=dict(size=16)),
                         showlegend=False,
                         hovermode='closest',
                         margin=dict(b=20,l=5,r=5,t=40),
                         annotations=[ dict(
                             text="Good: Clear network structure with meaningful node sizing",
                             showarrow=False,
                             xref="paper", yref="paper",
                             x=0.005, y=-0.002,
                             xanchor='left', yanchor='bottom',
                             font=dict(size=12)
                         )],
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

        return fig

    def create_spatial_examples(self):
        """Create spatial data visualization examples"""
        data = self.generate_spatial_data()

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Good: Choropleth Map', 'Good: Business Performance Heatmap'),
            specs=[[{"type": "geo"}, {"type": "xy"}]]
        )

        # Good example: Choropleth map
        fig.add_trace(
            go.Choropleth(
                locations=data['states'],
                z=data['sales_by_state'],
                locationmode='USA-states',
                colorscale='Blues',
                colorbar_title="Sales ($)",
            ),
            row=1, col=1
        )

        # Good example: Business performance heatmap with meaningful labels
        # Create structured business data
        departments = ['Sales', 'Marketing', 'IT', 'HR', 'Finance', 'Operations']
        metrics = ['Q1 Performance', 'Q2 Performance', 'Q3 Performance', 'Q4 Performance',
                  'Budget Efficiency', 'Customer Satisfaction', 'Employee Retention']

        # Create realistic business performance data (0-100 scale)
        np.random.seed(42)  # For reproducible results
        heatmap_data = []
        for dept in departments:
            dept_scores = []
            for metric in metrics:
                if 'Performance' in metric:
                    # Performance metrics: 70-95 range
                    score = np.random.normal(82, 8)
                elif 'Budget' in metric:
                    # Budget efficiency: 60-90 range
                    score = np.random.normal(75, 10)
                elif 'Customer' in metric:
                    # Customer satisfaction: 75-95 range
                    score = np.random.normal(85, 6)
                else:  # Employee retention
                    # Retention: 80-98 range
                    score = np.random.normal(89, 5)

                # Ensure values are within reasonable bounds
                score = max(50, min(100, score))
                dept_scores.append(score)
            heatmap_data.append(dept_scores)

        fig.add_trace(
            go.Heatmap(
                z=heatmap_data,
                x=metrics,
                y=departments,
                colorscale='RdYlGn',  # Red-Yellow-Green for performance
                showscale=True,
                colorbar=dict(
                    title=dict(text="Performance Score"),
                    x=1.02
                ),
                text=[[f'{val:.1f}' for val in row] for row in heatmap_data],
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False,
                hovertemplate='<b>%{y}</b><br>%{x}<br>Score: %{z:.1f}<extra></extra>'
            ),
            row=1, col=2
        )

        # Update layout for better appearance
        fig.update_geos(scope="usa")
        fig.update_layout(
            title_text="Spatial Data Visualization Examples",
            height=600,
            xaxis2=dict(
                title="Business Metrics",
                tickangle=45,
                side="bottom"
            ),
            yaxis2=dict(
                title="Departments",
                tickmode="array",
                tickvals=list(range(len(departments))),
                ticktext=departments
            )
        )

        return fig

    def save_plots_to_pdf(self):
        """Save all plots to a comprehensive PDF with notes"""
        doc = SimpleDocTemplate("BUS_672_Data_Visualization_Examples.pdf", pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1
        )
        story.append(Paragraph("BUS 672 - Data Visualization Types", title_style))
        story.append(Paragraph("Comprehensive Examples with Best Practices", styles['Heading2']))
        story.append(Spacer(1, 20))

        # Introduction
        intro_text = """
        This document demonstrates various data visualization types with examples of good practices,
        common pitfalls, and importance highlighting techniques. Each section covers different data
        types and their appropriate visualization methods.
        """
        story.append(Paragraph(intro_text, styles['Normal']))
        story.append(Spacer(1, 20))

        # Create and save each plot type
        plot_functions = [
            (self.create_categorical_examples, "Categorical Data Visualizations"),
            (self.create_ordinal_examples, "Ordinal Data Visualizations"),
            (self.create_continuous_examples, "Continuous Data Visualizations"),
            (self.create_temporal_examples, "Temporal Data Visualizations"),
            (self.create_network_examples, "Network/Relational Data Visualizations"),
            (self.create_spatial_examples, "Spatial Data Visualizations")
        ]

        for plot_func, title in plot_functions:
            # Add section title
            story.append(Paragraph(title, styles['Heading2']))

            # Create plot
            fig = plot_func()

            # Save plot as image
            img_bytes = fig.to_image(format="png", width=800, height=600)
            img_buffer = io.BytesIO(img_bytes)

            # Add image to PDF
            img = Image(img_buffer, width=6*inch, height=4.5*inch)
            story.append(img)
            story.append(Spacer(1, 20))

            # Add notes for each section
            self.add_section_notes(story, title, styles)
            story.append(Spacer(1, 30))

        # Build PDF
        doc.build(story)
        print("PDF saved as 'BUS_672_Data_Visualization_Examples.pdf'")

    def add_section_notes(self, story, section_title, styles):
        """Add detailed notes for each visualization section"""
        notes = {
            "Categorical Data Visualizations": [
                "Best Practices: Use consistent colors, clear labels, avoid truncated y-axis",
                "Common Pitfalls: Too many categories, overuse of colors, misleading scales",
                "Use Cases: Comparing categories, showing rankings, displaying proportions"
            ],
            "Ordinal Data Visualizations": [
                "Best Practices: Maintain logical order, use ordered bar charts or slope charts",
                "Common Pitfalls: Using alphabetical instead of logical order",
                "Use Cases: Survey scales, rankings, education levels, time comparisons"
            ],
            "Continuous Data Visualizations": [
                "Best Practices: Appropriate bin sizes, clear axis labels, handle overplotting",
                "Common Pitfalls: Wrong bin sizes, overplotting, unclear scales",
                "Use Cases: Correlations, distributions, outlier identification"
            ],
            "Temporal Data Visualizations": [
                "Best Practices: Use lines for continuous time series, show seasonality",
                "Common Pitfalls: Using bars for continuous time, uneven intervals",
                "Use Cases: Trend analysis, time comparisons, seasonal patterns"
            ],
            "Network/Relational Data Visualizations": [
                "Best Practices: Clear node sizing, meaningful connections, avoid hairball networks",
                "Common Pitfalls: Overly dense networks, unclear relationships",
                "Use Cases: Social networks, process flows, entity relationships"
            ],
            "Spatial Data Visualizations": [
                "Best Practices: Appropriate projections, normalized data, clear color scales",
                "Common Pitfalls: Rainbow color scales, misleading projections, raw vs. normalized",
                "Use Cases: Geographic comparisons, density mapping, location patterns"
            ]
        }

        section_notes = notes.get(section_title, [])
        for note in section_notes:
            story.append(Paragraph(f"• {note}", styles['Normal']))

def main():
    """Main execution function"""
    print("Creating comprehensive data visualization examples...")

    # Create visualization examples
    viz = DataVisualizationExamples()

    # Generate and display each plot type
    print("Generating categorical examples...")
    cat_fig = viz.create_categorical_examples()
    cat_fig.show()

    print("Generating ordinal examples...")
    ord_fig = viz.create_ordinal_examples()
    ord_fig.show()

    print("Generating continuous examples...")
    cont_fig = viz.create_continuous_examples()
    cont_fig.show()

    print("Generating temporal examples...")
    temp_fig = viz.create_temporal_examples()
    temp_fig.show()

    print("Generating network examples...")
    net_fig = viz.create_network_examples()
    net_fig.show()

    print("Generating spatial examples...")
    spatial_fig = viz.create_spatial_examples()
    spatial_fig.show()

    # Create comprehensive PDF
    print("Creating comprehensive PDF with all examples and notes...")
    viz.save_plots_to_pdf()

    print("All visualizations created successfully!")
    print("Check 'BUS_672_Data_Visualization_Examples.pdf' for the complete guide.")

if __name__ == "__main__":
    main()