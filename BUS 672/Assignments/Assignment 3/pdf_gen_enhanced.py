import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import io

# Set matplotlib style for professional visualizations
plt.style.use('seaborn-v0_8-darkgrid')

# Generate spoofed data for each analytics type
def generate_retrospective_data():
    """Generate historical sales data"""
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='M')
    sales = np.random.randint(50000, 150000, size=len(dates))
    # Add trend
    trend = np.linspace(0, 30000, len(dates))
    sales = sales + trend
    return pd.DataFrame({'date': dates, 'sales': sales})

def generate_predictive_data():
    """Generate customer churn probability data"""
    customers = [f'Customer_{i}' for i in range(1, 101)]
    features = {
        'customer_id': customers,
        'months_active': np.random.randint(1, 60, 100),
        'support_tickets': np.random.randint(0, 20, 100),
        'usage_hours': np.random.randint(5, 200, 100),
        'churn_probability': np.random.uniform(0.05, 0.95, 100)
    }
    return pd.DataFrame(features)

def generate_prescriptive_data():
    """Generate inventory optimization data"""
    products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
    current_stock = [150, 300, 75, 200, 180]
    recommended_stock = [200, 250, 120, 180, 220]
    holding_cost = [5, 3, 8, 4, 6]
    stockout_cost = [50, 30, 80, 45, 55]

    return pd.DataFrame({
        'product': products,
        'current_stock': current_stock,
        'recommended_stock': recommended_stock,
        'holding_cost': holding_cost,
        'stockout_cost': stockout_cost
    })

def create_retrospective_viz(data):
    """Create enhanced visualization for retrospective analytics"""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot with enhanced styling
    line = ax.plot(data['date'], data['sales'],
                   marker='o', linewidth=2.5, markersize=6,
                   color='#3498db', markerfacecolor='#2980b9',
                   markeredgecolor='white', markeredgewidth=1.5,
                   label='Monthly Sales')

    # Add trend line
    z = np.polyfit(range(len(data)), data['sales'], 1)
    p = np.poly1d(z)
    ax.plot(data['date'], p(range(len(data))),
            linestyle='--', linewidth=2, color='#e74c3c',
            alpha=0.8, label='Trend Line')

    # Fill area under curve
    ax.fill_between(data['date'], data['sales'], alpha=0.2, color='#3498db')

    # Enhanced labels and styling
    ax.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax.set_ylabel('Sales ($)', fontsize=13, fontweight='bold')
    ax.set_title('Historical Sales Performance (Retrospective Analytics)',
                 fontsize=15, fontweight='bold', pad=20)

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Enhanced grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close()
    return buf

def create_predictive_viz(data):
    """Create enhanced visualization for predictive analytics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Create bins for churn probability
    bins = [0, 0.3, 0.7, 1.0]
    labels = ['Low Risk', 'Medium Risk', 'High Risk']
    data['risk_category'] = pd.cut(data['churn_probability'], bins=bins, labels=labels)

    # Left plot: Bar chart with enhanced styling
    risk_counts = data['risk_category'].value_counts()
    colors = ['#27ae60', '#f39c12', '#e74c3c']

    bars = ax1.bar(risk_counts.index, risk_counts.values,
                   color=colors, edgecolor='#2c3e50', linewidth=2,
                   alpha=0.85, width=0.6)

    ax1.set_xlabel('Risk Category', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
    ax1.set_title('Customer Churn Risk Distribution', fontsize=13, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_ylim(0, max(risk_counts.values) * 1.15)

    # Add value labels on bars with enhanced styling
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Add percentage labels
    total = sum(risk_counts.values)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        pct = (height/total) * 100
        ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{pct:.1f}%',
                ha='center', va='center', fontweight='bold',
                fontsize=10, color='white')

    # Right plot: Distribution of churn probabilities
    ax2.hist(data['churn_probability'], bins=20,
             color='#9b59b6', alpha=0.7, edgecolor='#2c3e50', linewidth=1.5)
    ax2.axvline(data['churn_probability'].mean(),
                color='#e74c3c', linestyle='--', linewidth=2.5,
                label=f'Mean: {data["churn_probability"].mean():.2f}')

    ax2.set_xlabel('Churn Probability', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Probability Distribution', fontsize=13, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.legend(fontsize=10, framealpha=0.9)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close()
    return buf

def create_prescriptive_viz(data):
    """Create enhanced visualization for prescriptive analytics"""
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # Top plot: Stock comparison with enhanced styling
    x = np.arange(len(data['product']))
    width = 0.35

    bars1 = ax1.bar(x - width/2, data['current_stock'], width,
                    label='Current Stock', color='#3498db',
                    edgecolor='#2c3e50', linewidth=2, alpha=0.85)
    bars2 = ax1.bar(x + width/2, data['recommended_stock'], width,
                    label='Recommended Stock', color='#2ecc71',
                    edgecolor='#2c3e50', linewidth=2, alpha=0.85)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Add difference indicators
    for i in range(len(data)):
        diff = data['recommended_stock'].iloc[i] - data['current_stock'].iloc[i]
        y_pos = max(data['current_stock'].iloc[i], data['recommended_stock'].iloc[i]) + 10
        color = '#27ae60' if diff > 0 else '#e74c3c'
        symbol = '▲' if diff > 0 else '▼'
        ax1.text(i, y_pos, f'{symbol}{abs(diff)}',
                ha='center', va='bottom', fontweight='bold',
                fontsize=9, color=color)

    ax1.set_xlabel('Product', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Stock Level (Units)', fontsize=12, fontweight='bold')
    ax1.set_title('Inventory Optimization Recommendations (Prescriptive Analytics)',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(data['product'], fontweight='bold')
    ax1.legend(fontsize=11, loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_ylim(0, max(data['recommended_stock'].max(), data['current_stock'].max()) * 1.2)

    # Bottom left: Cost comparison
    total_holding = data['current_stock'] * data['holding_cost']
    total_stockout_risk = (data['recommended_stock'] - data['current_stock']).clip(lower=0) * data['stockout_cost']

    x2 = np.arange(len(data['product']))
    bars3 = ax2.barh(x2, total_holding, color='#e67e22',
                     edgecolor='#2c3e50', linewidth=1.5, alpha=0.85,
                     label='Holding Cost')

    for i, bar in enumerate(bars3):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2,
                f'${width:.0f}',
                ha='left', va='center', fontweight='bold', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax2.set_yticks(x2)
    ax2.set_yticklabels(data['product'])
    ax2.set_xlabel('Cost ($)', fontsize=11, fontweight='bold')
    ax2.set_title('Current Holding Costs', fontsize=12, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, axis='x', linestyle='--')

    # Bottom right: Potential savings
    potential_savings = total_stockout_risk - (data['recommended_stock'] - data['current_stock']).abs() * data['holding_cost']
    colors_savings = ['#27ae60' if x > 0 else '#e74c3c' for x in potential_savings]

    bars4 = ax3.barh(x2, potential_savings, color=colors_savings,
                     edgecolor='#2c3e50', linewidth=1.5, alpha=0.85)

    for i, bar in enumerate(bars4):
        width = bar.get_width()
        x_pos = width + (5 if width > 0 else -5)
        ha = 'left' if width > 0 else 'right'
        ax3.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'${width:.0f}',
                ha=ha, va='center', fontweight='bold', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax3.set_yticks(x2)
    ax3.set_yticklabels(data['product'])
    ax3.set_xlabel('Net Impact ($)', fontsize=11, fontweight='bold')
    ax3.set_title('Optimization Impact', fontsize=12, fontweight='bold', pad=10)
    ax3.axvline(x=0, color='#2c3e50', linestyle='-', linewidth=2)
    ax3.grid(True, alpha=0.3, axis='x', linestyle='--')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close()
    return buf

def create_pdf():
    """Create PDF with all analytics types"""
    filename = 'analytics_types_report_enhanced.pdf'
    doc = SimpleDocTemplate(filename, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor='#2c3e50',
        spaceAfter=30,
        alignment=TA_CENTER
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor='#34495e',
        spaceAfter=12,
        spaceBefore=12
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=12
    )

    # Title
    story.append(Paragraph("Three Types of Analytics", title_style))
    story.append(Spacer(1, 0.2*inch))

    # Introduction
    intro_text = """This report demonstrates the three primary types of analytics used in
    business intelligence and data science: Retrospective Analytics, Predictive Analytics,
    and Prescriptive Analytics. Each type serves a unique purpose in data-driven decision making."""
    story.append(Paragraph(intro_text, body_style))
    story.append(Spacer(1, 0.3*inch))

    # === RETROSPECTIVE ANALYTICS ===
    story.append(Paragraph("1. Retrospective Analytics (Descriptive)", heading_style))

    retro_desc = """<b>What is it?</b> Retrospective analytics, also known as descriptive
    analytics, examines historical data to understand what happened in the past. It provides
    insights into trends, patterns, and performance metrics using data visualization, reporting,
    and statistical analysis."""
    story.append(Paragraph(retro_desc, body_style))

    retro_usecase = """<b>Use Cases:</b><br/>
    • Financial reporting and performance dashboards<br/>
    • Sales trend analysis and revenue tracking<br/>
    • Website traffic and user behavior analysis<br/>
    • Customer demographics and segmentation<br/>
    • Operational efficiency metrics and KPI monitoring"""
    story.append(Paragraph(retro_usecase, body_style))
    story.append(Spacer(1, 0.2*inch))

    # Add retrospective visualization
    retro_data = generate_retrospective_data()
    retro_img = create_retrospective_viz(retro_data)
    story.append(Image(retro_img, width=6.5*inch, height=3.8*inch))
    story.append(PageBreak())

    # === PREDICTIVE ANALYTICS ===
    story.append(Paragraph("2. Predictive Analytics", heading_style))

    pred_desc = """<b>What is it?</b> Predictive analytics uses statistical models, machine
    learning algorithms, and historical data to forecast future outcomes. It answers the question
    "What is likely to happen?" by identifying patterns and relationships in data to make
    probabilistic predictions."""
    story.append(Paragraph(pred_desc, body_style))

    pred_usecase = """<b>Use Cases:</b><br/>
    • Customer churn prediction and retention strategies<br/>
    • Demand forecasting for inventory management<br/>
    • Credit risk assessment and loan default prediction<br/>
    • Predictive maintenance for equipment failures<br/>
    • Sales forecasting and revenue projections<br/>
    • Healthcare outcome predictions and disease progression"""
    story.append(Paragraph(pred_usecase, body_style))
    story.append(Spacer(1, 0.2*inch))

    # Add predictive visualization
    pred_data = generate_predictive_data()
    pred_img = create_predictive_viz(pred_data)
    story.append(Image(pred_img, width=6.5*inch, height=2.8*inch))
    story.append(PageBreak())

    # === PRESCRIPTIVE ANALYTICS ===
    story.append(Paragraph("3. Prescriptive Analytics", heading_style))

    presc_desc = """<b>What is it?</b> Prescriptive analytics goes beyond prediction to recommend
    specific actions. It uses optimization algorithms, simulation, and decision analysis to suggest
    the best course of action among multiple alternatives. It answers "What should we do?" by
    evaluating various scenarios and their potential outcomes."""
    story.append(Paragraph(presc_desc, body_style))

    presc_usecase = """<b>Use Cases:</b><br/>
    • Supply chain optimization and route planning<br/>
    • Dynamic pricing strategies for maximum profit<br/>
    • Resource allocation and workforce scheduling<br/>
    • Marketing campaign optimization and budget allocation<br/>
    • Investment portfolio optimization<br/>
    • Clinical decision support and treatment recommendations<br/>
    • Energy grid management and load balancing"""
    story.append(Paragraph(presc_usecase, body_style))
    story.append(Spacer(1, 0.2*inch))

    # Add prescriptive visualization
    presc_data = generate_prescriptive_data()
    presc_img = create_prescriptive_viz(presc_data)
    story.append(Image(presc_img, width=6.5*inch, height=3.2*inch))

    # Build PDF (no summary section added)
    doc.build(story)
    print(f"Enhanced PDF created successfully: {filename}")

if __name__ == "__main__":
    create_pdf()
