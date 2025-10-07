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
    """Create visualization for retrospective analytics"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['date'], data['sales'], marker='o', linewidth=2, markersize=4)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Sales ($)', fontsize=12)
    ax.set_title('Historical Sales Performance (Retrospective Analytics)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

def create_predictive_viz(data):
    """Create visualization for predictive analytics"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bins for churn probability
    bins = [0, 0.3, 0.7, 1.0]
    labels = ['Low Risk', 'Medium Risk', 'High Risk']
    data['risk_category'] = pd.cut(data['churn_probability'], bins=bins, labels=labels)
    
    risk_counts = data['risk_category'].value_counts()
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    ax.bar(risk_counts.index, risk_counts.values, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Risk Category', fontsize=12)
    ax.set_ylabel('Number of Customers', fontsize=12)
    ax.set_title('Customer Churn Risk Distribution (Predictive Analytics)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(risk_counts.values):
        ax.text(i, v + 1, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

def create_prescriptive_viz(data):
    """Create visualization for prescriptive analytics"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(data['product']))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, data['current_stock'], width, label='Current Stock', 
                   color='#3498db', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, data['recommended_stock'], width, label='Recommended Stock',
                   color='#2ecc71', edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Product', fontsize=12)
    ax.set_ylabel('Stock Level (Units)', fontsize=12)
    ax.set_title('Inventory Optimization Recommendations (Prescriptive Analytics)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(data['product'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

def create_pdf():
    """Create PDF with all analytics types"""
    filename = 'analytics_types_report.pdf'
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
    story.append(Image(retro_img, width=6*inch, height=3.6*inch))
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
    story.append(Image(pred_img, width=6*inch, height=3.6*inch))
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
    story.append(Image(presc_img, width=6*inch, height=3.6*inch))
    story.append(Spacer(1, 0.3*inch))
    
    # Conclusion
    story.append(Paragraph("Summary", heading_style))
    conclusion = """These three types of analytics build upon each other to create a comprehensive 
    approach to data-driven decision making. Retrospective analytics tells us what happened, 
    predictive analytics forecasts what might happen, and prescriptive analytics recommends what 
    actions to take. Organizations that effectively leverage all three types gain significant 
    competitive advantages through better insights and optimized decision-making processes."""
    story.append(Paragraph(conclusion, body_style))
    
    # Build PDF
    doc.build(story)
    print(f"PDF created successfully: {filename}")

if __name__ == "__main__":
    create_pdf()