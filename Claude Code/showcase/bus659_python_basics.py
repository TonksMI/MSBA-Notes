"""
BUS 659 - Python Basics for Business Analytics
Converted from Jupyter Notebook for showcase

Description: Foundational Python concepts for business data analysis
Topics: Data structures, pandas fundamentals, basic visualization
Target Audience: Business managers with no coding background
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BusinessPythonBasics:
    """
    Introduction to Python for Business Analytics
    
    This class covers essential Python concepts for business professionals:
    - Data types and structures for business data
    - Pandas for data manipulation
    - Basic visualizations for business insights
    - Practical business applications
    """
    
    def __init__(self):
        self.sample_data = None
        
    def python_fundamentals_for_business(self):
        """
        Core Python concepts every business analyst needs
        
        Business Applications:
        - Variables for storing business metrics
        - Lists for product catalogs, customer segments
        - Dictionaries for structured business data
        - Functions for reusable business calculations
        """
        print("=" * 50)
        print("PYTHON FUNDAMENTALS FOR BUSINESS")
        print("=" * 50)
        
        # 1. Variables and Basic Data Types
        print("\n1. BUSINESS VARIABLES")
        print("-" * 20)
        
        # Revenue metrics
        monthly_revenue = 250000
        quarterly_growth = 0.125  # 12.5%
        company_name = "TechCorp Analytics"
        is_profitable = True
        
        print(f"Company: {company_name}")
        print(f"Monthly Revenue: ${monthly_revenue:,}")
        print(f"Quarterly Growth: {quarterly_growth:.1%}")
        print(f"Profitable: {is_profitable}")
        
        # 2. Lists for Business Data
        print("\n2. BUSINESS LISTS")
        print("-" * 20)
        
        # Product portfolio
        products = ["Analytics Platform", "Data Warehouse", "BI Tools", "ML Models"]
        quarterly_sales = [75000, 90000, 65000, 120000]
        
        print(f"Product Portfolio: {products}")
        print(f"Q1 Sales by Product: {quarterly_sales}")
        
        # Calculate total sales
        total_sales = sum(quarterly_sales)
        print(f"Total Q1 Sales: ${total_sales:,}")
        
        # 3. Dictionaries for Structured Business Data
        print("\n3. BUSINESS DATA STRUCTURES")
        print("-" * 20)
        
        # Customer information
        customer_profile = {
            "customer_id": "CUST_001",
            "company": "Fortune 500 Corp",
            "industry": "Finance",
            "contract_value": 500000,
            "start_date": "2024-01-01",
            "products": ["Analytics Platform", "ML Models"]
        }
        
        print("Customer Profile:")
        for key, value in customer_profile.items():
            print(f"  {key}: {value}")
        
        # 4. Functions for Business Calculations
        print("\n4. BUSINESS FUNCTIONS")
        print("-" * 20)
        
        def calculate_customer_lifetime_value(monthly_revenue, retention_months, margin=0.3):
            """Calculate Customer Lifetime Value for business planning"""
            return monthly_revenue * retention_months * margin
        
        def calculate_roi(investment, return_amount):
            """Calculate Return on Investment for business decisions"""
            return (return_amount - investment) / investment
        
        # Example calculations
        clv = calculate_customer_lifetime_value(25000, 24, 0.35)
        roi = calculate_roi(100000, 150000)
        
        print(f"Customer Lifetime Value: ${clv:,.0f}")
        print(f"Marketing Campaign ROI: {roi:.1%}")
        
        return {
            'revenue': monthly_revenue,
            'growth': quarterly_growth,
            'products': products,
            'sales': quarterly_sales,
            'customer': customer_profile,
            'clv': clv,
            'roi': roi
        }
    
    def pandas_for_business_data(self):
        """
        Pandas fundamentals for business data analysis
        
        Business Applications:
        - Loading sales data from Excel/CSV
        - Filtering for specific customers/products
        - Calculating business metrics
        - Grouping data for insights
        """
        print("\n" + "=" * 50)
        print("PANDAS FOR BUSINESS DATA ANALYSIS")
        print("=" * 50)
        
        # Create sample business dataset
        np.random.seed(42)
        
        # Generate realistic business data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        customers = ['Enterprise Corp', 'SMB Solutions', 'Startup Inc', 'Global Industries', 'Local Business']
        products = ['Analytics Pro', 'Data Warehouse', 'ML Suite', 'BI Dashboard']
        regions = ['North', 'South', 'East', 'West']
        
        business_data = []
        for i in range(500):
            record = {
                'date': np.random.choice(dates),
                'customer': np.random.choice(customers),
                'product': np.random.choice(products),
                'region': np.random.choice(regions),
                'revenue': np.random.normal(25000, 8000),
                'units_sold': np.random.randint(1, 20),
                'sales_rep': f"Rep_{np.random.randint(1, 11)}"
            }
            business_data.append(record)
        
        self.sample_data = pd.DataFrame(business_data)
        self.sample_data['revenue'] = self.sample_data['revenue'].clip(lower=5000)  # No negative revenue
        
        print("1. BUSINESS DATA OVERVIEW")
        print("-" * 30)
        print(f"Dataset shape: {self.sample_data.shape}")
        print(f"Date range: {self.sample_data['date'].min()} to {self.sample_data['date'].max()}")
        print(f"Total revenue: ${self.sample_data['revenue'].sum():,.0f}")
        print(f"Average deal size: ${self.sample_data['revenue'].mean():,.0f}")
        
        # Display first few records
        print("\nSample Data:")
        print(self.sample_data.head())
        
        # 2. Business Filtering and Analysis
        print("\n2. BUSINESS FILTERING")
        print("-" * 30)
        
        # High-value customers (deals > $30k)
        high_value_deals = self.sample_data[self.sample_data['revenue'] > 30000]
        print(f"High-value deals (>$30k): {len(high_value_deals)}")
        print(f"High-value revenue: ${high_value_deals['revenue'].sum():,.0f}")
        
        # Top performing product
        product_performance = self.sample_data.groupby('product')['revenue'].agg(['sum', 'count', 'mean'])
        product_performance.columns = ['Total Revenue', 'Deal Count', 'Average Deal Size']
        product_performance = product_performance.sort_values('Total Revenue', ascending=False)
        
        print("\nProduct Performance:")
        print(product_performance)
        
        # 3. Regional Analysis
        print("\n3. REGIONAL BUSINESS INSIGHTS")
        print("-" * 30)
        
        regional_analysis = self.sample_data.groupby('region').agg({
            'revenue': ['sum', 'mean', 'count'],
            'units_sold': 'sum'
        }).round(0)
        
        regional_analysis.columns = ['Total Revenue', 'Avg Deal Size', 'Deal Count', 'Total Units']
        regional_analysis = regional_analysis.sort_values('Total Revenue', ascending=False)
        
        print("Regional Performance:")
        print(regional_analysis)
        
        # 4. Time-based Analysis
        print("\n4. MONTHLY TRENDS")
        print("-" * 30)
        
        self.sample_data['month'] = self.sample_data['date'].dt.to_period('M')
        monthly_trends = self.sample_data.groupby('month')['revenue'].sum()
        
        print("Monthly Revenue Trends:")
        for month, revenue in monthly_trends.items():
            print(f"  {month}: ${revenue:,.0f}")
        
        return self.sample_data, product_performance, regional_analysis, monthly_trends
    
    def business_visualizations(self):
        """
        Essential visualizations for business presentations
        
        Chart Types for Business:
        - Bar charts for product/regional comparisons
        - Line charts for trends over time
        - Pie charts for market share
        - Scatter plots for correlations
        """
        if self.sample_data is None:
            self.pandas_for_business_data()
        
        print("\n" + "=" * 50)
        print("BUSINESS VISUALIZATIONS")
        print("=" * 50)
        
        # Set up the visualization style
        plt.style.use('seaborn-v0_8')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Business Performance Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Product Revenue Comparison (Bar Chart)
        product_revenue = self.sample_data.groupby('product')['revenue'].sum().sort_values(ascending=False)
        
        bars = ax1.bar(product_revenue.index, product_revenue.values, color='skyblue', alpha=0.8)
        ax1.set_title('Revenue by Product Line', fontweight='bold')
        ax1.set_ylabel('Revenue ($)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}', ha='center', va='bottom')
        
        # 2. Monthly Revenue Trend (Line Chart)
        monthly_revenue = self.sample_data.groupby(self.sample_data['date'].dt.to_period('M'))['revenue'].sum()
        
        ax2.plot(range(len(monthly_revenue)), monthly_revenue.values, marker='o', linewidth=3, color='green')
        ax2.set_title('Monthly Revenue Trend', fontweight='bold')
        ax2.set_ylabel('Revenue ($)')
        ax2.set_xlabel('Month')
        ax2.grid(True, alpha=0.3)
        
        # 3. Regional Market Share (Pie Chart)
        regional_revenue = self.sample_data.groupby('region')['revenue'].sum()
        
        ax3.pie(regional_revenue.values, labels=regional_revenue.index, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Revenue Distribution by Region', fontweight='bold')
        
        # 4. Deal Size vs Units Sold (Scatter Plot)
        ax4.scatter(self.sample_data['units_sold'], self.sample_data['revenue'], alpha=0.6, color='coral')
        ax4.set_title('Deal Size vs Units Sold', fontweight='bold')
        ax4.set_xlabel('Units Sold')
        ax4.set_ylabel('Revenue ($)')
        
        # Add trend line
        z = np.polyfit(self.sample_data['units_sold'], self.sample_data['revenue'], 1)
        p = np.poly1d(z)
        ax4.plot(self.sample_data['units_sold'], p(self.sample_data['units_sold']), 
                "r--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        
        print("Generated business visualizations:")
        print("1. Product Revenue Comparison")
        print("2. Monthly Revenue Trends")  
        print("3. Regional Market Share")
        print("4. Deal Size vs Units Correlation")
        
        return fig
    
    def business_insights_and_recommendations(self):
        """
        Generate actionable business insights from data analysis
        
        Business Value:
        - Identify top-performing products for investment
        - Recognize regional opportunities
        - Understand seasonal trends for planning
        - Optimize sales strategies based on data
        """
        if self.sample_data is None:
            self.pandas_for_business_data()
        
        print("\n" + "=" * 50)
        print("BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("=" * 50)
        
        # Key Performance Indicators
        total_revenue = self.sample_data['revenue'].sum()
        total_deals = len(self.sample_data)
        avg_deal_size = self.sample_data['revenue'].mean()
        
        # Product analysis
        product_performance = self.sample_data.groupby('product').agg({
            'revenue': ['sum', 'count', 'mean']
        })
        product_performance.columns = ['total_revenue', 'deal_count', 'avg_deal_size']
        top_product = product_performance['total_revenue'].idxmax()
        
        # Regional analysis
        regional_performance = self.sample_data.groupby('region')['revenue'].sum()
        top_region = regional_performance.idxmax()
        
        # Customer analysis
        customer_performance = self.sample_data.groupby('customer')['revenue'].sum().sort_values(ascending=False)
        top_customer = customer_performance.index[0]
        
        insights = {
            'kpis': {
                'total_revenue': total_revenue,
                'total_deals': total_deals,
                'avg_deal_size': avg_deal_size
            },
            'top_performers': {
                'product': top_product,
                'region': top_region,
                'customer': top_customer
            },
            'recommendations': [
                f"Focus marketing investment on '{top_product}' - highest revenue generator",
                f"Expand sales team in '{top_region}' region - strongest market performance",
                f"Develop customer success program for '{top_customer}' - top revenue contributor",
                "Investigate correlation between units sold and deal size for pricing optimization",
                "Monitor monthly trends to identify seasonal patterns for capacity planning"
            ]
        }
        
        print("KEY PERFORMANCE INDICATORS:")
        print(f"  Total Revenue: ${insights['kpis']['total_revenue']:,.0f}")
        print(f"  Total Deals: {insights['kpis']['total_deals']:,}")
        print(f"  Average Deal Size: ${insights['kpis']['avg_deal_size']:,.0f}")
        
        print("\nTOP PERFORMERS:")
        print(f"  Best Product: {insights['top_performers']['product']}")
        print(f"  Best Region: {insights['top_performers']['region']}")
        print(f"  Best Customer: {insights['top_performers']['customer']}")
        
        print("\nBUSINESS RECOMMENDATIONS:")
        for i, rec in enumerate(insights['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        return insights

def run_business_python_tutorial():
    """
    Complete Python tutorial for business professionals
    """
    print("🐍 PYTHON FOR BUSINESS ANALYTICS - COMPLETE TUTORIAL")
    print("=" * 60)
    print("Target Audience: Business managers with no coding background")
    print("Learning Objectives: Data analysis, visualization, business insights")
    print("=" * 60)
    
    # Initialize the tutorial
    tutorial = BusinessPythonBasics()
    
    # Run all sections
    fundamentals = tutorial.python_fundamentals_for_business()
    data_analysis = tutorial.pandas_for_business_data()
    visualizations = tutorial.business_visualizations()
    insights = tutorial.business_insights_and_recommendations()
    
    print("\n" + "=" * 60)
    print("TUTORIAL COMPLETE - KEY TAKEAWAYS")
    print("=" * 60)
    print("✅ Python fundamentals for business data")
    print("✅ Pandas for data manipulation and analysis")
    print("✅ Business visualizations for presentations")
    print("✅ Data-driven insights and recommendations")
    print("\nNext Steps: Apply these concepts to your business data!")
    
    return tutorial, fundamentals, data_analysis, visualizations, insights

if __name__ == "__main__":
    tutorial, fundamentals, data_analysis, visualizations, insights = run_business_python_tutorial()
    plt.show()