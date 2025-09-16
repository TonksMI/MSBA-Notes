import streamlit as st
import os
import pandas as pd
from pathlib import Path
import re

# Page configuration
st.set_page_config(
    page_title="Matthew Tonks - Academic Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .course-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def count_files_by_extension(directory, extension):
    """Count files with specific extension in directory"""
    try:
        return len(list(Path(directory).rglob(f"*.{extension}")))
    except:
        return 0

def get_file_content(file_path):
    """Read file content safely"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return "Unable to read file content"

def render_markdown_with_latex(content):
    """Render markdown content with LaTeX support"""
    # Basic LaTeX rendering for display math
    content = re.sub(r'\$\$(.*?)\$\$', r'$$\1$$', content, flags=re.DOTALL)
    st.markdown(content, unsafe_allow_html=True)

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Matthew Tonks - Academic Portfolio</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Portfolio overview metrics
    repo_path = Path("../")
    
    # Calculate metrics
    total_md_files = count_files_by_extension(repo_path, "md")
    total_notebooks = count_files_by_extension(repo_path, "ipynb")
    total_r_files = count_files_by_extension(repo_path, "Rmd")
    
    # Main navigation
    page = st.sidebar.selectbox(
        "Select Section",
        ["Portfolio Overview", "CPSC 540 - Statistical ML", "BUS 659 - ML for Managers", 
         "BUS 671 - Data Management", "BUS 672 - Data Visualization", "Personal Story Portfolio"]
    )
    
    if page == "Portfolio Overview":
        show_portfolio_overview(total_md_files, total_notebooks, total_r_files)
    elif page == "CPSC 540 - Statistical ML":
        show_cpsc_540()
    elif page == "BUS 659 - ML for Managers":
        show_bus_659()
    elif page == "BUS 671 - Data Management":
        show_bus_671()
    elif page == "BUS 672 - Data Visualization":
        show_bus_672()
    elif page == "Personal Story Portfolio":
        show_personal_story()

def show_portfolio_overview(md_files, notebooks, r_files):
    st.header("Academic Portfolio Overview")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Markdown Files", md_files)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Jupyter Notebooks", notebooks)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("R Markdown Files", r_files)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Active Courses", 4)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Course overview
    st.subheader("Course Portfolio")
    
    courses = [
        {
            "name": "CPSC 540 - Statistical Machine Learning I",
            "description": "Advanced statistical ML theory with comprehensive mathematical foundations, Bayesian inference, and modern applications.",
            "highlights": ["Mathematical Rigor", "Bayesian Methods", "Advanced Topics", "LaTeX Documentation"],
            "files": "50+ comprehensive notes"
        },
        {
            "name": "BUS 659 - Machine Learning for Managers",
            "description": "Business-focused ML applications with practical implementation in Python and R for managerial decision-making.",
            "highlights": ["Business Applications", "ROI Analysis", "Python/R Implementation", "No-Code Background"],
            "files": "5 comprehensive topics"
        },
        {
            "name": "BUS 671 - Managing Data for Analysis",
            "description": "Database systems and data pipeline management focusing on SQL and modern data storage solutions.",
            "highlights": ["SQL Mastery", "ETL Processes", "NoSQL Systems", "Data Architecture"],
            "files": "SQL fundamentals + advanced"
        },
        {
            "name": "BUS 672 - Data Visualization for Business",
            "description": "Professional data visualization and presentation skills with focus on business storytelling and audience engagement.",
            "highlights": ["Business Storytelling", "Tableau/ggplot", "Presentation Skills", "Hero's Journey Framework"],
            "files": "Personal story + visualization theory"
        }
    ]
    
    for course in courses:
        st.markdown(f"""
        <div class="course-card">
            <h3>{course['name']}</h3>
            <p>{course['description']}</p>
            <p><strong>Key Highlights:</strong> {' • '.join(course['highlights'])}</p>
            <p><strong>Content:</strong> {course['files']}</p>
        </div>
        """, unsafe_allow_html=True)

def show_cpsc_540():
    st.header("CPSC 540 - Statistical Machine Learning I")
    
    st.markdown("""
    **Advanced graduate course in statistical machine learning theory and application**
    
    This course covers comprehensive mathematical foundations with proper LaTeX notation, 
    Bayesian and Frequentist inference, and advanced topics beyond standard ML curricula.
    """)
    
    # Topic selection
    topic_type = st.selectbox(
        "Select Content Type",
        ["Background Knowledge", "Class Notes", "Advanced Topics", "Assignments"]
    )
    
    if topic_type == "Background Knowledge":
        st.subheader("Mathematical Foundations")
        
        topics = {
            "Linear Algebra": "../CPSC 540/Assisted Notes/00-Background-Linear-Algebra.md",
            "Probability Theory": "../CPSC 540/Assisted Notes/00-Background-Probability-Theory.md", 
            "Statistics Review": "../CPSC 540/Assisted Notes/00-Background-Statistics-Review.md"
        }
        
        selected_topic = st.selectbox("Choose Background Topic", list(topics.keys()))
        
        if st.button("Load Content"):
            content = get_file_content(topics[selected_topic])
            if content != "Unable to read file content":
                st.markdown("### " + selected_topic)
                render_markdown_with_latex(content[:5000] + "..." if len(content) > 5000 else content)
            else:
                st.error("Could not load content")
    
    elif topic_type == "Advanced Topics":
        st.subheader("Advanced Applications")
        
        advanced_topics = {
            "Deep Learning Fundamentals": "Comprehensive neural network theory with business applications",
            "Natural Language Processing": "LLMs and business text analysis applications", 
            "Computer Vision": "Image analysis for business intelligence",
            "Reinforcement Learning": "Business decision optimization and strategy",
            "Time Series Forecasting": "Business forecasting and trend analysis",
            "Ensemble Methods": "Model optimization and performance enhancement"
        }
        
        for topic, description in advanced_topics.items():
            with st.expander(topic):
                st.write(description)
                st.info("Click to explore comprehensive coverage with mathematical rigor and business applications")

def show_bus_659():
    st.header("BUS 659 - Machine Learning for Managers")
    
    st.markdown("""
    **Business-focused machine learning without heavy technical prerequisites**
    
    Designed for managers to understand when and how ML models contribute to business objectives.
    Includes practical implementation in Python with comprehensive business ROI analysis.
    """)
    
    # Enhanced showcase with actual content
    topics = {
        "Topic 1: Uncertainty & Bias-Variance": {
            "description": "Understanding model uncertainty and the fundamental bias-variance tradeoff in business contexts",
            "business_value": "Risk assessment, model selection, performance optimization",
            "implementation": "Python scikit-learn with business case studies"
        },
        "Topic 2: Linear Regression for Business": {
            "description": "Linear regression applications with focus on business interpretation and ROI analysis",
            "business_value": "Sales forecasting, pricing optimization, resource allocation",
            "implementation": "statsmodels, sklearn with financial modeling"
        },
        "Topic 3: Classification Models": {
            "description": "Logistic regression and classification for business decision-making",
            "business_value": "Customer segmentation, churn prediction, risk classification",
            "implementation": "Binary and multi-class classification with business metrics"
        },
        "Topic 4: Decision Trees & Random Forests": {
            "description": "Tree-based models for interpretable business decision-making",
            "business_value": "Rule-based decisions, feature importance, explainable AI",
            "implementation": "sklearn with business rule extraction"
        },
        "Topic 5: Regularization & Feature Selection": {
            "description": "Lasso, Ridge regression for robust business models",
            "business_value": "Model simplification, cost reduction, key driver identification",
            "implementation": "glmnet-style regularization with business interpretation"
        }
    }
    
    selected_topic = st.selectbox("Select Topic", list(topics.keys()))
    
    topic_info = topics[selected_topic]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Description")
        st.write(topic_info["description"])
        
        st.subheader("Business Value")
        st.success(topic_info["business_value"])
    
    with col2:
        st.subheader("Implementation")
        st.info(topic_info["implementation"])
        
        if st.button("View Python Implementation"):
            st.code(f"""
# {selected_topic} - Business Implementation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Load business dataset
data = pd.read_csv('business_data.csv')

# Business-focused feature engineering
X = data[['revenue', 'customers', 'market_share', 'satisfaction']]
y = data['success_metric']

# Train-test split with business logic
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Model implementation with business interpretation
# (Specific implementation varies by topic)

# Business ROI analysis
roi_improvement = calculate_business_roi(model_predictions)
print(f"Projected ROI improvement: {roi_improvement:.1%}")
            """, language="python")

def show_bus_671():
    st.header("BUS 671 - Managing Data for Analysis")
    
    st.markdown("""
    **Database systems and data pipeline management for modern business**
    
    Covers SQL fundamentals through advanced data retrieval, ETL processes, and NoSQL systems
    for enterprise data management.
    """)
    
    # SQL showcase with examples
    st.subheader("SQL Mastery Portfolio")
    
    sql_examples = {
        "Basic Data Retrieval": """
-- Customer analysis with business metrics
SELECT 
    customer_id,
    SUM(order_value) as total_revenue,
    COUNT(*) as order_count,
    AVG(order_value) as avg_order_value,
    MAX(order_date) as last_order_date
FROM orders 
WHERE order_date >= '2024-01-01'
GROUP BY customer_id
HAVING total_revenue > 1000
ORDER BY total_revenue DESC;
        """,
        "Advanced Joins & Subqueries": """
-- Customer lifetime value analysis
WITH customer_metrics AS (
    SELECT 
        c.customer_id,
        c.acquisition_date,
        SUM(o.order_value) as lifetime_value,
        COUNT(o.order_id) as total_orders,
        DATEDIFF(MAX(o.order_date), c.acquisition_date) as customer_lifespan
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_id, c.acquisition_date
)
SELECT 
    CASE 
        WHEN lifetime_value > 5000 THEN 'High Value'
        WHEN lifetime_value > 1000 THEN 'Medium Value'
        ELSE 'Low Value'
    END as customer_segment,
    COUNT(*) as customer_count,
    AVG(lifetime_value) as avg_lifetime_value
FROM customer_metrics
GROUP BY customer_segment;
        """,
        "Business Intelligence Queries": """
-- Monthly revenue trend with YoY comparison
SELECT 
    DATE_FORMAT(order_date, '%Y-%m') as month,
    SUM(order_value) as monthly_revenue,
    LAG(SUM(order_value), 12) OVER (ORDER BY DATE_FORMAT(order_date, '%Y-%m')) as prev_year_revenue,
    ROUND(
        (SUM(order_value) - LAG(SUM(order_value), 12) OVER (ORDER BY DATE_FORMAT(order_date, '%Y-%m'))) 
        / LAG(SUM(order_value), 12) OVER (ORDER BY DATE_FORMAT(order_date, '%Y-%m')) * 100, 2
    ) as yoy_growth_percent
FROM orders 
GROUP BY DATE_FORMAT(order_date, '%Y-%m')
ORDER BY month;
        """
    }
    
    selected_sql = st.selectbox("Select SQL Example", list(sql_examples.keys()))
    
    st.subheader(f"Example: {selected_sql}")
    st.code(sql_examples[selected_sql], language="sql")
    
    # NoSQL and modern data systems
    st.subheader("Modern Data Systems")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **NoSQL Implementations**
        - MongoDB document storage for customer profiles
        - Redis caching for real-time analytics
        - Cassandra for time-series business metrics
        """)
    
    with col2:
        st.markdown("""
        **ETL Pipeline Architecture**
        - Data extraction from multiple business systems
        - Transformation with business rule validation
        - Loading into enterprise data warehouse
        """)

def show_bus_672():
    st.header("BUS 672 - Data Visualization for Business")
    
    st.markdown("""
    **Professional data visualization and business presentation skills**
    
    Focus on creating business-ready visualizations, understanding audience needs, 
    and developing compelling data stories for executive presentations.
    """)
    
    # Visualization examples
    st.subheader("Visualization Portfolio")
    
    # Sample data for demonstration
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Executive Dashboard", "Analytical Deep-Dive", "Storytelling Narrative"]
    )
    
    if viz_type == "Executive Dashboard":
        st.subheader("Executive Dashboard Example")
        
        # Sample business metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Revenue", "$2.4M", "12.5%")
        with col2:
            st.metric("Customers", "1,247", "8.2%")
        with col3:
            st.metric("Conversion", "3.2%", "-0.3%")
        with col4:
            st.metric("AOV", "$157", "5.1%")
        
        # Sample chart
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        revenue = [180, 195, 220, 240, 235, 260]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(months, revenue, marker='o', linewidth=3, color='#1f77b4')
        ax.set_title('Monthly Revenue Trend', fontsize=14, fontweight='bold')
        ax.set_ylabel('Revenue ($K)')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    elif viz_type == "Analytical Deep-Dive":
        st.subheader("Customer Segmentation Analysis")
        
        # Sample segmentation data
        np.random.seed(42)
        data = {
            'High Value': np.random.normal(250, 50, 100),
            'Medium Value': np.random.normal(150, 30, 200), 
            'Low Value': np.random.normal(75, 20, 300)
        }
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Distribution plot
        for segment, values in data.items():
            ax1.hist(values, alpha=0.7, label=segment, bins=20)
        ax1.set_title('Customer Value Distribution by Segment')
        ax1.set_xlabel('Customer Lifetime Value ($)')
        ax1.legend()
        
        # Box plot
        ax2.boxplot([data['High Value'], data['Medium Value'], data['Low Value']], 
                   labels=['High', 'Medium', 'Low'])
        ax2.set_title('Value Distribution Comparison')
        ax2.set_ylabel('Customer Lifetime Value ($)')
        
        st.pyplot(fig)

def show_personal_story():
    st.header("Personal Story Portfolio - Hero's Journey Framework")
    
    st.markdown("""
    **Professional storytelling development for business interviews**
    
    Comprehensive development of a personal story using the hero's journey framework,
    with detailed scoring rubrics and optimization for interview presentations.
    """)
    
    # Story versions
    story_version = st.selectbox(
        "Select Story Version",
        ["Final Optimized Version (92/100)", "500-Word Interview Version", "Memory Guide", "Scoring Analysis"]
    )
    
    if story_version == "Final Optimized Version (92/100)":
        st.subheader("Final Story - Interview Ready")
        
        story_content = """
**Opening Hook:**
Hi, my name is Matthew Tonks. Throughout my career, I realized that the more I succeed with technology, the less I need to rely on it. This created the opportunity to ask better questions and to have the courage to search for answers when I don't have them.

**The Challenge:**
My first role out of college was at a small consulting firm doing data pipeline implementations. My first months were hands on training with a tool called Informatica.

As training wrapped up, I was preparing for my first project interview when a huge red flag appeared. I saw the resume that was being sent to the client. It described someone who had been working for years at a high level with these tools. I was being sent headfirst into a project with barely 2 months of training on a tool I didn't know existed until after I joined.

**The Decision:**
I had a decision to make. I wanted the experience desperately since it was my first job and I wanted to succeed more than anything.

Pressure makes diamonds.

**The Heroic Actions:**
But here's where something clicked. Instead of grinding through more 16-hour days, I needed to work differently.

First, I took ownership of all technical aspects. I couldn't rely on our QA person, so I had to ensure everything was bulletproof for deployment.

Then I did something that probably saved our contract: I started communicating directly with the client's team. No more hiding behind project management. I talked to their developers, hosted knowledge transfer sessions, and got usable feedback.

**The Triumph:**
The breakthrough came when the client realized the major improvements our framework would provide and how we completely covered their file processing needs.

The demo went better than we thought possible. But more importantly, I figured out how to navigate impossible projects. After this, I knew my career would not be defined by technical skills, but by my adaptability and my ability to learn.
        """
        
        st.markdown(story_content)
        
        # Scoring breakdown
        st.subheader("Story Score: 92/100")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Content Quality: 95/100**
            - Hero's journey completion: 9.5/10
            - Professional growth: 10/10
            - Business relevance: 10/10
            - Learning articulation: 9/10
            """)
        
        with col2:
            st.markdown("""
            **Professional Presentation: 90/100**
            - Technical writing: 9/10
            - Professional tone: 10/10
            - Memorability: 9.5/10
            - Interview readiness: 9.5/10
            """)
    
    elif story_version == "Memory Guide":
        st.subheader("10-Section Memory Structure")
        
        sections = [
            ("The Hook", "Technology paradox sets transformation theme", "Success with tech = less reliance"),
            ("The Setup", "Establish challenge and context", "First job = Informatica training = college difficulty"),
            ("The Crisis", "The inflection point - everything changes", "Resume lies = 2 months vs years of experience"),
            ("The Theme", "Memorable quote", "Pressure makes diamonds"),
            ("The Challenge", "Impossible project conditions", "2-person team = PM + useless QA"),
            ("The Hole", "Everything working against success", "24 hours meetings + 16-hour days = exhausted"),
            ("The Breakthrough", "The turning point", "Something clicked = work differently"),
            ("The Action", "Specific heroic actions", "Direct client communication = saved contract"),
            ("The Success", "Client breakthrough", "Client realized improvements = breakthrough"),
            ("The Learning", "Personal growth and insight", "Career = adaptability + learning, not tech skills")
        ]
        
        for i, (title, description, trigger) in enumerate(sections, 1):
            with st.expander(f"Section {i}: {title}"):
                st.write(f"**Key Message**: {description}")
                st.write(f"**Memory Trigger**: _{trigger}_")
                st.progress(i/10)

# Run the app
if __name__ == "__main__":
    main()