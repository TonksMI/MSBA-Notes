"""
Matthew Tonks - Academic Portfolio Showcase
Interactive Streamlit application showcasing academic excellence in ML, analytics, and data science
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Page configuration
st.set_page_config(
    page_title="Matthew Tonks - Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-blue: #1f77b4;
        --secondary-blue: #4a90e2;
        --accent-green: #2ecc71;
        --accent-red: #e74c3c;
        --light-bg: #f8f9fa;
        --dark-text: #2c3e50;
    }

    /* Header styling */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        color: var(--primary-blue);
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    /* Card components */
    .custom-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 6px solid var(--primary-blue);
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }

    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }

    .metric-box {
        background: linear-gradient(135deg, #e8f4fd 0%, #d4e9ff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.1);
        border: 2px solid #b3d9ff;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: var(--primary-blue);
        margin: 0.5rem 0;
    }

    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Story section */
    .story-card {
        background: linear-gradient(135deg, #fff5f5 0%, #ffe5e5 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 6px solid var(--accent-red);
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Code blocks */
    .stCodeBlock {
        border-radius: 10px;
        border: 1px solid #e1e4e8;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        border: 2px solid var(--primary-blue);
        background-color: var(--primary-blue);
        color: white;
        font-weight: 600;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background-color: var(--secondary-blue);
        border-color: var(--secondary-blue);
        transform: scale(1.05);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        background-color: #f0f0f0;
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background-color: var(--light-bg);
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def create_metric_card(label, value, delta=None, icon="📊"):
    """Create a custom metric card"""
    delta_html = f'<div style="color: #2ecc71; font-size: 1rem; margin-top: 0.5rem;">{delta}</div>' if delta else ''

    return f"""
    <div class="metric-box">
        <div style="font-size: 2rem;">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """

def render_course_card(title, description, highlights, metrics, icon="🎓"):
    """Render a course showcase card"""
    highlights_html = " • ".join(highlights)

    return f"""
    <div class="custom-card">
        <h3>{icon} {title}</h3>
        <p style="font-size: 1.1rem; color: #555; margin: 1rem 0;">{description}</p>
        <p style="margin: 0.5rem 0;"><strong>🎯 Key Strengths:</strong> {highlights_html}</p>
    </div>
    """

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">Matthew Tonks</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">📊 Academic Portfolio | Statistical Machine Learning & Business Analytics</p>', unsafe_allow_html=True)

    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 🎯 Navigation")
        st.markdown("---")

        page = st.radio(
            "Select Section",
            [
                "🏠 Overview",
                "🔬 CPSC 540 - Statistical ML",
                "💼 BUS 659 - ML for Managers",
                "🗄️ BUS 671 - Data Management",
                "📈 BUS 672 - Visualization",
                "🎭 Personal Story",
                "🐍 Code Showcase",
                "📊 Interactive Demos"
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### 📈 Quick Stats")
        st.metric("Courses", "4", help="Active graduate courses")
        st.metric("Notes", "50+", help="Comprehensive documentation pages")
        st.metric("Story Score", "92/100", help="Professional interview story")

    # Route to pages
    if page == "🏠 Overview":
        show_overview()
    elif page == "🔬 CPSC 540 - Statistical ML":
        show_cpsc540()
    elif page == "💼 BUS 659 - ML for Managers":
        show_bus659()
    elif page == "🗄️ BUS 671 - Data Management":
        show_bus671()
    elif page == "📈 BUS 672 - Visualization":
        show_bus672()
    elif page == "🎭 Personal Story":
        show_personal_story()
    elif page == "🐍 Code Showcase":
        show_code_showcase()
    elif page == "📊 Interactive Demos":
        show_interactive_demos()

def show_overview():
    """Portfolio overview page"""
    st.header("🎓 Academic Excellence Portfolio")

    st.markdown("""
    Welcome to my comprehensive academic portfolio showcasing advanced coursework in **statistical machine learning**,
    **business analytics**, and **data science applications**. This portfolio demonstrates both theoretical mastery
    and practical implementation across multiple domains.
    """)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(create_metric_card("Documentation", "150+", "Markdown & Notes", "📄"), unsafe_allow_html=True)

    with col2:
        st.markdown(create_metric_card("Notebooks", "30+", "Python & R", "📓"), unsafe_allow_html=True)

    with col3:
        st.markdown(create_metric_card("Active Courses", "4", "Graduate Level", "🎯"), unsafe_allow_html=True)

    with col4:
        st.markdown(create_metric_card("Story Score", "92/100", "+19 improvement", "🎭"), unsafe_allow_html=True)

    st.markdown("---")

    # Course showcase
    st.subheader("🎓 Course Portfolio")

    courses = [
        {
            "title": "CPSC 540 - Statistical Machine Learning I",
            "description": "Advanced graduate course with rigorous mathematical foundations, Bayesian inference, and cutting-edge ML applications with comprehensive LaTeX documentation.",
            "highlights": ["Mathematical Rigor", "Bayesian Methods", "Advanced Topics", "LaTeX Documentation"],
            "metrics": {"Notes": "50+ pages", "Topics": "12 advanced", "Framework": "Bayesian + Frequentist"},
            "icon": "🔬"
        },
        {
            "title": "BUS 659 - Machine Learning for Managers",
            "description": "Business-focused ML applications designed for non-technical managers with emphasis on ROI analysis and practical Python implementation.",
            "highlights": ["Business ROI", "No-Code Background", "Python Implementation", "Cost-Benefit Analysis"],
            "metrics": {"Topics": "5 comprehensive", "Focus": "Business value", "Tools": "Python + R"},
            "icon": "💼"
        },
        {
            "title": "BUS 671 - Managing Data for Analysis",
            "description": "Enterprise data management covering SQL mastery, ETL processes, and modern data architecture for business intelligence.",
            "highlights": ["SQL Expertise", "ETL Pipelines", "NoSQL Systems", "Cloud Architecture"],
            "metrics": {"Skills": "SQL advanced", "Systems": "Multiple DB", "Tools": "Modern stack"},
            "icon": "🗄️"
        },
        {
            "title": "BUS 672 - Data Visualization for Business",
            "description": "Professional visualization and storytelling for business presentations with personal story development using hero's journey framework.",
            "highlights": ["Storytelling", "Executive Dashboards", "Presentation Skills", "Hero's Journey"],
            "metrics": {"Story": "92/100", "Framework": "Hero's journey", "Focus": "Business comm"},
            "icon": "📈"
        }
    ]

    for course in courses:
        st.markdown(render_course_card(
            course["title"],
            course["description"],
            course["highlights"],
            course["metrics"],
            course["icon"]
        ), unsafe_allow_html=True)

        # Display metrics
        cols = st.columns(len(course["metrics"]))
        for i, (key, value) in enumerate(course["metrics"].items()):
            with cols[i]:
                st.metric(key, value)

        st.markdown("<br>", unsafe_allow_html=True)

    # Value proposition
    st.markdown("---")
    st.subheader("💡 Key Strengths")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **🔬 Academic Rigor**
        - Mathematical foundations with LaTeX
        - Advanced statistical theory
        - Comprehensive documentation
        - Beyond standard curricula
        """)

    with col2:
        st.markdown("""
        **💼 Business Focus**
        - ROI analysis for every technique
        - Non-technical communication
        - Practical implementations
        - Real-world applications
        """)

    with col3:
        st.markdown("""
        **🚀 Technical Skills**
        - Python & R proficiency
        - SQL & database systems
        - Data visualization tools
        - Modern ML frameworks
        """)

def show_cpsc540():
    """CPSC 540 course page"""
    st.header("🔬 CPSC 540 - Statistical Machine Learning I")

    st.markdown("""
    **Advanced graduate course in statistical machine learning theory and rigorous mathematical foundations**

    This course represents the pinnacle of academic rigor in ML, covering comprehensive mathematical foundations,
    Bayesian and Frequentist inference, and advanced topics beyond standard curricula.
    """)

    tabs = st.tabs(["📚 Foundations", "📖 Class Notes", "🚀 Advanced Topics", "📝 Resources"])

    with tabs[0]:
        st.subheader("🧮 Mathematical Foundations")

        foundations = {
            "Linear Algebra": {
                "topics": ["Vector Spaces & Norms", "Matrix Operations", "Eigendecomposition", "SVD", "Quadratic Forms"],
                "business": "PCA, recommendation systems, portfolio optimization",
                "depth": "Complete proofs with LaTeX notation"
            },
            "Probability Theory": {
                "topics": ["Distributions", "Bayesian Inference", "Maximum Likelihood", "Information Theory", "Stochastic Processes"],
                "business": "Risk modeling, A/B testing, fraud detection",
                "depth": "Advanced probability with business applications"
            },
            "Statistics Review": {
                "topics": ["Hypothesis Testing", "Confidence Intervals", "Regression", "Non-parametric Methods", "Bootstrap"],
                "business": "Statistical significance, causal inference",
                "depth": "Modern statistical methods for data science"
            }
        }

        selected = st.selectbox("Choose Foundation Topic", list(foundations.keys()))
        info = foundations[selected]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**🎯 Key Topics:**")
            for topic in info['topics']:
                st.markdown(f"• {topic}")

            st.markdown(f"\n**📊 Depth:** {info['depth']}")

        with col2:
            st.markdown(f"**💼 Business Applications:**")
            st.info(info['business'])

            st.markdown("**📄 Documentation:**")
            st.success("Comprehensive notes with proper LaTeX mathematical notation")

    with tabs[1]:
        st.subheader("📖 Class-by-Class Notes")

        st.markdown("""
        Sequential notes following the complete course curriculum with rigorous mathematical treatment.
        Each class builds upon previous foundations with comprehensive examples and derivations.
        """)

        class_topics = [
            "Class 00: Course Philosophy",
            "Class 01: Mathematical Review & ML Introduction",
            "Class 03: Probability Theory & Graph Theory",
            "Class 04: Generalized Linear Models",
            "Class 10: MCMC & Monte Carlo Methods",
            "Class 12: Causal Inference with DAGs"
        ]

        for topic in class_topics:
            with st.expander(topic):
                st.write("Comprehensive coverage with mathematical derivations and business applications")
                st.code("# LaTeX mathematical notation throughout", language="latex")

    with tabs[2]:
        st.subheader("🚀 Advanced Applications")

        advanced = {
            "Deep Learning": {"roi": "60% cost reduction in customer service", "tech": "Backpropagation, CNNs, RNNs"},
            "NLP & LLMs": {"roi": "80% time reduction in document processing", "tech": "Transformers, BERT, GPT"},
            "Computer Vision": {"roi": "40% defect reduction in manufacturing", "tech": "Classification, Detection, Segmentation"},
            "Reinforcement Learning": {"roi": "15% revenue increase in dynamic pricing", "tech": "Q-Learning, Policy Gradients"},
            "Time Series": {"roi": "25% inventory reduction", "tech": "ARIMA, State Space, Neural Forecasting"},
            "Ensemble Methods": {"roi": "20% accuracy improvement", "tech": "Random Forests, Gradient Boosting, Stacking"}
        }

        cols = st.columns(2)
        for i, (topic, details) in enumerate(advanced.items()):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="custom-card">
                    <h4>🎯 {topic}</h4>
                    <p><strong>ROI Example:</strong> {details['roi']}</p>
                    <p><strong>Techniques:</strong> {details['tech']}</p>
                </div>
                """, unsafe_allow_html=True)

    with tabs[3]:
        st.subheader("📝 Learning Resources")

        st.markdown("""
        **📚 Comprehensive Documentation:**
        - 50+ pages of mathematical foundations
        - Proper LaTeX notation throughout
        - Business applications for every concept
        - Advanced topics beyond standard curricula

        **🔧 Implementation Focus:**
        - Python implementations with sklearn, scipy
        - R implementations with modern packages
        - Bayesian methods with Stan and brms
        - Visualization with matplotlib and ggplot
        """)

def show_bus659():
    """BUS 659 course page"""
    st.header("💼 BUS 659 - Machine Learning for Managers")

    st.markdown("""
    **Business-focused machine learning designed for managers without technical backgrounds**

    This course bridges ML concepts and business applications with comprehensive ROI analysis.
    """)

    topics = {
        "Uncertainty & Bias-Variance": {
            "description": "Understanding model uncertainty and the bias-variance tradeoff",
            "value": "Risk assessment, model selection, performance optimization",
            "roi": "Model accuracy +15%, Decision confidence +25%, Risk reduction 30%",
            "tools": "Python scikit-learn, cross-validation, business metrics"
        },
        "Linear Regression": {
            "description": "Predictive modeling with business interpretation",
            "value": "Sales forecasting, pricing optimization, resource allocation",
            "roi": "Forecasting accuracy +20%, Pricing revenue +12%, Resource efficiency +18%",
            "tools": "statsmodels, sklearn, financial modeling"
        },
        "Classification Models": {
            "description": "Customer segmentation and risk assessment",
            "value": "Churn prediction, customer targeting, risk classification",
            "roi": "Churn reduction 25%, Targeting efficiency +30%, Risk accuracy +40%",
            "tools": "Logistic regression, classification metrics, confusion matrices"
        },
        "Decision Trees & Random Forests": {
            "description": "Interpretable business decision-making",
            "value": "Explainable AI, automated rules, feature importance",
            "roi": "Decision automation 60% time savings, Rule clarity +80%",
            "tools": "sklearn decision trees, rule extraction, business logic"
        },
        "Regularization": {
            "description": "Robust models with feature selection",
            "value": "Model simplification, cost reduction, key driver ID",
            "roi": "Features reduced 50%, Costs down 35%, Key drivers identified",
            "tools": "Lasso, Ridge, ElasticNet, feature selection"
        }
    }

    selected = st.selectbox("🎯 Select Business Topic", list(topics.keys()))
    info = topics[selected]

    st.markdown(f"""
    <div class="custom-card">
        <h3>📊 {selected}</h3>
        <p style="font-size: 1.1rem;">{info['description']}</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**💼 Business Value:**")
        st.info(info['value'])

        st.markdown("**🔧 Tools & Implementation:**")
        st.code(info['tools'], language="text")

    with col2:
        st.markdown("**📈 ROI Metrics:**")
        st.success(info['roi'])

        if st.button("🐍 View Python Implementation"):
            st.code(f"""
# {selected} - Business Implementation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load business data
data = pd.read_csv('business_data.csv')

# Business-focused features
X = data[['revenue', 'customers', 'satisfaction']]
y = data['success_metric']

# Model training with business interpretation
# ... specific implementation ...

# Business ROI calculation
roi_improvement = calculate_business_roi(predictions)
print(f"ROI improvement: {{roi_improvement:.1%}}")
            """, language="python")

def show_bus671():
    """BUS 671 course page"""
    st.header("🗄️ BUS 671 - Managing Data for Analysis")

    st.markdown("""
    **Enterprise database systems and data pipeline management**

    Comprehensive coverage of SQL, ETL processes, and modern data architecture.
    """)

    tab1, tab2, tab3 = st.tabs(["💾 SQL Expertise", "🔄 ETL Pipelines", "☁️ Modern Systems"])

    with tab1:
        st.subheader("SQL Mastery Portfolio")

        sql_type = st.radio(
            "Select SQL Complexity",
            ["Basic Queries", "Advanced Analytics", "Business Intelligence"],
            horizontal=True
        )

        if sql_type == "Basic Queries":
            st.code("""
-- Customer Revenue Analysis
SELECT
    customer_id,
    company_name,
    SUM(order_value) as lifetime_value,
    COUNT(*) as total_orders,
    AVG(order_value) as avg_order_value,
    MAX(order_date) as last_purchase
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE order_date >= '2024-01-01'
GROUP BY customer_id, company_name
HAVING lifetime_value > 10000
ORDER BY lifetime_value DESC;
            """, language="sql")

        elif sql_type == "Advanced Analytics":
            st.code("""
-- Year-over-Year Growth Analysis with Window Functions
WITH monthly_revenue AS (
    SELECT
        DATE_FORMAT(order_date, '%Y-%m') as month,
        SUM(order_value) as revenue,
        COUNT(DISTINCT customer_id) as customers
    FROM orders
    GROUP BY DATE_FORMAT(order_date, '%Y-%m')
)
SELECT
    month,
    revenue,
    LAG(revenue, 12) OVER (ORDER BY month) as prev_year,
    ROUND(
        (revenue - LAG(revenue, 12) OVER (ORDER BY month))
        / LAG(revenue, 12) OVER (ORDER BY month) * 100, 2
    ) as yoy_growth_pct
FROM monthly_revenue
ORDER BY month;
            """, language="sql")

        else:  # Business Intelligence
            st.code("""
-- Executive Dashboard KPIs
WITH performance_metrics AS (
    SELECT
        'Current Month' as period,
        SUM(CASE WHEN MONTH(order_date) = MONTH(CURRENT_DATE)
            THEN order_value ELSE 0 END) as revenue,
        COUNT(DISTINCT customer_id) as customers
    FROM orders
    WHERE order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 12 MONTH)
)
SELECT
    period,
    CONCAT('$', FORMAT(revenue, 0)) as revenue_formatted,
    customers,
    ROUND(revenue / customers, 2) as revenue_per_customer
FROM performance_metrics;
            """, language="sql")

        st.info(f"**Business Value:** {['Customer analytics', 'Growth tracking', 'Executive reporting'][['Basic Queries', 'Advanced Analytics', 'Business Intelligence'].index(sql_type)]}")

    with tab2:
        st.subheader("ETL Pipeline Architecture")

        st.markdown("""
        **🔄 Complete Data Pipeline Design:**

        **Extract:**
        - Multi-source integration (CRM, ERP, Web Analytics)
        - API connections and file processing
        - Real-time and batch extraction

        **Transform:**
        - Data quality validation
        - Business rule application
        - Schema normalization
        - Aggregation and enrichment

        **Load:**
        - Data warehouse loading
        - Data mart distribution
        - Incremental updates
        - Error handling and logging
        """)

        # Simple ETL visualization
        fig, ax = plt.subplots(figsize=(10, 4))
        stages = ['Extract\n(Source Systems)', 'Transform\n(Business Rules)', 'Load\n(Data Warehouse)', 'Analyze\n(Business Intelligence)']
        y = [1, 1, 1, 1]
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        ax.barh(range(len(stages)), [1]*len(stages), color=colors, alpha=0.7)
        ax.set_yticks(range(len(stages)))
        ax.set_yticklabels(stages)
        ax.set_xlim(0, 1)
        ax.set_title('ETL Pipeline Flow', fontweight='bold', fontsize=14)
        ax.set_xlabel('Process Flow →')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])

        st.pyplot(fig)

    with tab3:
        st.subheader("Modern Data Systems")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **🗄️ NoSQL Systems:**
            - **MongoDB:** Document storage for customer profiles
            - **Redis:** Real-time analytics caching
            - **Cassandra:** Time-series business metrics
            - **Elasticsearch:** Full-text search and logs

            **⚡ Real-Time Processing:**
            - Apache Kafka for streaming
            - Event-driven architecture
            - Real-time dashboards
            """)

        with col2:
            st.markdown("""
            **☁️ Cloud Platforms:**
            - **AWS:** RDS, Redshift, S3 integration
            - **Google Cloud:** BigQuery analytics
            - **Azure:** Synapse data warehousing

            **📊 Analytics Tools:**
            - Tableau for visualization
            - Power BI for reporting
            - Python/R for analysis
            """)

def show_bus672():
    """BUS 672 course page"""
    st.header("📈 BUS 672 - Data Visualization for Business")

    st.markdown("""
    **Professional data visualization and business storytelling**

    Creating executive-ready visualizations and compelling data narratives.
    """)

    viz_type = st.selectbox(
        "Select Visualization Type",
        ["📊 Executive Dashboard", "🔍 Analytical Deep-Dive", "📈 Trend Analysis"]
    )

    if viz_type == "📊 Executive Dashboard":
        st.subheader("Executive Performance Dashboard")

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Revenue", "$2.4M", "12.5% ↗️")
        with col2:
            st.metric("Customers", "1,247", "8.2% ↗️")
        with col3:
            st.metric("Conversion", "3.2%", "-0.3% ↘️")
        with col4:
            st.metric("AOV", "$157", "5.1% ↗️")

        # Charts
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Revenue trend
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        revenue = [200, 215, 240, 235, 260, 280]
        target = [210, 220, 230, 240, 250, 260]

        ax1.plot(months, revenue, marker='o', linewidth=3, color='#1f77b4', label='Actual')
        ax1.plot(months, target, marker='s', linewidth=2, color='#ff7f0e', linestyle='--', label='Target')
        ax1.set_title('Revenue Performance', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Revenue ($K)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Regional performance
        regions = ['North', 'South', 'East', 'West']
        performance = [95, 102, 88, 105]

        bars = ax2.bar(regions, performance, color=['#d62728' if p < 100 else '#2ca02c' for p in performance], alpha=0.7)
        ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Target')
        ax2.set_title('Regional Performance (% of Target)', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Performance (%)')
        ax2.legend()

        for bar, perf in zip(bars, performance):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{perf}%', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)

        st.info("**Design Principles:** Clarity, actionability, executive focus, visual hierarchy")

    elif viz_type == "🔍 Analytical Deep-Dive":
        st.subheader("Customer Segmentation Analysis")

        # Generate sample data
        np.random.seed(42)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        segments = {'High Value': np.random.normal(250, 50, 100),
                   'Medium Value': np.random.normal(150, 30, 200),
                   'Low Value': np.random.normal(75, 20, 300)}

        for segment, values in segments.items():
            ax1.hist(values, alpha=0.6, label=segment, bins=20)
        ax1.set_title('Customer Value Distribution', fontweight='bold')
        ax1.set_xlabel('Lifetime Value ($)')
        ax1.set_ylabel('Count')
        ax1.legend()

        ax2.boxplot([segments['High Value'], segments['Medium Value'], segments['Low Value']],
                   labels=['High', 'Medium', 'Low'])
        ax2.set_title('Value Distribution Comparison', fontweight='bold')
        ax2.set_ylabel('Lifetime Value ($)')

        plt.tight_layout()
        st.pyplot(fig)

    else:  # Trend Analysis
        st.subheader("Business Trend Analysis")

        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=180, freq='D')
        trend = np.linspace(100, 150, 180) + np.random.normal(0, 5, 180)

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(dates, trend, linewidth=2, color='#1f77b4', alpha=0.8)
        ax.fill_between(dates, trend-10, trend+10, alpha=0.2, color='#1f77b4')
        ax.set_title('6-Month Business Trend with Confidence Band', fontweight='bold', fontsize=14)
        ax.set_xlabel('Date')
        ax.set_ylabel('Metric Value')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

def show_personal_story():
    """Personal story showcase"""
    st.header("🎭 Personal Story - Hero's Journey Framework")

    st.markdown("""
    **Interview-ready professional story with 92/100 score**

    Developed using hero's journey framework with comprehensive optimization and memory techniques.
    """)

    tabs = st.tabs(["🏆 Final Story", "📊 Score Evolution", "🧠 Memory Guide"])

    with tabs[0]:
        st.markdown("""
        <div class="story-card">
            <h4>🎯 The Complete Story</h4>
        </div>
        """, unsafe_allow_html=True)

        story = """
**Opening Hook:**
Hi, my name is Matthew Tonks. Throughout my career, I realized that the more I succeed with technology,
the less I need to rely on it. This created the opportunity to ask better questions and to have the courage
to search for answers when I don't have them.

**The Challenge:**
My first role out of college was at a small consulting firm doing data pipeline implementations. As training
wrapped up, I saw the resume being sent to the client. It described someone with years of experience.
I had barely 2 months of training on tools I didn't know existed until after I joined.

**The Decision:**
I wanted the experience desperately. It was my first job. I wanted to succeed more than anything.

**Pressure makes diamonds.**

**The Breakthrough:**
Instead of grinding through more 16-hour days, something clicked. I needed to work differently.

First, I took ownership of all technical aspects. Then I did something that probably saved our contract:
I started communicating directly with the client's team. No more hiding behind project management.

**The Triumph:**
The client realized the major improvements our framework would provide. The demo went better than we
thought possible. But more importantly, I figured out how to navigate impossible projects.

**The Growth:**
After this, I knew my career would not be defined by technical skills, but by my adaptability and
my ability to learn.
        """

        st.markdown(story)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Score", "92/100", "+19")
        with col2:
            st.metric("Time", "4-5 min", "Perfect")
        with col3:
            st.metric("Hero's Journey", "9.5/10", "Complete")
        with col4:
            st.metric("Interview Ready", "9.5/10", "Optimized")

    with tabs[1]:
        st.subheader("Story Score Evolution")

        versions = ['Original', 'Hero Added', 'Learning Enhanced', 'Polished', 'Final']
        scores = [73, 84, 88, 91, 92]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(versions, scores, color=['#d62728', '#ff7f0e', '#2ca02c', '#17becf', '#1f77b4'])
        ax.set_ylabel('Score (/100)')
        ax.set_title('Story Development Journey', fontweight='bold', fontsize=14)
        ax.set_ylim(70, 95)

        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{score}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("""
        **Improvement Areas:**
        - ✅ Complete hero's journey arc
        - ✅ Clear inflection point (resume deception)
        - ✅ Memorable quote ("Pressure makes diamonds")
        - ✅ Specific heroic actions
        - ✅ Professional growth insight
        """)

    with tabs[2]:
        st.subheader("10-Section Memory Framework")

        sections = [
            ("Hook", "Success with tech = less reliance"),
            ("Setup", "First job = Informatica training"),
            ("Crisis", "Resume lies = experience gap"),
            ("Theme", "Pressure makes diamonds"),
            ("Challenge", "2-person team + impossible deadlines"),
            ("Breakthrough", "Work differently, not harder"),
            ("Action", "Direct client communication"),
            ("Success", "Client breakthrough moment"),
            ("Growth", "Adaptability > technical skills")
        ]

        for i, (title, trigger) in enumerate(sections, 1):
            with st.expander(f"Section {i}: {title}"):
                st.markdown(f"**Memory Trigger:** _{trigger}_")
                st.progress(i/len(sections))

def show_code_showcase():
    """Code showcase page"""
    st.header("🐍 Code Showcase - Academic Work Converted to Python")

    st.markdown("""
    **Professional Python implementations of academic projects**

    Demonstrates versatility across programming languages and practical skills.
    """)

    showcases = {
        "Redfin Real Estate Analysis": {
            "description": "Advanced house price prediction using Elastic Net and Linear Regression",
            "original": "R Markdown (MGSC 310 Project)",
            "techniques": ["Elastic Net Regularization", "Cross-Validation", "Feature Engineering", "ROI Analysis"],
            "value": "Automated property valuation, investment identification, pricing optimization"
        },
        "Business Python Fundamentals": {
            "description": "Python tutorial for business managers with no coding background",
            "original": "Jupyter Notebook (BUS 659)",
            "techniques": ["Pandas Analysis", "Business Visualizations", "KPI Calculations", "ROI Modeling"],
            "value": "Data-driven decisions, automated reporting, business insights"
        }
    }

    selected = st.selectbox("Select Showcase Project", list(showcases.keys()))
    info = showcases[selected]

    st.markdown(f"""
    <div class="custom-card">
        <h3>📊 {selected}</h3>
        <p><strong>Description:</strong> {info['description']}</p>
        <p><strong>Converted from:</strong> {info['original']}</p>
        <p><strong>Business Value:</strong> {info['value']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**🔧 Techniques Applied:**")
    for tech in info['techniques']:
        st.markdown(f"• {tech}")

    with st.expander("📝 View Sample Code"):
        st.code("""
# Professional Python Implementation
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score

# Load business data
data = pd.read_csv('business_data.csv')

# Feature engineering
X = engineer_features(data)
y = data['target']

# Model with business interpretation
model = ElasticNet(alpha=0.1, l1_ratio=0.5)
scores = cross_val_score(model, X, y, cv=5)

print(f"Cross-validation score: {scores.mean():.3f}")
        """, language="python")

def show_interactive_demos():
    """Interactive demos page"""
    st.header("📊 Interactive Demos & Visualizations")

    st.markdown("**Live demonstrations of analytical techniques**")

    demo = st.selectbox(
        "Select Demo",
        ["Business Analytics Dashboard", "ML Model Comparison", "ROI Calculator"]
    )

    if demo == "Business Analytics Dashboard":
        st.subheader("Live Business Dashboard")

        np.random.seed(42)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Revenue", "$2.4M", "12.5%")
        with col2:
            st.metric("Customers", "1,247", "8.2%")
        with col3:
            st.metric("Conversion", "3.2%", "-0.3%")
        with col4:
            st.metric("AOV", "$157", "5.1%")

        dates = pd.date_range('2024-01-01', periods=12, freq='M')
        revenue = np.random.normal(500000, 50000, 12).cumsum()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(dates, revenue, marker='o', linewidth=3, color='#1f77b4')
        ax.set_title('Revenue Growth Trend', fontweight='bold', fontsize=14)
        ax.set_ylabel('Revenue ($)')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    elif demo == "ML Model Comparison":
        st.subheader("Model Performance Comparison")

        models = ['Linear\nRegression', 'Random\nForest', 'Gradient\nBoosting', 'Neural\nNetwork']
        accuracy = [0.82, 0.89, 0.91, 0.87]
        training_time = [0.5, 2.3, 5.1, 8.7]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.bar(models, accuracy, color='skyblue', alpha=0.8)
        ax1.set_title('Model Accuracy', fontweight='bold')
        ax1.set_ylabel('Accuracy Score')
        ax1.set_ylim(0, 1)

        ax2.bar(models, training_time, color='lightcoral', alpha=0.8)
        ax2.set_title('Training Time', fontweight='bold')
        ax2.set_ylabel('Time (minutes)')

        plt.tight_layout()
        st.pyplot(fig)

        st.success("**Recommendation:** Gradient Boosting offers best accuracy (91%) for critical decisions")

    else:  # ROI Calculator
        st.subheader("ML ROI Calculator")

        col1, col2 = st.columns(2)

        with col1:
            investment = st.number_input("Implementation Cost ($)", 10000, 1000000, 100000)
            time_saved = st.slider("Time Saved (%)", 0, 100, 30)
            accuracy = st.slider("Accuracy Improvement (%)", 0, 100, 20)

        with col2:
            current_cost = st.number_input("Current Process Cost ($)", 50000, 5000000, 500000)
            employees = st.number_input("Employees Affected", 1, 1000, 10)

            # Calculate ROI
            annual_savings = current_cost * (time_saved/100)
            accuracy_value = current_cost * (accuracy/100) * 0.5
            total_benefit = annual_savings + accuracy_value
            roi = ((total_benefit - investment) / investment) * 100

            st.metric("Annual Savings", f"${annual_savings:,.0f}")
            st.metric("Accuracy Value", f"${accuracy_value:,.0f}")
            st.metric("ROI", f"{roi:.1f}%")

            if roi > 100:
                st.success("🎯 Excellent ROI - Strongly recommended")
            elif roi > 50:
                st.info("💡 Good ROI - Recommended with caveats")
            else:
                st.warning("⚠️ Low ROI - Consider alternatives")

# Run the application
if __name__ == "__main__":
    main()
