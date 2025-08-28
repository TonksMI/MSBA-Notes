# Topic 1: SQL Basics and Data Retrieval

## Learning Objectives
By the end of this topic, students should understand:
- Fundamental database concepts and relational data models
- How to write basic SQL queries to retrieve business data
- Filtering, sorting, and formatting techniques for business analysis
- Best practices for data retrieval in business contexts

## 1. Database Fundamentals for Business

### What is a Database?
A **database** is an organized collection of structured information stored electronically. For business applications, databases serve as the foundation for:
- Customer relationship management (CRM)
- Enterprise resource planning (ERP)
- Financial reporting and analytics
- Supply chain management
- E-commerce platforms

### Relational Database Concepts

#### Tables and Relationships
**Table Structure:**
```
Customers Table:
┌─────────────┬──────────────┬─────────────┬──────────────┐
│ customer_id │ first_name   │ last_name   │ email        │
├─────────────┼──────────────┼─────────────┼──────────────┤
│ 1           │ John         │ Smith       │ j.smith@...  │
│ 2           │ Sarah        │ Johnson     │ s.johnson@...│
│ 3           │ Mike         │ Brown       │ m.brown@...  │
└─────────────┴──────────────┴─────────────┴──────────────┘
```

**Key Concepts:**
- **Primary Key**: Unique identifier (customer_id)
- **Foreign Key**: References another table's primary key
- **Row/Record**: Individual data entry (one customer)
- **Column/Field**: Data attribute (first_name, email)

#### Business Entity Relationships
```
One-to-Many: One Customer → Many Orders
Many-to-Many: Many Products ↔ Many Orders (through Order_Items)
One-to-One: One Customer → One Customer_Profile
```

### Sample Business Database Schema
```
Customers (customer_id, first_name, last_name, email, phone, registration_date)
Products (product_id, product_name, category, price, stock_quantity)
Orders (order_id, customer_id, order_date, total_amount, status)
Order_Items (order_item_id, order_id, product_id, quantity, unit_price)
```

## 2. Basic SQL Query Structure

### The SELECT Statement
**Basic Syntax:**
```sql
SELECT column1, column2, column3
FROM table_name;
```

**Business Example:**
```sql
-- Retrieve customer contact information
SELECT first_name, last_name, email, phone
FROM customers;
```

### Common Business Queries

#### Customer List
```sql
-- Get all customers with their basic information
SELECT 
    customer_id,
    first_name || ' ' || last_name AS full_name,
    email,
    registration_date
FROM customers;
```

#### Product Catalog
```sql
-- Retrieve product information for catalog
SELECT 
    product_name,
    category,
    price,
    CASE 
        WHEN stock_quantity > 0 THEN 'In Stock'
        ELSE 'Out of Stock'
    END AS availability
FROM products;
```

#### Order Summary
```sql
-- Daily order summary for management dashboard
SELECT 
    order_date,
    COUNT(*) AS total_orders,
    SUM(total_amount) AS daily_revenue
FROM orders
WHERE order_date = CURRENT_DATE
GROUP BY order_date;
```

## 3. Filtering Data with WHERE

### Basic Filtering
**Syntax:**
```sql
SELECT columns
FROM table
WHERE condition;
```

#### Comparison Operators
```sql
-- Customers who registered recently
SELECT first_name, last_name, registration_date
FROM customers
WHERE registration_date >= '2023-01-01';

-- High-value orders
SELECT order_id, customer_id, total_amount
FROM orders
WHERE total_amount > 500;

-- Specific product categories
SELECT product_name, price
FROM products
WHERE category = 'Electronics';
```

#### Pattern Matching
```sql
-- Find customers with Gmail addresses
SELECT first_name, last_name, email
FROM customers
WHERE email LIKE '%@gmail.com';

-- Products starting with 'Pro'
SELECT product_name, price
FROM products
WHERE product_name LIKE 'Pro%';

-- Phone numbers with specific area code
SELECT first_name, last_name, phone
FROM customers
WHERE phone LIKE '555-%';
```

### Advanced Filtering

#### Multiple Conditions
```sql
-- Active customers with high-value recent orders
SELECT c.first_name, c.last_name, o.order_date, o.total_amount
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_date >= '2023-01-01'
  AND o.total_amount > 1000
  AND o.status = 'Completed';
```

#### IN and NOT IN
```sql
-- Orders from specific regions
SELECT order_id, customer_id, total_amount
FROM orders
WHERE customer_id IN (
    SELECT customer_id 
    FROM customers 
    WHERE state IN ('CA', 'NY', 'TX')
);

-- Products except discontinued ones
SELECT product_name, price
FROM products
WHERE product_id NOT IN (
    SELECT product_id 
    FROM discontinued_products
);
```

#### NULL Handling
```sql
-- Customers without phone numbers
SELECT first_name, last_name, email
FROM customers
WHERE phone IS NULL;

-- Complete customer profiles only
SELECT first_name, last_name, email, phone
FROM customers
WHERE phone IS NOT NULL
  AND email IS NOT NULL;
```

## 4. Sorting and Organizing Results

### ORDER BY Clause
```sql
-- Customers by registration date (newest first)
SELECT first_name, last_name, registration_date
FROM customers
ORDER BY registration_date DESC;

-- Products by price (lowest to highest)
SELECT product_name, category, price
FROM products
ORDER BY price ASC;

-- Multiple column sorting
SELECT first_name, last_name, registration_date
FROM customers
ORDER BY last_name ASC, first_name ASC;
```

### Business Sorting Examples

#### Top Customers by Spending
```sql
SELECT 
    c.first_name,
    c.last_name,
    SUM(o.total_amount) as total_spent
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.first_name, c.last_name
ORDER BY total_spent DESC;
```

#### Best Selling Products
```sql
SELECT 
    p.product_name,
    p.category,
    SUM(oi.quantity) as total_sold
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY p.product_id, p.product_name, p.category
ORDER BY total_sold DESC;
```

## 5. Limiting and Sampling Results

### LIMIT Clause
```sql
-- Top 10 customers by spending
SELECT 
    c.first_name,
    c.last_name,
    SUM(o.total_amount) as total_spent
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.first_name, c.last_name
ORDER BY total_spent DESC
LIMIT 10;
```

### Business Applications

#### Executive Dashboard Queries
```sql
-- Today's top 5 orders
SELECT 
    order_id,
    customer_id,
    total_amount,
    order_date
FROM orders
WHERE DATE(order_date) = CURRENT_DATE
ORDER BY total_amount DESC
LIMIT 5;

-- Most popular products this month
SELECT 
    p.product_name,
    COUNT(*) as order_frequency
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
JOIN orders o ON oi.order_id = o.order_id
WHERE o.order_date >= DATE_TRUNC('month', CURRENT_DATE)
GROUP BY p.product_id, p.product_name
ORDER BY order_frequency DESC
LIMIT 10;
```

## 6. Data Formatting and Presentation

### String Functions
```sql
-- Format customer names for reports
SELECT 
    UPPER(last_name) || ', ' || first_name AS formatted_name,
    LOWER(email) as email_lowercase,
    LENGTH(phone) as phone_digits
FROM customers;

-- Clean and format phone numbers
SELECT 
    first_name,
    last_name,
    REGEXP_REPLACE(phone, '[^0-9]', '', 'g') as clean_phone
FROM customers
WHERE phone IS NOT NULL;
```

### Date Functions
```sql
-- Format dates for business reports
SELECT 
    order_id,
    TO_CHAR(order_date, 'YYYY-MM-DD') as formatted_date,
    TO_CHAR(order_date, 'Month DD, YYYY') as display_date,
    EXTRACT(YEAR FROM order_date) as order_year,
    EXTRACT(MONTH FROM order_date) as order_month
FROM orders;

-- Calculate customer tenure
SELECT 
    first_name,
    last_name,
    registration_date,
    CURRENT_DATE - registration_date as days_as_customer,
    DATE_PART('year', AGE(CURRENT_DATE, registration_date)) as years_as_customer
FROM customers;
```

### Numeric Functions
```sql
-- Format financial data for reports
SELECT 
    product_name,
    ROUND(price, 2) as formatted_price,
    CEIL(price) as price_ceiling,
    FLOOR(price) as price_floor
FROM products;

-- Calculate business metrics
SELECT 
    order_id,
    total_amount,
    ROUND(total_amount * 1.08, 2) as total_with_tax,
    CASE 
        WHEN total_amount > 1000 THEN 'Premium'
        WHEN total_amount > 500 THEN 'Standard'
        ELSE 'Basic'
    END as order_tier
FROM orders;
```

## 7. Business-Specific Query Patterns

### Customer Analysis Queries

#### New Customer Acquisition
```sql
-- Monthly new customer acquisition
SELECT 
    EXTRACT(YEAR FROM registration_date) as year,
    EXTRACT(MONTH FROM registration_date) as month,
    COUNT(*) as new_customers
FROM customers
WHERE registration_date >= '2023-01-01'
GROUP BY EXTRACT(YEAR FROM registration_date), 
         EXTRACT(MONTH FROM registration_date)
ORDER BY year, month;
```

#### Customer Segmentation
```sql
-- Segment customers by order frequency
SELECT 
    c.customer_id,
    c.first_name,
    c.last_name,
    COUNT(o.order_id) as total_orders,
    CASE 
        WHEN COUNT(o.order_id) >= 10 THEN 'Frequent Buyer'
        WHEN COUNT(o.order_id) >= 5 THEN 'Regular Customer'
        WHEN COUNT(o.order_id) >= 1 THEN 'Occasional Buyer'
        ELSE 'No Purchases'
    END as customer_segment
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.first_name, c.last_name;
```

### Sales Analysis Queries

#### Revenue Trending
```sql
-- Daily revenue for the last 30 days
SELECT 
    DATE(order_date) as sale_date,
    COUNT(*) as total_orders,
    SUM(total_amount) as daily_revenue,
    AVG(total_amount) as avg_order_value
FROM orders
WHERE order_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(order_date)
ORDER BY sale_date;
```

#### Product Performance
```sql
-- Product performance analysis
SELECT 
    p.category,
    COUNT(DISTINCT oi.order_id) as orders_containing_category,
    SUM(oi.quantity) as total_units_sold,
    SUM(oi.quantity * oi.unit_price) as category_revenue
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
JOIN orders o ON oi.order_id = o.order_id
WHERE o.order_date >= '2023-01-01'
GROUP BY p.category
ORDER BY category_revenue DESC;
```

## 8. Performance Considerations

### Writing Efficient Queries

#### Use Specific Column Names
```sql
-- Instead of this:
SELECT * FROM customers;

-- Use this:
SELECT customer_id, first_name, last_name, email FROM customers;
```

#### Leverage Indexes
```sql
-- Queries on indexed columns are faster
SELECT * FROM customers WHERE customer_id = 12345;  -- Fast (primary key)
SELECT * FROM orders WHERE order_date >= '2023-01-01';  -- Fast if indexed
SELECT * FROM customers WHERE phone LIKE '%555%';  -- Slow without index
```

#### Limit Result Sets
```sql
-- For large tables, always consider limiting results
SELECT first_name, last_name, email
FROM customers
WHERE registration_date >= '2023-01-01'
ORDER BY registration_date DESC
LIMIT 1000;  -- Don't retrieve more than needed
```

## 9. Common Business Use Cases

### Daily Operations Queries

#### Inventory Management
```sql
-- Low stock alert
SELECT 
    product_name,
    stock_quantity,
    CASE 
        WHEN stock_quantity = 0 THEN 'OUT OF STOCK'
        WHEN stock_quantity < 10 THEN 'LOW STOCK'
        ELSE 'ADEQUATE'
    END as stock_status
FROM products
WHERE stock_quantity < 20
ORDER BY stock_quantity ASC;
```

#### Order Fulfillment
```sql
-- Orders pending shipment
SELECT 
    o.order_id,
    c.first_name || ' ' || c.last_name as customer_name,
    o.order_date,
    o.total_amount,
    o.status
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.status IN ('Pending', 'Processing')
ORDER BY o.order_date ASC;
```

### Reporting and Analytics

#### Monthly Business Summary
```sql
-- Monthly summary for executive dashboard
SELECT 
    'Total Customers' as metric,
    COUNT(*) as value
FROM customers
WHERE registration_date <= CURRENT_DATE

UNION ALL

SELECT 
    'New Customers This Month' as metric,
    COUNT(*) as value
FROM customers
WHERE registration_date >= DATE_TRUNC('month', CURRENT_DATE)

UNION ALL

SELECT 
    'Total Orders This Month' as metric,
    COUNT(*) as value
FROM orders
WHERE order_date >= DATE_TRUNC('month', CURRENT_DATE);
```

## 10. Best Practices for Business Queries

### Data Quality Considerations
```sql
-- Always check for data quality issues
SELECT 
    COUNT(*) as total_records,
    COUNT(email) as records_with_email,
    COUNT(phone) as records_with_phone,
    COUNT(*) - COUNT(email) as missing_emails
FROM customers;
```

### Documentation and Comments
```sql
-- Monthly customer acquisition report
-- Generated: [Current Date]
-- Purpose: Track customer growth trends for marketing team
SELECT 
    DATE_TRUNC('month', registration_date) as month,
    COUNT(*) as new_customers,
    SUM(COUNT(*)) OVER (ORDER BY DATE_TRUNC('month', registration_date)) as cumulative_customers
FROM customers
WHERE registration_date >= '2023-01-01'
GROUP BY DATE_TRUNC('month', registration_date)
ORDER BY month;
```

### Error Handling
```sql
-- Defensive querying with NULL handling
SELECT 
    customer_id,
    COALESCE(first_name, 'Unknown') as first_name,
    COALESCE(last_name, 'Unknown') as last_name,
    COALESCE(email, 'No email provided') as email
FROM customers
WHERE customer_id IS NOT NULL;
```

## Key Takeaways

### 1. Start with Business Questions
- Always begin with "What business problem are we solving?"
- Structure queries to answer specific business needs
- Consider the audience for your query results

### 2. Write Readable Queries
- Use clear, descriptive column aliases
- Format queries consistently
- Add comments for complex logic

### 3. Consider Performance
- Limit result sets when appropriate
- Use indexed columns in WHERE clauses
- Test query performance on production-sized data

### 4. Validate Results
- Check for expected data ranges and patterns
- Verify calculations make business sense
- Cross-check critical numbers with known values

## Practice Exercises

### Exercise 1: Customer Analysis
Write queries to answer these business questions:
1. How many customers registered each month in 2023?
2. Which customers haven't placed an order in the last 6 months?
3. What's the average time between registration and first purchase?

### Exercise 2: Product Performance
Create queries for:
1. Top 10 best-selling products by revenue
2. Products that haven't sold in the last 90 days
3. Average order value by product category

### Exercise 3: Operational Reports
Build queries for daily operations:
1. Orders requiring immediate attention (high value, VIP customers)
2. Inventory items requiring reorder (low stock alerts)
3. Customer service priorities (recent high-value customers with issues)

---

**Next Topic**: [Advanced SQL Joins and Relationships](./Topic-02-Advanced-SQL-Joins-and-Relationships.md) - Learn how to combine data from multiple tables for comprehensive business analysis.