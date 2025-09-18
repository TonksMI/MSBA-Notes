#!/usr/bin/env python3
"""
Inventory Optimization and Location Recommendation System

This script provides comprehensive inventory optimization across branches
and intelligent location recommendations using predictive analytics.

Features:
- Multi-branch inventory optimization
- Predictive demand forecasting
- Location-based recommendations
- Weight-based shipping optimization
- Performance analytics and visualization

Author: AI Assistant
Version: 1.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

class InventoryOptimizer:
    """
    Main class for inventory optimization and location recommendations
    """
    
    def __init__(self):
        self.inventory_data = None
        self.demand_predictor = None
        self.location_optimizer = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.optimization_results = {}
        
    def load_data(self, data_source=None):
        """
        Load inventory data from various sources
        
        Args:
            data_source: DataFrame, CSV file path, or None for sample data
        """
        if data_source is None:
            self.inventory_data = self._generate_sample_data()
        elif isinstance(data_source, str):
            self.inventory_data = pd.read_csv(data_source)
        elif isinstance(data_source, pd.DataFrame):
            self.inventory_data = data_source.copy()
        else:
            raise ValueError("Invalid data source type")
            
        print(f"Loaded {len(self.inventory_data)} inventory records")
        return self.inventory_data
    
    def _generate_sample_data(self, n_records=10000):
        """
        Generate realistic sample inventory data for demonstration
        """
        np.random.seed(42)
        
        # Generate branch and product data
        branches = [f"BR_{i:03d}" for i in range(1, 51)]  # 50 branches
        products = [f"PR_{i:04d}" for i in range(1, 501)]  # 500 products
        
        # Product categories for realistic patterns
        categories = ['Electronics', 'Clothing', 'Home', 'Sports', 'Books', 'Food']
        
        data = []
        for _ in range(n_records):
            branch = np.random.choice(branches)
            product = np.random.choice(products)
            category = np.random.choice(categories)
            
            # Weight varies by category
            base_weight = {
                'Electronics': 2.0, 'Clothing': 0.5, 'Home': 5.0,
                'Sports': 1.5, 'Books': 0.3, 'Food': 1.0
            }[category]
            
            weight = np.random.normal(base_weight, base_weight * 0.3)
            weight = max(0.1, weight)  # Minimum weight
            
            # Demand influenced by seasonality and location
            base_demand = np.random.poisson(20)
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * (datetime.now().timetuple().tm_yday / 365))
            location_factor = np.random.uniform(0.7, 1.3)
            
            current_stock = np.random.poisson(base_demand * seasonal_factor * location_factor)
            demand_forecast = int(base_demand * seasonal_factor * location_factor)
            
            # Branch coordinates for distance calculations
            branch_coords = {
                'lat': np.random.uniform(25, 49),  # US latitude range
                'lon': np.random.uniform(-125, -65)  # US longitude range
            }
            
            data.append({
                'mdm_product_key': product,
                'mdm_branch_key': branch,
                'weight': round(weight, 2),
                'category': category,
                'current_stock': current_stock,
                'demand_forecast': demand_forecast,
                'reorder_point': max(5, demand_forecast // 4),
                'max_capacity': demand_forecast * 2,
                'unit_cost': np.random.uniform(10, 200),
                'selling_price': np.random.uniform(15, 300),
                'branch_lat': round(branch_coords['lat'], 4),
                'branch_lon': round(branch_coords['lon'], 4),
                'lead_time_days': np.random.randint(1, 15),
                'last_updated': datetime.now() - timedelta(days=np.random.randint(0, 30))
            })
        
        return pd.DataFrame(data)
    
    def preprocess_data(self):
        """
        Preprocess data for machine learning models
        """
        if self.inventory_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Create features for prediction
        df = self.inventory_data.copy()
        
        # Encode categorical variables
        categorical_cols = ['category', 'mdm_branch_key', 'mdm_product_key']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
        
        # Create time-based features
        df['days_since_update'] = (datetime.now() - pd.to_datetime(df['last_updated'])).dt.days
        df['stock_turnover_ratio'] = df['demand_forecast'] / (df['current_stock'] + 1)
        df['capacity_utilization'] = df['current_stock'] / df['max_capacity']
        df['profit_margin'] = (df['selling_price'] - df['unit_cost']) / df['unit_cost']
        
        # Stock status indicators
        df['is_understocked'] = (df['current_stock'] < df['reorder_point']).astype(int)
        df['is_overstocked'] = (df['current_stock'] > df['max_capacity'] * 0.8).astype(int)
        
        self.processed_data = df
        return df
    
    def train_demand_predictor(self, target_col='demand_forecast'):
        """
        Train machine learning models to predict demand
        """
        if not hasattr(self, 'processed_data'):
            self.preprocess_data()
        
        df = self.processed_data
        
        # Feature selection for demand prediction
        feature_cols = [
            'weight', 'current_stock', 'reorder_point', 'max_capacity',
            'unit_cost', 'selling_price', 'lead_time_days', 'days_since_update',
            'stock_turnover_ratio', 'capacity_utilization', 'profit_margin',
            'category_encoded', 'mdm_branch_key_encoded', 'mdm_product_key_encoded',
            'branch_lat', 'branch_lon'
        ]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        self.demand_predictor = {}
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Evaluate model
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.demand_predictor[name] = {
                'model': model,
                'mae': mae,
                'mse': mse,
                'r2': r2,
                'feature_importance': dict(zip(feature_cols, model.feature_importances_))
            }
            
            print(f"{name} Model Performance:")
            print(f"  MAE: {mae:.2f}")
            print(f"  MSE: {mse:.2f}")
            print(f"  R²: {r2:.3f}")
            print()
        
        return self.demand_predictor
    
    def predict_demand(self, model_name='RandomForest'):
        """
        Predict demand for all products across all branches
        """
        if not hasattr(self, 'demand_predictor') or model_name not in self.demand_predictor:
            raise ValueError(f"Model {model_name} not trained. Call train_demand_predictor() first.")
        
        df = self.processed_data
        feature_cols = [
            'weight', 'current_stock', 'reorder_point', 'max_capacity',
            'unit_cost', 'selling_price', 'lead_time_days', 'days_since_update',
            'stock_turnover_ratio', 'capacity_utilization', 'profit_margin',
            'category_encoded', 'mdm_branch_key_encoded', 'mdm_product_key_encoded',
            'branch_lat', 'branch_lon'
        ]
        
        X = df[feature_cols]
        X_scaled = self.scaler.transform(X)
        
        model = self.demand_predictor[model_name]['model']
        predictions = model.predict(X_scaled)
        
        df['predicted_demand'] = predictions
        df['demand_variance'] = abs(df['predicted_demand'] - df['demand_forecast'])
        
        return df[['mdm_product_key', 'mdm_branch_key', 'predicted_demand', 'demand_variance']]
    
    def optimize_inventory_allocation(self, optimization_method='minimize_cost'):
        """
        Optimize inventory allocation across branches
        """
        if not hasattr(self, 'processed_data'):
            self.preprocess_data()
        
        df = self.processed_data
        
        # Group by product to optimize allocation
        results = []
        
        for product in df['mdm_product_key'].unique():
            product_data = df[df['mdm_product_key'] == product].copy()
            
            if optimization_method == 'minimize_cost':
                allocation = self._minimize_cost_allocation(product_data)
            elif optimization_method == 'maximize_service_level':
                allocation = self._maximize_service_level(product_data)
            else:
                allocation = self._balanced_allocation(product_data)
            
            results.extend(allocation)
        
        self.optimization_results = pd.DataFrame(results)
        return self.optimization_results
    
    def _minimize_cost_allocation(self, product_data):
        """
        Minimize total cost including holding and shortage costs
        """
        branches = product_data['mdm_branch_key'].tolist()
        demands = product_data['predicted_demand'].tolist() if 'predicted_demand' in product_data.columns else product_data['demand_forecast'].tolist()
        current_stocks = product_data['current_stock'].tolist()
        weights = product_data['weight'].tolist()
        
        total_stock = sum(current_stocks)
        total_demand = sum(demands)
        
        # If total stock exceeds demand, redistribute optimally
        if total_stock >= total_demand:
            # Allocate based on demand ratio
            allocations = [int(d * total_stock / total_demand) for d in demands]
        else:
            # Prioritize high-demand branches
            demand_priority = np.array(demands) / sum(demands)
            allocations = [int(p * total_stock) for p in demand_priority]
        
        results = []
        for i, branch in enumerate(branches):
            results.append({
                'mdm_product_key': product_data.iloc[0]['mdm_product_key'],
                'mdm_branch_key': branch,
                'current_allocation': current_stocks[i],
                'optimal_allocation': allocations[i],
                'reallocation_needed': allocations[i] - current_stocks[i],
                'weight_impact': weights[i] * abs(allocations[i] - current_stocks[i]),
                'optimization_method': 'minimize_cost'
            })
        
        return results
    
    def _maximize_service_level(self, product_data):
        """
        Maximize service level by ensuring stock availability
        """
        branches = product_data['mdm_branch_key'].tolist()
        demands = product_data['predicted_demand'].tolist() if 'predicted_demand' in product_data.columns else product_data['demand_forecast'].tolist()
        current_stocks = product_data['current_stock'].tolist()
        weights = product_data['weight'].tolist()
        reorder_points = product_data['reorder_point'].tolist()
        
        total_stock = sum(current_stocks)
        
        # Ensure each branch has at least reorder point stock
        allocations = []
        remaining_stock = total_stock
        
        # First pass: ensure minimum stock levels
        for i, (demand, reorder_point) in enumerate(zip(demands, reorder_points)):
            min_allocation = max(reorder_point, demand)
            allocation = min(min_allocation, remaining_stock)
            allocations.append(allocation)
            remaining_stock -= allocation
        
        # Second pass: distribute remaining stock based on demand
        if remaining_stock > 0:
            total_additional_demand = sum(max(0, d - a) for d, a in zip(demands, allocations))
            if total_additional_demand > 0:
                for i, (demand, current_alloc) in enumerate(zip(demands, allocations)):
                    additional_need = max(0, demand - current_alloc)
                    additional_allocation = int(remaining_stock * additional_need / total_additional_demand)
                    allocations[i] += additional_allocation
        
        results = []
        for i, branch in enumerate(branches):
            results.append({
                'mdm_product_key': product_data.iloc[0]['mdm_product_key'],
                'mdm_branch_key': branch,
                'current_allocation': current_stocks[i],
                'optimal_allocation': allocations[i],
                'reallocation_needed': allocations[i] - current_stocks[i],
                'weight_impact': weights[i] * abs(allocations[i] - current_stocks[i]),
                'optimization_method': 'maximize_service_level'
            })
        
        return results
    
    def _balanced_allocation(self, product_data):
        """
        Balanced allocation considering both cost and service level
        """
        cost_allocation = self._minimize_cost_allocation(product_data)
        service_allocation = self._maximize_service_level(product_data)
        
        # Weighted average of both approaches
        weight_cost = 0.6
        weight_service = 0.4
        
        results = []
        for i, branch in enumerate(product_data['mdm_branch_key']):
            cost_alloc = cost_allocation[i]['optimal_allocation']
            service_alloc = service_allocation[i]['optimal_allocation']
            
            balanced_alloc = int(weight_cost * cost_alloc + weight_service * service_alloc)
            
            results.append({
                'mdm_product_key': product_data.iloc[0]['mdm_product_key'],
                'mdm_branch_key': branch,
                'current_allocation': product_data.iloc[i]['current_stock'],
                'optimal_allocation': balanced_alloc,
                'reallocation_needed': balanced_alloc - product_data.iloc[i]['current_stock'],
                'weight_impact': product_data.iloc[i]['weight'] * abs(balanced_alloc - product_data.iloc[i]['current_stock']),
                'optimization_method': 'balanced'
            })
        
        return results
    
    def recommend_locations(self, product_key=None, n_recommendations=5):
        """
        Recommend optimal locations for products based on demand and logistics
        """
        if not hasattr(self, 'processed_data'):
            self.preprocess_data()
        
        df = self.processed_data
        
        if product_key:
            df = df[df['mdm_product_key'] == product_key]
        
        # Calculate location scores
        df['demand_score'] = df['predicted_demand'] if 'predicted_demand' in df.columns else df['demand_forecast']
        df['demand_score'] = (df['demand_score'] - df['demand_score'].min()) / (df['demand_score'].max() - df['demand_score'].min())
        
        df['capacity_score'] = 1 - df['capacity_utilization']
        df['profit_score'] = (df['profit_margin'] - df['profit_margin'].min()) / (df['profit_margin'].max() - df['profit_margin'].min())
        
        # Weight-based logistics score (lighter is better for shipping)
        df['logistics_score'] = 1 / (1 + df['weight'])
        df['logistics_score'] = (df['logistics_score'] - df['logistics_score'].min()) / (df['logistics_score'].max() - df['logistics_score'].min())
        
        # Combined score
        df['location_score'] = (
            0.3 * df['demand_score'] +
            0.25 * df['capacity_score'] +
            0.25 * df['profit_score'] +
            0.2 * df['logistics_score']
        )
        
        # Get top recommendations
        recommendations = df.nlargest(n_recommendations, 'location_score')[
            ['mdm_product_key', 'mdm_branch_key', 'location_score', 'demand_score', 
             'capacity_score', 'profit_score', 'logistics_score', 'weight']
        ]
        
        return recommendations
    
    def cluster_branches_by_performance(self, n_clusters=5):
        """
        Cluster branches based on performance metrics
        """
        if not hasattr(self, 'processed_data'):
            self.preprocess_data()
        
        df = self.processed_data
        
        # Aggregate metrics by branch
        branch_metrics = df.groupby('mdm_branch_key').agg({
            'demand_forecast': 'mean',
            'current_stock': 'mean',
            'capacity_utilization': 'mean',
            'stock_turnover_ratio': 'mean',
            'profit_margin': 'mean',
            'weight': 'mean',
            'branch_lat': 'first',
            'branch_lon': 'first'
        }).reset_index()
        
        # Features for clustering
        feature_cols = ['demand_forecast', 'current_stock', 'capacity_utilization', 
                       'stock_turnover_ratio', 'profit_margin', 'weight']
        
        X = branch_metrics[feature_cols]
        X_scaled = StandardScaler().fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        branch_metrics['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Analyze clusters
        cluster_summary = branch_metrics.groupby('cluster').agg({
            'demand_forecast': 'mean',
            'current_stock': 'mean',
            'capacity_utilization': 'mean',
            'stock_turnover_ratio': 'mean',
            'profit_margin': 'mean',
            'weight': 'mean',
            'mdm_branch_key': 'count'
        }).round(2)
        
        cluster_summary.columns = ['avg_demand', 'avg_stock', 'avg_capacity_util', 
                                  'avg_turnover', 'avg_profit_margin', 'avg_weight', 'branch_count']
        
        return branch_metrics, cluster_summary
    
    def calculate_shipping_optimization(self):
        """
        Optimize shipping routes based on weight and distance
        """
        if not hasattr(self, 'optimization_results'):
            self.optimize_inventory_allocation()
        
        df = self.optimization_results
        
        # Filter items that need reallocation
        reallocation_needed = df[df['reallocation_needed'] != 0].copy()
        
        # Calculate shipping priorities
        reallocation_needed['shipping_priority'] = (
            abs(reallocation_needed['reallocation_needed']) * 
            reallocation_needed['weight_impact']
        )
        
        # Sort by priority
        shipping_plan = reallocation_needed.sort_values('shipping_priority', ascending=False)
        
        return shipping_plan[['mdm_product_key', 'mdm_branch_key', 'reallocation_needed', 
                             'weight_impact', 'shipping_priority']]
    
    def generate_analytics_report(self):
        """
        Generate comprehensive analytics report
        """
        if not hasattr(self, 'processed_data'):
            self.preprocess_data()
        
        df = self.processed_data
        
        report = {
            'summary_statistics': {
                'total_products': df['mdm_product_key'].nunique(),
                'total_branches': df['mdm_branch_key'].nunique(),
                'total_inventory_value': (df['current_stock'] * df['unit_cost']).sum(),
                'average_weight_per_item': df['weight'].mean(),
                'total_weight': (df['current_stock'] * df['weight']).sum()
            },
            'inventory_health': {
                'understocked_items': df['is_understocked'].sum(),
                'overstocked_items': df['is_overstocked'].sum(),
                'optimal_items': len(df) - df['is_understocked'].sum() - df['is_overstocked'].sum(),
                'average_capacity_utilization': df['capacity_utilization'].mean()
            },
            'profitability': {
                'average_profit_margin': df['profit_margin'].mean(),
                'total_potential_revenue': (df['current_stock'] * df['selling_price']).sum(),
                'high_margin_products': (df['profit_margin'] > df['profit_margin'].quantile(0.75)).sum()
            }
        }
        
        # Model performance if available
        if hasattr(self, 'demand_predictor') and self.demand_predictor:
            best_model = max(self.demand_predictor.items(), key=lambda x: x[1]['r2'])
            report['model_performance'] = {
                'best_model': best_model[0],
                'r2_score': best_model[1]['r2'],
                'mae': best_model[1]['mae']
            }
        
        return report
    
    def visualize_results(self, save_plots=False):
        """
        Create visualizations for analysis results
        """
        if not hasattr(self, 'processed_data'):
            self.preprocess_data()
        
        df = self.processed_data
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Inventory Optimization Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Inventory Status Distribution
        status_counts = [
            df['is_understocked'].sum(),
            df['is_overstocked'].sum(),
            len(df) - df['is_understocked'].sum() - df['is_overstocked'].sum()
        ]
        axes[0, 0].pie(status_counts, labels=['Understocked', 'Overstocked', 'Optimal'], 
                       autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Inventory Status Distribution')
        
        # 2. Demand vs Current Stock
        axes[0, 1].scatter(df['demand_forecast'], df['current_stock'], alpha=0.6, c=df['weight'], cmap='viridis')
        axes[0, 1].plot([df['demand_forecast'].min(), df['demand_forecast'].max()], 
                        [df['demand_forecast'].min(), df['demand_forecast'].max()], 'r--', alpha=0.7)
        axes[0, 1].set_xlabel('Demand Forecast')
        axes[0, 1].set_ylabel('Current Stock')
        axes[0, 1].set_title('Demand vs Current Stock (colored by weight)')
        
        # 3. Weight Distribution by Category
        df.boxplot(column='weight', by='category', ax=axes[0, 2])
        axes[0, 2].set_title('Weight Distribution by Category')
        axes[0, 2].set_xlabel('Category')
        axes[0, 2].set_ylabel('Weight')
        
        # 4. Capacity Utilization
        axes[1, 0].hist(df['capacity_utilization'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(df['capacity_utilization'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["capacity_utilization"].mean():.2f}')
        axes[1, 0].set_xlabel('Capacity Utilization')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Capacity Utilization Distribution')
        axes[1, 0].legend()
        
        # 5. Profit Margin vs Stock Turnover
        axes[1, 1].scatter(df['profit_margin'], df['stock_turnover_ratio'], alpha=0.6)
        axes[1, 1].set_xlabel('Profit Margin')
        axes[1, 1].set_ylabel('Stock Turnover Ratio')
        axes[1, 1].set_title('Profit Margin vs Stock Turnover')
        
        # 6. Feature Importance (if model is trained)
        if hasattr(self, 'demand_predictor') and self.demand_predictor:
            best_model_name = max(self.demand_predictor.items(), key=lambda x: x[1]['r2'])[0]
            importance = self.demand_predictor[best_model_name]['feature_importance']
            
            # Get top 10 features
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            features, importances = zip(*top_features)
            
            axes[1, 2].barh(range(len(features)), importances)
            axes[1, 2].set_yticks(range(len(features)))
            axes[1, 2].set_yticklabels(features)
            axes[1, 2].set_xlabel('Feature Importance')
            axes[1, 2].set_title(f'Top 10 Feature Importances ({best_model_name})')
        else:
            axes[1, 2].text(0.5, 0.5, 'Train model first\nto see feature importance', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Feature Importance (Model Not Trained)')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('inventory_analysis_dashboard.png', dpi=300, bbox_inches='tight')
            print("Dashboard saved as 'inventory_analysis_dashboard.png'")
        
        plt.show()


def main():
    """
    Example usage of the Inventory Optimization System
    """
    print("Inventory Optimization and Location Recommendation System")
    print("=" * 60)
    
    # Initialize the optimizer
    optimizer = InventoryOptimizer()
    
    # Load sample data
    print("1. Loading sample data...")
    data = optimizer.load_data()
    print(f"   Loaded {len(data)} records")
    print(f"   Products: {data['mdm_product_key'].nunique()}")
    print(f"   Branches: {data['mdm_branch_key'].nunique()}")
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    processed_data = optimizer.preprocess_data()
    print("   Data preprocessing completed")
    
    # Train demand prediction models
    print("\n3. Training demand prediction models...")
    models = optimizer.train_demand_predictor()
    
    # Make predictions
    print("\n4. Making demand predictions...")
    predictions = optimizer.predict_demand()
    print(f"   Predictions generated for {len(predictions)} items")
    
    # Optimize inventory allocation
    print("\n5. Optimizing inventory allocation...")
    allocation_results = optimizer.optimize_inventory_allocation('balanced')
    print(f"   Allocation optimized for {allocation_results['mdm_product_key'].nunique()} products")
    
    # Get location recommendations
    print("\n6. Generating location recommendations...")
    recommendations = optimizer.recommend_locations(n_recommendations=10)
    print("   Top 5 location recommendations:")
    print(recommendations.head()[['mdm_product_key', 'mdm_branch_key', 'location_score']].to_string(index=False))
    
    # Cluster analysis
    print("\n7. Performing branch clustering analysis...")
    branch_clusters, cluster_summary = optimizer.cluster_branches_by_performance()
    print("   Cluster summary:")
    print(cluster_summary.to_string())
    
    # Shipping optimization
    print("\n8. Calculating shipping optimization...")
    shipping_plan = optimizer.calculate_shipping_optimization()
    if len(shipping_plan) > 0:
        print(f"   {len(shipping_plan)} items need reallocation")
        print("   Top 5 shipping priorities:")
        print(shipping_plan.head()[['mdm_product_key', 'mdm_branch_key', 'reallocation_needed']].to_string(index=False))
    else:
        print("   No reallocations needed")
    
    # Generate analytics report
    print("\n9. Generating analytics report...")
    report = optimizer.generate_analytics_report()
    print("   Analytics Summary:")
    for section, metrics in report.items():
        print(f"   {section.replace('_', ' ').title()}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"     {metric.replace('_', ' ').title()}: {value:.2f}")
            else:
                print(f"     {metric.replace('_', ' ').title()}: {value}")
    
    # Create visualizations
    print("\n10. Creating visualizations...")
    print("    Generating dashboard...")
    optimizer.visualize_results(save_plots=True)
    
    print("\nInventory optimization analysis completed!")
    
    # Return optimizer for further analysis
    return optimizer


if __name__ == "__main__":
    # Run the main analysis
    optimizer = main()
    
    # Additional example: Product-specific analysis
    print("\n" + "="*60)
    print("ADDITIONAL ANALYSIS EXAMPLES")
    print("="*60)
    
    # Example 1: Analyze specific product
    sample_product = optimizer.inventory_data['mdm_product_key'].iloc[0]
    print(f"\nExample 1: Analyzing product {sample_product}")
    product_recommendations = optimizer.recommend_locations(product_key=sample_product, n_recommendations=5)
    print("Top recommendations for this product:")
    print(product_recommendations.to_string(index=False))
    
    # Example 2: Find products that need urgent attention
    print(f"\nExample 2: Products needing urgent attention")
    urgent_items = optimizer.processed_data[
        (optimizer.processed_data['is_understocked'] == 1) | 
        (optimizer.processed_data['capacity_utilization'] > 0.9)
    ].nlargest(5, 'demand_forecast')[['mdm_product_key', 'mdm_branch_key', 'current_stock', 'demand_forecast']]
    print(urgent_items.to_string(index=False))
    
    # Example 3: Branch performance analysis
    print(f"\nExample 3: Branch performance analysis")
    branch_performance = optimizer.processed_data.groupby('mdm_branch_key').agg({
        'current_stock': 'sum',
        'demand_forecast': 'sum',
        'profit_margin': 'mean',
        'capacity_utilization': 'mean'
    }).round(2)
    
    print("Top 5 performing branches by total demand:")
    print(branch_performance.nlargest(5, 'demand_forecast').to_string())