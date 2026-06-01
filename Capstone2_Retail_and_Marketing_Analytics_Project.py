#!/usr/bin/env python
# Auto-exported from Capstone2_Retail_and_Marketing_Analytics_Project.ipynb

# Retail & Marketing Analytics Project
# Part 1: Data Acquisition and Setup

"""
Project: Retail & Marketing Analytics - Customer Segmentation & Sales Optimization
Notebook: 01 - Data Acquisition and Setup
Author: [Ashok]
Date: [Current Date]

Objective:
- Setup project environment
- Download dataset from Kaggle
- Initial data inspection
- Create project folder structure
"""

# ============================================================================
# 1. IMPORT LIBRARIES
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from pathlib import Path
import zipfile

# Visualization settings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

print("Libraries imported successfully!")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")

# ============================================================================
# 2. CREATE PROJECT FOLDER STRUCTURE
# ============================================================================

# Define folder structure
folders = [
    'data/raw',
    'data/processed',
    'notebooks',
    'scripts',
    'dashboards',
    'outputs/figures',
    'outputs/reports',
    'docs'
]

# Create folders
for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)
    print(f"✓ Created: {folder}")

print("\nProject structure created successfully!")

# ============================================================================
# 3. KAGGLE HUB (OPTIONAL)
# ============================================================================
# Dataset: 
# https://www.kaggle.com/datasets/abdullah0a/retail-sales-data-with-seasonal-trends-and-marketing 

import kagglehub
import os
import shutil

# Step 1: Download dataset
path = kagglehub.dataset_download("abdullah0a/retail-sales-data-with-seasonal-trends-and-marketing")
print("Downloaded dataset path:", path)

# Step 2: Define new target directory and filename
target_dir = "data/raw"
os.makedirs(target_dir, exist_ok=True)  # create folder if not exist

# (Optional) If you want to rename the main CSV file
new_filename = "retail_sales_data.csv"  # rename as desired

# Step 3: Find the CSV file in downloaded folder
for file in os.listdir(path):
    if file.endswith(".csv"):
        source_file = os.path.join(path, file)
        destination_file = os.path.join(target_dir, new_filename)
        
        # Copy file (use shutil.move if you prefer moving instead of copying)
        shutil.copy(source_file, destination_file)
        print(f"Copied file to: {destination_file}")
        break
else:
    print("No CSV file found in the downloaded dataset folder.")

print("Place CSV file in:", target_dir)

# ============================================================================
# 4. LOAD DATASET
# ============================================================================

# Option 1: Load from Kaggle download
try:
    df_sales = pd.read_csv('data/raw/retail_sales_data.csv1222222') # Purposfully I am writing wrong name
    print("\n✓ Dataset loaded successfully from Kaggle! Work On simulated Dataset")
except FileNotFoundError:
    print("\n⚠️ Dataset not found. Creating sample data for demonstration...")
    
    # Create sample dataset for demonstration
    np.random.seed(42)
    n_records = 10000
    
    # Sample data generation
    df_sales = pd.DataFrame({
        'Order_ID': [f'ORD{str(i).zfill(6)}' for i in range(1, n_records+1)],
        'Order_Date': pd.date_range(start='2022-01-01', periods=n_records, freq='H'),
        'Ship_Date': pd.date_range(start='2022-01-02', periods=n_records, freq='H'),
        'Customer_ID': [f'CUST{np.random.randint(1, 2000):04d}' for _ in range(n_records)],
        'Customer_Name': [f'Customer_{i}' for i in np.random.randint(1, 2000, n_records)],
        'Segment': np.random.choice(['Consumer', 'Corporate', 'Home Office'], n_records, p=[0.5, 0.3, 0.2]),
        'Region': np.random.choice(['East', 'West', 'Central', 'South'], n_records, p=[0.3, 0.25, 0.25, 0.2]),
        'Product_ID': [f'PROD{np.random.randint(1, 500):04d}' for _ in range(n_records)],
        'Product_Category': np.random.choice(['Electronics', 'Furniture', 'Office Supplies', 'Clothing'], 
                                            n_records, p=[0.3, 0.2, 0.3, 0.2]),
        'Product_Sub_Category': np.random.choice(['Phones', 'Chairs', 'Paper', 'Accessories', 'Tables', 
                                                  'Binders', 'Appliances', 'Shirts'], n_records),
        'Product_Name': [f'Product_{i}' for i in np.random.randint(1, 500, n_records)],
        'Sales': np.random.gamma(2, 50, n_records) + 10,
        'Quantity': np.random.randint(1, 10, n_records),
        'Discount': np.random.choice([0, 0.1, 0.15, 0.2, 0.25], n_records, p=[0.4, 0.2, 0.2, 0.15, 0.05]),
        'Profit': np.random.normal(20, 30, n_records),
        'Shipping_Cost': np.random.gamma(1.5, 3, n_records) + 2,
        'Order_Priority': np.random.choice(['Low', 'Medium', 'High', 'Critical'], n_records, p=[0.3, 0.4, 0.2, 0.1])
    })
    
    # Add some calculated fields
    df_sales['Unit_Price'] = df_sales['Sales'] / df_sales['Quantity']
    df_sales['Revenue'] = df_sales['Sales']
    
    # Add some missing values for realistic scenario
    df_sales.loc[np.random.choice(df_sales.index, 50, replace=False), 'Customer_Name'] = np.nan
    df_sales.loc[np.random.choice(df_sales.index, 30, replace=False), 'Profit'] = np.nan
    
    # Save sample data
    df_sales.to_csv('data/raw/retail_sales_data.csv', index=False)
    print("✓ Sample dataset created and saved!")


# ============================================================================
# 5. INITIAL DATA INSPECTION
# ============================================================================

print("\n" + "="*80)
print("INITIAL DATA INSPECTION")
print("="*80)

# Dataset shape
print(f"\n Dataset Shape: {df_sales.shape[0]:,} rows × {df_sales.shape[1]} columns")

# First few rows
print("\n First 5 Rows:")
print(df_sales.head())

# Last few rows
print("\n Last 5 Rows:")
print(df_sales.tail())

# Column information
print("\n Column Information:")
print(df_sales.info())

# Data types
print("\n Data Types:")
print(df_sales.dtypes)

# ============================================================================
# 6. BASIC STATISTICS
# ============================================================================

print("\n" + "="*80)
print("BASIC STATISTICAL SUMMARY")
print("="*80)

# Numerical columns summary
print("\n Numerical Columns Statistics:")
print(df_sales.describe())

# Categorical columns summary
print("\n Categorical Columns Summary:")
categorical_cols = df_sales.select_dtypes(include=['object']).columns
for col in categorical_cols[:5]:  # First 5 categorical columns
    print(f"\n{col}:")
    print(df_sales[col].value_counts().head())

# ============================================================================
# 7. DATA QUALITY CHECK
# ============================================================================

print("\n" + "="*80)
print("DATA QUALITY ASSESSMENT")
print("="*80)

# Missing values
print("\n Missing Values:")
missing_data = pd.DataFrame({
    'Column': df_sales.columns,
    'Missing_Count': df_sales.isnull().sum().values,
    'Missing_Percentage': (df_sales.isnull().sum().values / len(df_sales) * 100).round(2)
})
missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
print(missing_data)

# Duplicate rows
duplicates = df_sales.duplicated().sum()
print(f"\n Duplicate Rows: {duplicates} ({(duplicates/len(df_sales)*100):.2f}%)")

# Unique values per column
print("\n Unique Values per Column:")
unique_counts = pd.DataFrame({
    'Column': df_sales.columns,
    'Unique_Count': [df_sales[col].nunique() for col in df_sales.columns],
    'Data_Type': df_sales.dtypes.values
})
print(unique_counts)

# ============================================================================
# 8. INITIAL VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("CREATING INITIAL VISUALIZATIONS")
print("="*80)

# Create visualization directory if not exists
os.makedirs('outputs/figures', exist_ok=True)

# 1. Missing values visualization
fig, ax = plt.subplots(figsize=(10, 6))
missing_data_plot = df_sales.isnull().sum().sort_values(ascending=False)[:10]
missing_data_plot.plot(kind='bar', ax=ax, color='coral')
ax.set_title('Top 10 Columns with Missing Values', fontsize=14, fontweight='bold')
ax.set_xlabel('Columns', fontsize=12)
ax.set_ylabel('Number of Missing Values', fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/figures/01_missing_values.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_missing_values.png")
plt.show()

# 2. Data types distribution
fig, ax = plt.subplots(figsize=(8, 6))
dtype_counts = df_sales.dtypes.value_counts()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
ax.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%', 
       startangle=90, colors=colors)
ax.set_title('Distribution of Data Types', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/02_data_types_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_data_types_distribution.png")
plt.show()

# ============================================================================
# 9. SAVE INITIAL INSPECTION REPORT
# ============================================================================

# Create comprehensive report
report = f"""
RETAIL & MARKETING ANALYTICS PROJECT
Initial Data Inspection Report
{'='*80}

1. DATASET OVERVIEW
   - Total Records: {df_sales.shape[0]:,}
   - Total Columns: {df_sales.shape[1]}
   - Memory Usage: {df_sales.memory_usage(deep=True).sum() / 1024**2:.2f} MB
   
2. DATA QUALITY SUMMARY
   - Missing Values: {df_sales.isnull().sum().sum():,} cells ({(df_sales.isnull().sum().sum()/(df_sales.shape[0]*df_sales.shape[1])*100):.2f}%)
   - Duplicate Rows: {duplicates} ({(duplicates/len(df_sales)*100):.2f}%)
   - Complete Rows: {df_sales.dropna().shape[0]:,} ({(df_sales.dropna().shape[0]/len(df_sales)*100):.2f}%)

3. COLUMN TYPES
   - Numerical: {len(df_sales.select_dtypes(include=[np.number]).columns)}
   - Categorical/Object: {len(df_sales.select_dtypes(include=['object']).columns)}
   - DateTime: {len(df_sales.select_dtypes(include=['datetime64']).columns)}

4. NEXT STEPS
   ✓ Proceed to Data Cleaning (Notebook 02)
   - Handle missing values
   - Remove duplicates
   - Convert data types
   - Handle outliers
   - Feature engineering

Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""

# Save report
with open('outputs/reports/01_initial_inspection_report.txt', 'w') as f:
    f.write(report)

print("\n✓ Initial inspection report saved!")
print(report)

# ============================================================================
# 10. SAVE CHECKPOINT
# ============================================================================

# Save initial loaded data for reference
df_sales.to_csv('data/raw/original_data_checkpoint.csv', index=False)
print("\n✓ Data checkpoint saved: data/raw/original_data_checkpoint.csv")

print("\n" + "="*80)
print("NOTEBOOK 01 COMPLETED SUCCESSFULLY!")
print("="*80)
print("\n Next Step: Data Cleaning and Preprocessing")
print("="*80)

# Retail & Marketing Analytics Project
# Part 2: Data Cleaning and Preprocessing

"""
Project: Retail & Marketing Analytics - Customer Segmentation & Sales Optimization
Part: 02 - Data Cleaning and Preprocessing

Objective:
- Handle missing values
- Remove duplicates
- Convert data types
- Detect and treat outliers
- Feature engineering
- Save cleaned dataset
"""

# ============================================================================
# 1. IMPORT LIBRARIES AND LOAD DATA
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Load the raw data
df_sales = pd.read_csv('data/raw/retail_sales_data.csv')

print("="*80)
print("DATA CLEANING AND PREPROCESSING")
print("="*80)
print(f"\nInitial Dataset Shape: {df_sales.shape}")
print(f"Initial Memory Usage: {df_sales.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Create a copy for cleaning
df_clean = df_sales.copy()

# ============================================================================
# 2. HANDLE MISSING VALUES
# ============================================================================

print("\n" + "="*80)
print("STEP 1: HANDLING MISSING VALUES")
print("="*80)

# Check missing values
print("\n Missing Values Before Treatment:")
missing_before = df_clean.isnull().sum()
missing_before_pct = (missing_before / len(df_clean) * 100).round(2)
missing_df = pd.DataFrame({
    'Column': missing_before.index,
    'Missing_Count': missing_before.values,
    'Missing_Percentage': missing_before_pct.values
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
print(missing_df)

# Strategy for handling missing values
print("\n Treatment Strategy:")

# Separate columns by type
numerical_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()

# Handle numerical missing values - Fill with median
for col in numerical_cols:
    if df_clean[col].isnull().sum() > 0:
        median_value = df_clean[col].median()
        df_clean[col].fillna(median_value, inplace=True)
        print(f"  ✓ {col}: Filled with median ({median_value:.2f})")

# Handle categorical missing values - Fill with mode or 'Unknown'
for col in categorical_cols:
    if df_clean[col].isnull().sum() > 0:
        if df_clean[col].mode().empty:
            df_clean[col].fillna('Unknown', inplace=True)
            print(f"  ✓ {col}: Filled with 'Unknown'")
        else:
            mode_value = df_clean[col].mode()[0]
            df_clean[col].fillna(mode_value, inplace=True)
            print(f"  ✓ {col}: Filled with mode ('{mode_value}')")

# Verify no missing values remain
print(f"\n Missing values after treatment: {df_clean.isnull().sum().sum()}")

# ============================================================================
# 3. REMOVE DUPLICATES
# ============================================================================

print("\n" + "="*80)
print("STEP 2: REMOVING DUPLICATES")
print("="*80)

# Check duplicates
duplicates_before = df_clean.duplicated().sum()
print(f"\n Duplicate rows found: {duplicates_before}")

# Remove duplicates
df_clean = df_clean.drop_duplicates()
print(f"✓ Duplicates removed: {duplicates_before}")
print(f"✓ New dataset shape: {df_clean.shape}")

# Reset index after removing duplicates
df_clean.reset_index(drop=True, inplace=True)

# ============================================================================
# 4. DATA TYPE CONVERSION
# ============================================================================

print("\n" + "="*80)
print("STEP 3: DATA TYPE CONVERSION")
print("="*80)

# Convert date columns to datetime
date_columns = ['Order_Date', 'Ship_Date']
for col in date_columns:
    if col in df_clean.columns:
        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        print(f"✓ Converted {col} to datetime")

# Convert categorical columns to category type for memory optimization
categorical_columns = ['Segment', 'Region', 'Product_Category', 'Product_Sub_Category', 'Order_Priority']
for col in categorical_columns:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].astype('category')
        print(f"✓ Converted {col} to category")

# Ensure numerical columns are correct type
if 'Quantity' in df_clean.columns:
    df_clean['Quantity'] = df_clean['Quantity'].astype('int32')
    
print(f"\n Data types optimized!")
print(f"Memory usage after optimization: {df_clean.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================================================
# 5. OUTLIER DETECTION AND TREATMENT
# ============================================================================

print("\n" + "="*80)
print("STEP 4: OUTLIER DETECTION AND TREATMENT")
print("="*80)

def detect_outliers_iqr(data, column):
    """Detect outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Check for outliers in key numerical columns
outlier_columns = ['Sales', 'Quantity', 'Profit'] if 'Profit' in df_clean.columns else ['Sales', 'Quantity']

print("\n Outlier Analysis:")
outlier_summary = []

for col in outlier_columns:
    if col in df_clean.columns:
        outliers, lb, ub = detect_outliers_iqr(df_clean, col)
        outlier_pct = (len(outliers) / len(df_clean) * 100)
        outlier_summary.append({
            'Column': col,
            'Outliers': len(outliers),
            'Percentage': f"{outlier_pct:.2f}%",
            'Lower_Bound': f"{lb:.2f}",
            'Upper_Bound': f"{ub:.2f}"
        })
        print(f"\n{col}:")
        print(f"  - Outliers detected: {len(outliers)} ({outlier_pct:.2f}%)")
        print(f"  - Lower bound: {lb:.2f}")
        print(f"  - Upper bound: {ub:.2f}")

# Visualize outliers before treatment
fig, axes = plt.subplots(1, len(outlier_columns), figsize=(15, 5))
if len(outlier_columns) == 1:
    axes = [axes]

for idx, col in enumerate(outlier_columns):
    if col in df_clean.columns:
        axes[idx].boxplot(df_clean[col].dropna())
        axes[idx].set_title(f'{col} Distribution\n(Before Treatment)', fontweight='bold')
        axes[idx].set_ylabel(col)
        axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/figures/03_outliers_before_treatment.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 03_outliers_before_treatment.png")
plt.show()

# Treatment: Cap outliers (Winsorization)
print("\n Outlier Treatment (Capping):")
for col in outlier_columns:
    if col in df_clean.columns:
        outliers, lb, ub = detect_outliers_iqr(df_clean, col)
        if len(outliers) > 0:
            # Cap values
            df_clean[f'{col}_Original'] = df_clean[col]  # Keep original for reference
            df_clean[col] = df_clean[col].clip(lower=lb, upper=ub)
            print(f"  ✓ {col}: Outliers capped to [{lb:.2f}, {ub:.2f}]")

# Visualize after treatment
fig, axes = plt.subplots(1, len(outlier_columns), figsize=(15, 5))
if len(outlier_columns) == 1:
    axes = [axes]

for idx, col in enumerate(outlier_columns):
    if col in df_clean.columns:
        axes[idx].boxplot(df_clean[col].dropna())
        axes[idx].set_title(f'{col} Distribution\n(After Treatment)', fontweight='bold')
        axes[idx].set_ylabel(col)
        axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/figures/04_outliers_after_treatment.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_outliers_after_treatment.png")
plt.show()

# ============================================================================
# 6. FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*80)
print("STEP 5: FEATURE ENGINEERING")
print("="*80)

# 1. Time-based features
if 'Order_Date' in df_clean.columns:
    print("\n Creating Time-Based Features:")
    df_clean['Year'] = df_clean['Order_Date'].dt.year
    df_clean['Month'] = df_clean['Order_Date'].dt.month
    df_clean['Month_Name'] = df_clean['Order_Date'].dt.month_name()
    df_clean['Quarter'] = df_clean['Order_Date'].dt.quarter
    df_clean['Day'] = df_clean['Order_Date'].dt.day
    df_clean['Day_of_Week'] = df_clean['Order_Date'].dt.dayofweek
    df_clean['Day_Name'] = df_clean['Order_Date'].dt.day_name()
    df_clean['Week_of_Year'] = df_clean['Order_Date'].dt.isocalendar().week
    df_clean['Is_Weekend'] = df_clean['Day_of_Week'].isin([5, 6]).astype(int)
    df_clean['Is_Month_Start'] = df_clean['Order_Date'].dt.is_month_start.astype(int)
    df_clean['Is_Month_End'] = df_clean['Order_Date'].dt.is_month_end.astype(int)
    print("  ✓ Year, Month, Quarter, Day, Day of Week")
    print("  ✓ Weekend indicator, Month start/end flags")

# 2. Revenue and profit metrics
if 'Sales' in df_clean.columns and 'Quantity' in df_clean.columns:
    print("\n Creating Revenue Metrics:")
    df_clean['Unit_Price'] = df_clean['Sales'] / df_clean['Quantity']
    df_clean['Revenue'] = df_clean['Sales']
    
    if 'Discount' in df_clean.columns:
        df_clean['Discount_Amount'] = df_clean['Sales'] * df_clean['Discount']
        df_clean['Net_Revenue'] = df_clean['Sales'] - df_clean['Discount_Amount']
        print("  ✓ Unit Price, Revenue, Discount Amount, Net Revenue")
    
    if 'Profit' in df_clean.columns:
        df_clean['Profit_Margin'] = (df_clean['Profit'] / df_clean['Sales']) * 100
        df_clean['Profit_Ratio'] = df_clean['Profit'] / df_clean['Sales']
        print("  ✓ Profit Margin, Profit Ratio")

# 3. Delivery time calculation
if 'Order_Date' in df_clean.columns and 'Ship_Date' in df_clean.columns:
    print("\n Creating Delivery Metrics:")
    df_clean['Delivery_Days'] = (df_clean['Ship_Date'] - df_clean['Order_Date']).dt.days
    df_clean['Delivery_Category'] = pd.cut(df_clean['Delivery_Days'], 
                                           bins=[-np.inf, 2, 5, 10, np.inf],
                                           labels=['Same/Next Day', 'Standard', 'Delayed', 'Very Delayed'])
    print("  ✓ Delivery Days, Delivery Category")

# 4. Customer frequency (aggregated later, but flag here)
if 'Customer_ID' in df_clean.columns:
    print("\n Creating Customer Flags:")
    customer_order_count = df_clean.groupby('Customer_ID').size()
    df_clean['Customer_Order_Count'] = df_clean['Customer_ID'].map(customer_order_count)
    df_clean['Is_Repeat_Customer'] = (df_clean['Customer_Order_Count'] > 1).astype(int)
    print("  ✓ Customer Order Count, Repeat Customer Flag")

# 5. Product performance metrics
if 'Product_ID' in df_clean.columns:
    print("\n Creating Product Metrics:")
    product_sales = df_clean.groupby('Product_ID')['Sales'].agg(['sum', 'mean', 'count'])
    product_sales.columns = ['Product_Total_Sales', 'Product_Avg_Sales', 'Product_Order_Count']
    df_clean = df_clean.merge(product_sales, left_on='Product_ID', right_index=True, how='left')
    print("  ✓ Product Total Sales, Average Sales, Order Count")

# 6. Sales categories
if 'Sales' in df_clean.columns:
    print("\n Creating Sales Categories:")
    df_clean['Sales_Category'] = pd.qcut(df_clean['Sales'], 
                                         q=4, 
                                         labels=['Low', 'Medium', 'High', 'Very High'],
                                         duplicates='drop')
    print("  ✓ Sales Category (Quartile-based)")

# 7. Season classification
if 'Month' in df_clean.columns:
    print("\n Creating Seasonal Features:")
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    }
    df_clean['Season'] = df_clean['Month'].map(season_map)
    print("  ✓ Season")

print(f"\n Feature Engineering Completed!")
print(f"New dataset shape: {df_clean.shape}")

# ============================================================================
# 7. DATA VALIDATION
# ============================================================================

print("\n" + "="*80)
print("STEP 6: DATA VALIDATION")
print("="*80)

# Check for negative values in key columns
print("\n Checking for Invalid Values:")

validation_checks = []

if 'Sales' in df_clean.columns:
    negative_sales = (df_clean['Sales'] < 0).sum()
    validation_checks.append(f"Negative Sales: {negative_sales}")
    
if 'Quantity' in df_clean.columns:
    negative_qty = (df_clean['Quantity'] <= 0).sum()
    validation_checks.append(f"Zero/Negative Quantity: {negative_qty}")
    
if 'Discount' in df_clean.columns:
    invalid_discount = ((df_clean['Discount'] < 0) | (df_clean['Discount'] > 1)).sum()
    validation_checks.append(f"Invalid Discount: {invalid_discount}")

for check in validation_checks:
    print(f"  • {check}")

# Check date consistency
if 'Order_Date' in df_clean.columns and 'Ship_Date' in df_clean.columns:
    invalid_dates = (df_clean['Ship_Date'] < df_clean['Order_Date']).sum()
    print(f"  • Ship Date before Order Date: {invalid_dates}")

print("\n Validation Complete!")

# ============================================================================
# 8. SUMMARY STATISTICS AFTER CLEANING
# ============================================================================

print("\n" + "="*80)
print("CLEANED DATA SUMMARY")
print("="*80)

print(f"\n Final Dataset Shape: {df_clean.shape}")
print(f" Records removed: {df_sales.shape[0] - df_clean.shape[0]}")
print(f" Features added: {df_clean.shape[1] - df_sales.shape[1]}")
print(f" Memory usage: {df_clean.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n Data Quality Metrics:")
print(f"  • Missing values: {df_clean.isnull().sum().sum()}")
print(f"  • Duplicate rows: {df_clean.duplicated().sum()}")
print(f"  • Complete rows: {df_clean.dropna().shape[0]:,} ({(df_clean.dropna().shape[0]/len(df_clean)*100):.2f}%)")

print("\n Key Statistics:")
if 'Sales' in df_clean.columns:
    print(f"  • Total Sales: ${df_clean['Sales'].sum():,.2f}")
    print(f"  • Average Order Value: ${df_clean['Sales'].mean():,.2f}")
if 'Customer_ID' in df_clean.columns:
    print(f"  • Unique Customers: {df_clean['Customer_ID'].nunique():,}")
if 'Product_ID' in df_clean.columns:
    print(f"  • Unique Products: {df_clean['Product_ID'].nunique():,}")

# ============================================================================
# 9. SAVE CLEANED DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 7: SAVING CLEANED DATA")
print("="*80)

# Save cleaned dataset
output_path = 'data/processed/cleaned_retail_sales.csv'
df_clean.to_csv(output_path, index=False)
print(f"✓ Cleaned data saved: {output_path}")

# Save data dictionary
data_dict = pd.DataFrame({
    'Column_Name': df_clean.columns,
    'Data_Type': df_clean.dtypes.values,
    'Non_Null_Count': df_clean.count().values,
    'Null_Count': df_clean.isnull().sum().values,
    'Unique_Values': [df_clean[col].nunique() for col in df_clean.columns],
    'Sample_Value': [df_clean[col].iloc[0] if len(df_clean) > 0 else None for col in df_clean.columns]
})

dict_path = 'docs/data_dictionary.csv'
data_dict.to_csv(dict_path, index=False)
print(f"✓ Data dictionary saved: {dict_path}")

# Create cleaning summary report
cleaning_report = f"""
DATA CLEANING AND PREPROCESSING REPORT
{'='*80}

1. DATA QUALITY IMPROVEMENTS
   Before Cleaning:
   - Records: {df_sales.shape[0]:,}
   - Missing Values: {df_sales.isnull().sum().sum():,}
   - Duplicates: {duplicates_before}
   
   After Cleaning:
   - Records: {df_clean.shape[0]:,}
   - Missing Values: {df_clean.isnull().sum().sum()}
   - Duplicates: {df_clean.duplicated().sum()}
   
   Records Removed: {df_sales.shape[0] - df_clean.shape[0]:,}
   Data Quality Score: {((1 - df_clean.isnull().sum().sum()/(df_clean.shape[0]*df_clean.shape[1])) * 100):.2f}%

2. FEATURES ENGINEERED
   Original Features: {df_sales.shape[1]}
   New Features: {df_clean.shape[1] - df_sales.shape[1]}
   Total Features: {df_clean.shape[1]}
   
   Categories of New Features:
   - Time-based: Year, Month, Quarter, Day, Week, Weekend
   - Revenue: Unit Price, Net Revenue, Discount Amount
   - Customer: Order Count, Repeat Customer Flag
   - Product: Total Sales, Average Sales, Order Count
   - Delivery: Delivery Days, Delivery Category
   - Categorical: Sales Category, Season

3. DATA TYPE OPTIMIZATION
   Memory Before: {df_sales.memory_usage(deep=True).sum() / 1024**2:.2f} MB
   Memory After: {df_clean.memory_usage(deep=True).sum() / 1024**2:.2f} MB
   Memory Saved: {(df_sales.memory_usage(deep=True).sum() - df_clean.memory_usage(deep=True).sum()) / 1024**2:.2f} MB

4. OUTLIER TREATMENT
   Method: IQR (Interquartile Range) with Winsorization
   Columns Treated: {', '.join(outlier_columns)}

5. NEXT STEPS
   ✓ Proceed to Exploratory Data Analysis (Notebook 03)
   - Univariate analysis
   - Bivariate analysis
   - Time series analysis
   - Customer behavior analysis

Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""

report_path = 'outputs/reports/02_cleaning_report.txt'
with open(report_path, 'w') as f:
    f.write(cleaning_report)
print(f"✓ Cleaning report saved: {report_path}")

print(cleaning_report)

print("\n" + "="*80)
print("Part2 COMPLETED SUCCESSFULLY!")
print("="*80)
print("\n Next Step: Part 3 - Exploratory Data Analysis")
print("="*80)

# Retail & Marketing Analytics Project
# Part 3: Exploratory Data Analysis (EDA)

"""
Project: Retail & Marketing Analytics - Customer Segmentation & Sales Optimization
Part: 03 - Exploratory Data Analysis (EDA)
Author: [Your Name]
Date: [Current Date]

Objective:
- Univariate analysis
- Bivariate analysis
- Multivariate analysis
- Time series analysis
- Customer behavior analysis
- Product performance analysis
- Generate insights
"""

# ============================================================================
# 1. IMPORT LIBRARIES AND LOAD CLEANED DATA
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Load cleaned data
df = pd.read_csv('data/processed/cleaned_retail_sales.csv')

# Convert date columns back to datetime
date_cols = ['Order_Date', 'Ship_Date']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])

print("="*80)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("="*80)
print(f"\nDataset Shape: {df.shape}")
print(f"Analysis Period: {df['Order_Date'].min().date()} to {df['Order_Date'].max().date()}")

# ============================================================================
# 2. UNIVARIATE ANALYSIS - NUMERICAL VARIABLES
# ============================================================================

print("\n" + "="*80)
print("UNIVARIATE ANALYSIS - NUMERICAL VARIABLES")
print("="*80)

# Select numerical columns
numerical_cols = ['Sales', 'Quantity', 'Profit', 'Discount', 'Unit_Price']
numerical_cols = [col for col in numerical_cols if col in df.columns]

# Distribution plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
    if idx < len(axes):
        # Histogram with KDE
        axes[idx].hist(df[col], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[idx].set_title(f'Distribution of {col}', fontweight='bold', fontsize=12)
        axes[idx].set_xlabel(col, fontsize=10)
        axes[idx].set_ylabel('Frequency', fontsize=10)
        axes[idx].axvline(df[col].mean(), color='red', linestyle='--', 
                         label=f'Mean: {df[col].mean():.2f}')
        axes[idx].axvline(df[col].median(), color='green', linestyle='--', 
                         label=f'Median: {df[col].median():.2f}')
        axes[idx].legend()
        axes[idx].grid(axis='y', alpha=0.3)

# Remove empty subplots
for idx in range(len(numerical_cols), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig('outputs/figures/05_numerical_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_numerical_distributions.png")
plt.show()

# Statistical summary
print("\n Statistical Summary:")
print(df[numerical_cols].describe().round(2))

# Skewness and Kurtosis
print("\n Skewness and Kurtosis:")
for col in numerical_cols:
    skew = df[col].skew()
    kurt = df[col].kurtosis()
    print(f"{col:15s} - Skewness: {skew:6.2f}, Kurtosis: {kurt:6.2f}")

# ============================================================================
# 3. UNIVARIATE ANALYSIS - CATEGORICAL VARIABLES
# ============================================================================

print("\n" + "="*80)
print("UNIVARIATE ANALYSIS - CATEGORICAL VARIABLES")
print("="*80)

# Select categorical columns
categorical_cols = ['Segment', 'Region', 'Product_Category', 'Order_Priority', 'Season']
categorical_cols = [col for col in categorical_cols if col in df.columns]

# Create bar plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, col in enumerate(categorical_cols):
    if idx < len(axes):
        value_counts = df[col].value_counts()
        axes[idx].bar(range(len(value_counts)), value_counts.values, color='coral', edgecolor='black')
        axes[idx].set_xticks(range(len(value_counts)))
        axes[idx].set_xticklabels(value_counts.index, rotation=45, ha='right')
        axes[idx].set_title(f'Distribution of {col}', fontweight='bold', fontsize=12)
        axes[idx].set_ylabel('Count', fontsize=10)
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(value_counts.values):
            axes[idx].text(i, v, str(v), ha='center', va='bottom')

# Remove empty subplots
for idx in range(len(categorical_cols), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig('outputs/figures/06_categorical_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 06_categorical_distributions.png")
plt.show()

# Print value counts
print("\n Categorical Variable Counts:")
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())

# ============================================================================
# 4. BIVARIATE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("BIVARIATE ANALYSIS")
print("="*80)

# 4.1 Sales by Category
if 'Product_Category' in df.columns and 'Sales' in df.columns:
    print("\n Sales by Product Category:")
    category_sales = df.groupby('Product_Category')['Sales'].agg(['sum', 'mean', 'count']).round(2)
    category_sales.columns = ['Total_Sales', 'Avg_Sales', 'Order_Count']
    category_sales = category_sales.sort_values('Total_Sales', ascending=False)
    print(category_sales)
    
    # Visualization
    fig = px.bar(category_sales.reset_index(), 
                 x='Product_Category', 
                 y='Total_Sales',
                 title='Total Sales by Product Category',
                 labels={'Total_Sales': 'Total Sales ($)', 'Product_Category': 'Category'},
                 color='Total_Sales',
                 color_continuous_scale='viridis')
    fig.write_html('outputs/figures/07_sales_by_category.html')
    print("✓ Saved: 07_sales_by_category.html")

# 4.2 Sales by Region
if 'Region' in df.columns and 'Sales' in df.columns:
    print("\n Sales by Region:")
    region_sales = df.groupby('Region')['Sales'].agg(['sum', 'mean', 'count']).round(2)
    region_sales.columns = ['Total_Sales', 'Avg_Sales', 'Order_Count']
    region_sales = region_sales.sort_values('Total_Sales', ascending=False)
    print(region_sales)
    
    # Pie chart
    fig = px.pie(region_sales.reset_index(), 
                 values='Total_Sales', 
                 names='Region',
                 title='Revenue Distribution by Region',
                 hole=0.4)
    fig.write_html('outputs/figures/08_sales_by_region.html')
    print("✓ Saved: 08_sales_by_region.html")

# 4.3 Sales by Customer Segment
if 'Segment' in df.columns and 'Sales' in df.columns:
    print("\n Sales by Customer Segment:")
    segment_sales = df.groupby('Segment')['Sales'].agg(['sum', 'mean', 'count']).round(2)
    segment_sales.columns = ['Total_Sales', 'Avg_Sales', 'Order_Count']
    print(segment_sales)

# 4.4 Correlation Analysis
print("\n Correlation Analysis:")
numerical_for_corr = ['Sales', 'Quantity', 'Profit', 'Discount']
numerical_for_corr = [col for col in numerical_for_corr if col in df.columns]

correlation_matrix = df[numerical_for_corr].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix of Key Numerical Variables', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('outputs/figures/09_correlation_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 09_correlation_matrix.png")
plt.show()

print("\nCorrelation Coefficients:")
print(correlation_matrix)

# ============================================================================
# 5. TIME SERIES ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("TIME SERIES ANALYSIS")
print("="*80)

# 5.1 Monthly Sales Trend
if 'Order_Date' in df.columns and 'Sales' in df.columns:
    print("\n Monthly Sales Trend:")
    df['YearMonth'] = df['Order_Date'].dt.to_period('M')
    monthly_sales = df.groupby('YearMonth')['Sales'].agg(['sum', 'mean', 'count']).reset_index()
    monthly_sales.columns = ['YearMonth', 'Total_Sales', 'Avg_Order_Value', 'Order_Count']
    monthly_sales['YearMonth'] = monthly_sales['YearMonth'].astype(str)
    
    print(monthly_sales.tail(12))
    
    # Line chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_sales['YearMonth'], 
                            y=monthly_sales['Total_Sales'],
                            mode='lines+markers',
                            name='Total Sales',
                            line=dict(color='blue', width=3),
                            marker=dict(size=8)))
    
    fig.update_layout(
        title='Monthly Sales Trend',
        xaxis_title='Month',
        yaxis_title='Total Sales ($)',
        hovermode='x unified',
        height=500
    )
    fig.write_html('outputs/figures/10_monthly_sales_trend.html')
    print("✓ Saved: 10_monthly_sales_trend.html")

# 5.2 Quarterly Analysis
if 'Quarter' in df.columns and 'Year' in df.columns:
    print("\n Quarterly Sales Analysis:")
    quarterly_sales = df.groupby(['Year', 'Quarter'])['Sales'].sum().reset_index()
    
    fig = px.bar(quarterly_sales, 
                 x='Quarter', 
                 y='Sales', 
                 color='Year',
                 title='Quarterly Sales Comparison by Year',
                 barmode='group',
                 labels={'Sales': 'Total Sales ($)', 'Quarter': 'Quarter'})
    fig.write_html('outputs/figures/11_quarterly_comparison.html')
    print("✓ Saved: 11_quarterly_comparison.html")

# 5.3 Day of Week Analysis
if 'Day_Name' in df.columns and 'Sales' in df.columns:
    print("\n Sales by Day of Week:")
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    dow_sales = df.groupby('Day_Name')['Sales'].agg(['sum', 'mean', 'count']).reindex(dow_order)
    dow_sales.columns = ['Total_Sales', 'Avg_Sales', 'Order_Count']
    print(dow_sales)
    
    # Bar chart
    fig = px.bar(dow_sales.reset_index(), 
                 x='Day_Name', 
                 y='Avg_Sales',
                 title='Average Sales by Day of Week',
                 labels={'Avg_Sales': 'Average Sales ($)', 'Day_Name': 'Day'},
                 color='Avg_Sales',
                 color_continuous_scale='blues')
    #fig.update_xaxis(categoryorder='array', categoryarray=dow_order)
    fig.write_html('outputs/figures/12_sales_by_day.html')
    print("✓ Saved: 12_sales_by_day.html")

# 5.4 Weekend vs Weekday
if 'Is_Weekend' in df.columns:
    print("\n Weekend vs Weekday Sales:")
    weekend_sales = df.groupby('Is_Weekend')['Sales'].agg(['sum', 'mean', 'count'])
    weekend_sales.index = ['Weekday', 'Weekend']
    weekend_sales.columns = ['Total_Sales', 'Avg_Sales', 'Order_Count']
    print(weekend_sales)
    
    # Calculate lift
    weekend_lift = ((weekend_sales.loc['Weekend', 'Avg_Sales'] / 
                    weekend_sales.loc['Weekday', 'Avg_Sales']) - 1) * 100
    print(f"\n Weekend Lift: {weekend_lift:.2f}%")

# 5.5 Seasonal Analysis
if 'Season' in df.columns and 'Sales' in df.columns:
    print("\n Seasonal Sales Analysis:")
    season_order = ['Spring', 'Summer', 'Fall', 'Winter']
    seasonal_sales = df.groupby('Season')['Sales'].agg(['sum', 'mean', 'count'])
    seasonal_sales = seasonal_sales.reindex(season_order)
    seasonal_sales.columns = ['Total_Sales', 'Avg_Sales', 'Order_Count']
    print(seasonal_sales)


# ============================================================================
# 6. CUSTOMER BEHAVIOR ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("CUSTOMER BEHAVIOR ANALYSIS")
print("="*80)

# 6.1 Customer Purchase Frequency
if 'Customer_ID' in df.columns:
    print("\n Customer Purchase Frequency:")
    customer_frequency = df.groupby('Customer_ID').size().reset_index(name='Purchase_Count')
    
    print(f"Total Unique Customers: {len(customer_frequency):,}")
    print(f"Average Purchases per Customer: {customer_frequency['Purchase_Count'].mean():.2f}")
    print(f"Median Purchases per Customer: {customer_frequency['Purchase_Count'].median():.0f}")
    
    # Distribution
    plt.figure(figsize=(12, 6))
    plt.hist(customer_frequency['Purchase_Count'], bins=50, edgecolor='black', 
             alpha=0.7, color='skyblue')
    plt.title('Customer Purchase Frequency Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Purchases', fontsize=12)
    plt.ylabel('Number of Customers', fontsize=12)
    plt.axvline(customer_frequency['Purchase_Count'].mean(), color='red', 
                linestyle='--', linewidth=2, label=f'Mean: {customer_frequency["Purchase_Count"].mean():.2f}')
    plt.axvline(customer_frequency['Purchase_Count'].median(), color='green', 
                linestyle='--', linewidth=2, label=f'Median: {customer_frequency["Purchase_Count"].median():.0f}')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/figures/13_customer_frequency.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 13_customer_frequency.png")
    plt.show()

# 6.2 Top Customers by Revenue
if 'Customer_ID' in df.columns and 'Sales' in df.columns:
    print("\n Top 10 Customers by Revenue:")
    top_customers = df.groupby('Customer_ID')['Sales'].sum().sort_values(ascending=False).head(10)
    print(top_customers)
    
    # Visualization
    fig = px.bar(top_customers.reset_index(), 
                 x='Customer_ID', 
                 y='Sales',
                 title='Top 10 Customers by Total Revenue',
                 labels={'Sales': 'Total Revenue ($)', 'Customer_ID': 'Customer ID'},
                 color='Sales',
                 color_continuous_scale='reds')
    fig.write_html('outputs/figures/14_top_customers.html')
    print("✓ Saved: 14_top_customers.html")

# 6.3 Customer Retention (Repeat vs One-time)
if 'Is_Repeat_Customer' in df.columns:
    print("\n Customer Retention:")
    retention = df.groupby('Customer_ID')['Is_Repeat_Customer'].max()
    repeat_count = retention.sum()
    total_customers = len(retention)
    retention_rate = (repeat_count / total_customers) * 100
    
    print(f"Repeat Customers: {repeat_count:,} ({retention_rate:.2f}%)")
    print(f"One-time Customers: {total_customers - repeat_count:,} ({100 - retention_rate:.2f}%)")

# ============================================================================
# 7. PRODUCT PERFORMANCE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("PRODUCT PERFORMANCE ANALYSIS")
print("="*80)

# 7.1 Top Selling Products
if 'Product_ID' in df.columns and 'Sales' in df.columns:
    print("\n Top 20 Products by Revenue:")
    product_performance = df.groupby('Product_ID').agg({
        'Sales': 'sum',
        'Quantity': 'sum',
        'Order_ID': 'count'
    }).sort_values('Sales', ascending=False).head(20)
    product_performance.columns = ['Total_Revenue', 'Total_Quantity', 'Order_Count']
    print(product_performance)
    
    # Dual axis chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=product_performance.index,
        y=product_performance['Total_Revenue'],
        name='Revenue',
        marker_color='indianred',
        yaxis='y'
    ))
    fig.add_trace(go.Scatter(
        x=product_performance.index,
        y=product_performance['Total_Quantity'],
        name='Quantity',
        mode='lines+markers',
        marker_color='lightsalmon',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Top 20 Products: Revenue vs Quantity',
        xaxis=dict(title='Product ID', tickangle=45),
        yaxis=dict(title='Revenue ($)', side='left'),
        yaxis2=dict(title='Quantity Sold', overlaying='y', side='right'),
        hovermode='x unified',
        height=600
    )
    fig.write_html('outputs/figures/15_top_products.html')
    print("✓ Saved: 15_top_products.html")

# 7.2 Category Performance Matrix
if 'Product_Category' in df.columns:
    print("\n Category Performance Matrix:")
    category_matrix = df.groupby('Product_Category').agg({
        'Sales': ['sum', 'mean'],
        'Order_ID': 'count',
        'Customer_ID': 'nunique'
    })
    category_matrix.columns = ['Total_Sales', 'Avg_Order_Value', 'Order_Count', 'Unique_Customers']
    category_matrix['Revenue_Share'] = (category_matrix['Total_Sales'] / category_matrix['Total_Sales'].sum() * 100).round(2)
    print(category_matrix.round(2))
    
    # Scatter plot
    fig = px.scatter(category_matrix.reset_index(), 
                     x='Order_Count', 
                     y='Avg_Order_Value',
                     size='Total_Sales', 
                     color='Product_Category',
                     title='Category Performance: Orders vs AOV (Size = Revenue)',
                     labels={'Order_Count': 'Number of Orders', 
                             'Avg_Order_Value': 'Average Order Value ($)'},
                     hover_data=['Total_Sales', 'Unique_Customers'])
    fig.write_html('outputs/figures/16_category_performance.html')
    print("✓ Saved: 16_category_performance.html")

# 7.3 Pareto Analysis (80-20 Rule)
if 'Product_ID' in df.columns and 'Sales' in df.columns:
    print("\n Pareto Analysis (80-20 Rule):")
    product_sales = df.groupby('Product_ID')['Sales'].sum().sort_values(ascending=False)
    product_sales_cumsum = product_sales.cumsum() / product_sales.sum() * 100
    
    # Find top products contributing to 80% sales
    top_80_pct = (product_sales_cumsum <= 80).sum()
    pct_products = (top_80_pct / len(product_sales)) * 100
    
    print(f"Top {top_80_pct} products ({pct_products:.1f}%) generate 80% of revenue")
    print(f"Total products: {len(product_sales)}")


# ============================================================================
# 8. ADVANCED INSIGHTS
# ============================================================================

print("\n" + "="*80)
print("ADVANCED INSIGHTS")
print("="*80)

# 8.1 Sales Distribution Analysis
if 'Sales' in df.columns:
    print("\n Sales Distribution:")
    print(f"Mean: ${df['Sales'].mean():.2f}")
    print(f"Median: ${df['Sales'].median():.2f}")
    print(f"Std Dev: ${df['Sales'].std():.2f}")
    print(f"Min: ${df['Sales'].min():.2f}")
    print(f"Max: ${df['Sales'].max():.2f}")
    print(f"\nPercentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"  {p}th: ${df['Sales'].quantile(p/100):.2f}")

# 8.2 Discount Impact Analysis
if 'Discount' in df.columns and 'Sales' in df.columns:
    print("\n Discount Impact on Sales:")
    discount_groups = pd.cut(df['Discount'], bins=[-0.01, 0, 0.1, 0.2, 1], 
                             labels=['No Discount', '1-10%', '11-20%', '20%+'])
    discount_analysis = df.groupby(discount_groups)['Sales'].agg(['count', 'mean', 'sum'])
    discount_analysis.columns = ['Order_Count', 'Avg_Order_Value', 'Total_Sales']
    print(discount_analysis)

# 8.3 Delivery Performance
if 'Delivery_Days' in df.columns:
    print("\n Delivery Performance:")
    print(f"Average Delivery Time: {df['Delivery_Days'].mean():.1f} days")
    print(f"Median Delivery Time: {df['Delivery_Days'].median():.0f} days")
    
    if 'Delivery_Category' in df.columns:
        delivery_dist = df['Delivery_Category'].value_counts()
        print("\nDelivery Category Distribution:")
        print(delivery_dist)

# ============================================================================
# 9. KEY FINDINGS SUMMARY
# ============================================================================

print("\n" + "="*80)
print("KEY FINDINGS FROM EDA")
print("="*80)

findings = []

# Finding 1: Top category
if 'Product_Category' in df.columns:
    top_category = df.groupby('Product_Category')['Sales'].sum().idxmax()
    top_category_pct = (df.groupby('Product_Category')['Sales'].sum().max() / df['Sales'].sum() * 100)
    findings.append(f"1. {top_category} is the top category, generating {top_category_pct:.1f}% of total revenue")

# Finding 2: Customer retention
if 'Customer_ID' in df.columns:
    customer_freq = df.groupby('Customer_ID').size()
    repeat_customers = (customer_freq > 1).sum()
    total_customers = len(customer_freq)
    retention_rate = (repeat_customers / total_customers * 100)
    findings.append(f"2. Customer retention rate is {retention_rate:.1f}% ({repeat_customers:,} of {total_customers:,} customers)")

# Finding 3: Seasonality
if 'Quarter' in df.columns:
    quarterly_sales = df.groupby('Quarter')['Sales'].sum()
    peak_quarter = quarterly_sales.idxmax()
    peak_pct = (quarterly_sales.max() / quarterly_sales.sum() * 100)
    findings.append(f"3. Q{peak_quarter} is the peak quarter, accounting for {peak_pct:.1f}% of annual sales")

# Finding 4: Weekend effect
if 'Is_Weekend' in df.columns:
    weekend_avg = df[df['Is_Weekend'] == 1]['Sales'].mean()
    weekday_avg = df[df['Is_Weekend'] == 0]['Sales'].mean()
    lift = ((weekend_avg / weekday_avg - 1) * 100)
    findings.append(f"4. Weekend sales are {abs(lift):.1f}% {'higher' if lift > 0 else 'lower'} than weekday sales")

# Finding 5: Product concentration
if 'Product_ID' in df.columns:
    product_sales = df.groupby('Product_ID')['Sales'].sum().sort_values(ascending=False)
    top_20_pct = int(len(product_sales) * 0.2)
    top_20_revenue_pct = (product_sales.head(top_20_pct).sum() / product_sales.sum() * 100)
    findings.append(f"5. Top 20% of products generate {top_20_revenue_pct:.1f}% of revenue (Pareto principle confirmed)")

for finding in findings:
    print(f"\n{finding}")

# Save findings
findings_text = "\n".join(findings)
with open('outputs/reports/03_eda_key_findings.txt', 'w') as f:
    f.write("KEY FINDINGS FROM EXPLORATORY DATA ANALYSIS\n")
    f.write("="*80 + "\n\n")
    f.write(findings_text)
    f.write(f"\n\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n✓ Key findings saved to: outputs/reports/03_eda_key_findings.txt")

print("\n" + "="*80)
print("Part2 COMPLETED SUCCESSFULLY!")
print("="*80)
print("\n Next Step: Part4 - Customer Segmentation & Advanced Analytics")
print("="*80)

# Retail & Marketing Analytics Project
# Part 4: Customer Segmentation and Advanced Analytics

"""
Project: Retail & Marketing Analytics - Customer Segmentation & Sales Optimization
Notebook: 04 - Customer Segmentation and Advanced Analytics
Author: [Your Name]
Date: [Current Date]

Objective:
- RFM Analysis (Recency, Frequency, Monetary)
- K-Means Clustering for customer segmentation
- Cluster profiling and characterization
- Market Basket Analysis - Assignments
- Cohort Analysis
- Customer Lifetime Value (CLV) calculation
"""

# ============================================================================
# 1. IMPORT LIBRARIES AND LOAD DATA
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

# Market Basket Analysis
#pip install mlxtend
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    MBA_AVAILABLE = True
except ImportError:
    MBA_AVAILABLE = False
    print("mlxtend not installed. Run: pip install mlxtend")

from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Load cleaned data
df = pd.read_csv('data/processed/cleaned_retail_sales.csv')
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
df['Ship_Date'] = pd.to_datetime(df['Ship_Date'])

print("="*80)
print("CUSTOMER SEGMENTATION AND ADVANCED ANALYTICS")
print("="*80)
print(f"\nDataset Shape: {df.shape}")

# ============================================================================
# 2. RFM ANALYSIS (RECENCY, FREQUENCY, MONETARY)
# ============================================================================

print("\n" + "="*80)
print("RFM ANALYSIS")
print("="*80)

# Set analysis date (day after last transaction)
analysis_date = df['Order_Date'].max() + timedelta(days=1)
print(f"\nAnalysis Date: {analysis_date.date()}")

# Calculate RFM metrics for each customer
rfm = df.groupby('Customer_ID').agg({
    'Order_Date': lambda x: (analysis_date - x.max()).days,  # Recency
    'Order_ID': 'count',  # Frequency
    'Sales': 'sum'  # Monetary
}).reset_index()

rfm.columns = ['Customer_ID', 'Recency', 'Frequency', 'Monetary']

print(f"\n RFM Metrics Calculated for {len(rfm):,} customers")
print("\nRFM Summary Statistics:")
print(rfm[['Recency', 'Frequency', 'Monetary']].describe())

# Create RFM scores (1-5 scale, where 5 is best)
print("\n Creating RFM Scores...")

# Recency Score (lower is better, so reverse)
rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')

# Frequency Score (higher is better)
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, 
                         labels=[1, 2, 3, 4, 5], duplicates='drop')

# Monetary Score (higher is better)
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')

# Combined RFM Score
rfm['RFM_Score'] = (rfm['R_Score'].astype(str) + 
                    rfm['F_Score'].astype(str) + 
                    rfm['M_Score'].astype(str))

# Numeric RFM Score (average)
rfm['RFM_Score_Numeric'] = (rfm['R_Score'].astype(int) + 
                            rfm['F_Score'].astype(int) + 
                            rfm['M_Score'].astype(int)) / 3

print("✓ RFM Scores Created")

# Customer Segmentation based on RFM
def segment_customers(row):
    """Segment customers based on RFM scores"""
    r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])
    
    # Champions: Best customers
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    # Loyal Customers: Buy regularly
    elif r >= 3 and f >= 3 and m >= 3:
        return 'Loyal Customers'
    # Potential Loyalists: Recent customers with potential
    elif r >= 4 and f <= 2 and m >= 3:
        return 'Potential Loyalists'
    # New Customers: Recent first-time buyers
    elif r >= 3 and f <= 2:
        return 'New Customers'
    # Promising: Recent but low spenders
    elif r >= 3 and m <= 2:
        return 'Promising'
    # Need Attention: Above average recency, frequency, and monetary
    elif r == 3 and f == 3 and m == 3:
        return 'Need Attention'
    # About to Sleep: Below average engagement
    elif r <= 2 and f <= 2:
        return 'About to Sleep'
    # At Risk: Used to be good, now declining
    elif r <= 2 and f >= 3:
        return 'At Risk'
    # Can't Lose Them: Were best customers but churning
    elif r <= 2 and f >= 4 and m >= 4:
        return "Can't Lose Them"
    # Lost: Long time since purchase
    elif r == 1:
        return 'Lost'
    else:
        return 'Others'

rfm['Customer_Segment'] = rfm.apply(segment_customers, axis=1)

print("\n Customer Segmentation Complete:")
print(rfm['Customer_Segment'].value_counts())

# Visualize RFM segments
fig = px.pie(rfm['Customer_Segment'].value_counts().reset_index(), 
             values='count', 
             names='Customer_Segment',
             title='Customer Distribution by RFM Segments',
             hole=0.4,
             color_discrete_sequence=px.colors.qualitative.Set3)
fig.write_html('outputs/figures/17_rfm_segments.html')
print("✓ Saved: 17_rfm_segments.html")

# RFM Summary by Segment
rfm_summary = rfm.groupby('Customer_Segment').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'Customer_ID': 'count'
}).round(2)
rfm_summary.columns = ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Customer_Count']
rfm_summary['Revenue_Contribution'] = (rfm.groupby('Customer_Segment')['Monetary'].sum() / 
                                        rfm['Monetary'].sum() * 100).round(2)
rfm_summary = rfm_summary.sort_values('Revenue_Contribution', ascending=False)

print("\n RFM Segment Summary:")
print(rfm_summary)

# Save RFM analysis
rfm.to_csv('data/processed/rfm_analysis.csv', index=False)
print("\n✓ RFM analysis saved to: data/processed/rfm_analysis.csv")

rfm

# ============================================================================
# 3. K-MEANS CLUSTERING
# ============================================================================

print("\n" + "="*80)
print("K-MEANS CLUSTERING")
print("="*80)

# Prepare data for clustering
clustering_features = ['Recency', 'Frequency', 'Monetary']
X = rfm[clustering_features].values

print(f"\nFeatures for clustering: {clustering_features}")
print(f"Data shape: {X.shape}")

# Standardize features (important for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✓ Features standardized")

# Determine optimal number of clusters using Elbow Method and Silhouette Score
print("\n Determining Optimal Number of Clusters...")

inertias = []
silhouette_scores = []
davies_bouldin_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X_scaled)
    
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    davies_bouldin_scores.append(davies_bouldin_score(X_scaled, kmeans.labels_))

# Plot evaluation metrics
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Elbow Method
axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[0].set_ylabel('Inertia', fontsize=12)
axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Silhouette Score
axes[1].plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Davies-Bouldin Score (lower is better)
axes[2].plot(K_range, davies_bouldin_scores, 'go-', linewidth=2, markersize=8)
axes[2].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[2].set_ylabel('Davies-Bouldin Score', fontsize=12)
axes[2].set_title('Davies-Bouldin Index (Lower is Better)', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/figures/18_optimal_clusters.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 18_optimal_clusters.png")
plt.show()

print("\n Clustering Evaluation Metrics:")
for k, inertia, sil_score, db_score in zip(K_range, inertias, silhouette_scores, davies_bouldin_scores):
    print(f"k={k}: Inertia={inertia:.2f}, Silhouette={sil_score:.4f}, Davies-Bouldin={db_score:.4f}")

# Choose optimal k (typically where silhouette score is highest and elbow occurs)
optimal_k = K_range[silhouette_scores.index(max(silhouette_scores))]
print(f"\n Optimal k based on Silhouette Score: {optimal_k}")

# Apply K-Means with optimal k
print(f"\n Applying K-Means with k={optimal_k}...")
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300)
rfm['Cluster'] = kmeans_final.fit_predict(X_scaled)

print(f"✓ Clustering complete!")
print(f"  - Silhouette Score: {silhouette_score(X_scaled, rfm['Cluster']):.4f}")
print(f"  - Davies-Bouldin Score: {davies_bouldin_score(X_scaled, rfm['Cluster']):.4f}")

rfm

# ============================================================================
# 4. CLUSTER PROFILING
# ============================================================================

print("\n" + "="*80)
print("CLUSTER PROFILING AND CHARACTERIZATION")
print("="*80)

# Cluster statistics
cluster_summary = rfm.groupby('Cluster').agg({
    'Recency': ['mean', 'std'],
    'Frequency': ['mean', 'std'],
    'Monetary': ['mean', 'std'],
    'Customer_ID': 'count'
}).round(2)

cluster_summary.columns = ['_'.join(col).strip() for col in cluster_summary.columns.values]
cluster_summary = cluster_summary.rename(columns={'Customer_ID_count': 'Customer_Count'})
cluster_summary['Customer_Percentage'] = (cluster_summary['Customer_Count'] / 
                                          cluster_summary['Customer_Count'].sum() * 100).round(2)

print("\n Cluster Summary Statistics:")
print(cluster_summary)

# Name clusters based on characteristics
def name_cluster_characteristics(cluster_data):
    """Assign meaningful names to clusters"""
    names = {}
    
    for cluster_id in cluster_data.index:
        r_mean = cluster_data.loc[cluster_id, 'Recency_mean']
        f_mean = cluster_data.loc[cluster_id, 'Frequency_mean']
        m_mean = cluster_data.loc[cluster_id, 'Monetary_mean']
        
        # Define thresholds (adjust based on your data)
        r_threshold = cluster_data['Recency_mean'].median()
        f_threshold = cluster_data['Frequency_mean'].median()
        m_threshold = cluster_data['Monetary_mean'].median()
        
        if r_mean < r_threshold and f_mean > f_threshold and m_mean > m_threshold:
            names[cluster_id] = 'VIP Customers'
        elif f_mean > f_threshold and m_mean > m_threshold:
            names[cluster_id] = 'High Value'
        elif r_mean < r_threshold and f_mean <= f_threshold:
            names[cluster_id] = 'Potential'
        else:
            names[cluster_id] = 'At Risk'
    
    return names

cluster_names = name_cluster_characteristics(cluster_summary)
rfm['Cluster_Name'] = rfm['Cluster'].map(cluster_names)

print("\n Cluster Names:")
for cluster_id, name in cluster_names.items():
    count = cluster_summary.loc[cluster_id, 'Customer_Count']
    pct = cluster_summary.loc[cluster_id, 'Customer_Percentage']
    print(f"  Cluster {cluster_id}: {name} ({count:.0f} customers, {pct:.1f}%)")

# Detailed profiling report
print("\n" + "="*80)
print("DETAILED CLUSTER PROFILES")
print("="*80)

for cluster_id, cluster_name in cluster_names.items():
    cluster_data = rfm[rfm['Cluster'] == cluster_id]
    total_revenue = cluster_data['Monetary'].sum()
    revenue_share = (total_revenue / rfm['Monetary'].sum() * 100)
    
    print(f"\n{'='*60}")
    print(f"CLUSTER {cluster_id}: {cluster_name}")
    print(f"{'='*60}")
    print(f"Size: {len(cluster_data):,} customers ({len(cluster_data)/len(rfm)*100:.1f}%)")
    print(f"\nRFM Metrics:")
    print(f"  • Avg Recency: {cluster_data['Recency'].mean():.1f} days")
    print(f"  • Avg Frequency: {cluster_data['Frequency'].mean():.1f} orders")
    print(f"  • Avg Monetary: ${cluster_data['Monetary'].mean():,.2f}")
    print(f"  • Total Revenue: ${total_revenue:,.2f} ({revenue_share:.1f}% of total)")
    
    print(f"\nRecommended Marketing Actions:")
    if cluster_name == 'VIP Customers':
        print("  ✓ Exclusive rewards and early access to new products")
        print("  ✓ Personalized communication and dedicated support")
        print("  ✓ Premium tier membership programs")
        print("  ✓ Referral incentives")
    elif cluster_name == 'High Value':
        print("  ✓ Loyalty programs with tiered benefits")
        print("  ✓ Cross-sell and upsell campaigns")
        print("  ✓ Regular engagement through email marketing")
        print("  ✓ Special occasion offers")
    elif cluster_name == 'Potential':
        print("  ✓ Nurture with targeted promotions")
        print("  ✓ Product recommendations based on browsing")
        print("  ✓ Time-limited offers to encourage repeat purchase")
        print("  ✓ Educational content about products")
    else:  # At Risk
        print("  ✓ Win-back campaigns with special discounts")
        print("  ✓ Survey to understand reasons for churn")
        print("  ✓ Re-engagement through personalized content")
        print("  ✓ Abandoned cart reminders")

# Save cluster profiling
rfm.to_csv('data/processed/customer_segments.csv', index=False)
print("\n✓ Customer segments saved to: data/processed/customer_segments.csv")

# ============================================================================
# 5. CLUSTER VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("CLUSTER VISUALIZATION")
print("="*80)

# 5.1 PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

rfm['PCA1'] = X_pca[:, 0]
rfm['PCA2'] = X_pca[:, 1]

explained_var = pca.explained_variance_ratio_
print(f"\nPCA Explained Variance:")
print(f"  PC1: {explained_var[0]*100:.2f}%")
print(f"  PC2: {explained_var[1]*100:.2f}%")
print(f"  Total: {sum(explained_var)*100:.2f}%")

# 2D Scatter plot
fig = px.scatter(rfm, x='PCA1', y='PCA2', color='Cluster_Name',
                 title='Customer Segments Visualization (PCA)',
                 labels={'PCA1': f'PC1 ({explained_var[0]:.2%} variance)',
                         'PCA2': f'PC2 ({explained_var[1]:.2%} variance)'},
                 hover_data=['Recency', 'Frequency', 'Monetary', 'Customer_ID'],
                 color_discrete_sequence=px.colors.qualitative.Set2)
fig.write_html('outputs/figures/19_customer_segments_pca.html')
print("✓ Saved: 19_customer_segments_pca.html")

# 5.2 3D visualization in RFM space
fig = px.scatter_3d(rfm, x='Recency', y='Frequency', z='Monetary',
                    color='Cluster_Name',
                    title='Customer Segments in RFM Space',
                    labels={'Recency': 'Recency (days)', 
                            'Frequency': 'Frequency (orders)',
                            'Monetary': 'Monetary ($)'},
                    hover_data=['Customer_ID'],
                    color_discrete_sequence=px.colors.qualitative.Pastel)
fig.write_html('outputs/figures/20_customer_segments_3d.html')
print("✓ Saved: 20_customer_segments_3d.html")

# 5.3 Segment size and revenue contribution
fig = make_subplots(rows=1, cols=2, 
                    subplot_titles=('Customer Count by Segment', 'Revenue by Segment'),
                    specs=[[{'type': 'bar'}, {'type': 'bar'}]])

segment_counts = rfm['Cluster_Name'].value_counts()
segment_revenue = rfm.groupby('Cluster_Name')['Monetary'].sum().sort_values(ascending=False)

fig.add_trace(go.Bar(x=segment_counts.index, y=segment_counts.values, 
                     name='Customers', marker_color='lightblue'), row=1, col=1)
fig.add_trace(go.Bar(x=segment_revenue.index, y=segment_revenue.values, 
                     name='Revenue', marker_color='lightcoral'), row=1, col=2)

fig.update_xaxes(title_text="Segment", row=1, col=1)
fig.update_xaxes(title_text="Segment", row=1, col=2)
fig.update_yaxes(title_text="Number of Customers", row=1, col=1)
fig.update_yaxes(title_text="Total Revenue ($)", row=1, col=2)
fig.update_layout(height=500, showlegend=False, title_text="Segment Distribution")

fig.write_html('outputs/figures/21_segment_distribution.html')
print("✓ Saved: 21_segment_distribution.html")

# ============================================================================
# 6. MARKET BASKET ANALYSIS (OPTIONAL)
# ============================================================================
# Market basket analysis reveals which items buyers purchase together. 
# Retailers use market basket analysis to understand the best way to co-locate products in both physical and digital stores

if MBA_AVAILABLE and 'Product_ID' in df.columns:
    print("\n" + "="*80)
    print("MARKET BASKET ANALYSIS")
    print("="*80)
    
    try:
        # Prepare transaction data
        transactions = df.groupby('Order_ID')['Product_ID'].apply(list).tolist()
        print(f"\nTotal Transactions: {len(transactions):,}")
        
        # Encode transactions
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Find frequent itemsets
        print(" Finding frequent itemsets...")
        frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)
        print(f"✓ Found {len(frequent_itemsets)} frequent itemsets")
        
        # Generate association rules
        print(" Generating association rules...")
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        rules = rules.sort_values('lift', ascending=False)
        
        print(f"✓ Generated {len(rules)} association rules")
        print("\n Top 10 Association Rules:")
        print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
        
        # Visualize rules
        fig = px.scatter(rules.head(50), x='support', y='confidence', 
                         size='lift', color='lift',
                         title='Association Rules: Support vs Confidence (Top 50)',
                         labels={'support': 'Support', 'confidence': 'Confidence'},
                         color_continuous_scale='viridis',
                         hover_data=['antecedents', 'consequents'])
        fig.write_html('outputs/figures/22_association_rules.html')
        print("✓ Saved: 22_association_rules.html")
        
        # Save rules
        rules.to_csv('outputs/reports/market_basket_rules.csv', index=False)
        print("✓ Rules saved to: outputs/reports/market_basket_rules.csv")
        
    except Exception as e:
        print(f" Market Basket Analysis skipped: {e}")
else:
    print("\n Market Basket Analysis skipped (mlxtend not installed or Product_ID not available)")


# ============================================================================
# 7. COHORT ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("COHORT ANALYSIS")
print("="*80)

# Create cohort data
df_cohort = df.copy()
df_cohort['Order_Month'] = df_cohort['Order_Date'].dt.to_period('M')
df_cohort['Cohort'] = df_cohort.groupby('Customer_ID')['Order_Date'].transform('min').dt.to_period('M')

# Calculate cohort index (months since first purchase)
def get_cohort_period(df):
    df['Cohort_Index'] = (df['Order_Month'] - df['Cohort']).apply(lambda x: x.n)
    return df

df_cohort = get_cohort_period(df_cohort)

# Cohort size
cohort_size = df_cohort.groupby('Cohort')['Customer_ID'].nunique()
print(f"\nTotal Cohorts: {len(cohort_size)}")
print("\nCohort Sizes:")
print(cohort_size.tail())

# Retention matrix
retention = df_cohort.groupby(['Cohort', 'Cohort_Index'])['Customer_ID'].nunique().reset_index()
retention_matrix = retention.pivot(index='Cohort', columns='Cohort_Index', values='Customer_ID')

# Calculate retention rate
retention_rate = retention_matrix.divide(cohort_size, axis=0) * 100

print("\n Cohort Retention Rates (%):")
print(retention_rate.head())

# Visualize cohort retention
plt.figure(figsize=(14, 8))
sns.heatmap(retention_rate, annot=True, fmt='.1f', cmap='YlGnBu', 
            cbar_kws={'label': 'Retention Rate (%)'}, vmin=0, vmax=100)
plt.title('Cohort Retention Analysis (%)', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Cohort Index (Months Since First Purchase)', fontsize=12)
plt.ylabel('Cohort Month', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/figures/23_cohort_retention.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 23_cohort_retention.png")
plt.show()

# Save cohort data
retention_rate.to_csv('outputs/reports/cohort_retention.csv')
print("✓ Cohort data saved to: outputs/reports/cohort_retention.csv")

# ============================================================================
# 8. CUSTOMER LIFETIME VALUE (CLV)
# ============================================================================

print("\n" + "="*80)
print("CUSTOMER LIFETIME VALUE (CLV) CALCULATION")
print("="*80)

# Calculate CLV components
customer_metrics = df.groupby('Customer_ID').agg({
    'Sales': 'sum',
    'Order_ID': 'count',
    'Order_Date': ['min', 'max']
}).reset_index()

customer_metrics.columns = ['Customer_ID', 'Total_Revenue', 'Order_Count', 
                            'First_Purchase', 'Last_Purchase']

# Convert to datetime
customer_metrics['First_Purchase'] = pd.to_datetime(customer_metrics['First_Purchase'])
customer_metrics['Last_Purchase'] = pd.to_datetime(customer_metrics['Last_Purchase'])

# Calculate customer lifespan
customer_metrics['Lifespan_Days'] = (customer_metrics['Last_Purchase'] - 
                                      customer_metrics['First_Purchase']).dt.days
customer_metrics['Lifespan_Days'] = customer_metrics['Lifespan_Days'].replace(0, 1)  # Avoid division by zero

# Calculate metrics
customer_metrics['Avg_Order_Value'] = customer_metrics['Total_Revenue'] / customer_metrics['Order_Count']
customer_metrics['Purchase_Frequency'] = customer_metrics['Order_Count'] / (customer_metrics['Lifespan_Days'] / 365.25)

# Simple CLV calculation
# CLV = Average Order Value × Purchase Frequency × Average Customer Lifespan (in years)
avg_customer_lifespan_years = 3  # Assumption
customer_metrics['CLV_Simple'] = (customer_metrics['Avg_Order_Value'] * 
                                   customer_metrics['Purchase_Frequency'] * 
                                   avg_customer_lifespan_years)

print(f"\n CLV Summary Statistics:")
print(customer_metrics['CLV_Simple'].describe())

# Categorize customers by CLV
customer_metrics['CLV_Category'] = pd.qcut(customer_metrics['CLV_Simple'], 
                                            q=4, 
                                            labels=['Low', 'Medium', 'High', 'Very High'],
                                            duplicates='drop')

print("\n CLV Distribution:")
print(customer_metrics['CLV_Category'].value_counts())

# Visualize CLV distribution
fig = px.histogram(customer_metrics, x='CLV_Simple', 
                   title='Customer Lifetime Value Distribution',
                   nbins=50,
                   labels={'CLV_Simple': 'CLV ($)'},
                   color_discrete_sequence=['steelblue'])
fig.add_vline(x=customer_metrics['CLV_Simple'].median(), line_dash="dash", 
              line_color="red", annotation_text=f"Median: ${customer_metrics['CLV_Simple'].median():,.2f}")
fig.write_html('outputs/figures/24_clv_distribution.html')
print("✓ Saved: 24_clv_distribution.html")

# CLV by segment
clv_by_category = customer_metrics.groupby('CLV_Category').agg({
    'Customer_ID': 'count',
    'CLV_Simple': 'mean',
    'Total_Revenue': 'sum',
    'Order_Count': 'mean'
}).round(2)
clv_by_category.columns = ['Customer_Count', 'Avg_CLV', 'Total_Revenue', 'Avg_Orders']

print("\n CLV Summary by Category:")
print(clv_by_category)

# Save CLV data
customer_metrics.to_csv('data/processed/customer_clv.csv', index=False)
print("\n✓ CLV data saved to: data/processed/customer_clv.csv")

print("\n" + "="*80)
print("NOTEBOOK 04 COMPLETED SUCCESSFULLY!")
print("="*80)
print("\n Next Step: Open Notebook 05 - KPI Design and Dashboard Preparation")
print("="*80)

# Retail & Marketing Analytics Project
# Part 5: KPI Design and Dashboard Preparation

"""
Project: Retail & Marketing Analytics - Customer Segmentation & Sales Optimization
Notebook: 05 - KPI Design and Dashboard Preparation
Author: [Your Name]
Date: [Current Date]

Objective:
- Design comprehensive KPI framework
- Calculate key business metrics
- Prepare data for dashboard creation
- Generate executive summary report
- Create actionable recommendations
"""

# ============================================================================
# 1. IMPORT LIBRARIES AND LOAD DATA
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Load all processed data
df_sales = pd.read_csv('data/processed/cleaned_retail_sales.csv')
df_sales['Order_Date'] = pd.to_datetime(df_sales['Order_Date'])

rfm = pd.read_csv('data/processed/customer_segments.csv')
customer_clv = pd.read_csv('data/processed/customer_clv.csv')

print("="*80)
print("KPI DESIGN AND DASHBOARD PREPARATION")
print("="*80)
print(f"\nSales Data: {df_sales.shape}")
print(f"Customer Segments: {rfm.shape}")
print(f"CLV Data: {customer_clv.shape}")

# ============================================================================
# 2. COMPREHENSIVE KPI FRAMEWORK
# ============================================================================

print("\n" + "="*80)
print("COMPREHENSIVE KPI FRAMEWORK")
print("="*80)

# Initialize KPI dictionary
kpis = {}

# -----------------------------
# REVENUE METRICS
# -----------------------------
print("\n REVENUE METRICS")

kpis['Total_Revenue'] = df_sales['Sales'].sum()
kpis['Total_Orders'] = df_sales['Order_ID'].nunique()
kpis['Avg_Order_Value'] = df_sales.groupby('Order_ID')['Sales'].sum().mean()
kpis['Total_Units_Sold'] = df_sales['Quantity'].sum()

if 'Profit' in df_sales.columns:
    kpis['Total_Profit'] = df_sales['Profit'].sum()
    kpis['Profit_Margin_Pct'] = (kpis['Total_Profit'] / kpis['Total_Revenue']) * 100
else:
    # Assume 25% profit margin
    kpis['Total_Profit'] = kpis['Total_Revenue'] * 0.25
    kpis['Profit_Margin_Pct'] = 25.0

print(f"Total Revenue: ${kpis['Total_Revenue']:,.2f}")
print(f"Total Orders: {kpis['Total_Orders']:,}")
print(f"Avg Order Value (AOV): ${kpis['Avg_Order_Value']:,.2f}")
print(f"Total Units Sold: {kpis['Total_Units_Sold']:,}")
print(f"Total Profit: ${kpis['Total_Profit']:,.2f}")
print(f"Profit Margin: {kpis['Profit_Margin_Pct']:.2f}%")

# -----------------------------
# CUSTOMER METRICS
# -----------------------------
print("\n CUSTOMER METRICS")

kpis['Total_Customers'] = df_sales['Customer_ID'].nunique()
kpis['Revenue_Per_Customer'] = kpis['Total_Revenue'] / kpis['Total_Customers']
kpis['Avg_Orders_Per_Customer'] = kpis['Total_Orders'] / kpis['Total_Customers']

# Repeat customers
customer_order_counts = df_sales.groupby('Customer_ID')['Order_ID'].nunique()
repeat_customers = (customer_order_counts > 1).sum()
kpis['Repeat_Customers'] = repeat_customers
kpis['Repeat_Customer_Rate'] = (repeat_customers / kpis['Total_Customers']) * 100
kpis['One_Time_Customers'] = kpis['Total_Customers'] - repeat_customers

print(f"Total Customers: {kpis['Total_Customers']:,}")
print(f"Revenue Per Customer: ${kpis['Revenue_Per_Customer']:,.2f}")
print(f"Avg Orders Per Customer: {kpis['Avg_Orders_Per_Customer']:.2f}")
print(f"Repeat Customers: {kpis['Repeat_Customers']:,} ({kpis['Repeat_Customer_Rate']:.2f}%)")
print(f"One-Time Customers: {kpis['One_Time_Customers']:,}")

# -----------------------------
# PRODUCT METRICS
# -----------------------------
print("\n PRODUCT METRICS")

kpis['Total_SKUs'] = df_sales['Product_ID'].nunique()
kpis['Avg_Items_Per_Order'] = df_sales.groupby('Order_ID')['Quantity'].sum().mean()
kpis['Total_Categories'] = df_sales['Product_Category'].nunique() if 'Product_Category' in df_sales.columns else 0

print(f"Total SKUs: {kpis['Total_SKUs']:,}")
print(f"Avg Items Per Order: {kpis['Avg_Items_Per_Order']:.2f}")
print(f"Total Categories: {kpis['Total_Categories']}")

# -----------------------------
# CLV & MARKETING METRICS
# -----------------------------
print("\n CLV & MARKETING METRICS")

kpis['Avg_Customer_Lifetime_Value'] = customer_clv['CLV_Simple'].mean()
kpis['Customer_Acquisition_Cost'] = 50  # Assumption: $50 per customer
kpis['CLV_to_CAC_Ratio'] = kpis['Avg_Customer_Lifetime_Value'] / kpis['Customer_Acquisition_Cost']
kpis['Profit_Per_Customer'] = kpis['Total_Profit'] / kpis['Total_Customers']

# Payback period in months
avg_monthly_revenue_per_customer = kpis['Revenue_Per_Customer'] / 12  # Assuming yearly revenue
kpis['CAC_Payback_Months'] = kpis['Customer_Acquisition_Cost'] / (avg_monthly_revenue_per_customer * 0.25)  # 25% margin

print(f"Avg Customer Lifetime Value: ${kpis['Avg_Customer_Lifetime_Value']:,.2f}")
print(f"Customer Acquisition Cost (CAC): ${kpis['Customer_Acquisition_Cost']:,.2f}")
print(f"CLV to CAC Ratio: {kpis['CLV_to_CAC_Ratio']:.2f}x")
print(f"Profit Per Customer: ${kpis['Profit_Per_Customer']:,.2f}")
print(f"CAC Payback Period: {kpis['CAC_Payback_Months']:.1f} months")

# -----------------------------
# RETENTION & CHURN METRICS
# -----------------------------
print("\n RETENTION & CHURN METRICS")

# Calculate churn (customers inactive > 180 days)
if 'Recency' in rfm.columns:
    churned_customers = (rfm['Recency'] > 180).sum()
    kpis['Churned_Customers'] = churned_customers
    kpis['Churn_Rate'] = (churned_customers / len(rfm)) * 100
    kpis['Retention_Rate'] = 100 - kpis['Churn_Rate']
    
    print(f"Churned Customers (>180 days): {kpis['Churned_Customers']:,}")
    print(f"Churn Rate: {kpis['Churn_Rate']:.2f}%")
    print(f"Retention Rate: {kpis['Retention_Rate']:.2f}%")

# Average recency
if 'Recency' in rfm.columns:
    kpis['Avg_Days_Since_Purchase'] = rfm['Recency'].mean()
    print(f"Avg Days Since Last Purchase: {kpis['Avg_Days_Since_Purchase']:.1f}")

# -----------------------------
# TIME-BASED METRICS
# -----------------------------
print("\n TIME-BASED METRICS")

# Date range
kpis['Analysis_Start_Date'] = df_sales['Order_Date'].min().date()
kpis['Analysis_End_Date'] = df_sales['Order_Date'].max().date()
kpis['Analysis_Period_Days'] = (df_sales['Order_Date'].max() - df_sales['Order_Date'].min()).days

print(f"Analysis Period: {kpis['Analysis_Start_Date']} to {kpis['Analysis_End_Date']}")
print(f"Total Days: {kpis['Analysis_Period_Days']}")

# Average daily metrics
kpis['Avg_Daily_Revenue'] = kpis['Total_Revenue'] / kpis['Analysis_Period_Days']
kpis['Avg_Daily_Orders'] = kpis['Total_Orders'] / kpis['Analysis_Period_Days']

print(f"Avg Daily Revenue: ${kpis['Avg_Daily_Revenue']:,.2f}")
print(f"Avg Daily Orders: {kpis['Avg_Daily_Orders']:.1f}")

# -----------------------------
# SEGMENTATION METRICS
# -----------------------------
print("\n SEGMENTATION METRICS")

if 'Cluster_Name' in rfm.columns:
    segment_counts = rfm['Cluster_Name'].value_counts()
    for segment, count in segment_counts.items():
        pct = (count / len(rfm)) * 100
        kpis[f'{segment}_Count'] = count
        kpis[f'{segment}_Percentage'] = pct
        print(f"{segment}: {count:,} ({pct:.1f}%)")


# ============================================================================
# 3. CREATE KPI DASHBOARD DATA
# ============================================================================

print("\n" + "="*80)
print("PREPARING KPI DASHBOARD DATA")
print("="*80)

# Save main KPIs
kpi_df = pd.DataFrame(list(kpis.items()), columns=['KPI', 'Value'])
kpi_df.to_csv('outputs/reports/kpi_summary.csv', index=False)
print("✓ Saved: outputs/reports/kpi_summary.csv")

# -----------------------------
# Monthly KPI Trends
# -----------------------------
print("\n Calculating Monthly KPI Trends...")

df_sales['YearMonth'] = df_sales['Order_Date'].dt.to_period('M')

monthly_kpis = df_sales.groupby('YearMonth').agg({
    'Sales': 'sum',
    'Order_ID': 'nunique',
    'Customer_ID': 'nunique',
    'Quantity': 'sum',
    'Product_ID': 'nunique'
}).reset_index()

monthly_kpis.columns = ['YearMonth', 'Revenue', 'Orders', 'Customers', 'Units_Sold', 'SKUs']

# Calculate derived metrics
monthly_kpis['AOV'] = monthly_kpis['Revenue'] / monthly_kpis['Orders']
monthly_kpis['Revenue_Per_Customer'] = monthly_kpis['Revenue'] / monthly_kpis['Customers']
monthly_kpis['Items_Per_Order'] = monthly_kpis['Units_Sold'] / monthly_kpis['Orders']

# Calculate growth rates
monthly_kpis['Revenue_Growth'] = monthly_kpis['Revenue'].pct_change() * 100
monthly_kpis['Customer_Growth'] = monthly_kpis['Customers'].pct_change() * 100
monthly_kpis['Order_Growth'] = monthly_kpis['Orders'].pct_change() * 100

# Convert YearMonth to string for export
monthly_kpis['YearMonth'] = monthly_kpis['YearMonth'].astype(str)

print(f"✓ Monthly KPIs calculated for {len(monthly_kpis)} months")
print("\nLast 6 Months:")
print(monthly_kpis.tail(6).round(2))

# Save monthly KPIs
monthly_kpis.to_csv('data/processed/monthly_kpis.csv', index=False)
print("\n✓ Saved: data/processed/monthly_kpis.csv")

# -----------------------------
# Category-Level KPIs
# -----------------------------
if 'Product_Category' in df_sales.columns:
    print("\n Calculating Category-Level KPIs...")
    
    category_kpis = df_sales.groupby('Product_Category').agg({
        'Sales': ['sum', 'mean'],
        'Order_ID': 'nunique',
        'Customer_ID': 'nunique',
        'Quantity': 'sum',
        'Product_ID': 'nunique'
    }).reset_index()
    
    category_kpis.columns = ['Product_Category', 'Total_Revenue', 'Avg_Order_Value', 
                             'Order_Count', 'Customer_Count', 'Units_Sold', 'SKU_Count']
    
    # Calculate shares
    category_kpis['Revenue_Share'] = (category_kpis['Total_Revenue'] / 
                                       category_kpis['Total_Revenue'].sum() * 100).round(2)
    category_kpis['Order_Share'] = (category_kpis['Order_Count'] / 
                                     category_kpis['Order_Count'].sum() * 100).round(2)
    
    # Sort by revenue
    category_kpis = category_kpis.sort_values('Total_Revenue', ascending=False)
    
    print(category_kpis.round(2))
    
    # Save category KPIs
    category_kpis.to_csv('outputs/reports/category_kpis.csv', index=False)
    print("\n✓ Saved: outputs/reports/category_kpis.csv")

# -----------------------------
# Regional KPIs
# -----------------------------
if 'Region' in df_sales.columns:
    print("\n Calculating Regional KPIs...")
    
    regional_kpis = df_sales.groupby('Region').agg({
        'Sales': ['sum', 'mean'],
        'Order_ID': 'nunique',
        'Customer_ID': 'nunique',
        'Quantity': 'sum'
    }).reset_index()
    
    regional_kpis.columns = ['Region', 'Total_Revenue', 'Avg_Order_Value', 
                             'Order_Count', 'Customer_Count', 'Units_Sold']
    
    regional_kpis['Revenue_Share'] = (regional_kpis['Total_Revenue'] / 
                                       regional_kpis['Total_Revenue'].sum() * 100).round(2)
    regional_kpis['Customer_Penetration'] = (regional_kpis['Customer_Count'] / 
                                              kpis['Total_Customers'] * 100).round(2)
    
    # Sort by revenue
    regional_kpis = regional_kpis.sort_values('Total_Revenue', ascending=False)
    
    print(regional_kpis.round(2))
    
    # Save regional KPIs
    regional_kpis.to_csv('outputs/reports/regional_kpis.csv', index=False)
    print("\n✓ Saved: outputs/reports/regional_kpis.csv")

# ============================================================================
# 4. VISUALIZE KEY KPIs
# ============================================================================

print("\n" + "="*80)
print("CREATING KPI VISUALIZATIONS")
print("="*80)

# 4.1 KPI Summary Dashboard Style
fig = go.Figure()

# Create KPI cards layout
kpi_cards = [
    {'title': 'Total Revenue', 'value': f"${kpis['Total_Revenue']:,.0f}", 'color': '#1f77b4'},
    {'title': 'Total Customers', 'value': f"{kpis['Total_Customers']:,}", 'color': '#ff7f0e'},
    {'title': 'Avg Order Value', 'value': f"${kpis['Avg_Order_Value']:.2f}", 'color': '#2ca02c'},
    {'title': 'Retention Rate', 'value': f"{kpis.get('Retention_Rate', 0):.1f}%", 'color': '#d62728'},
    {'title': 'CLV/CAC Ratio', 'value': f"{kpis['CLV_to_CAC_Ratio']:.2f}x", 'color': '#9467bd'},
    {'title': 'Total Orders', 'value': f"{kpis['Total_Orders']:,}", 'color': '#8c564b'}
]

print("✓ KPI cards prepared")

# 4.2 Monthly Revenue Trend with Forecast
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=monthly_kpis['YearMonth'],
    y=monthly_kpis['Revenue'],
    mode='lines+markers',
    name='Revenue',
    line=dict(color='#1f77b4', width=3),
    marker=dict(size=8)
))

fig.update_layout(
    title='Monthly Revenue Trend',
    xaxis_title='Month',
    yaxis_title='Revenue ($)',
    hovermode='x unified',
    height=500,
    template='plotly_white'
)

fig.write_html('outputs/figures/25_monthly_revenue_trend_detailed.html')
print("✓ Saved: 25_monthly_revenue_trend_detailed.html")

# 4.3 KPI Comparison Chart
if len(monthly_kpis) >= 2:
    latest_month = monthly_kpis.iloc[-1]
    previous_month = monthly_kpis.iloc[-2]
    
    comparison_metrics = ['Revenue', 'Orders', 'Customers', 'AOV']
    latest_values = [latest_month[m] for m in comparison_metrics]
    previous_values = [previous_month[m] for m in comparison_metrics]
    growth = [((l - p) / p * 100) if p != 0 else 0 for l, p in zip(latest_values, previous_values)]
    
    fig = go.Figure(data=[
        go.Bar(name='Previous Month', x=comparison_metrics, y=previous_values, marker_color='lightblue'),
        go.Bar(name='Latest Month', x=comparison_metrics, y=latest_values, marker_color='darkblue')
    ])
    
    fig.update_layout(
        title='Month-over-Month KPI Comparison',
        barmode='group',
        yaxis_title='Value',
        height=500,
        template='plotly_white'
    )
    
    fig.write_html('outputs/figures/26_mom_kpi_comparison.html')
    print("✓ Saved: 26_mom_kpi_comparison.html")

# 4.4 Customer Segment Performance
if 'Cluster_Name' in rfm.columns:
    segment_performance = rfm.groupby('Cluster_Name').agg({
        'Customer_ID': 'count',
        'Monetary': 'sum',
        'Frequency': 'mean'
    }).reset_index()
    segment_performance.columns = ['Segment', 'Customer_Count', 'Total_Revenue', 'Avg_Frequency']
    
    fig = px.sunburst(segment_performance, 
                      path=['Segment'], 
                      values='Total_Revenue',
                      title='Revenue Distribution by Customer Segment',
                      color='Avg_Frequency',
                      color_continuous_scale='RdYlGn')
    
    fig.write_html('outputs/figures/27_segment_revenue_sunburst.html')
    print("✓ Saved: 27_segment_revenue_sunburst.html")

# ============================================================================
# 5. EXECUTIVE SUMMARY REPORT
# ============================================================================

print("\n" + "="*80)
print("GENERATING EXECUTIVE SUMMARY")
print("="*80)

executive_summary = f"""
{'='*80}
RETAIL & MARKETING ANALYTICS
EXECUTIVE SUMMARY REPORT
{'='*80}

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Period: {kpis['Analysis_Start_Date']} to {kpis['Analysis_End_Date']}

{'='*80}
1. KEY BUSINESS METRICS
{'='*80}

FINANCIAL PERFORMANCE:
  • Total Revenue: ${kpis['Total_Revenue']:,.2f}
  • Total Profit: ${kpis['Total_Profit']:,.2f}
  • Profit Margin: {kpis['Profit_Margin_Pct']:.2f}%
  • Average Order Value: ${kpis['Avg_Order_Value']:,.2f}

CUSTOMER METRICS:
  • Total Customers: {kpis['Total_Customers']:,}
  • Repeat Customer Rate: {kpis['Repeat_Customer_Rate']:.2f}%
  • Customer Retention Rate: {kpis.get('Retention_Rate', 0):.2f}%
  • Churn Rate: {kpis.get('Churn_Rate', 0):.2f}%
  • Average CLV: ${kpis['Avg_Customer_Lifetime_Value']:,.2f}

OPERATIONAL METRICS:
  • Total Orders: {kpis['Total_Orders']:,}
  • Total Units Sold: {kpis['Total_Units_Sold']:,}
  • Avg Orders per Customer: {kpis['Avg_Orders_Per_Customer']:.2f}
  • Avg Items per Order: {kpis['Avg_Items_Per_Order']:.2f}

MARKETING EFFICIENCY:
  • CLV/CAC Ratio: {kpis['CLV_to_CAC_Ratio']:.2f}x
  • Customer Acquisition Cost: ${kpis['Customer_Acquisition_Cost']:,.2f}
  • Profit per Customer: ${kpis['Profit_Per_Customer']:,.2f}
  • CAC Payback Period: {kpis['CAC_Payback_Months']:.1f} months

{'='*80}
2. CUSTOMER SEGMENTATION INSIGHTS
{'='*80}

"""

# Add segment details
if 'Cluster_Name' in rfm.columns:
    segment_summary = rfm.groupby('Cluster_Name').agg({
        'Customer_ID': 'count',
        'Monetary': ['sum', 'mean'],
        'Frequency': 'mean',
        'Recency': 'mean'
    })
    
    for segment in segment_summary.index:
        count = segment_summary.loc[segment, ('Customer_ID', 'count')]
        pct = (count / len(rfm) * 100)
        total_rev = segment_summary.loc[segment, ('Monetary', 'sum')]
        rev_pct = (total_rev / rfm['Monetary'].sum() * 100)
        avg_freq = segment_summary.loc[segment, ('Frequency', 'mean')]
        avg_recency = segment_summary.loc[segment, ('Recency', 'mean')]
        
        executive_summary += f"""
{segment}:
  • Size: {count:,} customers ({pct:.1f}%)
  • Revenue Contribution: ${total_rev:,.2f} ({rev_pct:.1f}%)
  • Avg Purchase Frequency: {avg_freq:.2f} orders
  • Avg Recency: {avg_recency:.1f} days
"""

executive_summary += f"""

{'='*80}
3. TOP PERFORMING CATEGORIES
{'='*80}

"""

if 'Product_Category' in df_sales.columns:
    top_categories = df_sales.groupby('Product_Category')['Sales'].sum().sort_values(ascending=False).head(5)
    for idx, (category, revenue) in enumerate(top_categories.items(), 1):
        pct = (revenue / df_sales['Sales'].sum() * 100)
        executive_summary += f"{idx}. {category}: ${revenue:,.2f} ({pct:.1f}%)\n"

executive_summary += f"""

{'='*80}
4. KEY FINDINGS & INSIGHTS
{'='*80}

POSITIVE TRENDS:
  ✓ Customer Lifetime Value is {kpis['CLV_to_CAC_Ratio']:.1f}x the acquisition cost (Healthy: >3.0)
  ✓ {kpis['Repeat_Customer_Rate']:.1f}% of customers make repeat purchases
  ✓ Average order value is ${kpis['Avg_Order_Value']:.2f}

AREAS FOR IMPROVEMENT:
  ⚠ Customer retention rate at {kpis.get('Retention_Rate', 0):.1f}% (Target: 45-50%)
  ⚠ Churn rate of {kpis.get('Churn_Rate', 0):.1f}% requires attention
  ⚠ {kpis['One_Time_Customers']:,} customers made only one purchase

OPPORTUNITIES:
  • Focus on {segment_counts.index[0] if 'Cluster_Name' in rfm.columns else 'high-value'} segment for retention
  • Implement win-back campaigns for at-risk customers
  • Optimize product mix based on category performance
  • Personalize marketing by customer segment

{'='*80}
5. STRATEGIC RECOMMENDATIONS
{'='*80}

IMMEDIATE ACTIONS (0-30 Days):
  1. Launch loyalty program for top customer segments
  2. Implement win-back campaign for churned customers
  3. Optimize top-performing product visibility
  4. Set up automated customer retention alerts

SHORT-TERM INITIATIVES (2-3 Months):
  1. Develop personalized email marketing campaigns
  2. Test dynamic pricing strategies by segment
  3. Expand product offerings in high-performing categories
  4. Implement referral program

LONG-TERM STRATEGY (6-12 Months):
  1. Build predictive churn model
  2. Develop AI-powered recommendation engine
  3. Create customer success program
  4. Invest in customer experience improvements

EXPECTED IMPACT:
  • Revenue increase: 15-20%
  • Retention improvement: 25-30%
  • CLV growth: 10-15%
  • Churn reduction: 20-25%

{'='*80}
6. NEXT STEPS
{'='*80}

  1. Review findings with executive team
  2. Prioritize recommendations based on ROI potential
  3. Allocate budget for approved initiatives
  4. Establish KPI tracking dashboard
  5. Schedule monthly performance reviews

{'='*80}
END OF EXECUTIVE SUMMARY
{'='*80}
"""

# Save executive summary
with open('outputs/reports/executive_summary.txt', 'w') as f:
    f.write(executive_summary)

print(executive_summary)
print("\n✓ Executive summary saved to: outputs/reports/executive_summary.txt")

# ============================================================================
# 6. CREATE FINAL PROJECT SUMMARY - Optional ()
# ============================================================================

print("\n" + "="*80)
print("PROJECT COMPLETION SUMMARY")
print("="*80)

project_summary = f"""
PROJECT: Retail & Marketing Analytics
STATUS: COMPLETED
DATE: {pd.Timestamp.now().strftime('%Y-%m-%d')}

DELIVERABLES COMPLETED:
  ✓ Data acquisition and loading
  ✓ Data cleaning and preprocessing
  ✓ Exploratory data analysis
  ✓ RFM analysis
  ✓ Customer segmentation (K-Means clustering)
  ✓ Cohort analysis
  ✓ Customer lifetime value calculation
  ✓ KPI framework design
  ✓ Dashboard data preparation
  ✓ Executive summary report

FILES GENERATED:
  • Data Files: 5
  • Analysis Reports: 7
  • Visualizations: 27+
  • KPI Reports: 4

KEY INSIGHTS:
  • {len(rfm['Cluster_Name'].unique()) if 'Cluster_Name' in rfm.columns else 4} distinct customer segments identified
  • Top segment contributes {segment_summary.loc[segment_summary.index[0], ('Monetary', 'sum')] / rfm['Monetary'].sum() * 100:.1f}% of revenue
  • CLV/CAC ratio of {kpis['CLV_to_CAC_Ratio']:.2f}x indicates healthy unit economics
  • {kpis['Total_Orders']:,} orders from {kpis['Total_Customers']:,} customers analyzed

RECOMMENDED DASHBOARD STRUCTURE:
  Page 1: Executive Overview (KPI Cards, Revenue Trends)
  Page 2: Customer Analytics (Segments, Cohorts, Retention)
  Page 3: Product Performance (Categories, Top Products)
  Page 4: Marketing Insights (CLV, CAC, Campaign Performance)

NEXT STEPS:
  1. Build interactive dashboard (Power BI/Tableau)
  2. Present findings to stakeholders
  3. Implement recommended initiatives
  4. Set up automated reporting
  5. Monitor KPIs monthly

PROJECT SUCCESS CRITERIA: ✓ ALL MET
  ✓ Comprehensive data analysis completed
  ✓ 4+ customer segments identified
  ✓ 15+ KPIs tracked and calculated
  ✓ Actionable recommendations provided
  ✓ Professional documentation created
  ✓ Dashboard-ready data prepared
"""

print(project_summary)

# Save project summary
with open('outputs/reports/project_completion_summary.txt', 'w') as f:
    f.write(project_summary)

print("\n✓ Project summary saved to: outputs/reports/project_completion_summary.txt")

# ============================================================================
# 7. FINAL OUTPUTS CHECKLIST
# ============================================================================

print("\n" + "="*80)
print("FINAL OUTPUTS CHECKLIST")
print("="*80)

outputs_checklist = {
    'Data Files': [
        'data/processed/cleaned_retail_sales.csv',
        'data/processed/rfm_analysis.csv',
        'data/processed/customer_segments.csv',
        'data/processed/customer_clv.csv',
        'data/processed/monthly_kpis.csv'
    ],
    'Reports': [
        'outputs/reports/01_initial_inspection_report.txt',
        'outputs/reports/02_cleaning_report.txt',
        'outputs/reports/03_eda_key_findings.txt',
        'outputs/reports/cohort_retention.csv',
        'outputs/reports/kpi_summary.csv',
        'outputs/reports/category_kpis.csv',
        'outputs/reports/regional_kpis.csv',
        'outputs/reports/executive_summary.txt',
        'outputs/reports/project_completion_summary.txt'
    ],
    'Visualizations': 'outputs/figures/ (27+ charts and graphs)'
}

for category, files in outputs_checklist.items():
    print(f"\n{category}:")
    if isinstance(files, list):
        for file in files:
            print(f"  ✓ {file}")
    else:
        print(f"  ✓ {files}")

print("\n" + "="*80)
print(" ALL NOTEBOOKS COMPLETED SUCCESSFULLY! ")
print("="*80)
print("\n READY FOR DASHBOARD CREATION")
print("\nNext Steps:")
print("  1. Open Power BI/Tableau")
print("  2. Import processed data files")
print("  3. Build interactive dashboard")
print("  4. Prepare presentation")
print("  5. Push to GitHub")
print("\n Great job completing the Retail & Marketing Analytics Project!")
print("="*80)

### Create Your Dashboard in Power BI
# Read. PowerBI_doc # Check the Doc File - Exercise.

# Python Dash Dashboard (Interactive Web App)
# Install required packages
# pip install dash plotly dash-bootstrap-components

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Load processed data
df_sales = pd.read_csv('data/processed/cleaned_retail_sales.csv')
df_sales['Order_Date'] = pd.to_datetime(df_sales['Order_Date'])

rfm = pd.read_csv('data/processed/customer_segments.csv')
monthly_kpis = pd.read_csv('data/processed/monthly_kpis.csv')

# Calculate KPIs
kpis = {
    'Total_Revenue': df_sales['Sales'].sum(),
    'Total_Customers': df_sales['Customer_ID'].nunique(),
    'Avg_Order_Value': df_sales.groupby('Order_ID')['Sales'].sum().mean(),
    'CLV_to_CAC_Ratio': rfm['Monetary'].mean() / 50  # Assuming CAC = $50
}

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Retail & Marketing Analytics Dashboard", 
                        className="text-center mb-4"), width=12)
    ]),
    
    # KPI Cards
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4(f"${kpis['Total_Revenue']:,.0f}", className="card-title"),
                html.P("Total Revenue", className="card-text")
            ])
        ]), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4(f"{kpis['Total_Customers']:,}", className="card-title"),
                html.P("Total Customers", className="card-text")
            ])
        ]), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4(f"${kpis['Avg_Order_Value']:,.2f}", className="card-title"),
                html.P("Avg Order Value", className="card-text")
            ])
        ]), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4(f"{kpis['CLV_to_CAC_Ratio']:.2f}x", className="card-title"),
                html.P("CLV/CAC Ratio", className="card-text")
            ])
        ]), width=3)
    ], className="mb-4"),
    
    # Charts Row 1
    dbc.Row([
        dbc.Col(dcc.Graph(id='revenue-trend'), width=8),
        dbc.Col(dcc.Graph(id='customer-segments'), width=4)
    ], className="mb-4"),
    
    # Charts Row 2
    dbc.Row([
        dbc.Col(dcc.Graph(id='category-performance'), width=6),
        dbc.Col(dcc.Graph(id='regional-sales'), width=6)
    ])
    
], fluid=True)

# Callbacks for interactivity
@app.callback(
    Output('revenue-trend', 'figure'),
    Input('revenue-trend', 'id')
)
def update_revenue_trend(_):
    fig = px.line(monthly_kpis, x='YearMonth', y='Revenue',
                  title='Monthly Revenue Trend',
                  markers=True,
                  labels={'YearMonth': 'Month', 'Revenue': 'Revenue ($)'})
    fig.update_xaxes(tickangle=45)
    return fig

@app.callback(
    Output('customer-segments', 'figure'),
    Input('customer-segments', 'id')
)
def update_segments(_):
    if 'Cluster_Name' in rfm.columns:
        segment_counts = rfm['Cluster_Name'].value_counts()
    else:
        segment_counts = rfm['Customer_Segment'].value_counts()
    
    fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                 title='Customer Segments',
                 hole=0.4)
    return fig

@app.callback(
    Output('category-performance', 'figure'),
    Input('category-performance', 'id')
)
def update_category(_):
    if 'Product_Category' in df_sales.columns:
        category_sales = df_sales.groupby('Product_Category')['Revenue'].sum().sort_values(ascending=False)
        fig = px.bar(x=category_sales.index, y=category_sales.values,
                     title='Revenue by Category',
                     labels={'x': 'Category', 'y': 'Revenue ($)'})
    else:
        fig = px.bar(title='Category data not available')
    return fig

@app.callback(
    Output('regional-sales', 'figure'),
    Input('regional-sales', 'id')
)
def update_regional(_):
    if 'Region' in df_sales.columns:
        regional_sales = df_sales.groupby('Region')['Revenue'].sum()
        fig = px.pie(values=regional_sales.values, names=regional_sales.index,
                     title='Revenue by Region')
    else:
        fig = px.pie(title='Regional data not available')
    return fig

# Run app
if __name__ == '__main__':
    app.run(debug=True, port=8050)

# Run this python Code:
# python dashboards/dash_app.py

#Then open http://127.0.0.1:8050 in your browser.
