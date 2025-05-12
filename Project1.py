# ======================================
# STEP 1: Import Required Libraries
# ======================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization aesthetics
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ======================================
# STEP 2: Load and Inspect Raw Data
# ======================================
df = pd.read_csv("Retail.csv", encoding='ISO-8859-1')

print("First 5 rows:\n", df.head())
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Info:\n")
df.info()

# ======================================
# STEP 3: Clean the Data
# ======================================
# Drop rows with missing CustomerID or Description
df = df[df['CustomerID'].notnull() & df['Description'].notnull()].copy()

# Drop duplicates
df = df.drop_duplicates()

# Filter out invalid Quantity or UnitPrice values
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)].copy()

# ======================================
# STEP 4: Handle Dates and Create New Features
# ======================================
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

df['InvoiceYear'] = df['InvoiceDate'].dt.year
df['InvoiceMonth'] = df['InvoiceDate'].dt.month
df['InvoiceHour'] = df['InvoiceDate'].dt.hour
df['InvoiceDayofweek'] = df['InvoiceDate'].dt.dayofweek
df['InvoiceDateOnly'] = df['InvoiceDate'].dt.date

# ======================================
# STEP 5: Calculate Total Transaction Value
# ======================================
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# ======================================
# STEP 6: Save Cleaned Data (Optional)
# ======================================
df.to_csv("Retail_Cleaned.csv", index=False)

# ======================================
# STEP 7: Perform Exploratory Data Analysis (EDA)
# ======================================

# 7.1 Distribution of Transaction Value
plt.figure()
sns.histplot(df['TotalPrice'], bins=100, kde=True, color='skyblue')
plt.title("Distribution of Transaction Value (TotalPrice)")
plt.xlabel("TotalPrice")
plt.ylabel("Count")
plt.xlim(0, df['TotalPrice'].quantile(0.95))
plt.tight_layout()
plt.show()

# 7.2 Top 10 Products by Revenue
top_products = df.groupby('Description')['TotalPrice'].sum().sort_values(ascending=False).head(10)
plt.figure()
sns.barplot(x=top_products.values, y=top_products.index, palette='viridis')
plt.title("Top 10 Revenue-Generating Products")
plt.xlabel("Total Revenue")
plt.ylabel("Product Description")
plt.tight_layout()
plt.show()

# 7.3 Purchase Frequency per Customer
invoice_counts = df.groupby('CustomerID')['InvoiceNo'].nunique()
plt.figure()
sns.histplot(invoice_counts, bins=50, kde=True, color='coral')
plt.title("Purchase Frequency per Customer")
plt.xlabel("Number of Purchases")
plt.ylabel("Number of Customers")
plt.xlim(0, invoice_counts.quantile(0.95))
plt.tight_layout()
plt.show()

# ======================================
# STEP 8: Total Revenue & Customer-Level Analysis
# ======================================
total_revenue = df['TotalPrice'].sum()
print("Total Revenue:", total_revenue)

customer_revenue = df.groupby('CustomerID')['TotalPrice'].sum().sort_values(ascending=False)
print("\nTop 5 Customers by Revenue:\n", customer_revenue.head())

# Save top 10 customers (optional)
customer_revenue.head(10).to_csv("Top_Customers.csv")

# ======================================
# STEP 9: Categorize Customers by Revenue Tier
# ======================================
def categorise_spending(revenue):
    if revenue >= 1000:
        return 'High'
    elif revenue >= 500:
        return 'Medium'
    else:
        return 'Low'

customer_revenue_category = customer_revenue.apply(categorise_spending)
df['customer_category'] = df['CustomerID'].map(customer_revenue_category).fillna('Unknown')

print("\nCustomer Category Mapping:\n", df[['CustomerID', 'customer_category']].head())

# ======================================
# STEP 10: RFM Analysis (Recency, Frequency, Monetary)
# ======================================
reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,   # Recency
    'InvoiceNo': 'nunique',                                      # Frequency
    'TotalPrice': 'sum'                                          # Monetary
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# ======================================
# STEP 11: RFM Scoring
# ======================================
rfm['R_Score'] = pd.qcut(rfm['Recency'], q=4, labels=[4, 3, 2, 1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=4, labels=[1, 2, 3, 4])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=4, labels=[1, 2, 3, 4])

# ======================================
# STEP 12: Segment Customers Based on RFM
# ======================================
rfm['RFM_Segment'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

rfm['RFM_Level'] = rfm['RFM_Segment'].apply(
    lambda x: 'Champions' if x == '444' else (
        'Loyal' if x[0] == '4' else (
            'At Risk' if x[0] == '1' else 'Others'
        )
    )
)

print("\nSample RFM Output:\n", rfm.head())

# Save RFM segments (optional)
rfm.to_csv("RFM_Segments.csv", index=False)

# ======================================
# STEP 13: More EDA - RFM Visualization
# ======================================

# 13.1 Recency Distribution
plt.figure()
sns.histplot(rfm['Recency'], bins=40, kde=True, color='green')
plt.title("Recency Distribution (Days Since Last Purchase)")
plt.xlabel("Recency (Days)")
plt.ylabel("Customer Count")
plt.tight_layout()
plt.show()

# 13.2 Revenue by Month
monthly_revenue = df.set_index('InvoiceDate').resample('M')['TotalPrice'].sum()
plt.figure()
monthly_revenue.plot(marker='o', color='purple')
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Total Revenue")
plt.grid(True)
plt.tight_layout()
plt.show()

# 13.3 Hour of Purchase
plt.figure()
sns.countplot(x='InvoiceHour', data=df, palette='coolwarm')
plt.title("Number of Transactions by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.show()

# 13.4 Day of Week Purchases
day_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
df['DayName'] = df['InvoiceDayofweek'].map(day_map)

plt.figure()
sns.countplot(x='DayName', data=df, order=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], palette='Set2')
plt.title("Transactions by Day of Week")
plt.xlabel("Day")
plt.ylabel("Transaction Count")
plt.tight_layout()
plt.show()

# 13.5 Correlation Matrix for RFM Features
plt.figure()
sns.heatmap(rfm[['Recency', 'Frequency', 'Monetary']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix: RFM Features")
plt.tight_layout()
plt.show()

git init

git remote add origin https://github.com/dixit-prashant/Marketing-Analytics-Pipeline

git add .
git commit -m "First commit"
git branch -M main
git push -u origin main

