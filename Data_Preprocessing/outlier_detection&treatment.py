import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Load dataset from Excel
df = pd.read_csv("C:/School Files/Depression Predictor Model/Datasets/missing_data_treatment.csv")

# Create IQR-based outlier flag
df["Outlier_IQR"] = False

# Dictionary to store IQR bounds
iqr_bounds = {}

# Detect outliers for each numeric feature
for col in df.select_dtypes(include=[float, int]).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Store bounds
    iqr_bounds[col] = {"Lower Bound": lower_bound, "Upper Bound": upper_bound}

    # Flag outliers
    df["Outlier_IQR"] |= (df[col] < lower_bound) | (df[col] > upper_bound)

# Convert bounds to DataFrame for display
iqr_bounds_df = pd.DataFrame(iqr_bounds).T

# Display IQR bounds
print("📊 IQR Bounds for Each Numeric Feature:\n")
print(iqr_bounds_df)

# Plot scatter plots
sns.set(style="whitegrid")

# Safely get numeric columns excluding the outlier flag
numeric_cols = df.select_dtypes(include=[float, int]).columns
numeric_cols = numeric_cols.drop("Outlier_IQR") if "Outlier_IQR" in numeric_cols else numeric_cols

# Plot scatter plot for each numeric column
for col in numeric_cols:
    plt.figure(figsize=(10, 4))
    plt.scatter(df.index, df[col], 
                c=df["Outlier_IQR"].map({True: 'red', False: 'blue'}),
                alpha=0.7)
    plt.title(f'Scatter Plot for {col} (Red = Outliers)')
    plt.xlabel("Row Index")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()


# Create a copy for cleaned data
df_cleaned = df.copy()

for col in df_cleaned.select_dtypes(include=[float, int]).columns:

    Q1 = df_cleaned[col].quantile(0.25)
    Q3 = df_cleaned[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Store bounds
    iqr_bounds[col] = {"Lower Bound": lower_bound, "Upper Bound": upper_bound}
    
# Drop rows flagged as outliers
df_no_outliers = df[~df["Outlier_IQR"]].copy()

print(f"\n✅ Rows after dropping outliers: {df_no_outliers.shape[0]}")
print(f"🗑️ Outlier rows removed: {df.shape[0] - df_no_outliers.shape[0]}")

# Count how many outliers were originally present
original_outlier_count = df["Outlier_IQR"].sum()
print(f"\n🔴 Total number of outliers originally: {original_outlier_count}")

# Count how many still remain in the cleaned dataset (should be 0)
remaining_outliers = df_no_outliers["Outlier_IQR"].sum()
print(f"🔎 Remaining outliers after cleaning: {remaining_outliers}")

#Check again to plot to confirm no outlier remains
sns.set(style="whitegrid")

# Select numeric columns (excluding Outlier_IQR if still present)
numeric_cols = df_no_outliers.select_dtypes(include=[float, int]).columns
if "Outlier_IQR" in numeric_cols:
    numeric_cols = numeric_cols.drop("Outlier_IQR")

# Plot scatter plots for each numeric column in the cleaned dataset
for col in numeric_cols:
    plt.figure(figsize=(10, 4))
    plt.scatter(df_no_outliers.index, df_no_outliers[col], color='blue', alpha=0.7)
    plt.title(f'Scatter Plot for {col} (Cleaned Data - No Outliers)')
    plt.xlabel("Row Index")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()



df.to_csv('C:/School Files/Depression Predictor Model/Datasets/outlier_treatment.csv', index=False)

