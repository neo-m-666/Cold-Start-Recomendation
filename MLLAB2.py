import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

data = pd.read_excel(r'/Users/mukil/Desktop/19CSE305_LabData_Set3.1.xlsx')
df = pd.DataFrame(data)
#DataType of each Attribute
print("Data Types for Each Attribute:")
print(df.dtypes)

#Range of numeric value

numeric_vars = df.select_dtypes(include=[np.number])
numeric_range = numeric_vars.describe()
print("Numeric Variables Range:")
print(numeric_range)

#Missing values in each attribute.

missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values)

#Mean and Variance 
numeric_mean = numeric_vars.mean()
numeric_variance = numeric_vars.var()
print("\nNumeric Variables Mean:")
print(numeric_mean)
print("\nNumeric Variables Variance:")
print(numeric_variance)

#Ploting Outliner Data
plt.figure(figsize=(12, 6))
plt.boxplot(numeric_vars.values, vert=False)
plt.xticks(range(1, len(numeric_vars.columns) + 1), numeric_vars.columns, rotation=45)
plt.title("Box Plot of Numeric Variables (Outliers)")
plt.show()

# categorical values

categorical_vars = df.select_dtypes(include=["object"])

print("Categorical Variables:")
print(categorical_vars.columns)

# 1. Identify attributes with missing values and their data types
missing_attributes = df.columns[df.isnull().any()]
data_types = df[missing_attributes].dtypes

# 2. missing values for numeric attributes
for attr in missing_attributes[data_types != 'object']: 
    if df[attr].hasnans:
        if df[attr].dtype == 'float64' or df[attr].dtype == 'int64':
            if df[attr].skew() < 1:  
                df[attr].fillna(df[attr].mean(), inplace=True)
            else:
                df[attr].fillna(df[attr].median(), inplace=True)


# missing values for categorical attributes
for attr in missing_attributes[data_types == 'object']:  
    if df[attr].hasnans:
        df[attr].fillna(df[attr].mode().iloc[0], inplace=True)

# Check if all missing values are filled
missing_values_after_imputation = df.isnull().sum()
print("Missing Values After Imputation:")
print(missing_values_after_imputation)

numeric_attributes = df.select_dtypes(include=[np.number])

# Choose and apply normalization techniques
# Min-Max Scaling
min_max_scaler = MinMaxScaler()
df[numeric_attributes.columns] = min_max_scaler.fit_transform(numeric_attributes)

# Z-Score Scaling (Standardization)
standard_scaler = StandardScaler()
df[numeric_attributes.columns] = standard_scaler.fit_transform(numeric_attributes)\

print(df)

vector1 = df.iloc[0, 1:]  # Assuming the first column is an identifier and should be excluded
vector2 = df.iloc[1, 1:]  # Assuming the first column is an identifier and should be excluded

# Reshape the vectors into a format suitable for cosine_similarity
vector1 = vector1.values.reshape(1, -1)
vector2 = vector2.values.reshape(1, -1)

# Calculate Cosine Similarity
cosine_sim = cosine_similarity(vector1, vector2)

# Print the Cosine Similarity value
print("Cosine Similarity:", cosine_sim[0, 0])

binary_attributes = ["attribute1", "attribute2", ...]  # Replace with actual binary attribute names
data = df.iloc[:20][binary_attributes]

# Calculate JC and SMC between pairs of vectors
n = len(data)
jc_matrix = np.zeros((n, n))
smc_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(i, n):
        intersection = (data.iloc[i] & data.iloc[j]).sum()
        union = (data.iloc[i] | data.iloc[j]).sum()
        jc_matrix[i, j] = jc_matrix[j, i] = intersection / union
        smc_matrix[i, j] = smc_matrix[j, i] = (data.iloc[i] == data.iloc[j]).sum() / len(data.columns)

# Calculate Cosine Similarity (COS) between pairs of vectors
cos_matrix = cosine_similarity(data)

# Create a heatmap to visualize the similarities
plt.figure(figsize=(12, 6))
plt.subplot(131)
sns.heatmap(jc_matrix, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=False, yticklabels=False)
plt.title("Jaccard Coefficient")

plt.subplot(132)
sns.heatmap(smc_matrix, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=False, yticklabels=False)
plt.title("Simple Matching Coefficient")

plt.subplot(133)
sns.heatmap(cos_matrix, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=False, yticklabels=False)
plt.title("Cosine Similarity")

plt.tight_layout()
plt.show()