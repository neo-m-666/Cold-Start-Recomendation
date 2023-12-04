import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

data = pd.read_csv(r'/Users/mukil/Desktop/water potability/waterPotability_updated.csv')
class_labels = data['ph']
data_without_labels = data.drop(columns=['ph'])
imputer = SimpleImputer(strategy='mean')
data_without_labels_imputed = pd.DataFrame(imputer.fit_transform(data_without_labels), columns=data_without_labels.columns)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_without_labels_imputed)

kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster'] = kmeans.fit_predict(scaled_data)

distortions = []
k_values = range(1, 32)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    distortions.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(k_values, distortions, marker='o', linestyle='-', color='b')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Average Euclidean Distance from Cluster Center')
plt.show()

chosen_k = 5
kmeans = KMeans(n_clusters=chosen_k, random_state=42)
data['cluster'] = kmeans.fit_predict(scaled_data)

print(data)

agg_clustering = AgglomerativeClustering(n_clusters=5, linkage='ward')
data['agg_cluster'] = agg_clustering.fit_predict(scaled_data)

plt.figure(figsize=(12, 8))
dendrogram(linkage(scaled_data, method='ward'), labels=data.index, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

class_labels = data['ph']
X = data.drop(columns=['ph'])
X_train, X_test, y_train, y_test = train_test_split(X, class_labels, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
sfs = SequentialFeatureSelector(model, n_features_to_select='best', direction='forward')
sfs.fit(X_train, y_train)

plt.plot(range(1, len(sfs.get_support()) + 1), sfs.get_metric_dict()[('cv_scores', 1)])
plt.title('Sequential Forward Selection')
plt.xlabel('Number of Features Selected')
plt.ylabel('Cross-validation Score')
plt.show()

selected_features = X.columns[sfs.get_support()]
print(f'Selected Features: {selected_features}')

class_labels = data['ph']
X = data.drop(columns=['ph'])
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

cumulative_variance_ratio = pca.explained_variance_ratio_.cumsum()
k = next(i for i, ratio in enumerate(cumulative_variance_ratio, 1) if ratio >= 0.95)
print(f'Number of features needed to capture 95% of variance: {k}')

plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', linestyle='-', color='b')
plt.title('Explained Variance Ratio')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.show()

K = 3
pca = PCA(n_components=K)
X_transformed = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_transformed, class_labels, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on the transformed dataset: {accuracy:.2f}')