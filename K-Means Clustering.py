# K-Means Clustering for Customer Segmentation

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv("Mall_Customers.csv")

# Display basic info
print("First 5 rows of dataset:")
print(data.head())
print("\nMissing values:\n", data.isnull().sum())

# Select features for clustering
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# Elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the elbow graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Apply KMeans with optimal clusters (from elbow method, typically 5)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Add cluster labels to original data
data['Cluster'] = y_kmeans

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X["Annual Income (k$)"], y=X["Spending Score (1-100)"],
                hue=y_kmeans, palette='Set1', s=100)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='yellow', label='Centroids', marker='*')
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()

# Print cluster summary
print("\nCluster Averages:")
print(data.groupby('Cluster')[["Annual Income (k$)", "Spending Score (1-100)"]].mean())

# Save the result to a CSV file
data.to_csv("clustered_customers.csv", index=False)
print("\nClustered data saved to 'clustered_customers.csv'")
