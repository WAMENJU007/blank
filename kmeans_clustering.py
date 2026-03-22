# Import necessary libraries
import numpy as np
from sklearn.cluster import KMeans

# Create a dataset representing customers with 2 features: amount spent and number of visits
# Dataset for 8 customers
customers = np.array([
    [500, 5],  # High spender, many visits
    [200, 2],  # Medium spender, few visits
    [100, 1],  # Low spender, few visits
    [800, 6],  # High spender, many visits
    [50, 1],   # Low spender, few visits
    [300, 3],  # Medium spender, medium visits
    [700, 4],  # High spender, medium visits
    [150, 2]   # Low spender, few visits
])

# Choose number of clusters (KMeans model with n_clusters = 2)
kmeans = KMeans(n_clusters=2)

# Train the model
kmeans.fit(customers)

# Display results
print("Customer Dataset:")
print(customers)
print("\nCluster Labels for each Customer:")
print(kmeans.labels_)

# Interpretation:
# Cluster 0: Low-value customers (lower spend and visits)
# Cluster 1: High-value customers (higher spend and visits)
# Based on labels, customers in Cluster 1 are high-value