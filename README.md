# Document-Clustering-using-KMeans-and-Hierarchical-Clustering
Performs document clustering using both the k-means and hierarchical clustering methods for easy comparison of their respective performances

# Cluster Analysis: K-Means
After testing with multiple k values between 1 and 10, I found the k value that presented the clearest separation of clusters to be 4. The clearer separation and relatively equal distribution of the clusters suggests reviews are being grouped into their appropriate clusters.

# Cluster Analysis: Hierarchical
I ran into significant issues while rendering the dendrogram - I tried reducing k value and setting linkage mode to "single," and adjusting recursion depth. The scatterplot of the hierarchical clusters while k was limited to 20 shows a majority of visible data points appearing in a single cluster. This suggested the clustering is highly imbalanced.
