import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


data = pd.read_csv('Mall_Customers.csv')
X = data.iloc[:,3:].values

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++',max_iter=300, n_init = 10,random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.scatter(X[:,0],X[:,1])
plt.show()
plt.plot (range(1,11),wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters=5, init = 'k-means++',max_iter = 300,n_init = 10, random_state =0)
y_kmeans = kmeans.fit_predict(X)


plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=50,c='red',label = "Cluster 1")
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=50,c='blue',label = "Cluster 2")
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=50,c='green',label = "Cluster 3")
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=50,c='black',label = "Cluster 4")
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=50,c='orange',label = "Cluster 5")

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=150,c='yellow',label = 'Centroids')
plt.title('Clusters of clients')
plt.xlabel("Annual Income(k$)")
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()
