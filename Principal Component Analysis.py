import sklearn.datasets as DS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb


def PCA(X,num_components):
  X_meaned = X-np.mean(X,axis=0)
  cov_mat = np.cov(X_meaned, rowvar = False)
  eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
  sorted_index = np.argsort(eigenvalues)[::-1]
  sorted_eigenvalues = eigenvalues[sorted_index]
  sorted_eigenvectors = eigenvectors[:,sorted_index]
  eigenvector_subset = sorted_eigenvectors[:,0:num_components]
  X_reduced = np.dot(eigenvector_subset.transpose(), X_meaned.transpose()).transpose()
  return X_reduced

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
data = pd.read_csv(url, names=['Sepal Length','Sepal Width', 'Petal Length','Petal Width', 'Target'])

X = data.iloc[:,0:4]
target = data.iloc[:,4]

reducedmat = PCA(X,2)
df = pd.DataFrame(reducedmat,columns = ['PC1','PC2'])
df = pd.concat([df,pd.DataFrame(target)],axis=1)
sb.scatterplot(data = df,x ='PC1',y='PC2',s=60,palette = 'icefire')
#sb.scatterplot(data = data) # check plotting this also you will see only essence of the data is plotted.


