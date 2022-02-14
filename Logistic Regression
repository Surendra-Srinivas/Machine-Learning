import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.model_selection import train_test_split

x=pd.read_csv('Social_Network_Ads.csv')
plt.scatter(x['Age'],x['Purchased'])
#plt.legend()
plt.show()
X_train, X_test, y_train, y_test = train_test_split(x['Age'], x['Purchased'],test_size=0.20)

def normalize(X):
  return X-X.mean()
  
def predict(X,b0,b1):
  return np.array([1/(1+math.exp(-1*b0+-1*b1*x)) for x in X])
  
def logistic_Regression(X,Y):
  X = normalize(X)
  b0 = 0
  b1 = 0
  L = 0.001
  epochs = 1500
  for e in range(epochs):
    y_pred = predict(X,b0,b1)
    D_b0 = -2*sum((Y-y_pred)*y_pred*(1-y_pred))
    D_b1 = -2*sum(X*(Y-y_pred)*(1-y_pred))
    b0 = b0-L*D_b0
    b1 = b1-L*D_b1
  return b0,b1
  
b0,b1 = logistic_Regression(X_train,y_train)
X_test_norm = normalize(X_test)
y_pred = predict(X_test_norm,b0,b1)
y_pred = [1 if p>=0.5 else 0 for p in y_pred]
plt.clf()
plt.scatter(X_test,y_test)
plt.scatter(X_test,y_pred,c="red")
plt.show()
accuracy = 0

for i in range(len(y_pred)):
    if y_pred[i] == y_test.iloc[i]:
        accuracy += 1
print(f"Accuracy = {accuracy / len(y_pred)}")
