import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm # Support Vector Classifier
from sklearn.metrics import classification_report

"""
    ID : Identifier
    Clump : Clump Thickness
    UnifSize : Uniformity of Cell Size
    UnifShape : Uniformity of Cell Shape
    MargAdh : Marginal Adhension
    SingEpiSize : Single epitheliail cell size
    BareNuc : Bare Nuclei
    BlanChorm : Blsnd Chromatin
    NormNucl : Normal Nucleoli
    Mit : Mitoses
    Class : Benign or Malignant
"""

data = pd.read_csv("/Users/surendrasrinivas/Downloads/cell_samples.csv")
#data.head()
#data.tail()
#data.shape
#data.size
#data.count()
#data['Class'].value_counts()

benign = data[data['Class']==2][0:200]
malignant = data[data['Class']==4][0:200]
#help(benign.plot)
#axes = benign.plot(kind = 'scatter', x = 'Clump', y = 'UnifSize',  color = 'blue', label="Benign")
#axes2 = malignant.plot(kind = 'scatter', x = 'Clump', y = 'UnifSize',  color = 'red', label="Malignant")
axes = benign.plot(kind = 'scatter', x = 'Clump', y = 'UnifSize',  color = 'blue', label="Benign")
axes2 = malignant.plot(kind = 'scatter', x = 'Clump', y = 'UnifSize',  color = 'red', label="Malignant", ax = axes)

data.dtypes

data = data[pd.to_numeric(data["BareNuc"], errors = 'coerce').notnull()]
data['BareNuc'] = data['BareNuc'].astype('int')
data.dtypes
data.columns

features = data[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize',
       'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
#data of 100 rows
# picked 9 columns out of 11
#independent variable 
x = np.asarray(features)

#dependent variable
y = np.asarray(data['Class'])
x[0:5]

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state = 4) 
"""X_train.shape
X_test.shape
y_train.shape
y_test.shape"""

classifier = svm.SVC(kernel="linear", gamma = "auto", C = 2)
classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)

print(classification_report(y_test, y_predict))

