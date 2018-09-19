"""
# Prediciting the convergence rate of a client based on past user behaviours.
# dependent variable is in binary form (eg., 1 = converged/0 = not or true/false) and hence, its a classification problem.
# Two classification techniques - Random Forest Classifier and Logistic Regression are used here.
"""
import numpy as np  
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LR
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

# Read input data- it is in .csv form
data = pd.read_csv('path/to/datafile')
hdr = list(data.columns.values)          # Get list of column names

# Create an output file to store predictions of each algorithm
outfile = pd.DataFrame(columns=['RF','LR'])

X = np.array(data[0:len(hdr)])  # All data except targets
y = np.array(data[len(hdr)])    # Target classes 

# Data preprocessing
le = preprocessing.LabelEncoder()
for i in range(0,len(hdr)):
     X[:,i] = le.fit_transform(X[:,i])

y[:] = le.fit_transform(y[:])

# Split the data as training and testing data - taking 80% train size (can be varied)
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.2, random_state=0)

# Classification using Random Forest Classifier
rf = RF(max_depth = 5, n_estimators = 10)
rf = rf.fit(X_train,y_train)
prediction = rf.predict(X_test)
print(accuracy_score(y_test, prediction))
outlen = len(prediction)
outfile['RF'] = np.random.randn(outlen)
outfile['RF'] = pd.DataFrame(prediction)

# Classification using Logistic Regression algorithm
logreg = LR()
logreg = logreg.fit(X_train,y_train)
prediction = logreg.predict(X_test)
print(accuracy_score(y_test, prediction))
outfile['LR'] = np.random.randn(outlen)
outfile['LR'] = pd.DataFrame(prediction)

# Write predictions to output file.
outfile.to_csv('prediction.csv', index = False, header = True)

print("Completed sucessfully!")
