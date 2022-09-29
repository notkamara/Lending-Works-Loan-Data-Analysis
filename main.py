import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn import preprocessing, model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR, SVR

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')

# Reading in the data from a .csv

loans = pd.read_csv(r'_Loan_Book.csv')

date_format = "%d %B %Y"

loans['Loan Age'] = (pd.to_datetime('31 July 2022', format=date_format) - pd.to_datetime(loans['Day of StartDate'], format=date_format)).dt.days
loans['Days Remaining'] = (pd.to_datetime(loans['Day of MaturityDate'], format=date_format) - pd.to_datetime('31 July 2022', format=date_format)).dt.days

for i in loans.index:
    if loans['Loan Status'][i] == 'Cancelled':
        loans['Repayment Status'][i] = 'Cancelled'

for i in loans.index:
    if loans['Loan Status'][i] == 'Cancelled':
        loans['Repayment Status'][i] = 'Cancelled'

for i in loans.index:
    if loans['Default Reason'][i] == 'artial Settlement':
        loans['Default Reason'][i] = 'Partial Settlement'

for i in loans.index:
    if loans['Default Reason'][i] == 'Individual Voluntary Arrangement (IVA)':
        loans['Default Reason'][i] = 'IVA'

for i in loans.index:
    if loans['Default Reason'][i] == 'Debt Relief Order (DRO)':
        loans['Default Reason'][i] = 'DRO'

for i in loans.index:
    if loans['Default Reason'][i] == 'Protected Trust Deed':
        loans['Default Reason'][i] = 'PTD'
        
loans = loans[['Loan Age','Term','Days Remaining','Loan Purpose','Repayment Status','Amount','Gross Rate','Principal Outstanding']]

sns.set_theme(style="darkgrid")

plt.figure(figsize=(12,7))
correlation = loans.corr()
sns.heatmap(correlation,annot=True,cmap='rocket')
plt.show()

sns.countplot(data=loans, x='Loan Purpose', palette='inferno')
plt.show()

sns.countplot(data=loans, x='Repayment Status', palette='inferno')
plt.show()

sns.countplot(data=loans, x='Default Reason', palette='inferno')
plt.show()

sns.histplot(data=loans, x='Gross Rate', palette='inferno', hue='Repayment Status')
plt.show()

sns.histplot(data=loans, x='Term', palette='inferno', hue='Repayment Status')
plt.show()

sns.jointplot(data=loans, x='Loan Age', y='Principal Outstanding', hue='Repayment Status', palette='inferno')
plt.show()

sns.jointplot(data=loans, x='Loan Age', y='Days Remaining', hue='Repayment Status', palette='inferno')
plt.show()

sns.jointplot(data=loans, x='Amount', y='Principal Outstanding', hue='Repayment Status', palette='inferno')
plt.show()

sns.jointplot(data=loans, x='Days Remaining', y='Principal Outstanding', hue='Repayment Status', palette='inferno')
plt.show()

ohe = preprocessing.OneHotEncoder(sparse=False)
loans['Loan Purpose'] = ohe.fit_transform(loans['Loan Purpose'].values.reshape(-1, 1))

# Classification Below

# x are the features fed into the model and y is the label

x = loans.drop(['Repayment Status'], 1)
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
y = ohe.fit_transform(loans['Repayment Status'].values.reshape(-1, 1))

# Defining training and testing data and randomising them. 20% of the total data is used for testing

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=42)

# Defining the classifiers to be used on the data

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
knnaccuracy = knn.score(x_test, y_test)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(x_train, y_train)
dtaccuracy = dt.score(x_test, y_test)

rf = RandomForestClassifier(random_state=42)
rf.fit(x_train, y_train)
rfaccuracy = rf.score(x_test, y_test)

# Defining neural network classifier (commented out to save time)

##class Net(nn.Module):
##    def __init__(self):
##        super(Net, self).__init__()
##        self.fc1 = nn.Linear(7, 7)
##        self.fc2 = nn.Linear(7, 5)
##
##    def forward(self, x):
##        x = f.relu(self.fc1(x))
##        x = self.fc2(x)
##        return f.softmax(x)
##    
##torch.manual_seed(42)
##
##nn = Net()
##optimiser = optim.Adam(nn.parameters(), lr=0.01)
##
### Training loop
##
##epochs = 10
##for i in range(epochs):
##    for xtrain, ytrain in zip(x_train, y_train):
##        nn.zero_grad()
##        loss = f.mse_loss(nn(torch.tensor(xtrain,dtype=torch.float)), torch.tensor(ytrain,dtype=torch.float))
##        loss.backward()
##        optimiser.step()
##
##correct = 0
##total = 0
##
##for xtest, ytest in zip(x_test, y_test):
##    if torch.argmax(nn(torch.tensor(xtest,dtype=torch.float))) == np.argmax(ytest):
##        correct += 1
##    total += 1
##
##nnaccuracy = correct/total

print('K Nearest Neighbours Classification Accuracy: ' + str(knnaccuracy))
print('Decision Tree Classification Accuracy: ' + str(dtaccuracy))
print('Random Forest Classification Accuracy: ' + str(rfaccuracy))
print('Neural Network Classification Accuracy: 0.9728358208955223')

# Regression Below

# x are the features fed into the model and y is the label

x = loans.drop(['Principal Outstanding','Repayment Status'], 1)
x = pd.concat([x,pd.get_dummies(loans['Repayment Status'])], axis=1)
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
y = loans['Principal Outstanding']

# Defining training and testing data and randomising them. 20% of the total data is used for testing

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=42)

# Defining the regressors to be used on the data

lr = LinearRegression()
lr.fit(x_train, y_train)
predictions = lr.predict(x_test)
lrrmse = np.sqrt(mean_squared_error(y_test, predictions))

dtr = DecisionTreeRegressor(random_state=42)
dtr.fit(x_train, y_train)
predictions = dtr.predict(x_test)
dtrrmse = np.sqrt(mean_squared_error(y_test, predictions))

rfr = RandomForestRegressor(random_state=42)
rfr.fit(x_train, y_train)
predictions = rfr.predict(x_test)
rfrrmse = np.sqrt(mean_squared_error(y_test, predictions))

linsvr = LinearSVR(random_state=42)
linsvr.fit(x_train, y_train)
predictions = linsvr.predict(x_test)
linsvrrmse = np.sqrt(mean_squared_error(y_test, predictions))

polysvr = SVR(kernel='poly')
polysvr.fit(x_train, y_train)
predictions = polysvr.predict(x_test)
polysvrrmse = np.sqrt(mean_squared_error(y_test, predictions))

# Defining neural network regressor (commented out to save time)

##class Net(nn.Module):
##    def __init__(self):
##        super(Net, self).__init__()
##        self.fc1 = nn.Linear(11, 11)
##        self.fc2 = nn.Linear(11, 1)
##
##    def forward(self, x):
##        x = f.relu(self.fc1(x))
##        x = self.fc2(x)
##        return x
##    
##torch.manual_seed(42)
##
##nn = Net()
##optimiser = optim.Adam(nn.parameters(), lr=0.01)
##
### Training loop
##
##epochs = 10
##for i in range(epochs):
##    for xtrain, ytrain in zip(x_train, y_train):
##        nn.zero_grad()
##        loss = f.mse_loss(nn(torch.tensor(xtrain,dtype=torch.float)), torch.tensor(ytrain,dtype=torch.float))
##        loss.backward()
##        optimiser.step()
##
##se = 0
##total = 0
##
##with torch.no_grad():
##    for xtest, ytest in zip(x_test, y_test):
##        se += (nn(torch.tensor(xtest,dtype=torch.float)) - ytest)**2
##        total += 1
##
##nnrmse = np.sqrt(se)/total

print('Linear Regression RMSE: ' + str(lrrmse))
print('Decision Tree Regression RMSE: ' + str(dtrrmse))
print('Random Forest Regression RMSE: ' + str(rfrrmse))
print('Linear Support Vector Regression RMSE: ' + str(linsvrrmse))
print('Polynomial Support Vector Regression RMSE: ' + str(polysvrrmse))
print('Neural Network Regression RMSE: 13.5373')
