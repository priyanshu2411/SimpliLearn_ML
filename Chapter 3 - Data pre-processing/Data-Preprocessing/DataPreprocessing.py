# -*- coding: utf-8 -*-
"""
CellStrat
"""

#==============================================================================
# First step to write the python program is to take benefit out of libraries
# already available. We will only focus on the data science related libraries.
#==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#==============================================================================
# #import data from the data file. In our case its Health.csv. 
#==============================================================================

healthData = pd.read_csv ('Health.csv')
print(healthData)
#==============================================================================
# All mathematical operations will be performed on the matrix, so now we create
# matrix for dependent variables and independent variables.
#==============================================================================


X = healthData.iloc [:,:-1].values
y = healthData.iloc [:,3].values

#==============================================================================
# Handle the missing values, we can see that in dataset there are some missing
# values, we will use strategy to impute mean of column values in these places
#==============================================================================

from sklearn.preprocessing import Imputer
# First create an Imputer
missingValueImputer = Imputer (missing_values = 'NaN', strategy = 'mean', 
                               axis = 0)
# Set which columns imputer should perform
missingValueImputer = missingValueImputer.fit (X[:,1:3])
# update values of X with new values
X[:,1:3] = missingValueImputer.transform(X[:,1:3])

#==============================================================================
# Encode the categorial data. So now instead of character values we will have
# corresponding numerical values
#==============================================================================

from sklearn.preprocessing import LabelEncoder
X_labelencoder = LabelEncoder()
X[:, 0] = X_labelencoder.fit_transform(X[:, 0])
print (X)

# for y
y_labelencoder = LabelEncoder ()
y = y_labelencoder.fit_transform (y)
print (y)

#==============================================================================
# Implementing OneHotEncoder to separate category variables into dummy 
# variables.
#==============================================================================

from sklearn.preprocessing import OneHotEncoder
X_onehotencoder = OneHotEncoder (categorical_features = [0])
X = X_onehotencoder.fit_transform(X).toarray()
print (X)

#==============================================================================
# split the dataset into training and test set. We will use 80/20 approach
#==============================================================================

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.2, 
                                                     random_state = 0)

#==============================================================================
# Feature scaling is to bring all the independent variables in a dataset into
# same scale, to avoid any variable dominating  the model. Here we will not 
# transform the dependent variables.
#==============================================================================

from sklearn.preprocessing import StandardScaler
independent_scalar = StandardScaler()
X_train = independent_scalar.fit_transform (X_train) #fit and transform
X_test = independent_scalar.transform (X_test) # only transform
print(X_train)
