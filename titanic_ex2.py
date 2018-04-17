# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 17:13:15 2018

@author: xiaoyu
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.preprocessing import Imputer
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


# read training data and test data
training_data_file = 'train.csv'
pred_raw_X = pd.read_csv(training_data_file)
y = pred_raw_X.Survived

test_data_file = 'test.csv'
test_raw_X = pd.read_csv(test_data_file)

pred_raw_X['training_set'] = True 
test_raw_X['training_set'] = False
X_full = pd.concat([pred_raw_X, test_raw_X]) #concatenate both dataframes prior to EDA

X_full.drop(['PassengerId', 'Survived'], axis=1, inplace=True)
# dropping variables useless
X_full.drop('Name', axis=1, inplace=True)
X_full.drop('Ticket', axis=1, inplace=True)

print('Traing + Test data:')
print(X_full.shape)
print(X_full.columns)

# using categorical data with One Hot Encoding
low_cardinality_cols = [cname for cname in X_full.columns if 
                                X_full[cname].nunique() < 10 and
                                X_full[cname].dtype == "object"]
numeric_cols = [cname for cname in X_full.columns if 
                                X_full[cname].dtype in ['int64', 'float64']]
my_cols = low_cardinality_cols + numeric_cols + ['training_set']
print('Low cardinality cols:', low_cardinality_cols)
print('Numeric & bool cols:', numeric_cols)
X_full = X_full[my_cols]
X_full[numeric_cols] = X_full[numeric_cols].apply(lambda x: x.fillna(x.median()),axis=0)
X_full[low_cardinality_cols] = X_full[low_cardinality_cols].apply(lambda x: x.fillna("None"),axis=0)

X_full = pd.get_dummies(X_full)

print('After one-hot-encoding & imputer')
print(X_full.shape)
print(X_full.columns)

# handling missing value;  (pay attention to fit_transform vs transform)
#my_imputer = Imputer()
#X_full = my_imputer.fit_transform(X_full)  # the return type is numpy array


X = X_full[X_full['training_set']==True]
X = X.drop('training_set', axis=1)
X_test = X_full[X_full['training_set']==False]
X_test = X_test.drop('training_set', axis=1)
print(X_full.shape)
print(X.shape, X_test.shape)

# split 75% + 25% (by default) as training + validation data set
train_X, val_X, train_y, val_y = train_test_split(X, y,test_size=0.3,random_state = 0)

'''
# check partial dependence plot
titanic_X_colns = ['Age','Fare']
t_model.fit(train_X[titanic_X_colns],train_y)
titanic_plots = plot_partial_dependence(t_model, features=[0,1], X=train_X[titanic_X_colns], 
                                        feature_names=titanic_X_colns, grid_resolution=8)
exit(0)
'''
# -----------------------------------------------------------------------------
# train the model using piplelines
my_pipeline = make_pipeline(Imputer(), GradientBoostingClassifier())
#my_pipeline.fit(train_X,train_y)
# use cross-validation, it is used to select model and adjust the parameters
scores = cross_val_score(my_pipeline, X, y, scoring='accuracy')
print(scores)
print('Mean accuracy score is %2f' %(scores.mean()))


# -----------------------------------------------------------------------------

my_pipeline.fit(X,y)
# get predicted result on training data
val_predictions = my_pipeline.predict(train_X)
print(accuracy_score(train_y, val_predictions))

# get predicted result on validation data
val_predictions = my_pipeline.predict(val_X)
print(accuracy_score(val_y, val_predictions))



# Use the model to make predictions
predicted_survived = my_pipeline.predict(X_test)
my_submission = pd.DataFrame({'PassengerId': test_raw_X.PassengerId, 'Survived': predicted_survived})
my_submission.to_csv('submission.csv', index=False)

'''last results:
accuracy_score
0.8997005988023952
0.8475336322869955

'''