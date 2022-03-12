
# import libraries 
import pyreadstat
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import shap 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

from numpy import mean
from numpy import std
from scipy.special import boxcox, inv_boxcox
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import KNNImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import seaborn as sns
import imblearn 
from sklearn.datasets import make_classification 
from imblearn.over_sampling import RandomOverSampler
#from keras.callbacks import EarlyStopping








print('The Sample Size is',len(X))


			
imputer = KNNImputer(n_neighbors=1)
X = imputer.fit_transform(X)

vif_data = pd.DataFrame()
vif_data["feature"] = sort_variables
vif_data["VIF"] = [variance_inflation_factor(X, i)
				  for i in range(len(sort_variables))]

count = 0
col = X.shape[1]

vari = sorted_vars



		




y,maxlog = stats.boxcox(y) 
pt = PowerTransformer(method = 'yeo-johnson')
pt.fit(X)
X = pt.transform(X)


		
		
space ={'num_leaves': sp_randint(6, 200), 
             'min_child_samples': sp_randint(50, 500), 
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8), 
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100],
           #  'weight_column': [weights/4],
             'n_estimators': [10, 50, 100, 200, 300, 400]
             
             }



count = 0

cv_outer = KFold(n_splits=5, shuffle=True, random_state=1)
# enumerate splits
outer_results = list()
for train_ix, test_ix in cv_outer.split(X):

	# split data
	X_train, X_test = X[train_ix, :], X[test_ix, :]
	y_train, y_test = y[train_ix], y[test_ix]
	# configure the cross-validation procedure
	cv_inner = KFold(n_splits=10, shuffle=True, random_state=1)
	# define the model
	model = LGBMRegressor()
	# define search space
	# define search
	search = RandomizedSearchCV(model, space, scoring='r2', cv=cv_inner, refit=True,n_iter = 50)
	#early_stopping = EarlyStopping( monitor='val_loss', min_delta=0, patience=0, verbose=0,
    #mode='auto', baseline=None, restore_best_weights=False)
    
    # execute search
	result = search.fit(X_train, y_train)
	# get the best performing model fit on the whole training set
	best_model = result.best_estimator_
	# create shap plots 
	explainer = shap.TreeExplainer(model = best_model, 
							  data = None, 
							   model_ouput = 'raw',
							   feature_pertubation = 'tree_path_dependent')
	if count == 0: 
		shap_values = explainer.shap_values(X_test)
		X_values = X_test
	else: 
		shap_values = np.vstack((shap_values,explainer.shap_values(X_test)))
		X_values = np.vstack((X_values, X_test))
	# evaluate model on the hold out dataset
	yhat = best_model.predict(X_test)
	yhat = inv_boxcox(yhat,maxlog)
	y_test = inv_boxcox(y_test,maxlog)
	# evaluate the model
	acc = mean_absolute_error(y_test, yhat)
	# store the result
	outer_results.append(acc)
	# report progress
	print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
	count = count + 1
# summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))

ab	= np.abs(shap_values) 
am = np.mean(ab,0)
sort = np.sort(am)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.scatter(yhat, y_test)
# plt.xlim(0,30)
# plt.ylim(0,30)
#plt.scatter(y_1,y_valid)
shap.summary_plot(shap_values, X_values,sort_variables,20)