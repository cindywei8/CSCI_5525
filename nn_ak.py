# import libraries 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#import shap 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

from scipy.special import boxcox, inv_boxcox
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import KNNImputer
import seaborn as sns
from sklearn.datasets import make_classification 

# import data
df = pd.read_csv("cardio_train.csv", sep=';')
print(df)
X = df.drop('cardio', axis=1)
y = df['cardio']

# fill in missing values
imputer = KNNImputer(n_neighbors=1)
X = imputer.fit_transform(X)		# should this be done on just x?

# transform data
#y,maxlog = stats.boxcox(y) 
pt = PowerTransformer(method = 'yeo-johnson')
pt.fit(X)
X = pt.transform(X)

# define hyperparameters to later search
param_dist = {	'hidden_layer_sizes':[(5,1),(5,3),(5,5)],
				'alpha':[1e-5],
				'random_state':[1]
             }


# k-fold the data
count = 0
outer_results = list()
cv_outer = KFold(n_splits=5, shuffle=True, random_state=1)
for train_ix, test_ix in cv_outer.split(X):

	# split data
	X_train, X_test = X[train_ix, :], X[test_ix, :]
	y_train, y_test = y[train_ix], y[test_ix]

	# configure the cross-validation procedure
	cv_inner = KFold(n_splits=10, shuffle=True, random_state=1)
	
	# define the model and search hyperparameters
	clf = MLPClassifier()
	#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
	search = RandomizedSearchCV(clf, param_dist, cv=cv_inner)
    
    # execute search and get best model
	result = search.fit(X_train, y_train)
	best_model = result.best_estimator_

	'''
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
	'''
	
	# evaluate model on the hold out dataset
	yhat = result.predict(X_test)
	#yhat = inv_boxcox(yhat,maxlog)
	#y_test = inv_boxcox(y_test,maxlog)

	# evaluate the model and store results
	acc = mean_absolute_error(y_test, yhat)
	outer_results.append(acc)
	
	# report progress
	print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
	count = count + 1

# summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (np.mean(outer_results), np.std(outer_results)))

'''
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
'''