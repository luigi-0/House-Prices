import pandas as pd
import numpy as np
import math

house_source_data = pd.read_csv("train.csv")
house_test = pd.read_csv("test.csv")

# The processed training and test sets
house_training = pd.read_pickle('house_training')
house_test_data = pd.read_pickle('house_test_data')

# Store the label for the training set.
house_labels = house_source_data['SalePrice'].copy()

# A function to calculate Root Mean Squared Logarithmic Error (RMSLE).
def rmsle(y, y_pred):
	assert len(y) == len(y_pred)
	terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
	return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

parameters = {"n_estimators": [100, 200, 300, 400, 500, 600, 700], 
			  "learning_rate":[1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001],
			  "max_depth":[3, 4, 5, 6, 7, 8, 9]}

Grid_GBR = GradientBoostingRegressor(subsample=1.0, random_state=0, loss='ls')

Grid_GBR.fit(house_training, house_labels)

Grid_GBR_CV = GridSearchCV(Grid_GBR, parameters,n_jobs=-1)
Grid_GBR_CV.fit(house_training , house_labels)

results = cross_val_score(Grid_GBR_CV, house_training, house_labels, cv=10, n_jobs =-1)
score = Grid_GBR_CV.score(house_training, house_labels)
rmse = rmsle(house_labels, Grid_GBR_CV.predict(house_training))

print("Results: {:.2f}% ({:.2f}%) {:.5f} {:.5f}".format(results.mean()*100, results.std()*100, score, rmse))





# best result 
we_the_best = "Results: {:.2f}% ({:.2f}%) {:.5f} {:.5f}".format(results.mean()*100, results.std()*100, score, rmse)