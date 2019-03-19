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

est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.01,
         max_depth=9,subsample=1.0, random_state=0, loss='ls').fit(house_training, house_labels)

results = cross_val_score(est, house_training, house_labels, cv=10, n_jobs =-1)
score = est.score(house_training, house_labels)
rmse = rmsle(house_labels, est.predict(house_training))

print("Results: {:.2f}% ({:.2f}%) {:.5f} {:.5f}".format(results.mean()*100, results.std()*100, score, rmse))



# best result 
we_the_best = "Results: {:.2f}% ({:.2f}%) {:.5f} {:.5f}".format(results.mean()*100, results.std()*100, score, rmse)
print("Results: {:.2f}% ({:.2f}%) {:.5f} {:.5f}".format(results.mean()*100, results.std()*100, score, rmse))