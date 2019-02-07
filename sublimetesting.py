import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

training_path = r"/Users/luisgranados/Documents/Kaggle/House Prices/train.csv"

test_path = r"/Users/luisgranados/Documents/Kaggle/House Prices/test.csv"

training_set = pd.read_csv(training_path)
test_set = pd.read_csv(test_path)
training_set.head()

# Store labels
house_labels = training_set['SalePrice']

class FeatureGenerator(BaseEstimator, TransformerMixin):
    """This class will create a dataset containing all the features, and generate new ones. """
    def __init__(self):
        self = self

    def generator(self, attribute_names):
        

# Create the full training set with generated features.
generated_training = FeatureGenerator()
training_set_prelim = generated_training.transform(training_set)

# Get the categorical data ready for encoding.
processing = training_set_prelim.copy()

drop_var = ['Name', 'Ticket', 'Cabin', 'Embarked', 'Title', 'Captain']
processing = processing.drop(drop_var, axis=1)

category_features = processing.select_dtypes(include='object')

numeric_features = processing.select_dtypes(exclude='object')

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """Turn the pandas dataframe into numpy arrays. You have to choose which features are converted."""
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X,y=None):
        return self
    def transform(self,X):
        return X[self.attribute_names].values

# Create a class to one-hot encode the categorical features using pandas.
class OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = pd.get_dummies(data=X, columns=self.attribute_names)
        return X

# Features to encode.
cat_attribs = ['Ticket_number_length','Pclass','Sex']

# Numeric features to convert into numpy arrays.
num_attribs = ['Age','Fare','SibSp','Parch']

# Categorical features to convert to numpy arrays.
cat_attribs_sel = ['Noble','Reverend','Military','Doctor','Ticket_number_length_1','Ticket_number_length_3',
                   'Ticket_number_length_4','Ticket_number_length_5','Ticket_number_length_6','Ticket_number_length_7',
                   'Pclass_1','Pclass_2','Pclass_3','Sex_male','Sex_female']

# Ordinal categorical features to encode.
ord_cat_attribs = ['ExterQual','ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
					'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'Electrical',
					'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish',
					'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence']

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy='median')),
    ('minmax_scaler', MinMaxScaler()),
])

cat_pipeline = Pipeline([
    ('generated_training', FeatureGenerator()),
    ('cat_encoder', OneHotEncoder(cat_attribs)),
    ('selector', DataFrameSelector(cat_attribs_sel)),
])

# Create final training set
full_pipeline = FeatureUnion([
    ('num_pipe', num_pipeline),
    ('cat_pipe', cat_pipeline),
])

titanic_tr_cleaned= full_pipeline.fit_transform(training_set.copy())

test_data = full_pipeline.transform(test_set)