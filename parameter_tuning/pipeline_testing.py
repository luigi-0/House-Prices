"""
Created on Mon Mar 25 18:00:45 2019

@author: luisgranados
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

house_source_data = pd.read_csv("train.csv")
house_test = pd.read_csv("test.csv")


# Store the label for the training set. 
house_labels = house_source_data['SalePrice'].copy()

# A function to calculate Root Mean Squared Logarithmic Error (RMSLE).
from sklearn.metrics import mean_squared_log_error

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_log_error(y, y_pred))

# Stage the ratings for the mapper function.
# Remember, these are ordinal features.
qual_rating_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual',
                'GarageCond', 'PoolQC']

qual_ordinal_ratings = [
        ('Ex',5), 
        ('Gd',4), 
        ('TA',3), 
        ('Fa',2), 
        ('Po',1), 
        ('NA',0)
    ]

# Exposure features need own mapping.
exposure_rating_feature = ['BsmtExposure']
exposure_ordinal_ratings = [
        ('Gd',4), 
        ('Av',3), 
        ('Mn',2), 
        ('No',1), 
        ('NA',0)
    ]

electrical_rating_feature = ['Electrical']
electrical_ordinal_ratings = [
        ('SBrkr',5),
        ('FuseA',4),
        ('FuseF',3),
        ('FuseP',2),
        ('Mix',1)
    ]

functional_rating_feature = ['Functional']
functional_ordinal_ratings = [
        ('Typ',7), 
        ('Min1',6), 
        ('Min2',5), 
        ('Mod',4), 
        ('Maj1',3),
        ('Maj2',2),
        ('Sev',1),
        ('Sal',0)
    ]

# Finish ratings
bsmt_finish_rating_features = ['BsmtFinType1', 'BsmtFinType2']
bsmtfin_ordinal_ratings = [
        ('GLQ',6), 
        ('ALQ',5), 
        ('BLQ',4), 
        ('Rec',3), 
        ('LwQ',2),
        ('Unf',1),
        ('NA',0)
    ]

grg_finish_rating_feature = ['GarageFinish']
grgfin_ordinal_ratings = [
        ('Fin',3), 
        ('RFn',2), 
        ('Unf',1), 
        ('NA',0)
    ]

paved_rating_feature = ['PavedDrive']
paved_ordinal_ratings = [
        ('Y',2), 
        ('P',1), 
        ('N',0)
    ]

fence_rating_feature = ['Fence']
fence_ordinal_ratings = [
        ('GdPrv',4),
        ('MnPrv',3),
        ('GdWo',2), 
        ('MnWw',1),
        ('NA',0)
    ]

alley_rating_feature = ['Alley']
alley_ordinal_ratings = [
        ('Pave',2),
        ('Grvl',1),
        ('NA',0)
    ]

utilities_rating_feature = ['Utilities']
utilities_ordinal_ratings = [
        ('AllPub',3),
        ('NoSewr',2),
        ('NoSeWa',1),
        ('ELO', 0)
    ]

# Select categorical features to one-hot-encode.
categorial_onehot_features = ['MSZoning',
    'Street',
    'LotShape',
    'LandContour',
    'LotConfig',
    'LandSlope',
    'Neighborhood',
    'Condition1',
    'Condition2',
    'BldgType',
    'HouseStyle',
    'RoofStyle',
    'RoofMatl',
    'Exterior1st',
    'Exterior2nd',
    'MasVnrType',
    'Foundation',
    'Heating',
    'CentralAir',
    'GarageType',
    'SaleType',
    'SaleCondition',
    'MSSubClass'
    ]

# Use this function for generating new ones. I can't generalize this part 
# because of all the potentially different ways to create each feature.
# but this will allow me to do the same generation for the test set.
def generator(dataframe):
    """Generate new features."""
    new_dataframe = dataframe.copy()
    #new_dataframe['The new one'] = new_dataframe['MSSubClass'] ** 2 

    return new_dataframe

def mapper(data_in, features, ratings):
    """Map ordinal ratings to numeric ranking."""
    counter = 0
    for feature in features:
        ordinal_mapping = [{
            "col":feature,    
            "mapping": ratings},
        ]

        encoder = ce.OrdinalEncoder(mapping = ordinal_mapping, 
                                    return_df = True, cols = feature)
        if counter == 0:
            df_train = encoder.fit_transform(data_in.copy())
        else:
            df_train = encoder.fit_transform(df_train)
        counter += 1
    return df_train

# Create the dataframe selector class for use in pipelines
class Mapper(BaseEstimator, TransformerMixin):
    """Generate any features and convert dataframe to numpy array."""
    def __init__(self):
        self = self
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Map ratings to numbers.
        X = mapper(X, qual_rating_features, qual_ordinal_ratings)
        X = mapper(X, bsmt_finish_rating_features, bsmtfin_ordinal_ratings)
        X = mapper(X, exposure_rating_feature, exposure_ordinal_ratings)
        X = mapper(X, electrical_rating_feature, electrical_ordinal_ratings)
        X = mapper(X, functional_rating_feature, functional_ordinal_ratings)
        X = mapper(X, grg_finish_rating_feature, grgfin_ordinal_ratings)
        X = mapper(X, paved_rating_feature, paved_ordinal_ratings)
        X = mapper(X, fence_rating_feature, fence_ordinal_ratings)
        X = mapper(X, alley_rating_feature, alley_ordinal_ratings)
        X = mapper(X, utilities_rating_feature, utilities_ordinal_ratings)
        return X

def data_staging(data_in):
    """Create preliminary training set."""
    data_frame = data_in.copy()
    
    # Fill in the null values with None.
    category_data = data_frame.copy().select_dtypes(exclude='number').fillna(value='None')

    # Fill in null values with 0.
    numeric_data = (data_frame
             .loc[:, data_frame.copy().columns != 'LotFrontage']
             .select_dtypes(include='number')
             .fillna(value=0)
            )

    # Re-merge the two datasets. Drop label and unneeded features.
    data_in_cleaned = (category_data
                         .merge(numeric_data, how='outer',
                                left_index=True, right_index=True)
                         .drop(columns=['SalePrice','MiscFeature', 'Id'], errors='ignore')
                        )
    data_in_cleaned = pd.concat([data_in_cleaned, data_in['LotFrontage']], axis=1)
    
    data_in_cleaned = mapper(data_in_cleaned, qual_rating_features, qual_ordinal_ratings)
    data_in_cleaned = mapper(data_in_cleaned, bsmt_finish_rating_features, bsmtfin_ordinal_ratings)
    data_in_cleaned = mapper(data_in_cleaned, exposure_rating_feature, exposure_ordinal_ratings)
    data_in_cleaned = mapper(data_in_cleaned, electrical_rating_feature, electrical_ordinal_ratings)
    data_in_cleaned = mapper(data_in_cleaned, functional_rating_feature, functional_ordinal_ratings)
    data_in_cleaned = mapper(data_in_cleaned, grg_finish_rating_feature, grgfin_ordinal_ratings)
    data_in_cleaned = mapper(data_in_cleaned, paved_rating_feature, paved_ordinal_ratings)
    data_in_cleaned = mapper(data_in_cleaned, fence_rating_feature, fence_ordinal_ratings)
    data_in_cleaned = mapper(data_in_cleaned, alley_rating_feature, alley_ordinal_ratings)
    data_in_cleaned = mapper(data_in_cleaned, utilities_rating_feature, utilities_ordinal_ratings)
    
    # Generate new features.
    data_in_cleaned = generator(data_in_cleaned)
    
    return data_in_cleaned

# Create the preliminary training set. This is for analysis.
# numeric attributes get pulled from here.
house_train_staging = data_staging(house_source_data.copy())
house_test_staging = data_staging(house_test.copy())

# Store all the numeric type column names for DataFrameSelector(). 
# MSSubClass gets encoded, so it gets droped from the numeric list of features to process.
numeric_features = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond',
 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt',
 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
 '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold',
 'LotFrontage','ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 
 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC',
 'BsmtExposure','Electrical','Functional','BsmtFinType1', 'BsmtFinType2',
 'GarageFinish','PavedDrive','Fence', 'Alley', 'Utilities',
 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual',
 'GarageCond', 'PoolQC', 'BsmtExposure', 'Electrical',
 'Functional', 'BsmtFinType1', 'BsmtFinType2',
 'GarageFinish', 'PavedDrive', 'Fence', 'Alley', 'Utilities']

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorial_onehot_features)])

house_train = preprocessor.fit_transform(house_train_staging.copy())
house_test_data = preprocessor.transform(house_test_staging.copy())

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

seed = 42
np.random.seed(seed)
############################## Lasso ##########################################
lasso_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('lasso', Lasso(random_state=np.random.seed(seed)))])

lasso_param_grid = {'lasso__alpha' : [143, 144, 145, 146],
                    'lasso__max_iter' : [31, 32, 33, 34]}

lasso_grid = GridSearchCV(lasso_pipe, param_grid=lasso_param_grid, cv=5, n_jobs=-1)

lasso_grid.fit(house_train_staging.copy(), house_labels)

lasso_grid_pred = lasso_grid.best_estimator_.named_steps['lasso'].predict(house_train)

lasso_rmse = rmsle(house_labels, lasso_grid_pred)

lasso_grid_best = lasso_grid.best_estimator_.named_steps['lasso']

print('Results: {:8f} {:8f}'.format(lasso_grid.best_score_, lasso_rmse))
############################## Ridge ##########################################
ridge_pipe = Pipeline(steps=[('preprocessor', preprocessor), 
                             ('rig', Ridge(random_state=np.random.seed(seed)))])

ridge_param_grid = {'rig__alpha' : [19, 20, 21.0]}

ridge_grid = GridSearchCV(ridge_pipe, param_grid=ridge_param_grid, cv=5, n_jobs=-1)

ridge_grid.fit(house_train_staging.copy(), house_labels)

ridge_grid_pred = ridge_grid.best_estimator_.named_steps['rig'].predict(house_train)

ridge_rmse = rmsle(house_labels, ridge_grid_pred)

ridge_grid_best = ridge_grid.best_estimator_.named_steps['rig']

print('Results: {:8f} {:8f}'.format(ridge_grid.best_score_, ridge_rmse))
############################## Elastic ########################################
elastic_pipe = Pipeline(steps=[('preprocessor', preprocessor), 
                               ('elastic', ElasticNet(normalize=False,random_state=np.random.seed(seed)))])

elastic_param_grid = {'elastic__alpha' : [0.4, 0.45, 0.5, 138],
                    'elastic__max_iter' : [30, 33, 35, 40,],
                    'elastic__l1_ratio' : [0.85, 0.95, 0.9, 1.0]}

elastic_grid = GridSearchCV(elastic_pipe, param_grid=elastic_param_grid, cv=5, n_jobs=-1)

elastic_grid.fit(house_train_staging.copy(), house_labels)

elastic_grid_pred = elastic_grid.best_estimator_.named_steps['elastic'].predict(house_train)

elastic_rmse = rmsle(house_labels, elastic_grid_pred)

elastic_grid_best = elastic_grid.best_estimator_.named_steps['elastic']

print('Results: {:8f} {:8f}'.format(elastic_grid.best_score_, elastic_rmse))
############################### SVM  ##########################################
from sklearn.svm import SVR

svm_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('svm', SVR(gamma='scale'))])

svm_param_grid =  {'svm__C' : [540000, 550000, 560000, 570000, 580000],
                   'svm__epsilon' : [3000, 3100, 3200, 3300, 3400]}

svm_grid = GridSearchCV(svm_pipe, param_grid=svm_param_grid, cv=5, n_jobs=-1)

svm_grid.fit(house_train_staging.copy(), house_labels)

svm_grid_pred = svm_grid.best_estimator_.named_steps['svm'].predict(house_train)

svm_rmse = rmsle(house_labels, svm_grid_pred)

svm_grid_best = svm_grid.best_estimator_.named_steps['svm']

print('Results: {:8f} {:8f}'.format(svm_grid.best_score_, svm_rmse))
############################ Decision tree ####################################
from sklearn.tree import DecisionTreeRegressor

dtr_pipe = Pipeline(steps=[('preprocessor', preprocessor), 
                           ('dtr', DecisionTreeRegressor(random_state=np.random.seed(seed)))])

dtr_param_grid = {'dtr__max_depth' : [8, 9, 10, 11],
                   'dtr__max_features': ['auto', 'sqrt', 'log2']}

dtr_grid = GridSearchCV(dtr_pipe, param_grid=dtr_param_grid, cv=5, n_jobs=-1)

dtr_grid.fit(house_train_staging.copy(), house_labels)

dtr_grid_pred = dtr_grid.best_estimator_.named_steps['dtr'].predict(house_train)

dtr_rmse = rmsle(house_labels, dtr_grid_pred)

# Store the tree to be used in Ada
tree_grid_cv = dtr_grid.best_estimator_.named_steps['dtr']

print('Results: {:8f} {:8f}'.format(dtr_grid.best_score_, dtr_rmse))
############################ Random Forest ####################################
from sklearn.ensemble import RandomForestRegressor

rf_pipe = Pipeline(steps=[('preprocessor', preprocessor), 
                          ('rf', RandomForestRegressor(random_state=np.random.seed(seed), n_jobs=-1))])

rf_param_grid =  {'rf__max_depth' : [14, 15, 16, 17],
                   'rf__n_estimators' : [500, 600, 700, 800],
                   'rf__max_features': ['auto', 'sqrt']}

rf_grid = GridSearchCV(rf_pipe, param_grid=rf_param_grid, cv=5, n_jobs=-1)

rf_grid.fit(house_train_staging.copy(), house_labels)

rf_grid_pred = rf_grid.best_estimator_.named_steps['rf'].predict(house_train)

rf_rmse = rmsle(house_labels, rf_grid_pred)

rf_grid_best = rf_grid.best_estimator_.named_steps['rf']

print('Results: {:8f} {:8f}'.format(rf_grid.best_score_, rf_rmse))
################################## GBR ########################################
from sklearn.ensemble import GradientBoostingRegressor

gbr_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('gbr', GradientBoostingRegressor(random_state=np.random.seed(seed)))])

gbr_param_grid = {'gbr__max_depth' : [2, 3, 4, 5],
                  'gbr__n_estimators' : [38, 39, 40],
                  'gbr__learning_rate' : [0.1, 0.2, 0.3],
                  'gbr__subsample' : [0.8, 0.9, 1.0]}

gbr_grid = GridSearchCV(gbr_pipe, param_grid=gbr_param_grid, cv=5, n_jobs=-1)

gbr_grid.fit(house_train_staging.copy(), house_labels)

gbr_grid_pred = gbr_grid.best_estimator_.named_steps['gbr'].predict(house_train)

gbr_rmse = rmsle(house_labels, gbr_grid_pred)

gbr_grid_best = gbr_grid.best_estimator_.named_steps['gbr']

print('Results: {:8f} {:8f}'.format(gbr_grid.best_score_, gbr_rmse))
################################## Ada ########################################
from sklearn.ensemble import AdaBoostRegressor

ada_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('ada', AdaBoostRegressor(tree_grid_cv,loss='square', random_state=np.random.seed(seed)))])

ada_param_grid = {'ada__n_estimators' : [450, 500, 550],
                  'ada__learning_rate' : [0.5, 0.6, 0.7]}

ada_grid = GridSearchCV(ada_pipe, param_grid=ada_param_grid, cv=5, n_jobs=-1)

ada_grid.fit(house_train_staging.copy(), house_labels)

ada_grid_pred = ada_grid.best_estimator_.named_steps['ada'].predict(house_train)

ada_rmse = rmsle(house_labels, ada_grid_pred)

ada_grid_best = ada_grid.best_estimator_.named_steps['ada']

print('Results: {:8f} {:8f}'.format(ada_grid.best_score_, ada_rmse))
############################# Neural Network ##################################
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.losses import mean_squared_logarithmic_error

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# define base model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(256, input_dim=np.shape(house_train)[1], 
                    kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss=mean_squared_logarithmic_error, optimizer='adam')
    return model  
    
nnr = KerasRegressor(build_fn=baseline_model, epochs=500, batch_size=128, verbose=0)

nnr.fit(house_train, house_labels)

nnr_pred = nnr.predict(house_train)

nn_rmse = rmsle(house_labels, nn_grid_pred)

print('Results: {:8f} {:8f}'.format(nn_grid.best_score_, nn_rmse))

######################## pre-trained nn model using google collab #############
from keras.models import model_from_json

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("nn_modelV1.h5")

loaded_model.compile(loss='mean_squared_error', optimizer='adam')
score = loaded_model.evaluate(house_train, house_labels) # kernel keeps crashing here
##################################### Stacker #################################
from mlxtend.regressor import StackingRegressor

# The StackingCVRegressor uses scikit-learn's check_cv
# internally, which doesn't support a random seed. Thus
# NumPy's random seed need to be specified explicitely for
# deterministic behavior
np.random.seed(RANDOM_SEED)

stack = StackingRegressor(regressors=(svm_grid_best, ridge_grid_best),
                            meta_regressor=rf_grid_best).fit(house_train, house_labels)

stack_rmse = rmsle(house_labels, stack.predict(house_train))

print('5-fold cross validation scores:\n')

for clf, label in zip([svr, ridge, stack], ['SVM', 'Ridge',
                                                'StackingRegressor']):
    scores = cross_val_score(clf, house_train, house_labels, cv=5)
    print("R^2 Score: %0.2f (+/- %0.2f) [%s]" % (
        scores.mean(), scores.std(), label))
    print(stack_rmse)
    





###############################################################################
# Create a function that will create dataframe for submission
# Note: This function assumes that you will be using the Scikit prediction method
def kaggle_submission(estimator, test_set, test_source, label, kaggle_id):
    """
    Create a csv to submit predictions to Kaggle.
    
    estimator: Sci-kit estimator used on training set.
    test_set: The test set.
    label: The name of the label; must be a string.
    test_source: The original training set, which contains the Id column.
    kaggle_id: The id column from the test set. 
    """
    predictions = estimator.predict(test_set)
    predictions = pd.DataFrame(predictions, columns=[label])
    submission = pd.DataFrame(test_source[kaggle_id])
    submission[label] = predictions[label]
    return submission

predictions = kaggle_submission(rf_grid.best_estimator_.named_steps['rf'], house_test_data, 
                                house_test, 'SalePrice', 'Id')

predictions.to_csv('rf_submission.csv', index=False)

# only uncomment this if your're getting better results
#house_training = preprocessor.fit_transform(house_train_staging.copy())
#house_test_data = preprocessor.fit_transform(house_test_staging.copy())

#pd.to_pickle(house_training, 'house_training')
#pd.to_pickle(house_test_data, 'house_test_data')




