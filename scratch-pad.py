"""Throwaway script."""
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import category_encoders as ce
file = "/Users/luisgranados/Documents/Kaggle/House Prices/train.csv"

data_in =  pd.read_csv(file)

house_descr = pd.read_table("/Users/luisgranados/Documents/Kaggle/House Prices/data_description.txt",
                    header=None, sep = '\t+',
                    engine='python',
                   skip_blank_lines = True)

class FeatureGenerator():
    """This class will create a dataset containing all the features, and generate new ones. """
    def __init__(self, dataframe, attribute_names):
        self.dataframe = dataframe
        self.attribute_names = attribute_names

    def generator(self, dataframe, attribute_names):
        """Generate new features."""
        self.dataframe[attribute_names] = dataframe["SalePrice"] ** 2 
        
        return dataframe

def generator(dataframe, attribute_names):
        """Generate new features."""
        for original, generated in attribute_names.items():
                dataframe[generated] = dataframe[original] ** 2 

        return dataframe

features_seed = {'SalePrice' : 'SalePrice_Squared', 'MSSubClass': 'MSSubClass_Doubled'}
a = generator(data_in, features_seed)

print(list(a.select_dtypes(include='number').columns))
b = data_in.copy()
c = b.values

print(c[:])

qual_rating_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual',
                'GarageCond', 'PoolQC']
qual_ordinal_map = [{
    "col":feature,  
    "mapping": [
        ('Ex',5), 
        ('Gd',4), 
        ('TA',3), 
        ('Fa',2), 
        ('Po',1), 
        ('NA',0)
    ]},
]

# These needs own mapping
exposure_rating_feature = ['BsmtExposure']
exposure_ordinal_map = [{
    "col":feature,  
    "mapping": [
        ('Gd',4), 
        ('Av',3), 
        ('Mn',2), 
        ('No',1), 
        ('NA',0)
    ]},
]

electrical_rating_feature = ['Electrical']
electrical_ordinal_map = [{
    "col":feature,  
    "mapping": [
        ('SBrkr',5),
        ('FuseA',4),
        ('FuseF',3),
        ('FuseP',2),
        ('Mix',1)
    ]},
]

functional_rating_feature = ['Functional']
functional_ordinal_map = [{
    "col":feature,  
    "mapping": [
        ('Typ',7), 
        ('Min1',6), 
        ('Min2',5), 
        ('Mod',4), 
        ('Maj1',3),
        ('Maj2',2),
        ('Sev',1),
        ('Sal',0)
    ]},
]

# Finish ratings

bsmt_finish_rating_feature = ['BsmtFinType1', 'BsmtFinType2']
bsmtfin_ordinal_map = [{
    "col":feature,  
    "mapping": [
        ('GLQ',6), 
        ('ALQ',5), 
        ('BLQ',4), 
        ('Rec',3), 
        ('LwQ',2),
        ('Unf',1),
        ('NA',0)
    ]},
]

grg_finish_rating_feature = ['GarageFinish']
grgfin_ordinal_map = [{
    "col":feature,  
    "mapping": [
        ('Fin',3), 
        ('RFn',2), 
        ('Unf',1), 
        ('NA',0)
    ]},
]

paved_rating_feature = ['PavedDrive']
paved_ordinal_map = [{
    "col":feature,  
    "mapping": [
        ('Y',2), 
        ('P',1), 
        ('N',0)
    ]},
]

fence_rating_feature = ['Fence']
fence_ordinal_map = [{
    "col":feature,  
    "mapping": [
        ('GdPrv',4),
        ('MnPrv',3),
        ('GdWo',2), 
        ('MnWw',1),
        ('NA',0)
    ]},
]


def mapper(data_in, features, ratings):
    """Map ratings to numeric ranking."""
    for feature in features:
        encoder = ce.OrdinalEncoder(mapping = ratings,
                                    return_df = True, cols = features)
        df_train = encoder.fit_transform(data_in)
    return df_train        
                
a = mapper(house_descr, qual_rating)
#a['BsmtFinType1'].head()

#a = mapper(house_explore, qual_rating)
a['ExterQual'].head()