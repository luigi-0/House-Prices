"""Throwaway script."""
import pandas as pd
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


#print(a.head())
