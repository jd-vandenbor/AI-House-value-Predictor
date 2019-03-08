#Main

import os, tarfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LinearRegression

class CustomBinarizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None,**fit_params):
        return self
    def transform(self, X):
        return LabelBinarizer().fit(X).transform(X)

#Custom transformer class to  add new attributes to house data
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            print(np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room].shape)
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

# attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
# housing_extra_attribs = attr_adder.transform(housing.values)

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

#Download housing data
def fetch_housing_data(housing_url, housing_path):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


#convert the data to a more readable pandas format
def load_housing_data(housing_path):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

#stratum split
def stratum_split(housing):
    # The following code creates
    # an income category attribute by dividing the median income by 1.5 (to limit the number of income
    # categories), and rounding up using ceil (to have discrete categories), and then merging all the categories
    # greater than 5 into category 5


    #create startum (catagories)
    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

    #split training set with stratum in mind
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
        
    #Now remove the income_cat attribute so the data is back to its original state
    for set_ in (strat_train_set, strat_test_set, housing):
        set_.drop("income_cat", axis=1, inplace=True)

    return strat_train_set, strat_test_set

def interpret_data(housing):
    housing.plot(kind="scatter", x="longitude", y="latitude")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    ) 
    plt.legend()
    plt.show()


# #Binarize a text column into many 
# # from sklearn.preprocessing import LabelBinarizer
# def binarize_ocean_proximity(housing):

#   encoder = LabelBinarizer(sparse_output=True) # make sparse_ouput=False for a dense NumPy Array. Boooooooo!
#   housing_cat_1hot = encoder.fit_transform(housing_cat)
#   housing_cat_1hot
#   ocean_labels = encoder.classes_ # just to use for reference to know what is what.




DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

fetch_housing_data(HOUSING_URL, HOUSING_PATH) #load data

housing = load_housing_data(HOUSING_PATH) # convert to pandas dataframe


#housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
#housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
#housing["population_per_household"]=housing["population"]/housing["households"]

# split test set with scikit-learn, instead we use the more acurate stratum split
#train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42) 

train_set, test_set = stratum_split(housing)
print(len(train_set))
print(len(test_set))

housing = train_set.drop("median_house_value", axis=1)
housing_labels = train_set["median_house_value"].copy()

#interpret_data(housing)
training_num = housing.drop("ocean_proximity", axis=1)
num_attribs = list(training_num)
cat_attribs = ["ocean_proximity"]
num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
#   housing_num_tr = num_pipeline.fit_transform(training_num)
#   print(housing_num_tr.shape)
cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', CustomBinarizer()),
    ])
#   cat = cat_pipeline.fit_transform(train_set)
#   print(cat.shape)
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)
print(housing_prepared.shape)

