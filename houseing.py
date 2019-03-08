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
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


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

class LabelBinarizerPipelineFriendly(LabelBinarizer):
    def fit(self, X, y=None):
        """this would allow us to fit the model based on the X input."""
        super(LabelBinarizerPipelineFriendly, self).fit(X)
    def transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).transform(X)

    def fit_transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).fit(X).transform(X)

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


#check mean squared error
#parameters: Prepared data, 
def check_mse_cross_validation(data, labels, model):
    #from sklearn.metrics import mean_squared_error
    scores = cross_val_score(model, data, labels, scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)
    return tree_rmse_scores
# display cross validated scores
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())



def main():
    #set up variables
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_PATH = os.path.join("datasets", "housing")
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

    fetch_housing_data(HOUSING_URL, HOUSING_PATH) #load data
    housing = load_housing_data(HOUSING_PATH) # convert to pandas dataframe

    #split training and test set with stratum split
    train_set, test_set = stratum_split(housing)

    #Putting the whole databse as just the training set and also splitting out the labels into it's own set
    housing = train_set.drop("median_house_value", axis=1)
    housing_labels = train_set["median_house_value"].copy()

    #-------------------- Set up Pipeline --------------------
    training_num = housing.drop("ocean_proximity", axis=1)
    num_attribs = list(training_num)
    print(num_attribs)
    cat_attribs = ["ocean_proximity"]
    num_pipeline = Pipeline([
            ('selector', DataFrameSelector(num_attribs)),
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
        ])
    cat_pipeline = Pipeline([
            ('selector', DataFrameSelector(cat_attribs)),
            ('label_binarizer', LabelBinarizerPipelineFriendly()),
        ])
    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])
    print(housing.shape)
    #-------------------------------------------------------
    housing_prepared = full_pipeline.fit_transform(housing) #run the pipeline

    #print(housing_prepared)
    print(housing_prepared.shape)
    print(housing_prepared.shape)

    #make decision tree model
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    lin_rmse_scores = check_mse_cross_validation(housing_prepared, housing_labels, lin_reg)
    #display_scores(lin_rmse_scores)

    #make decision tree model
    from sklearn.tree import DecisionTreeRegressor
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(housing_prepared, housing_labels)
    tree_rmse_scores = check_mse_cross_validation(housing_prepared, housing_labels, tree_reg)
    #display_scores(tree_rmse_scores)

    #make random forrest model
    from sklearn.ensemble import RandomForestRegressor
    forest_reg = RandomForestRegressor()
    forest_reg.fit(housing_prepared, housing_labels)
    forest_rmse_scores = check_mse_cross_validation(housing_prepared, housing_labels, forest_reg)
    #display_scores(forest_rmse_scores)

    #find the best hyperparameter fit --- we should look up more information on this
    from sklearn.model_selection import GridSearchCV
    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]
    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(housing_prepared, housing_labels)
    grid_search.best_params_

    # Test your model on the test set
    final_model = grid_search.best_estimator_
    X_test = test_set.drop("median_house_value", axis=1)
    y_test = test_set["median_house_value"].copy()
    X_test_prepared = full_pipeline.transform(X_test)
    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse) 
    print(final_rmse)

main()