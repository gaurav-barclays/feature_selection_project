
# %load q04_select_from_model/build.py
# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')
np.random.seed(9)
def select_from_model (data):
    X = data.iloc[:,:-1]
    Y = data['SalePrice']
   # print(Y)
    model = RandomForestClassifier()
    rfe = SelectFromModel(model)
    fitmod = rfe.fit(X, Y)
    return X.columns.values[fitmod.get_support()].tolist()

# Your solution code here
#a=select_from_model(data)
#print((a))
