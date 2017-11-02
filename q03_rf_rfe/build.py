# %load q03_rf_rfe/build.py
# Default imports
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# Your solution code here

def rf_rfe(data):
    X = data.iloc[:,:-1]
   # print(X)
    Y = data['SalePrice']
    model = RandomForestClassifier()
    rfe = RFE(model, n_features_to_select=18)
    fitmod = rfe.fit(X, Y)

    return X.columns.values[fitmod.support_].tolist()


#a=rf_rfe(data)
#print((a))
