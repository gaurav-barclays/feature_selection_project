# %load q05_forward_selected/build.py
# Default imports
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('data/house_prices_multivariate.csv')

model = LinearRegression()

#print(data.info())
# Your solution code here
def forward_selected(data, model):
    features = data.iloc[:,-1]
    print(len(features.column.values))

forward_selected(data, model)
