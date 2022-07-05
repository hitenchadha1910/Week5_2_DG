import numpy as np
import pandas as pd
import pickle

dataset = pd.read_csv("Salary Prediction based on Position and Experience.csv")
X = dataset.iloc[:,[1,2]].values
y = dataset.iloc[:,-1].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()

regressor.fit(X,y)
pickle.dump(regressor, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[6,8]]))