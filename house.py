import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def dataProcessing(Data):
	sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
	return sel.fit_transform(Data)


Data1 = pd.read_csv('data/train.csv')
Data2 = pd.read_csv('data/test.csv')

fcols = ['LotArea', 'OverallQual', 'YearBuilt', '1terQual', 'HeatingQC',
       'GrLivArea', 'KitchenQual', 'TotRmsAbvGrd', 'GarageArea']

reg = RandomForestRegressor(max_depth=8, random_state=0, n_estimators=150)

X = Data1[fcols]
y = Data1.SalePrice
test_x = Data2[fcols]
hid = Data2.Id

"""train_x, val_x, train_y, val_y = train_test_split(X, y, test_size = 0.4, random_state = 0)

reg.fit(train_x, train_y)
pred = reg.predict(val_x)"""

reg.fit(X, y)
pred = reg.predict(test_x)

"""error = mean_absolute_error(y, pred)
print(error)"""

df = pd.DataFrame({'Id':hid, 'SalePrice':pred})
df.to_csv('result.csv', sep=',', encoding='utf-8')



