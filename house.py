import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeRegressor

"""
feature reducted:
Street, Alley, LotShape, LandContour, Utilities, LotConfig, LandSlope, Condition1, Condition2,
BldgType, YearRemodAdd, RoofStyle, RoofMatl, Exterior2nd, 1terCond, BsmtCond, BsmtFinType2, Heating,
Ce1tralAir, GarageQual, GarageCond, 2avedDrive, PoolArea, PoolQC, Fence, MiscFeature, 

"""

def dataProcessing(Data):
	sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
	return sel.fit_transform(Data)


Data1 = pd.read_csv('data/train.csv')
Data2 = pd.read_csv('data/test.csv')

Train = dataProcessing(Data1)

fcols = ['MSSubClass', 'LotArea']

"""X = Data1[fcols]
y = Data1.SalePrice
val_x = Data2[fcols]

reg = DecisionTreeRegressor()
reg.fit(X, y)"""
