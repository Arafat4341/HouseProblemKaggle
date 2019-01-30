import pandas as pd
from sklearn.feature_selection import VarianceThreshold

def dataProcessing():
	sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
	sel.fit_transform(Data1)
	print(Data1.shape)

Data1 = pd.read_csv('data/train.csv')
Data2 = pd.read_csv('data/test.csv')
y = Data1.SalePrice

dataProcessing()
