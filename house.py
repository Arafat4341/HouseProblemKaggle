import pandas as pd

Data1 = pd.read_csv('data/train.csv')
Data2 = pd.read_csv('data/test.csv')

fcols = ['']


y = Data1.SalePrice