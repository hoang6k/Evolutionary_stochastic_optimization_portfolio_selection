import pandas as pd

data_xlsx = pd.read_excel('raw_data\\dulieu2018.xlsx', index_col=None)
# data_xlsx.reindex(index=data.index[::-1])
data_xlsx = data_xlsx.iloc[::-1]
data_xlsx.to_csv('data\\dulieu2018.csv', index=False)