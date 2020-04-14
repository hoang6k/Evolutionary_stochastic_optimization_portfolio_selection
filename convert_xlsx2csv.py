import pandas as pd


path = 'raw_data\\28_HVTC_DHBK_2.2.xlsx'
data_xlsx = pd.read_excel(path, index_col=None, skiprows=1)
# data_xlsx.reindex(index=data.index[::-1])
data_xlsx = data_xlsx.iloc[::-1]
data_xlsx.to_csv('data\\' + path[path.rfind('\\') + 1:-4] + 'csv', index=False)


# data_csv = pd.read_csv('raw_data\\input.csv', index_col=None)
# data_csv = data_csv.iloc[::-1]
# data_csv.to_csv('data\\input.csv', index=False)
