import os
from glob import glob
import pandas as pd


def create_data_table(year, start, end):
	print('\n\n-----------------' + year)
	files_name = glob('raw_data/' + year + '/*.csv')
	files_name.sort()
	stocks_code = [name[name.rfind('_') + 1:-4] for name in files_name]
	days = []
	close_set = []
	for idx, name in enumerate(files_name):
		code = stocks_code[idx].upper()
		df = pd.read_csv(name)
		df = df[['<DTYYYYMMDD>', '<CloseFixed>']]
		df = df.rename(columns={'<DTYYYYMMDD>': 'DTYYYYMMDD', '<CloseFixed>': code})
		try:
			start_idx = df.query("DTYYYYMMDD == " + start).head().index[0]
		except IndexError:
			print('Folder ' + year + ', file ' + code + ' doesnot have start date ' + start)
		try:
			end_idx = df.query("DTYYYYMMDD == " + end).head().index[0]
		except IndexError:
			print('Folder ' + year + ', file ' + code + ' doesnot have end date ' + start)
		if not idx:
			close_set.append(df[['DTYYYYMMDD']][end_idx:start_idx + 1])
		df = df[end_idx:start_idx + 1]
		close_set.append(df[[code]])
		days.append(start_idx - end_idx + 1)

	days = set(days)
	if len(days) > 1:
		print('Something wrong!!!')
		print(days)
	else:
		df = pd.concat(close_set, axis=1)
		df.reset_index(drop=True, inplace=True)
		print(df[:5])
		if not df.isnull().values.any():
			df.to_csv('data/' + year + '.csv')
		else:
			print('****************Missing data at year ' + year)


if __name__=="__main__":
	with open('expiry_date.txt', 'r') as f:
		expiry_date = f.readlines()
	f.close()
	for item in expiry_date:
		year, start, end = item.split()
		create_data_table(year, start, end)