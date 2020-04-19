import pandas as pd

path = 'result/result_dulieudetai_markovitz.csv'
df = pd.read_csv(path)
print(df.iloc[:,0])
df = df.round({'lambda': 4})
print(df.iloc[:,0])
df.to_csv(path, index=False)