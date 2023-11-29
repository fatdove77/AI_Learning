import pandas as pd
df = pd.read_csv('泰坦尼克.csv',index_col=0)
print(df.head())
print(df.describe())
age = pd.cut(df['年龄'],[0,25,120])
fare = pd.cut(df['费用'],2)
print(age)
print(df.pivot_table('是否生还',index=['船舱等级',fare],columns=['性别',age]))
# arr = df.values
# print(arr)