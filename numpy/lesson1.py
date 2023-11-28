import pandas as pd
import numpy as np
i1 = ['1号','2号','3号','4号']
i2 = ['1号','2号','3号','6号']
v1 = [ [10,None],[20,'男'],[30,'男'],[40,'女'] ]
v2 = [1,2,3,6]
c1 = ['年龄','性别']
c2 = ['牌照']
df1 = pd.DataFrame(v1,index=i1,columns=c1)
df2 = pd.DataFrame(v2,index=i2,columns=c2)
print(df1)
print(df2)
df1['加法'] = df1['年龄']+df2['牌照']
print(df1.isnull())

# v1 = [ [10,'女'],[20,'男'],[30,'男'],[40,'女'] ]
# i1 = ['1号','2号','3号','4号']
# c1 = ['年龄','性别']
# v2 = [ [10,'是'],[20,'是'],[30,'否'],[40,'是'] ]
# i2 = ['1号','2号','3号','4号']
# c2 = ['牌照','已婚']
# v3 = [ [50,'男',5,'是'],[60,'女',6,'是'], ]
# i3 = ['5号','6号']
# c3 = ['年龄','性别','牌照','已婚']
# df1 = pd.DataFrame(v1,index=i1,columns=c1)
# df2 = pd.DataFrame(v2,index=i2,columns=c2)
# df3 = pd.DataFrame(v3,index=i3,columns=c3)
# print(df1)
# print(df2)
# print(df3)
# df12 = pd.concat([df1,df2],axis=1)
# print(df12)
# df123 = pd.concat([df12,df3])
# print(df123)