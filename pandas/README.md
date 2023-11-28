# 对象的创建

导入pandas import pandas as pd

pandas中以numpy数组的方式存储，在pd中对象就是对象，python中对象叫字典

## 1.1一维对象的创建

### 1.1.1字典创建法

numpy中可以通过np.array()将python列表转化为numpy数组

pandas中可以通过pd.Series()函数，将python字典转化为series对象

```python
dict_v = {'a':0,'b':1}
sr = pd.Series(dict_v)
print(sr)
>>>
a    0
b    1
dtype: int64
```



### 1.1.2数组创建法

直接传值，values(列表，数组，张量)都可以,第二个参数是index（索引，键）

```python
//列表
v = [1,1.1,2.4,1.3]
k = ['a','b','c','d']
sr = pd.Series(v,index = k)
print(sr)
>>>
a    1.0
b    1.1
c    2.4
d    1.3
dtype: float64

//数组
v = np.array([1,1.1,2.4,1.3])
k = ['a','b','c','d']
sr = pd.Series(v,index = k)
print(sr)
>>>
a    1.0
b    1.1
c    2.4
d    1.3
dtype: float64
```



## 1.2一维对象的属性

无论用列表数组还是张量创建，values属性输出的都是数组，所以pandas是建立在numpy基础上创建的库

```python
v = [1,1.1,2.4,1.3]
k = ['a','b','c','d']
sr = pd.Series(v,index = k)
print(sr)
print(sr.values)
print(sr.index)
>>>
a    1.0
b    1.1
c    2.4
d    1.3
dtype: float64
[1.  1.1 2.4 1.3]
Index(['a', 'b', 'c', 'd'], dtype='object')
```





## 1.3而为对象的创建

### 1.3.1字典创建法

每一列是特征 每一行是个体

```python
v1 = [61,15,23,12]
v2 = ['male','female','male','male']
k1 = ['1号','2号','3号','4号']
sr1 = pd.Series(v1,index = k1)
sr2 = pd.Series(v2,index = k1)
print(sr1)
print(sr2)
//两个series的键是一样的
df = pd.DataFrame({'年龄':sr1,'性别':sr2}) //列标签：值
print(df)

>>>
1号    61
2号    15
3号    23
4号    12
dtype: int64
1号      male
2号    female
3号      male
4号      male
dtype: object
    年龄   性别
1号  61  male
2号  15  female
3号  23  male
4号  12  male

//如果有缺失值
v1 = [61,15,23,12]
v2 = ['male','female','male','male']
k1 = ['1号','2号','3号','4号']
k2 = ['1号','2号','3号','6号']
sr1 = pd.Series(v1,index = k1)
sr2 = pd.Series(v2,index = k2)
print(sr1)
print(sr2)
df = pd.DataFrame({'年龄':sr1,'性别':sr2})
print(df)
>>>
      年龄      性别
1号  61.0    male
2号  15.0  female
3号  23.0    male
4号  12.0     NaN
6号   NaN    male

```



### 1.3.2数组创建法

第一个参数values数组，第二个参数行标签index，第三个参数列标签columns

```python
v = np.array([ [34,'male'],[26,'female'],[44,'male'],[24,'male'] ]) //我们之前说过数组只能存储一种数据类型，这里是数组默默把数组转化成了字符串
i = ['1号','2号','3号','4号']
c = ['年龄','性别']
df = pd.DataFrame(v,index=i,columns=c)
print(df)

>>>
    年龄      性别
1号  34    male
2号  26  female
3号  44    male
4号  24    male
```



## 1.4二维对象的属性

```python
print(df.values)
print(df.index)
print(df.columns)
>>>
[['34' 'male']
 ['26' 'female']
 ['44' 'male']
 ['24' 'male']]
Index(['1号', '2号', '3号', '4号'], dtype='object')
Index(['年龄', '性别'], dtype='object')


//提取完整属性
arr = df.values
arr = arr[:,0].astype(int)  //提取第0列
print(arr)
```





# 对象的索引

pandas索引分为显示索引和隐式索引，作者开发显示索引器loc和隐式索引器iloc

## 2.1一维对象的索引

### 2.1.1访问元素

```python
i = ['1号','2号','3号','4号']
v = [21,222,3,4]
sr = pd.Series(v,index=i)
print(sr)
print(sr['1号'])  //这里显示索引和隐式索引不会混淆 所以直接用
>>>
1号     21
2号    222
3号      3
4号      4
dtype: int64
21

//花式索引
print(sr[['1号','3号']])
>>>
1号    21
3号     3
dtype: int64

//修改元素
sr['3号'] = 100
```



```
i = ['1号','2号','3号','4号']
v = [21,222,3,4]
sr = pd.Series(v,index=i)
print(sr)
print(sr[0]) //print(sr.iloc[0])
print(sr[[0,2]])
>>>
1号     21
2号    222
3号      3
4号      4
dtype: int64
21
1号    21
3号     3
dtype: int64

```



### 2.1.2访问切片

```python
i = ['1号','2号','3号','4号']
v = [21,222,3,4]
sr = pd.Series(v,index=i)
print(sr)
print(sr.loc['1号':'3号'])
>>>
1号     21
2号    222
3号      3
4号      4
dtype: int64
1号     21
2号    222
3号      3
dtype: int64



i = ['1号','2号','3号','4号']
v = [21,222,3,4]
sr = pd.Series(v,index=i)
print(sr)
print(sr.iloc[1:3])
>>>
1号     21
2号    222
3号      3
4号      4
dtype: int64
2号    222
3号      3
dtype: int64

```

赋值只是绑定变量，没有开启新的内存空间，如果需要新建内存空间使用.copy()





## 2.2二维对象索引 

### 2.2.1访问元素

```python
//显示索引
i = ['1号','2号','3号','4号']
v1 = [21,222,3,4]
v2 = ['male','female','male','male']
sr1 = pd.Series(v1,index=i)
sr2 = pd.Series(v2,index=i)
df = pd.DataFrame({'年龄':sr1,'性别':sr2})
print(df)
>>>年龄      性别
1号   21    male
2号  222  female
3号    3    male
4号    4    male
#访问元素
print(df.loc['1号','年龄'])
>>>21
print(df.loc[['1号','3号'],['年龄','性别']])
>>> 年龄    性别
1号  21  male
3号   3  male
```

**numpy中花式索引输出的是一个向量，但是pandas的行列标签不能丢失，所以输出一个二维对象**

```python
print(df.iloc[0,0])
print(df.iloc[[0,2],[1,0]])
>>>  年龄      性别
1号   21    male
2号  222  female
3号    3    male
4号    4    male
21
      性别  年龄
1号  male  21
3号  male   3

```



### 2.2.2访问切片

```python
print(df.iloc[0:3,:])
>>>
     年龄      性别
1号   21    male
2号  222  female
3号    3    male
4号    4    male

print(df.loc['1号':'3号',:])
>>>  年龄      性别
1号   21    male
2号  222  female
3号    3    male
```

在显示索引中，提取矩阵的行或列还有一种简编写法

print(df.loc[’1号‘])行

print(df[’年龄’]) 列





# 对象的变形

## 3.1对象的转置

有时候提供的大数据很畸形，行是特征，列是个体，这笔需要进行转置

```python
v = np.array([ [21,22,34,35],['male','female,','male','male'] ])
c = ['1号','2号','3号','4号']
v1 = [21,222,3,4]
i = ['年龄','性别']
df = pd.DataFrame(v,index=i,columns=c)
>> 1号       2号    3号    4号
年龄    21       22    34    35
性别  male  female,  male  male
```

```python
df = df.T
print(df)
>>>年龄       性别
1号  21     male
2号  22  female,
3号  34     male
4号  35     male
```



## 3.2对象的翻转

```python
#左右翻转
print(df.iloc[:,::-1])
>>> 性别  年龄
1号     male  21
2号  female,  22
3号     male  34
4号     male  35

#上下翻转
print(df.iloc[::-1,:])
>>>年龄       性别
4号  35     male
3号  34     male
2号  22  female,
1号  21     male
```



### 3.3对象的重塑

```python
i = ['1号','2号','3号','4号']
v1 = [21,222,3,4]
v2 = ['male','female,','male','male']
v3 = [1,0,1,1]
sr1 = pd.Series(v1,index=i)
sr2 = pd.Series(v2,index=i)
sr3 = pd.Series(v3,index=i)
df = pd.DataFrame({'年龄':sr1,'性别':sr2})
print(df)
df['牌照'] = sr3
print(df)
>>>
 年龄       性别
1号   21     male
2号  222  female,
3号    3     male
4号    4     male
     年龄       性别  牌照
1号   21     male   1
2号  222  female,   0
3号    3     male   1
4号    4     male   1
```

```python
sr4 = df['性别']
print(sr4)
>>>1号       male
2号    female,
3号       male
4号       male
Name: 性别, dtype: object
```



### 3.4对象的拼接

### 3.4.1一维对象合并

pd.concat()与np.concatenate()函数语法相似

```python
i1 = ['1号','2号','3号','4号']
i2 = ['5号','6号','7号']
v1 = [21,222,3,4]
v2 = [12,2,23]
sr1 = pd.Series(v1,index=i1)
sr2 = pd.Series(v2,index=i2)
print(pd.concat([sr1,sr2]))
>>>
1号     21
2号    222
3号      3
4号      4
5号     12
6号      2
7号     23
dtype: int64


sr3 = pd.concat([sr1,sr2])
print(sr3.index.is_unique) //检验索引有没有重复
>>>True
```

### 3.4.2一维对象和二位对象合并

```python
i1 = ['1号','2号','3号','4号']
v1 = ['male','female','male','male']
v2 = [21,222,3,4]
v3 = [1,0,1,1]
sr1 = pd.Series(v1,index=i1)
sr2 = pd.Series(v2,index=i1)
sr3 = pd.Series(v3,index=i1)
df = pd.DataFrame({'性别':sr1,'年龄':sr2})
print(df)
>>> 性别   年龄
1号     male   21
2号  female,  222
3号     male    3
4号     male    4
#添加一列
df['牌照'] = sr3
print(df)
 性别   年龄  牌照
1号    male   21   1
2号  female  222   0
3号    male    3   1
4号    male    4   1
#加上一行
df.loc['4号'] = ['male',23,1]
print(df)
>>> 性别   年龄  牌照
1号    male   21   1
2号  female  222   0
3号    male    3   1
4号    male   23   1
```



### 3.4.3二位对象的合并

二位对象合并依然使用pd.concat()只不过多了一个axis属性

```python
v1 = [ [10,'女'],[20,'男'],[30,'男'],[40,'女'] ]
i1 = ['1号','2号','3号','4号']
c1 = ['年龄','性别']
v2 = [ [10,'是'],[20,'是'],[30,'否'],[40,'是'] ]
i2 = ['1号','2号','3号','4号']
c2 = ['牌照','已婚']
v3 = [ [50,'男',5,'是'],[60,'女',6,'是'], ]
i3 = ['5号','6号']
c3 = ['年龄','性别','牌照','已婚']
df1 = pd.DataFrame(v1,index=i1,columns=c1)
df2 = pd.DataFrame(v2,index=i2,columns=c2)
df3 = pd.DataFrame(v3,index=i3,columns=c3)
print(df1)
print(df2)
print(df3)
>>> 年龄 性别
1号  10  女
2号  20  男
3号  30  男
4号  40  女
    牌照 已婚
1号  10  是
2号  20  是
3号  30  否
4号  40  是
    年龄 性别  牌照 已婚
5号  50  男   5  是
6号  60  女   6  是

#合并列对象
df12 = pd.concat([df1,df2],axis=1)
print(df12)
>>>
年龄 性别  牌照 已婚
1号  10  女  10  是
2号  20  男  20  是
3号  30  男  30  否
4号  40  女  40  是

#合并行对象
df123 = pd.concat([df12,df3])
print(df123)
>>>
 年龄 性别  牌照 已婚
1号  10  女  10  是
2号  20  男  20  是
3号  30  男  30  否
4号  40  女  40  是
5号  50  男   5  是
6号  60  女   6  是
```





# 对象的运算

## 4.1对象系数运算

```python
sr = pd.Series([1,2,3],index=['1号','2号','3号'])
print(sr)
print(sr+10) //对值进行运算
>>>1号    1
2号    2
3号    3
dtype: int64
1号    11
2号    12
3号    13
dtype: int64
```



二维对象运算

```python
i = ['1号','2号','3号','4号']
v1 = [21,222,3,4]
v2 = ['male','female','male','male']
sr1 = pd.Series(v1,index=i)
sr2 = pd.Series(v2,index=i)
df = pd.DataFrame({'年龄':sr1,'性别':sr2})
print(df)
print(df['年龄']+10)
>>>
年龄      性别
1号   21    male
2号  222  female
3号    3    male
4号    4    male
1号     31
2号    232
3号     13
4号     14
Name: 年龄, dtype: int64
```



## 4.2对象与对象之间运算

### 4.2.1一维对象之间运算

```python
i1 = ['1号','2号','3号','4号']
i2 = ['1号','2号','3号']
v1 = [21,222,3,4]
v2 = [1,2,3]
sr1 = pd.Series(v1,index=i1)
sr2 = pd.Series(v2,index=i2)
print(sr1)
print(sr2)
print(sr1+sr2)
>>>1号     21
2号    222
3号      3
4号      4
dtype: int64
1号    1
2号    2
3号    3
dtype: int64
1号     22.0
2号    224.0
3号      6.0
4号      NaN
dtype: float64
```

### 4.2.2 二维对象之间运算

```python
i1 = ['1号','2号','3号','4号']
i2 = ['1号','2号','3号','6号']
v1 = [ [10,'女'],[20,'男'],[30,'男'],[40,'女'] ]
v2 = [1,2,3,6]
c1 = ['年龄','性别']
c2 = ['牌照']
df1 = pd.DataFrame(v1,index=i1,columns=c1)
df2 = pd.DataFrame(v2,index=i2,columns=c2)
print(df1)
print(df2)
>>>年龄 性别
1号  10  女
2号  20  男
3号  30  男
4号  40  女
    牌照
1号   1
2号   2
3号   3
6号   6


#加法
df1['加法'] = df1['年龄']+df2['牌照']
print(df1)
>>>年龄 性别    加法
1号  10  女  11.0
2号  20  男  22.0
3号  30  男  33.0
4号  40  女   NaN


```



# 对象的缺失值

## 5.1 发现缺失值

```python
sr.isnull()
#以布尔型数组存储

#df
>>>年龄    性别    加法
1号  10  None  11.0
2号  20     男  22.0
3号  30     男  33.0
4号  40     女   NaN
df.isnull()
>>>
年龄     性别     加法
1号  False   True  False
2号  False  False  False
3号  False  False  False
4号  False  False   True


```



## 5.2剔除缺失值

.dropna()方法 直接把那一行剔除

.dropna(how='all') 一行全是Nan才剔除



## 5.3填充缺失值

### 5.3.1一维对象

```python
sr.fillna(0)//用0填充
sr.fillna(np.mean(sr)) //用均值填充
sr.fillna(method = 'ffill') //用前值填充
sr.fillna(method='bfill') //后值填充
```

### 5.3.2

```python
dr.fillna(0)//用0填充
dr.fillna(np.mean(df)) //用均值填充 nanna'y
dr.fillna(method = 'ffill') //用前值填充
dr.fillna(method='bfill') //后值填充
```





